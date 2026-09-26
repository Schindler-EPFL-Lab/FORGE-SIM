from collections.abc import Generator
from pathlib import Path

import drjit as dr
import mitsuba as mi

from rebel_shape.data_processing.masked_pose_image import MaskedPoseImage
from rebel_shape.data_processing.pose_scene_loader import PoseSceneLoader
from rebel_shape.data_processing.vggt_pose_scene_loader import VGGTPoseSceneLoader
from rebel_shape.plugin_wrappers.perspective_camera import PerspectiveCamera
from rebel_shape.plugin_wrappers.sampler import Sampler

_COLOUR_WITH_MASK_FORMATS = frozenset(
    {mi.Bitmap.PixelFormat.RGBA, mi.Bitmap.PixelFormat.XYZA}
)
"""
Source pixel formats a scene image may have: three colour channels plus the alpha
channel that carries the object mask.

`has_alpha()` alone is not enough, because it also holds for `PixelFormat.YA`. A
grey+alpha source survives the conversion to RGBA (the grey value is replicated into
R, G and B) and would then be optimized against as if it were a colour photograph.
"""


class ImageSceneLoader:
    """
    Loads RGB, object masks, foreground masks along with their associated camera poses.
    They are stored in separate lists and the order is guaranteed to match. They are
    further split into training and evaluation sets. Composites RGB images with the
    rendered background using object masks.

    RGB images are assumed to be sRGB gamma corrected. Foreground masks are assumed to
    be 8-bit linear grayscale images.

    The images are of type `mi.TensorXf` since this is the native format for Mitsuba
    operations (gradient computation, etc.), while the camera information and pose is
    encapsulated in `PerspectiveCamera`.
    """

    def __init__(
        self,
        scene_path: Path,
        pose_scene_loader: PoseSceneLoader,
        use_foreground_masks: bool = False,
        environment_map_path: Path | None = None,
        output_dir: Path | None = None,
        spp: int = 16,
    ) -> None:
        """
        Loads the ThermoScenes images, foreground masks, and cameras poses from the
        directory `scene_path`. Composites RGB images with the background rendered from
        `environment_map_path` using `spp` samples per pixel. Choosing a higher value
        than the default (`16`) has limited benefit. If the path is not provided, a
        solid white background is used. Optionally stores the composited images in
        `output_dir` if provided.

        The camera pixel format (`RGB` vs `LUMINANCE` for mono variants) is taken
        from `pose_scene_loader`; configure it there before passing in.

        :raises FileNotFoundError: If the scene path does not exist.
        :raises RuntimeError: If a presented image is not a colour image carrying its
            object mask in an alpha channel.
        """

        self._pose_loader = pose_scene_loader
        self._spp = spp
        self._use_foreground_masks = use_foreground_masks

        self._train_images: list[MaskedPoseImage] = []
        self._eval_images: list[MaskedPoseImage] = []

        background_scene = self._create_background_scene(environment_map_path)

        # Iterate through frames from transform_data
        for frame in self._pose_loader._transform_data.get("frames", []):
            frame = self._pose_loader.get_frame(frame=frame)

            rgb_image_filename = Path(frame["file_path"])
            image_rgb_path = self._pose_loader._scene_path / rgb_image_filename

            foreground_mask_path: Path | None = None
            if "foreground_mask_file_path" in frame:
                foreground_mask_path = self._pose_loader._scene_path / Path(
                    frame["foreground_mask_file_path"]
                )
            if not image_rgb_path.exists():
                raise RuntimeError(
                    f"Image {image_rgb_path} referenced in transform.json "
                    + "does not exist."
                )

            if foreground_mask_path is not None and not foreground_mask_path.exists():
                raise RuntimeError(
                    f"Foreground mask {foreground_mask_path} referenced in "
                    + "transform.json is missing in dataset."
                )

            # The alpha channel carries the object mask. `convert()` synthesises an
            # opaque alpha for a source that has none and replicates a grey value into
            # R, G and B, so both an absent mask and a greyscale source can only be
            # detected on the source bitmap: the converted image always reports four
            # colour channels.
            image_bitmap = mi.Bitmap(str(image_rgb_path))
            if image_bitmap.pixel_format() not in _COLOUR_WITH_MASK_FORMATS:
                raise RuntimeError(
                    f"Object mask image is not found in {image_rgb_path}. Its pixel "
                    f"format is {image_bitmap.pixel_format().name}, which is not a "
                    "colour format carrying an alpha channel to use as the object "
                    "mask."
                )

            mitsuba_image_rgba = mi.TensorXf(
                image_bitmap.convert(
                    pixel_format=mi.Bitmap.PixelFormat.RGBA,
                    component_format=mi.Struct.Type.Float32,
                    srgb_gamma=True,
                )
            )

            mitsuba_foreground_mask = mi.TensorXf(
                dr.zeros_like(mitsuba_image_rgba[:, :, 0:1])
            )
            if foreground_mask_path is not None:
                # Linear 8-bit grayscale: Mitsuba auto-tags an 8-bit PNG/JPEG as sRGB
                # on bit depth alone, so `srgb_gamma=False` on `convert()` alone still
                # applies a bogus sRGB->linear decode (byte 128 -> 0.216 instead of
                # 0.502). Mark the source linear first so values are read verbatim.
                foreground_mask_bitmap = mi.Bitmap(str(foreground_mask_path))
                foreground_mask_bitmap.set_srgb_gamma(False)
                mitsuba_foreground_mask = mi.TensorXf(
                    foreground_mask_bitmap.convert(
                        component_format=mi.Struct.Type.Float32, srgb_gamma=False
                    )
                )
            if mitsuba_foreground_mask.shape[-1] != 1:
                raise RuntimeError(
                    f"Foreground mask image {foreground_mask_path} is not a "
                    + "single channel grayscale image."
                )

            is_train = False
            if (
                len(self._train_images) < len(self._pose_loader.train_filenames)
                and self._pose_loader.train_filenames[len(self._train_images)]
                == rgb_image_filename.name
            ):
                is_train = True
                camera = self._pose_loader.train_cameras[len(self._train_images)]
            elif (
                len(self._eval_images) < len(self._pose_loader.eval_filenames)
                and self._pose_loader.eval_filenames[len(self._eval_images)]
                == rgb_image_filename.name
            ):
                camera = self._pose_loader.eval_cameras[len(self._eval_images)]
            else:
                raise ValueError(
                    f"Image {rgb_image_filename.name} does not specify whether "
                    "it is a training ('train') or evaluation ('eval') image"
                )

            object_mask = mitsuba_image_rgba[:, :, 3:4]

            composited_image_masked_pose = self._generate_image(
                background_scene=background_scene,
                camera=camera,
                mitsuba_foreground_mask=mitsuba_foreground_mask,
                mitsuba_image_rgba=mitsuba_image_rgba,
                object_mask=object_mask,
            )

            if output_dir is not None:
                bitmap = mi.Bitmap(composited_image_masked_pose.original_image)
                filename = f"composited_frame_{Path(rgb_image_filename).stem}.exr"
                bitmap.write(str(output_dir / filename))

            # Separate into train and eval based on the filename
            if is_train:
                self._train_images.append(composited_image_masked_pose)
            else:
                self._eval_images.append(composited_image_masked_pose)

    def _generate_image(
        self,
        background_scene: "mi.Scene",
        camera: PerspectiveCamera,
        mitsuba_image_rgba: "mi.TensorXf",
        object_mask: "mi.TensorXf",
        mitsuba_foreground_mask: "mi.TensorXf",
    ) -> MaskedPoseImage:
        # render the background and apply it for the image
        background = mi.render(
            scene=background_scene, sensor=camera.mitsuba_sensor, spp=self._spp
        )

        image = mitsuba_image_rgba[:, :, 0:3]
        composited_image = image * object_mask + background * (1.0 - object_mask)

        # optionally apply the foreground mask
        if self._use_foreground_masks:
            used_foreground_mask = 1.0 - mitsuba_foreground_mask
        else:
            used_foreground_mask = dr.ones_like(mitsuba_foreground_mask)
        return MaskedPoseImage(
            image=composited_image,
            camera=camera,
            mask=used_foreground_mask,
        )

    @classmethod
    def from_thermoscene(
        cls,
        scene_path: Path,
        sensor_sampler: Sampler = Sampler(),
        use_foreground_masks: bool = False,
        align_cameras_xz: bool = True,
        environment_map_path: Path | None = None,
        output_dir: Path | None = None,
        spp: int = 16,
        pixel_format: PerspectiveCamera.PixelFormat = PerspectiveCamera.PixelFormat.RGB,
    ) -> "ImageSceneLoader":
        """
        Loads the ThermoScenes images, foreground masks, and cameras poses from the
        directory `scene_path`. Optionally aligns camera poses so they lie on the
        xz-plane if `align_cameras_xz` is `True`. Each camera has `sensor_samples` for
        rays sampling. Composites RGB images with the background rendered from
        `environment_map_path` using `spp` samples per pixel. Choosing a higher value
        than the default (`16`) has limited benefit. If the path is not provided, a
        solid white background is used. Optionally stores the composited images in
        `output_dir` if provided.

        :raises FileNotFoundError: If the scene path does not exist.
        :raises RuntimeError: If a presented image is not a colour image carrying its
            object mask in an alpha channel.
        """
        pose_loader = PoseSceneLoader(
            scene_path=scene_path,
            sensor_sampler=sensor_sampler,
            align_cameras_xz=align_cameras_xz,
            pixel_format=pixel_format,
        )

        return cls(
            scene_path,
            pose_loader,
            use_foreground_masks,
            environment_map_path,
            output_dir,
            spp,
        )

    @classmethod
    def from_vggt(
        cls,
        scene_path: Path,
        sensor_sampler: Sampler = Sampler(),
        use_foreground_masks: bool = False,
        align_cameras_xz: bool = True,
        environment_map_path: Path | None = None,
        output_dir: Path | None = None,
        spp: int = 16,
        pixel_format: PerspectiveCamera.PixelFormat = PerspectiveCamera.PixelFormat.RGB,
    ) -> "ImageSceneLoader":
        """
        Loads the ThermoScenes images, foreground masks, and cameras poses from the
        directory `scene_path`. Optionally aligns camera poses so they lie on the
        xz-plane if `align_cameras_xz` is `True`. Each camera has `sensor_samples` for
        rays sampling. Composites RGB images with the background rendered from
        `environment_map_path` using `spp` samples per pixel. Choosing a higher value
        than the default (`16`) has limited benefit. If the path is not provided, a
        solid white background is used. Optionally stores the composited images in
        `output_dir` if provided.

        :raises FileNotFoundError: If the scene path does not exist.
        :raises RuntimeError: If a presented image is not a colour image carrying its
            object mask in an alpha channel.
        """
        pose_loader = VGGTPoseSceneLoader(
            scene_path=scene_path,
            sensor_sampler=sensor_sampler,
            align_cameras_xz=align_cameras_xz,
            pixel_format=pixel_format,
        )

        return cls(
            scene_path,
            pose_loader,
            use_foreground_masks,
            environment_map_path,
            output_dir,
            spp,
        )

    @staticmethod
    def _create_background_scene(
        environment_map_path: Path | None,
    ) -> "mi.Scene":
        """
        Builds a scene which consists of only the environment_map which is loaded from
        `environment_map_path`. It is useful for background generation.

        :raises FileNotFoundError if the environment map file cannot be found
        """
        if environment_map_path is not None and not environment_map_path.exists():
            raise FileNotFoundError(
                f"Environment map not found at {environment_map_path}"
            )

        environment_map = (
            {"type": "envmap", "filename": str(environment_map_path)}
            if environment_map_path is not None
            else {
                "type": "constant",
                "radiance": {
                    "type": "rgb",
                    "value": 1.0,
                },
            }
        )

        # Create a simple scene with just the environment map
        scene_dict = {
            "type": "scene",
            "integrator": {"type": "path"},
            "environment": environment_map,
        }
        return mi.load_dict(scene_dict)

    @property
    def tf_cameras(self) -> "mi.ScalarTransform4f":
        """Return the transformation which were applied to camera poses"""
        return self._pose_loader.tf_cameras

    @property
    def train_images(self) -> list[MaskedPoseImage]:
        """List of images for the training"""
        return self._train_images

    @property
    def train_cameras(self) -> list[PerspectiveCamera]:
        """List of camera poses for the training set with RGB pixel format"""
        return self._pose_loader.train_cameras

    @property
    def train_data(
        self,
    ) -> Generator[
        tuple[str, MaskedPoseImage],
        None,
        None,
    ]:
        """
        Yields tuples of (image filename, image) for training data.
        """
        for (filename, _), image in zip(
            self._pose_loader.train_data, self._train_images
        ):
            yield filename, image

    @property
    def eval_cameras(self) -> list[PerspectiveCamera]:
        """List of camera poses for the evaluation set with RGB pixel format"""
        return self._pose_loader.eval_cameras

    @property
    def eval_images(self) -> list[MaskedPoseImage]:
        """List of images for the training"""
        return self._eval_images

    @property
    def eval_data(
        self,
    ) -> Generator[
        tuple[str, MaskedPoseImage],
        None,
        None,
    ]:
        """
        Yields tuples of (image filename, image) for eval data.
        """
        for (filename, _), image in zip(self._pose_loader.eval_data, self._eval_images):
            yield filename, image
