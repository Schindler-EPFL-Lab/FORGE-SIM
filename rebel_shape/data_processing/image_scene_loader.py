from pathlib import Path
from typing import Generator

import drjit as dr
import mitsuba as mi

from rebel_shape.data_processing.masked_pose_image import MaskedPoseImage
from rebel_shape.data_processing.pose_scene_loader import PoseSceneLoader
from rebel_shape.plugin_wrappers.perspective_camera import PerspectiveCamera
from rebel_shape.plugin_wrappers.sampler import Sampler


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
        sensor_sampler: Sampler = Sampler(),
        use_foreground_masks: bool = False,
        align_cameras_xz: bool = True,
        environment_map_path: Path | None = None,
        output_dir: Path | None = None,
        spp: int = 16,
    ) -> None:
        """
        Loads the ThermoScenes images, foreground masks, and cameras poses from the
        directory `scene_path`. Optionally aligns camera poses so they lie on the
        xy-plane if `align_cameras_xz` if `True`. Each camera has `sensor_samples` for
        rays sampling. Composites RGB images with the background rendered from
        `environment_map_path` using `spp` samples per pixel. Choosing a higher value
        than the default (`16`) has limited benefit. If the path is not provided, a
        solid white background is used. Optionally stores the composited images in
        `output_dir` if provided.

        :raises FileNotFoundError: If the scene path does not exist.
        :raises RuntimeError: If the presented images does not have object masks as
            alpha channels.
        """

        self._pose_loader = PoseSceneLoader(
            scene_path=scene_path,
            sensor_sampler=sensor_sampler,
            align_cameras_xz=align_cameras_xz,
        )

        self._train_images: list[MaskedPoseImage] = []
        self._eval_images: list[MaskedPoseImage] = []

        rgb_folder = self._pose_loader._scene_path / "images"
        foreground_mask_folder = self._pose_loader._scene_path / "foreground_masks"

        image_rgb_paths = self._pose_loader._load_image_paths_from_folder(rgb_folder)
        foreground_mask_paths = self._pose_loader._load_image_paths_from_folder(
            foreground_mask_folder
        )

        background_scene = self._create_background_scene(environment_map_path)

        # Iterate through frames from transform_data
        for frame in self._pose_loader._transform_data.get("frames", []):
            rgb_image_filename = Path(frame["file_path"]).name
            foreground_mask_filename = (
                Path(frame["foreground_mask_file_path"]).name
                if foreground_mask_folder.exists()
                else None
            )

            if rgb_image_filename not in image_rgb_paths:
                raise RuntimeError(
                    f"Image {rgb_image_filename} referenced in transform.json "
                    + "is missing in dataset."
                )

            if (
                foreground_mask_filename is not None
                and foreground_mask_filename not in foreground_mask_paths
            ):
                raise RuntimeError(
                    f"Foreground mask {foreground_mask_filename} referenced in "
                    + "transform.json is missing in dataset."
                )

            image_rgb_path = image_rgb_paths[rgb_image_filename]
            mitsuba_image_rgba = mi.TensorXf(
                mi.Bitmap(str(image_rgb_path)).convert(
                    component_format=mi.Struct.Type.Float32, srgb_gamma=True
                )
            )
            if mitsuba_image_rgba.shape[-1] != 4:
                raise RuntimeError(
                    f"Object mask image is not found in {image_rgb_path}"
                )

            mitsuba_foreground_mask = mi.TensorXf(
                dr.zeros_like(mitsuba_image_rgba[:, :, 0:1])
            )
            if foreground_mask_filename is not None:
                foreground_mask_path = foreground_mask_paths[foreground_mask_filename]
                mitsuba_foreground_mask = mi.TensorXf(
                    mi.Bitmap(str(foreground_mask_path)).convert(
                        component_format=mi.Struct.Type.Float32, srgb_gamma=False
                    )
                )
            if mitsuba_foreground_mask.shape[-1] != 1:
                raise RuntimeError(
                    f"Foreground mask image {foreground_mask_filename} is not a "
                    + "single channel grayscale image."
                )

            is_train = False
            if (
                len(self._train_images) < len(self._pose_loader.train_filenames)
                and self._pose_loader.train_filenames[len(self._train_images)]
                == rgb_image_filename
            ):
                is_train = True
                camera = self._pose_loader.train_cameras[len(self._train_images)]
            elif (
                len(self._eval_images) < len(self._pose_loader.eval_filenames)
                and self._pose_loader.eval_filenames[len(self._eval_images)]
                == rgb_image_filename
            ):
                camera = self._pose_loader.eval_cameras[len(self._eval_images)]
            else:
                raise ValueError(
                    f"Image {rgb_image_filename} does not specify whether "
                    "it is a training ('train') or evaluation ('eval') image"
                )

            # render the background and apply it for the image
            background = mi.render(
                scene=background_scene, sensor=camera.mitsuba_sensor, spp=spp
            )

            image = mitsuba_image_rgba[:, :, 0:3]
            object_mask = mitsuba_image_rgba[:, :, 3:4]
            composited_image = image * object_mask + background * (1.0 - object_mask)

            if output_dir is not None:
                bitmap = mi.Bitmap(composited_image)
                filename = f"composited_frame_{Path(rgb_image_filename).stem}.exr"
                bitmap.write(str(output_dir / filename))

            # optionally apply the foreground mask
            if use_foreground_masks:
                used_foreground_mask = 1.0 - mitsuba_foreground_mask
            else:
                used_foreground_mask = dr.ones_like(mitsuba_foreground_mask)
            composited_image_masked_pose = MaskedPoseImage(
                image=composited_image,
                camera=camera,
                mask=used_foreground_mask,
            )

            # Separate into train and eval based on the filename
            if is_train:
                self._train_images.append(composited_image_masked_pose)
            else:
                self._eval_images.append(composited_image_masked_pose)

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
