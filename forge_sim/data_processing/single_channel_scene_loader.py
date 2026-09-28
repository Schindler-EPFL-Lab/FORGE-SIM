import json
from pathlib import Path

import mitsuba as mi

from forge_sim import logger
from forge_sim.data_processing.masked_pose_image import MaskedPoseImage
from forge_sim.data_processing.pose_scene_loader import PoseSceneLoader
from forge_sim.data_processing.vggt_pose_scene_loader import VGGTPoseSceneLoader
from forge_sim.plugin_wrappers.perspective_camera import PerspectiveCamera
from forge_sim.plugin_wrappers.sampler import Sampler


class SingleChannelImageSceneLoader:
    """
    Loads single channel images and their associated camera poses. They are stored in
    separate lists and the order is guaranteed to match. Images are further split into
    training and evaluation sets.

    No gamma correction is applied to loaded single channel images. The images are of
    type `mi.TensorXf` since this is the native format for Mitsuba operations (gradient
    computation, etc.), while the camera information and pose is encapsulated in
    `PerspectiveCamera`.

    If a single-channel image is stored with 3 or 4 channels (RGB/RGBA), it is converted
    to luminance using linear Rec.709 weights (0.2126, 0.7152, 0.0722). This matches the
    linear loading path (`srgb_gamma=False`) used for scalar images.
    """

    def __init__(
        self,
        scene_path: Path,
        image_folder: str,
        image_key: str,
        sensor_sampler: Sampler = Sampler(),
        align_cameras_xz: bool = True,
        pose_loader_cls: type[PoseSceneLoader] = PoseSceneLoader,
        allow_missing_images: bool = False,
    ):
        """
        Loads the ThermoScenes single-channel images and associated camera poses from
        `scene_path`/`image_folder`. The `image_key` for each single-channel image must
        be included in every frame of `transforms.json`, and its value should be the
        relative path to the corresponding single-channel image. For example, the
        `image_key` for thermal images is `thermal_file_path`.

        Optionally aligns camera poses to the xy-plane if `align_cameras_xz` is set to
        `True`. Each camera has `sensor_samples` for rays sampling.

        Every frame is required to carry a single-channel image. Set
        `allow_missing_images` to `True` to instead skip, with a warning naming the
        frame, the frames whose `image_key` is missing, blank, or refers to a file
        absent from `image_folder`; this shrinks the train/eval splits and is meant
        for VGGT-style datasets where only a subset of frames carry single-channel
        annotations.

        :raises FileNotFoundError: If `scene_path` or `image_folder` does not exist.
        :raises KeyError: If a frame has no `image_key` entry, unless
            `allow_missing_images` is set.
        :raises RuntimeError: If a frame's `image_key` entry is blank, or references a
            single-channel image that is absent from `image_folder`, unless
            `allow_missing_images` is set.
        :raises ValueError: If train/eval split inference is inconsistent with
            `transforms.json` ordering.
        """
        self._pose_loader = pose_loader_cls(
            scene_path=scene_path,
            sensor_sampler=sensor_sampler,
            align_cameras_xz=align_cameras_xz,
        )

        self._train_images: list[MaskedPoseImage] = []
        self._eval_images: list[MaskedPoseImage] = []

        semantic_folder = self._pose_loader._scene_path / image_folder
        image_paths = self._pose_loader._load_image_paths_from_folder(semantic_folder)

        # Indices into the pose loader's per-split lists. They advance for every frame
        # we see in transforms.json so that they remain in sync with the pose loader's
        # enumeration even when we skip a frame for lack of a single-channel image.
        train_idx = 0
        eval_idx = 0

        for frame in self._pose_loader._transform_data.get("frames", []):
            frame = self._pose_loader.get_frame(frame=frame)
            rgb_image_filename = Path(frame["file_path"]).name

            if (
                train_idx < len(self._pose_loader.train_filenames)
                and self._pose_loader.train_filenames[train_idx] == rgb_image_filename
            ):
                is_train = True
                camera = self._pose_loader.train_cameras[train_idx]
                train_idx += 1
            elif (
                eval_idx < len(self._pose_loader.eval_filenames)
                and self._pose_loader.eval_filenames[eval_idx] == rgb_image_filename
            ):
                is_train = False
                camera = self._pose_loader.eval_cameras[eval_idx]
                eval_idx += 1
            else:
                raise ValueError(
                    f"Image {rgb_image_filename} does not specify whether "
                    "it is a training (`train`) or evaluation (`eval`) image, "
                    "or the ordering between `transforms.json` and the split "
                    "filenames is inconsistent."
                )

            image_ref = frame.get(image_key)
            if image_ref is None or image_ref == "":
                if not allow_missing_images:
                    # An absent key used to fail on the lookup itself, while a blank
                    # one reached the missing-image check below with an empty
                    # filename. Both stay hard failures, of their original types.
                    if image_ref is None:
                        raise KeyError(
                            f"Frame {rgb_image_filename} has no '{image_key}' entry "
                            "in transforms.json."
                        )
                    raise RuntimeError(
                        f"Frame {rgb_image_filename} has a blank '{image_key}' entry "
                        "in transforms.json."
                    )
                logger.warning(
                    f"Frame {rgb_image_filename} has no '{image_key}' entry; "
                    "skipping single-channel image for this view."
                )
                continue

            image_filename = Path(str(image_ref)).name
            if image_filename not in image_paths:
                if not allow_missing_images:
                    raise RuntimeError(
                        f"Single-channel image {image_filename} (from '{image_key}') "
                        "referenced in transforms.json is missing in "
                        f"{semantic_folder}."
                    )
                logger.warning(
                    f"Single-channel image {image_filename} (from '{image_key}') "
                    f"referenced in transforms.json is missing in {semantic_folder}; "
                    "skipping this view."
                )
                continue

            # Linear scalar fields (FLIR thermal, masks): an 8-bit PNG/JPEG is
            # auto-tagged sRGB by Mitsuba on bit depth alone, so a plain convert()
            # would apply a bogus sRGB->linear decode that darkens every mid-tone.
            # Mark the source linear first so the stored values are read verbatim.
            bitmap = mi.Bitmap(str(image_paths[image_filename]))
            bitmap.set_srgb_gamma(False)
            mitsuba_image = mi.TensorXf(
                bitmap.convert(
                    component_format=mi.Struct.Type.Float32, srgb_gamma=False
                )
            )
            if mitsuba_image.ndim == 2:
                mitsuba_image = mitsuba_image[:, :, None]
            if mitsuba_image.shape[-1] >= 3:
                # Convert multi-channel inputs (RGB/RGBA grayscales) to luminance.
                rgb = mitsuba_image[:, :, :3]
                mitsuba_image = (
                    rgb[:, :, 0:1] * 0.2126
                    + rgb[:, :, 1:2] * 0.7152
                    + rgb[:, :, 2:3] * 0.0722
                )
            elif mitsuba_image.shape[-1] != 1:
                # Two channels: a grayscale image with alpha, whose first channel is
                # the value we want.
                mitsuba_image = mitsuba_image[:, :, 0:1]

            camera.pixel_format = PerspectiveCamera.PixelFormat.LUMINANCE
            masked_pose_image = MaskedPoseImage(
                image=mitsuba_image, camera=camera, mask=None
            )
            if is_train:
                self._train_images.append(masked_pose_image)
            else:
                self._eval_images.append(masked_pose_image)

        # Ensure every pose-loader camera reports luminance, including those whose
        # frame had no single-channel image (used for evaluation/rendering elsewhere).
        for camera in self._pose_loader.train_cameras + self._pose_loader.eval_cameras:
            camera.pixel_format = PerspectiveCamera.PixelFormat.LUMINANCE

    @classmethod
    def thermal_scene(
        cls,
        scene_path: Path,
        sensor_sampler: Sampler = Sampler(),
        align_cameras_xz: bool = True,
    ) -> "SingleChannelImageSceneLoader":
        """
        Instantiates the class when the single-channel images are thermal images.
        Thermal images are assumed to be located in the `thermal` folder. The
        `image_key` is `thermal_file_path`.
        """
        return cls(
            image_folder="thermal",
            image_key="thermal_file_path",
            scene_path=scene_path,
            sensor_sampler=sensor_sampler,
            align_cameras_xz=align_cameras_xz,
        )

    @classmethod
    def thermal_scene_vggt(
        cls,
        scene_path: Path,
        sensor_sampler: Sampler = Sampler(),
        align_cameras_xz: bool = True,
    ) -> "SingleChannelImageSceneLoader":
        """
        Instantiates the class for VGGT-style scenes with thermal images. Only a
        subset of the frames of such a scene carries a thermal image, so the frames
        without one are skipped with a warning rather than rejected.
        """
        return cls(
            image_folder="thermal",
            image_key="thermal_file_path",
            scene_path=scene_path,
            sensor_sampler=sensor_sampler,
            align_cameras_xz=align_cameras_xz,
            pose_loader_cls=VGGTPoseSceneLoader,
            allow_missing_images=True,
        )

    @classmethod
    def windows_masks_scene(
        cls,
        scene_path: Path,
        sensor_sampler: Sampler = Sampler(),
        align_cameras_xz: bool = True,
    ) -> "SingleChannelImageSceneLoader":
        """
        Instantiates the class when the single-channel images are windows mask images.
        Windows mask images are assumed to be located in the `windows_masks` folder. The
        `image_key` is `windows_mask_file_path`.
        """
        return cls(
            image_folder="windows_masks",
            image_key="windows_mask_file_path",
            scene_path=scene_path,
            sensor_sampler=sensor_sampler,
            align_cameras_xz=align_cameras_xz,
        )

    @staticmethod
    def _select_vggt_masks_source(scene_path: Path) -> tuple[str, str]:
        """
        Detect which VGGT facade mask key and folder to use.

        The function inspects the first frame under `frames[*].rgb` and returns a tuple
        of `(image_key, image_folder)`.

        Supported keys (in order):
        - `windows_mask_file_path`
        - `masks_facade`
        - `masks_facades`

        The returned folder is resolved from the value path (e.g. `masks/foo.png`
        -> `masks`). If the value path has no parent folder, `windows_masks` is used
        for `windows_mask_file_path` and `masks` otherwise.

        :raises ValueError: If `transforms.json` is missing/empty, not VGGT-formatted,
            or no supported mask key is present.
        """
        transforms_path = scene_path / "transforms.json"
        with open(transforms_path, "r", encoding="utf-8") as file:
            transforms_data = json.load(file)

        frames = transforms_data.get("frames", [])
        if not frames:
            raise ValueError("transforms.json has no frames to inspect for masks.")

        if not isinstance(frames[0], dict) or "rgb" not in frames[0]:
            raise ValueError("transforms.json does not appear to be VGGT format.")

        candidate_keys = [
            "windows_mask_file_path",
            "masks_facade",
            "masks_facades",
        ]
        for frame in frames:
            rgb_frame = frame.get("rgb", {})
            for key in candidate_keys:
                if key in rgb_frame and isinstance(rgb_frame[key], str):
                    value_path = Path(rgb_frame[key])
                    if value_path.parent.name:
                        return key, value_path.parent.name
                    if key == "windows_mask_file_path":
                        return key, "windows_masks"
                    return key, "masks"
        raise ValueError(
            "Could not resolve a facade mask key in VGGT transforms.json. "
            "Expected one of: windows_mask_file_path, masks_facade, masks_facades."
        )

    @classmethod
    def windows_masks_scene_vggt(
        cls,
        scene_path: Path,
        sensor_sampler: Sampler = Sampler(),
        align_cameras_xz: bool = True,
    ) -> "SingleChannelImageSceneLoader":
        """
        Instantiates the class for VGGT-style scenes with windows masks. Only a
        subset of the frames of such a scene carries a mask, so the frames without
        one are skipped with a warning rather than rejected.
        """
        image_key, image_folder = cls._select_vggt_masks_source(scene_path)
        return cls(
            image_folder=image_folder,
            image_key=image_key,
            scene_path=scene_path,
            sensor_sampler=sensor_sampler,
            align_cameras_xz=align_cameras_xz,
            pose_loader_cls=VGGTPoseSceneLoader,
            allow_missing_images=True,
        )

    @staticmethod
    def _is_vggt_transforms(scene_path: Path) -> bool:
        """
        Return True if `transforms.json` appears to be VGGT-style.

        VGGT datasets store per-frame data under `frames[*].rgb`, while ThermoScenes
        uses flat frame objects. This helper inspects the first frame to decide.

        :raises ValueError: If `transforms.json` exists but is not valid JSON.
        """
        transforms_path = scene_path / "transforms.json"
        with open(transforms_path, "r", encoding="utf-8") as file:
            transforms_data = json.load(file)
        frames = transforms_data.get("frames", [])
        if not frames:
            return False
        first_frame = frames[0]
        return isinstance(first_frame, dict) and "rgb" in first_frame

    @classmethod
    def thermal_scene_auto(
        cls,
        scene_path: Path,
        sensor_sampler: Sampler = Sampler(),
        align_cameras_xz: bool = True,
    ) -> "SingleChannelImageSceneLoader":
        """Instantiates the class for ThermoScenes or VGGT thermal scenes."""
        if cls._is_vggt_transforms(scene_path):
            return cls.thermal_scene_vggt(
                scene_path=scene_path,
                sensor_sampler=sensor_sampler,
                align_cameras_xz=align_cameras_xz,
            )
        return cls.thermal_scene(
            scene_path=scene_path,
            sensor_sampler=sensor_sampler,
            align_cameras_xz=align_cameras_xz,
        )

    @classmethod
    def windows_masks_scene_auto(
        cls,
        scene_path: Path,
        sensor_sampler: Sampler = Sampler(),
        align_cameras_xz: bool = True,
    ) -> "SingleChannelImageSceneLoader":
        """Instantiates the class for ThermoScenes or VGGT windows mask scenes."""
        if cls._is_vggt_transforms(scene_path):
            return cls.windows_masks_scene_vggt(
                scene_path=scene_path,
                sensor_sampler=sensor_sampler,
                align_cameras_xz=align_cameras_xz,
            )
        return cls.windows_masks_scene(
            scene_path=scene_path,
            sensor_sampler=sensor_sampler,
            align_cameras_xz=align_cameras_xz,
        )

    @property
    def train_images(self) -> list[MaskedPoseImage]:
        """List of images from the train data split"""
        return self._train_images

    @property
    def eval_images(self) -> list[MaskedPoseImage]:
        """List of images from the eval data split"""
        return self._eval_images
