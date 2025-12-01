from pathlib import Path

import mitsuba as mi

from rebel_shape.data_processing.masked_pose_image import MaskedPoseImage
from rebel_shape.data_processing.pose_scene_loader import PoseSceneLoader
from rebel_shape.plugin_wrappers.perspective_camera import PerspectiveCamera
from rebel_shape.plugin_wrappers.sampler import Sampler


class SingleChannelImageSceneLoader:
    """
    Loads single channel images and their associated camera poses. They are stored in
    separate lists and the order is guaranteed to match. Images are further split into
    training and evaluation sets.

    No gamma correction is applied to loaded single channel images. The images are of
    type `mi.TensorXf` since this is the native format for Mitsuba operations (gradient
    computation, etc.), while the camera information and pose is encapsulated in
    `PerspectiveCamera`.
    """

    def __init__(
        self,
        scene_path: Path,
        image_folder: str,
        image_key: str,
        sensor_sampler: Sampler = Sampler(),
        align_cameras_xz: bool = True,
    ):
        """
        Loads the ThermoScenes single-channel images and associated camera poses from
        `scene_path`/`image_folder`. The `image_key` for each single-channel image must
        be included in every frame of `transforms.json`, and its value should be the
        relative path to the corresponding single-channel image. For example, the
        `image_key` for thermal images is `thermal_file_path`.

        Optionally aligns camera poses to the xy-plane if `align_cameras_xz` is set to
        `True`. Each camera has `sensor_samples` for rays sampling.

        :raises `FileNotFoundError`** If `scene_path` or `image_folder` does not exist.
        :raises `RuntimeError`** If any frame in `transforms.json` does not include
        `image_key`.
        """
        self._pose_loader = PoseSceneLoader(
            scene_path=scene_path,
            sensor_sampler=sensor_sampler,
            align_cameras_xz=align_cameras_xz,
        )

        self._train_images: list[MaskedPoseImage] = []
        self._eval_images: list[MaskedPoseImage] = []

        semantic_folder = self._pose_loader._scene_path / image_folder
        image_paths = self._pose_loader._load_image_paths_from_folder(semantic_folder)

        # Explicit indices into pose loader splits
        train_idx = 0
        eval_idx = 0

        # Iterate through frames from transform_data
        for frame in self._pose_loader._transform_data.get("frames", []):
            rgb_image_filename = Path(frame["file_path"]).name
            image_filename = Path(frame[image_key]).name

            if image_filename not in image_paths:
                raise RuntimeError(
                    f"Image {image_filename} referenced in transform.json "
                    + "is missing in dataset."
                )
            # Note that sRGB gamma is not applied to the images as are scalar images
            image_path = image_paths[image_filename]
            mitsuba_image = mi.TensorXf(
                mi.Bitmap(str(image_path)).convert(
                    component_format=mi.Struct.Type.Float32, srgb_gamma=False
                )
            )
            if mitsuba_image.ndim == 2:
                mitsuba_image = mitsuba_image[:, :, None]
            if mitsuba_image.shape[-1] != 1:
                raise RuntimeError(
                    f"Image {image_filename} is not a single channel image."
                )

            is_train = False

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
                camera = self._pose_loader.eval_cameras[eval_idx]
                eval_idx += 1
            else:
                raise ValueError(
                    f"Image {rgb_image_filename} does not specify whether "
                    + "it is a training (`train`) or evaluation (`eval`) image, "
                    + "or the ordering between `transforms.json` and the split "
                    + "filenames is inconsistent."
                )

            camera.pixel_format = PerspectiveCamera.PixelFormat.LUMINANCE

            mitsuba_image_masked_pose = MaskedPoseImage(
                image=mitsuba_image, camera=camera, mask=None
            )
            # Separate into train and eval based on the filename
            if is_train:
                self._train_images.append(mitsuba_image_masked_pose)
            else:
                self._eval_images.append(mitsuba_image_masked_pose)

        # During training we the cameras are luminant
        self._train_cameras = self._pose_loader.train_cameras.copy()
        self._eval_cameras = self._pose_loader.eval_cameras.copy()
        for camera in self._train_cameras + self._eval_cameras:
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

    @property
    def train_images(self) -> list[MaskedPoseImage]:
        """List of images from the train data split"""
        return self._train_images

    @property
    def eval_images(self) -> list[MaskedPoseImage]:
        """List of images from the eval data split"""
        return self._eval_images
