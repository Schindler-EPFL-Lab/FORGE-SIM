from pathlib import Path
from typing import Generator

import drjit as dr
import mitsuba as mi

from rebel_shape.data_processing.pose_scene_loader import PoseSceneLoader
from rebel_shape.plugin_wrappers.perspective_camera import PerspectiveCamera
from rebel_shape.plugin_wrappers.sampler import Sampler


class ImageSceneLoader:
    """
    Loads RGB and foreground masks along with their associated camera poses. They are
    stored in separate lists and the order is guaranteed to match. Images are further
    split into training and evaluation sets.

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
        align_cameras_xz: bool = True,
    ):
        """
        Loads the ThermoScenes images, foreground masks, and cameras poses from the
        directory `scene_path`. Optionally aligns camera poses so they lie on the
        xy-plane if `align_cameras_xz` if `True`. Each camera has `sensor_samples` for
        rays sampling.

        :raises FileNotFoundError: If the scene path does not exist.
        """

        self._pose_loader = PoseSceneLoader(
            scene_path=scene_path,
            sensor_sampler=sensor_sampler,
            align_cameras_xz=align_cameras_xz,
        )

        self._train_images: list["mi.TensorXf"] = []
        self._train_foreground_masks: list["mi.TensorXf"] = []

        self._eval_images: list["mi.TensorXf"] = []
        self._eval_foreground_masks: list["mi.TensorXf"] = []

        rgb_folder = self._pose_loader._scene_path / "images"
        foreground_mask_folder = self._pose_loader._scene_path / "foreground_masks"

        image_rgb_paths = self._pose_loader._load_image_paths_from_folder(rgb_folder)
        foreground_mask_paths = self._pose_loader._load_image_paths_from_folder(
            foreground_mask_folder
        )

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
            mitsuba_image_rgb = mi.TensorXf(
                mi.Bitmap(str(image_rgb_path)).convert(
                    component_format=mi.Struct.Type.Float32, srgb_gamma=True
                )
            )

            mitsuba_foreground_mask = mi.TensorXf(
                dr.zeros_like(mitsuba_image_rgb[:, :, 0:1])
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

            # Separate into train and eval based on the filename
            if rgb_image_filename in self._pose_loader._train_rgb_filenames:
                self._train_images.append(mitsuba_image_rgb)
                self._train_foreground_masks.append(mitsuba_foreground_mask)

            elif rgb_image_filename in self._pose_loader._eval_rgb_filenames:
                self._eval_images.append(mitsuba_image_rgb)
                self._eval_foreground_masks.append(mitsuba_foreground_mask)

            else:
                raise ValueError(
                    f"Image {rgb_image_filename} does not specify whether "
                    "it is a training ('train') or evaluation ('eval') image"
                )

    @property
    def tf_cameras(self) -> "mi.ScalarTransform4f":
        """Return the transformation which were applied to camera poses"""
        return self._pose_loader.tf_cameras

    @property
    def train_cameras(self) -> list[PerspectiveCamera]:
        """List of camera poses for the training set with RGB pixel format"""
        return self._pose_loader.train_cameras

    @property
    def train_data(
        self,
    ) -> Generator[
        tuple[str, PerspectiveCamera, "mi.TensorXf", "mi.TensorXf"], None, None
    ]:
        """
        Yields tuples of (image filename, camera, image, foreground_mask) for training
        data.
        """
        for (filename, camera), image, foreground_mask in zip(
            self._pose_loader.train_data,
            self._train_images,
            self._train_foreground_masks,
        ):
            yield filename, camera, image, foreground_mask

    @property
    def eval_cameras(self) -> list[PerspectiveCamera]:
        """List of camera poses for the evaluation set with RGB pixel format"""
        return self._pose_loader.eval_cameras

    @property
    def eval_data(
        self,
    ) -> Generator[
        tuple[str, PerspectiveCamera, "mi.TensorXf", "mi.TensorXf"], None, None
    ]:
        """
        Yields tuples of (image filename, camera, image, foreground_mask) for eval data.
        """
        for (filename, camera), image, foreground_mask in zip(
            self._pose_loader.eval_data,
            self._eval_images,
            self._eval_foreground_masks,
        ):
            yield filename, camera, image, foreground_mask
