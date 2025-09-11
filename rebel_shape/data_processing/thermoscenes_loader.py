import json
from pathlib import Path

import mitsuba as mi
import numpy as np

from rebel_shape.plugin_wrappers.perspective_camera import PerspectiveCamera
from rebel_shape.plugin_wrappers.sampler import Sampler


class ThermoScenesLoader:
    """
    Specialized loader for the ThermoScenes dataset.

    It loads RGB, thermal images, and foreground masks along with their associated
    camera poses. They are stored in separate lists and the order is guaranteed to
    match. Images are further split into training and evaluation sets.

    Thermal images are assumed to be 8-bit linear grayscale images so no gamma
    correction is applied to them. RGB images are assumed to be sRGB gamma corrected.
    Foreground masks are assumed to be 8-bit linear grayscale images.

    The images are of type `mi.TensorXf` since this is the native format for Mitsuba
    operations (gradient computation, etc.), while the camera information and pose
    is encapsulated in `PerspectiveCamera`.

    When loading poses from files it transforms them into a reference configuration
    which is assumed to be the one from the first camera. Optionally, camera positions
    can be used to align the poses to lie approximately in the xz-plane (positive
    y-axis is down).
    """

    def __init__(
        self,
        scene_path: Path,
        align_cameras_xz: bool = True,
        sensor_sampler: Sampler = Sampler(),
    ) -> None:
        """
        Loads the ThermoScenes dataset from the directory `scene_path`.

        Camera poses produced by tools like COLMAP deal in relative coordinates without
        a clear reference frame. This loader assumes that the first camera is
        the reference. Furthermore, "real" camera poses are likely to lie on the
        ground plane) and setting `align_cameras_xz` to `True` will align the cameras
        to the xz-plane.

        Mitsuba sensors have an associated sampler; when loading the camera poses
        from the dataset, the `PerspectiveCamera` instance is initialized with the
        provided `sensor_sampler`.

        :raises FileNotFoundError: If the scene path does not exist.
        """
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene path {scene_path} does not exist.")

        self._scene_path = scene_path

        self._train_images_rgb: list["mi.TensorXf"] = []
        self._train_foreground_masks: list["mi.TensorXf | None"] = []
        self._train_images_thermal: list["mi.TensorXf"] = []
        self._train_cameras: list[PerspectiveCamera] = []

        self._eval_images_rgb: list["mi.TensorXf"] = []
        self._eval_foreground_masks: list["mi.TensorXf | None"] = []
        self._eval_images_thermal: list["mi.TensorXf"] = []
        self._eval_cameras: list[PerspectiveCamera] = []

        self._sensor_sampler = sensor_sampler
        self._tf_cameras = mi.ScalarTransform4f()
        self._load_data()

        if align_cameras_xz:
            self._align_cameras_to_xz_plane()

    def _load_data(self) -> None:
        """
        Loads images and their associated camera poses into private class variables.

        Most camera related data, like OpenCV corrections for distortion, is not used.
        The base `.json` file that is used to obtain camera poses compatible with
        `nerfstudio` and transformations are necessary to match Mitsuba's convention.

        The dataset is assumed to be in the following format:
        - `images` folder contains RGB images
        - `thermal` folder contains thermal images
        - `foreground_masks` optional folder contains foreground masks
        - `transforms.json` file contains camera poses and other metadata

        :raises FileNotFoundError: If the transformation file or rgb image folder or
            thermal image folder does not exist.

        :raises RuntimeError: If the rgb or thermal image referenced in the
            transformation file does not exist.

        :raises ValueError: If the image names do not contain "train" or "eval".
        """

        def load_image_paths_from_folder(folder: Path) -> dict[str, Path]:
            """
            Loads and returns a dictionary mapping image filenames to their full paths
            from the specified folder.
            """
            # Supported image extensions
            image_extensions = (".JPG", ".jpg", ".JPEG", ".jpeg", ".PNG", ".png")
            if not folder.exists():
                return {}
            image_filenames = sorted(
                [
                    image.name
                    for image in folder.iterdir()
                    if image.suffix in image_extensions
                ]
            )
            return {image: folder / image for image in image_filenames}

        rgb_folder = self._scene_path / "images"
        thermal_folder = self._scene_path / "thermal"
        foreground_mask_folder = self._scene_path / "foreground_masks"
        transform_path = self._scene_path / "transforms.json"

        # Load the transform.json file
        with open(transform_path, "r") as f:
            transform_data = json.load(f)

        image_width = int(transform_data["w"])
        image_height = int(transform_data["h"])

        focal_length_x = float(transform_data["fl_x"])
        fov_x = 2.0 * np.arctan(image_width / (2.0 * focal_length_x)) * 180.0 / np.pi

        # Transformation applied to the reference nerfstudio format
        applied_transform = np.eye(4)
        if "applied_transform" in transform_data:
            applied_transform[:3, :] = np.array(transform_data["applied_transform"])
        applied_transform_inv = np.linalg.inv(applied_transform)

        image_rgb_paths = load_image_paths_from_folder(rgb_folder)
        image_thermal_paths = load_image_paths_from_folder(thermal_folder)
        foreground_mask_paths = load_image_paths_from_folder(foreground_mask_folder)

        inv_reference_pose_matrix: mi.ScalarTransform4f | None = None

        # Iterate through frames from transform_data
        for frame in transform_data.get("frames", []):
            rgb_image_filename = Path(frame["file_path"]).name
            thermal_image_filename = Path(frame["thermal_file_path"]).name
            foreground_mask_filename: None | str = (
                Path(frame["foreground_mask_file_path"]).name
                if foreground_mask_folder.exists()
                else None
            )

            if rgb_image_filename not in image_rgb_paths:
                raise RuntimeError(
                    f"Image {rgb_image_filename} referenced in transform.json "
                    + "is missing in dataset."
                )

            if thermal_image_filename not in image_thermal_paths:
                raise RuntimeError(
                    f"Image {thermal_image_filename} referenced in transform.json "
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

            # Note that sRGB gamma is not applied to thermal images as they are 8-bit
            # linear grayscale images
            image_thermal_path = image_thermal_paths[thermal_image_filename]
            mitsuba_image_thermal = mi.TensorXf(
                mi.Bitmap(str(image_thermal_path)).convert(
                    component_format=mi.Struct.Type.Float32, srgb_gamma=False
                )
            )
            if mitsuba_image_thermal.shape[-1] != 1:
                raise RuntimeError(
                    f"Thermal image {thermal_image_filename} is not a single channel "
                    "grayscale image."
                )

            mitsuba_foreground_mask = None
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

            # Mitsuba, COLMAP and nerfstudio use different conventions for camera frame
            # COLMAP (OpenCV):  X - right,  Y - down,   Z - forward
            # Nerfstudio:       X - right,  Y - up,     Z - backward
            # Mitsuba 3:        X - left,   Y - up,     Z - forward

            # Details of COLMAP format (world to camera position and quaternion)
            # https://colmap.github.io/format.html#images-txt
            # What nerfstudio does with COLMAP data (start line 390)
            # https://github.com/nerfstudio-project/nerfstudio/blob/73fe54dda0b743616854fc839889d955522e0e68/nerfstudio/process_data/colmap_utils.py

            # Nerfstudio exported transform_matrix
            pose_matrix = np.array(frame["transform_matrix"])
            # Apply the inverse of applied_transform to convert poses
            # to the COLMAP global frame
            pose_matrix = applied_transform_inv @ pose_matrix

            # Going from Nerfstudio to Mitsuba requires flipping the sign of x and z
            pose_matrix = mi.ScalarTransform4f(pose_matrix)
            # scale(-1, 1, -1) should not be taken into account when calculating
            # _extra_transforms because the scaling changes only the camera local
            # coordinates but not the camera poses
            pose_matrix = pose_matrix.scale(mi.ScalarPoint3f(-1.0, 1.0, -1.0))

            # Transform into the reference frame
            if inv_reference_pose_matrix is None:
                inv_reference_pose_matrix = pose_matrix.inverse()

                self._tf_cameras = inv_reference_pose_matrix @ self._tf_cameras

            pose_matrix = inv_reference_pose_matrix @ pose_matrix

            # Create a camera object from pose, fov and resolution
            camera = PerspectiveCamera(
                tf_camera_pose=pose_matrix,
                camera_resolution=(image_width, image_height),
                fov=fov_x,
                fov_axis=PerspectiveCamera.FovAxis.X,
                sampler=self._sensor_sampler,
            )

            # Separate into train and eval based on the filename
            if "train" in rgb_image_filename.lower():
                self._train_images_rgb.append(mitsuba_image_rgb)
                self._train_images_thermal.append(mitsuba_image_thermal)
                self._train_foreground_masks.append(mitsuba_foreground_mask)
                self._train_cameras.append(camera)

            elif "eval" in rgb_image_filename.lower():
                self._eval_images_rgb.append(mitsuba_image_rgb)
                self._eval_images_thermal.append(mitsuba_image_thermal)
                self._eval_foreground_masks.append(mitsuba_foreground_mask)
                self._eval_cameras.append(camera)

            else:
                raise ValueError(
                    f"Image {rgb_image_filename} does not specify whether "
                    "it is a training ('train') or evaluation ('eval') image"
                )

    def _align_cameras_to_xz_plane(self) -> None:
        """
        Aligns all camera positions and orientations to the x-z plane by:
        1. Finding the best-fit plane for all camera positions
        2. Computing the rotation needed to align this plane with the x-z plane
        3. Applying this rotation to all camera poses

        Modifies `self._train_cameras` and `self._eval_cameras` pose matrices such that
        they lie in the defined plane.
        """
        positions = np.array(
            [
                camera.tf_pose_matrix.matrix.numpy()[:3, 3]
                for camera in self._train_cameras + self._eval_cameras
            ]
        )

        # Fit a plane to the camera positions
        # The normal of the plane is the eigenvector with the smallest eigenvalue
        centered = positions - np.mean(positions, axis=0)
        _, _, vh = np.linalg.svd(centered)
        plane_normal = vh[2, :]  # Third eigenvector is normal to the best-fit plane
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        # The negative y-axis is our target for the plane normal (upwards)
        target_normal = np.array([0.0, -1.0, 0.0])

        # Compute rotation axis and angle to align plane normal with y-axis
        rotation_axis = np.cross(plane_normal, target_normal)
        if np.linalg.norm(rotation_axis) < 1e-6:
            # Vectors are already aligned or opposite
            if np.dot(plane_normal, target_normal) > 0.0:
                return  # Already aligned
            else:
                # Opposite directions, rotate 180° around x-axis
                rotation_axis = np.array([1.0, 0.0, 0.0])
                rotation_angle = np.pi
        else:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            cos_angle = np.dot(plane_normal, target_normal)
            rotation_angle = np.arccos(np.clip(cos_angle, -1.0 + 1e-12, 1.0 - 1e-12))

            if rotation_angle >= 0.5 * np.pi:
                rotation_angle -= np.pi

        tf_alignment_matrix = mi.ScalarTransform4f().rotate(
            axis=mi.ScalarPoint3f(rotation_axis),
            angle=rotation_angle * 180.0 / np.pi,  # Convert to degrees for Mitsuba
        )

        # update self transforms global since the camera poses have been changed
        self._tf_cameras = tf_alignment_matrix @ self._tf_cameras

        # Apply the transformation to all cameras
        for camera in self._train_cameras + self._eval_cameras:
            camera.tf_pose_matrix = tf_alignment_matrix @ camera.tf_pose_matrix

    @property
    def tf_cameras(self) -> "mi.ScalarTransform4f":
        """Return the transformation which were applied to camera poses"""
        return self._tf_cameras

    @property
    def train_images_rgb(self) -> list["mi.TensorXf"]:
        """List of RGB images in the training set"""
        return self._train_images_rgb

    @property
    def train_images_thermal(self) -> list["mi.TensorXf"]:
        """List of thermal images in the training set"""
        return self._train_images_thermal

    @property
    def train_foreground_masks(self) -> list["mi.TensorXf | None"]:
        """List of foreground masks in the training set"""
        return self._train_foreground_masks

    @property
    def train_cameras(self) -> list[PerspectiveCamera]:
        """List of camera poses for the training set with RGB pixel format"""
        return self._train_cameras

    @property
    def train_cameras_thermal(self) -> list[PerspectiveCamera]:
        """
        List of cameras for the training set with a luminance pixel format.

        Thermal optimization is carried out with single channel ("mono") mode in
        Mitsuba while the cameras are RGB by default. This property copies the internal
        RGB cameras and makes them grayscale to avoid runtime warnings from Mitsuba.
        """
        thermal_cameras = self._train_cameras.copy()
        for camera in thermal_cameras:
            camera.pixel_format = PerspectiveCamera.PixelFormat.LUMINANCE

        return thermal_cameras

    @property
    def eval_images_rgb(self) -> list["mi.TensorXf"]:
        """List of RGB images in the evaluation set"""
        return self._eval_images_rgb

    @property
    def eval_images_thermal(self) -> list["mi.TensorXf"]:
        """List of thermal images in the evaluation set"""
        return self._eval_images_thermal

    @property
    def eval_foreground_masks(self) -> list["mi.TensorXf | None"]:
        """List of foreground masks in the eval set"""
        return self._eval_foreground_masks

    @property
    def eval_cameras(self) -> list[PerspectiveCamera]:
        """List of camera poses for the evaluation set with RGB pixel format"""
        return self._eval_cameras

    @property
    def eval_cameras_thermal(self) -> list[PerspectiveCamera]:
        """
        List of cameras for the evaluation set with a luminance pixel format.

        Thermal optimization is carried out with single channel ("mono") mode in
        Mitsuba while the cameras are RGB by default. This property copies the internal
        RGB cameras and makes them grayscale to avoid runtime warnings from Mitsuba.
        """
        thermal_cameras = self._eval_cameras.copy()
        for camera in thermal_cameras:
            camera.pixel_format = PerspectiveCamera.PixelFormat.LUMINANCE

        return thermal_cameras

    def get_train_data_rgb(self) -> list[tuple[PerspectiveCamera, "mi.TensorXf"]]:
        return list(zip(self.train_cameras, self.train_images_rgb))

    def get_train_data_thermal(self) -> list[tuple[PerspectiveCamera, "mi.TensorXf"]]:
        return list(zip(self.train_cameras, self.train_images_thermal))

    def get_train_data(
        self,
    ) -> list[tuple[PerspectiveCamera, "mi.TensorXf", "mi.TensorXf"]]:
        return list(
            zip(self.train_cameras, self.train_images_rgb, self.train_images_thermal)
        )

    def get_eval_data_rgb(self) -> list[tuple[PerspectiveCamera, "mi.TensorXf"]]:
        return list(zip(self.eval_cameras, self.eval_images_rgb))

    def get_eval_data_thermal(self) -> list[tuple[PerspectiveCamera, "mi.TensorXf"]]:
        return list(zip(self.eval_cameras, self.eval_images_thermal))

    def get_eval_data(
        self,
    ) -> list[tuple[PerspectiveCamera, "mi.TensorXf", "mi.TensorXf"]]:
        return list(
            zip(self.eval_cameras, self.eval_images_rgb, self.eval_images_thermal)
        )
