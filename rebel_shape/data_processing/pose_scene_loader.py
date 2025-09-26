import json
from pathlib import Path
from typing import Generator

import mitsuba as mi
import numpy as np

from rebel_shape.plugin_wrappers.perspective_camera import PerspectiveCamera
from rebel_shape.plugin_wrappers.sampler import Sampler


class PoseSceneLoader:
    """
    Loads camera poses from transforms.json in `scene_path`. They are further split
    into training and evaluation sets. It transforms the poses into a reference
    configuration which is assumed to be the one from the first camera. Optionally,
    camera positions can be used to align the poses to lie approximately in the
    xz-plane (positive y-axis is down).
    """

    def __init__(
        self,
        scene_path: Path,
        sensor_sampler: Sampler = Sampler(),
        align_cameras_xz: bool = True,
    ):
        """
        Loads the poses from the directory `scene_path`.

        Camera poses produced by tools like COLMAP deal in relative coordinates without
        a clear reference frame. This loader assumes that the first camera is
        the reference. Furthermore, "real" camera poses are likely to lie on the
        ground plane) and setting `align_cameras_xz` to `True` will align the cameras
        to the xz-plane. Each camera has `sensor_samples` for rays sampling.

        Mitsuba sensors have an associated sampler; when loading the camera poses
        from the dataset, the `PerspectiveCamera` instance is initialized with the
        provided `sensor_sampler`.

        :raises FileNotFoundError: If the scene path does not exist.
        """
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene path {scene_path} does not exist.")

        self._scene_path = scene_path
        self._sensor_sampler = sensor_sampler
        self._tf_cameras = mi.ScalarTransform4f()

        self._train_cameras: list[PerspectiveCamera] = []
        self._train_rgb_filenames: list[str] = []

        self._eval_cameras: list[PerspectiveCamera] = []
        self._eval_rgb_filenames: list[str] = []

        transform_path = self._scene_path / "transforms.json"

        # Load the transform.json file
        with open(transform_path, "r") as f:
            self._transform_data = json.load(f)

        image_width = int(self._transform_data["w"])
        image_height = int(self._transform_data["h"])

        focal_length_x = float(self._transform_data["fl_x"])
        fov_x = 2.0 * np.arctan(image_width / (2.0 * focal_length_x)) * 180.0 / np.pi

        # Transformation applied to the reference nerfstudio format
        applied_transform = np.eye(4)
        if "applied_transform" in self._transform_data:
            applied_transform[:3, :] = np.array(
                self._transform_data["applied_transform"]
            )
        applied_transform_inv = np.linalg.inv(applied_transform)

        inv_reference_pose_matrix: mi.ScalarTransform4f | None = None
        # Iterate through frames from transform_data
        for frame in self._transform_data.get("frames", []):
            rgb_image_filename = Path(frame["file_path"]).name

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
                self._train_cameras.append(camera)
                self._train_rgb_filenames.append(rgb_image_filename)

            elif "eval" in rgb_image_filename.lower():
                self._eval_cameras.append(camera)
                self._eval_rgb_filenames.append(rgb_image_filename)
            else:
                raise ValueError(
                    f"Image {rgb_image_filename} does not specify whether "
                    "it is a training ('train') or evaluation ('eval') image"
                )

        if align_cameras_xz:
            self._align_cameras_to_xz_plane()

    @staticmethod
    def _load_image_paths_from_folder(folder: Path) -> dict[str, Path]:
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
    def train_cameras(self) -> list[PerspectiveCamera]:
        """List of camera poses for the training set with RGB pixel format"""
        return self._train_cameras

    @property
    def train_filenames(self) -> list[str]:
        """List of RGB images filenames in the training set"""
        return self._train_rgb_filenames

    @property
    def train_data(
        self,
    ) -> Generator[tuple[str, PerspectiveCamera], None, None]:
        """
        Yields tuples of training RGB image filenames and their corresponding
        PerspectiveCamera objects.
        """

        for filename, camera in zip(self._train_rgb_filenames, self._train_cameras):
            yield filename, camera

    @property
    def eval_cameras(self) -> list[PerspectiveCamera]:
        """List of camera poses for the evaluation set with RGB pixel format"""
        return self._eval_cameras

    @property
    def eval_filenames(self) -> list[str]:
        """List of RGB images filenames in the eval set"""
        return self._eval_rgb_filenames

    @property
    def eval_data(
        self,
    ) -> Generator[tuple[str, PerspectiveCamera], None, None]:
        """
        Yields tuples of eval RGB image filenames and their corresponding
        PerspectiveCamera objects.
        """
        for filename, camera in zip(self._eval_rgb_filenames, self._eval_cameras):
            yield filename, camera
