from typing import Any

import mitsuba as mi
import numpy as np
import numpy.typing as npt

from forge_sim.data_processing.pose_scene_loader import PoseSceneLoader


class VGGTPoseSceneLoader(PoseSceneLoader):
    """
    `PoseSceneLoader` specialization for VGGT-produced `transforms.json` files.
    Per-frame data is nested under an `"rgb"` key rather than at the top level of
    each frame entry, and `transform_matrix` may be either a 3x4 OpenCV
    world-to-camera matrix (inverted here to camera-to-world) or an already
    camera-to-world 4x4 matrix.
    """

    def get_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        """Returns the frame record nested under `frame`'s `"rgb"` key."""
        return frame["rgb"]

    def get_pose_matrix(self, frame: dict[str, Any]) -> npt.NDArray[np.float64]:
        """
        Extracts the camera-to-world pose matrix from `frame`'s `transform_matrix`.

        A 3x4 matrix is treated as an OpenCV world-to-camera matrix, padded with a
        homogeneous row and inverted to camera-to-world; a 4x4 matrix is returned
        as-is (already camera-to-world).

        :returns: A `(4, 4)` camera-to-world pose matrix, still in the dataset's
            native coordinate convention.
        :raises ValueError: If `transform_matrix` is neither 3x4 nor 4x4.
        """
        pose_matrix = np.array(frame["transform_matrix"], dtype=np.float64)

        if pose_matrix.shape == (3, 4):
            new_row = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
            # VGGT commonly stores OpenCV world-to-camera in 3x4 form.
            pose_matrix = np.vstack((pose_matrix, new_row))
            return np.linalg.inv(pose_matrix)

        if pose_matrix.shape == (4, 4):
            return pose_matrix

        raise ValueError(
            "VGGT transform_matrix must be 3x4 (world-to-camera) or 4x4 "
            f"(camera-to-world), got shape {pose_matrix.shape}."
        )

    def preprocess_pose_matrix(
        self,
        pose_matrix: npt.NDArray[np.float64],
        applied_transform_inv: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """VGGT poses are already converted to camera-to-world in get_pose_matrix."""
        _ = applied_transform_inv
        return pose_matrix

    def pose_matrix_to_mitsuba(
        self,
        pose_matrix: npt.NDArray[np.float64],
    ) -> "mi.ScalarTransform4f":
        """
        Convert OpenCV camera coordinates to Mitsuba camera coordinates.

        OpenCV uses X right / Y down / Z forward, Mitsuba uses
        X left / Y up / Z forward, so x and y axes are flipped.
        """
        tf_pose = mi.ScalarTransform4f(pose_matrix.tolist())
        return tf_pose.scale(mi.ScalarPoint3f(-1.0, -1.0, 1.0))

    def get_image_info(self, frame: dict[str, Any]) -> tuple[int, int, float]:
        """Returns `(image_width, image_height, fov_x)` read directly from `frame`."""
        image_width = int(frame["w"])
        image_height = int(frame["h"])
        fov_x = float(frame["fovx"])
        return image_width, image_height, fov_x
