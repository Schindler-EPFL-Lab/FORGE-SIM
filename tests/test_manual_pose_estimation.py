import json
import tempfile
import unittest
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np
from scipy.spatial.transform import Rotation

from rebel_shape.data_processing.manual_initialization import (
    load_bounding_bbox_transforms,
    pose_params_to_bbox_transforms,
    save_bounding_bbox_transforms,
)


class TestManualPoseEstimation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mi.set_variant("llvm_ad_rgb")

        local_dir = Path(__file__).parent.resolve()
        cls.dataset_path = local_dir / "data/bbox_transforms/buildingA-spring.json"

    @staticmethod
    def _tf_new_pose(flat_params: np.ndarray) -> "mi.Transform4f":
        """
        Mirrors the chain assembled by `PoseOptimizer._shape_update_callback` so the
        converter can be cross-checked against the optimizer's own pose math without
        instantiating an optimizer.
        """
        tx, ty, tz = flat_params[0:3]
        sx, sy, sz = flat_params[3:6]
        rx_deg, ry_deg, rz_deg = np.rad2deg(flat_params[6:9])
        return (
            mi.Transform4f()
            .translate(mi.Point3f(float(tx), float(ty), float(tz)))
            .rotate(axis=mi.Point3f(1, 0, 0), angle=float(rx_deg))
            .rotate(axis=mi.Point3f(0, 1, 0), angle=float(ry_deg))
            .rotate(axis=mi.Point3f(0, 0, 1), angle=float(rz_deg))
            .scale(mi.Point3f(float(sx), float(sy), float(sz)))
        )

    def _assert_converter_round_trips(
        self,
        flat_params: np.ndarray,
        tf_cameras: "mi.ScalarTransform4f",
        initial_sphere_radius: float,
    ) -> None:
        """
        Converter output must satisfy the invariant

            tf_cameras @ load(saved_bbox) @ v == tf_new_pose @ (r * v)

        for every vertex v on a unit sphere, so the produced JSON is
        substitutable for a hand-authored bbox_transforms file.
        """
        translation, rotation, scale = pose_params_to_bbox_transforms(
            flat_params=mi.Float(flat_params.tolist()),
            tf_cameras=tf_cameras,
            initial_sphere_radius=initial_sphere_radius,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tf.json"
            save_bounding_bbox_transforms(
                bbox_transforms_path=path,
                translation=translation,
                rotation=rotation,
                scale=scale,
            )
            tf_bbox = load_bounding_bbox_transforms(path)

        lhs = (mi.Transform4f(tf_cameras) @ tf_bbox).matrix
        rhs = (
            self._tf_new_pose(flat_params)
            @ mi.Transform4f().scale(
            mi.Point3f(
                initial_sphere_radius,
                initial_sphere_radius,
                initial_sphere_radius,
            ),
        )
        ).matrix

        self.assertTrue(
            dr.allclose(lhs, rhs, atol=1e-4),
            msg=f"LHS=\n{np.array(lhs)}\nRHS=\n{np.array(rhs)}",
        )

    def test_loading(self) -> None:
        """Reference fixture must decode to the same 4x4 we shipped the tests with."""
        tf_loaded = load_bounding_bbox_transforms(self.dataset_path)
        tf_expected = mi.Transform4f(
            [
                (
                    2.6316609382629395,
                    -3.4852536856533334e-08,
                    0.1921636462211609,
                    0.7498473525047302,
                ),
                (
                    -0.09508349001407623,
                    1.994611382484436,
                    0.4876357614994049,
                    -0.3293049931526184,
                ),
                (
                    -0.2992772161960602,
                    -0.6337091326713562,
                    1.5348427295684814,
                    5.261859893798828,
                ),
                (0.0, 0.0, 0.0, 1.0),
            ],
        )

        self.assertTrue(dr.allclose(tf_loaded.matrix, tf_expected.matrix))

    def test_save_round_trip(self) -> None:
        translation = (1.25, -0.5, 3.0)
        rotation = (10.0, -25.0, 45.0)
        scale = (1.5, 2.0, 0.75)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "pose.json"
            save_bounding_bbox_transforms(
                bbox_transforms_path=output_path,
                translation=translation,
                rotation=rotation,
                scale=scale,
            )

            self.assertTrue(output_path.exists())

            with open(output_path, "r") as f:
                raw = json.load(f)
            self.assertEqual(tuple(raw["translation"]), translation)
            self.assertEqual(tuple(raw["rotation"]), rotation)
            self.assertEqual(tuple(raw["scale"]), scale)

            tf_loaded = load_bounding_bbox_transforms(output_path)

        expected = np.eye(4)
        expected[:3, :3] = np.diag(scale)
        expected[:3, :3] = (
            Rotation.from_euler("xyz", rotation, degrees=True).as_matrix()
            @ expected[:3, :3]
        )
        expected[:3, 3] = translation
        self.assertTrue(
            dr.allclose(tf_loaded.matrix, mi.Transform4f(expected).matrix),
        )

    def test_converter_identity_pose_and_cameras(self) -> None:
        flat_params = np.array(
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32,
        )
        self._assert_converter_round_trips(
            flat_params=flat_params,
            tf_cameras=mi.ScalarTransform4f(),
            initial_sphere_radius=1.0,
        )

    def test_converter_nontrivial_pose_and_camera_rotation(self) -> None:
        flat_params = np.array(
            [0.7, -1.1, 2.3, 1.2, 0.9, 1.4, 0.3, -0.5, 0.9], dtype=np.float32,
        )
        tf_cameras = (
            mi.ScalarTransform4f()
            .rotate(axis=mi.ScalarPoint3f(0, 1, 0), angle=37.0)
            .rotate(axis=mi.ScalarPoint3f(1, 0, 0), angle=-12.0)
        )
        self._assert_converter_round_trips(
            flat_params=flat_params,
            tf_cameras=tf_cameras,
            initial_sphere_radius=2.5,
        )

    def test_converter_rigid_camera_alignment_with_translation(self) -> None:
        flat_params = np.array(
            [-0.25, 0.8, -1.4, 1.05, 0.95, 1.0, 0.1, 0.2, -0.15], dtype=np.float32,
        )
        tf_cameras = mi.ScalarTransform4f().translate(
            mi.ScalarPoint3f(0.4, -0.2, 0.6),
        ) @ mi.ScalarTransform4f().rotate(axis=mi.ScalarPoint3f(0, 0, 1), angle=25.0)
        self._assert_converter_round_trips(
            flat_params=flat_params,
            tf_cameras=tf_cameras,
            initial_sphere_radius=3.0,
        )

    def test_converter_clips_out_of_range_params(self) -> None:
        """
        The pose optimizer clips scale to [scale_epsilon, 1/scale_epsilon] and
        rotation (in radians) to [-π, π]; the converter must replicate both so
        downstream users get the pose that was actually rendered, not the raw
        pre-clip state.
        """
        scale_epsilon = 1e-3
        flat_params = np.array(
            [0.1, 0.2, 0.3, 5e-4, 2e3, -1.0, 5.0, -5.0, 0.0], dtype=np.float32,
        )
        clipped_scale = np.clip(
            flat_params[3:6], scale_epsilon, 1.0 / scale_epsilon,
        )

        _, _, scale = pose_params_to_bbox_transforms(
            flat_params=mi.Float(flat_params.tolist()),
            tf_cameras=mi.ScalarTransform4f(),
            initial_sphere_radius=1.0,
            scale_epsilon=scale_epsilon,
        )

        # For identity tf_cameras and radius 1, the clipped scale flows straight
        # through to the output without being mixed with rotation.
        np.testing.assert_allclose(scale, clipped_scale, atol=1e-5)
