import json
import tempfile
import unittest
from pathlib import Path

import mitsuba as mi
import numpy as np
import numpy.testing as npt

from rebel_shape.data_processing.get_rgb_and_transformation import (
    SaveFormats,
    export_transforms_from_cameras_opencv,
    render_and_save_reference_set,
)
from rebel_shape.plugin_wrappers.perspective_camera import PerspectiveCamera


class TestGetRgbAndTransformation(unittest.TestCase):
    """Tests for JSON export and file outputs produced by get_rgb_and_transformation."""

    @classmethod
    def setUpClass(cls):
        mi.set_variant("llvm_ad_rgb")

    def test_export_transforms_json_structure_and_pose_rule(self):
        """
        Check exported transforms.json schema and that each pose follows the expected
        export rule.
        """
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            out_json = out_dir / "transforms.json"

            sensor_count = 1
            image_resolution = 64
            prefix = "ref"

            camera = PerspectiveCamera(
                tf_camera_pose=mi.ScalarTransform4f(
                    np.array(
                        [[1, 0, 0, 4], [0, 1, 0, 8], [0, 0, 1, 12], [0, 0, 0, 1]],
                        dtype=float,
                    )
                ),
                camera_resolution=(image_resolution, image_resolution),
            )
            opencv_pose = np.array(
                [[-1, 0, 0, 4], [0, 1, 0, 8], [0, 0, -1, 12], [0, 0, 0, 1]], dtype=float
            )

            export_transforms_from_cameras_opencv(
                cameras=[camera],
                out_path=out_json,
                image_prefix=prefix,
                save_format=SaveFormats.EXR,
            )
            data = json.loads(out_json.read_text())
            got = np.array(data["frames"][0]["transform_matrix"], dtype=float)
            npt.assert_allclose(got, opencv_pose, atol=1e-6, rtol=1e-6)

            # JSON fields existence / values
            self.assertEqual(data["camera_model"], "OPENCV")
            self.assertEqual(data["w"], image_resolution)
            self.assertEqual(data["h"], image_resolution)
            self.assertIn("fl_x", data)
            self.assertIn("cx", data)
            self.assertEqual(len(data["frames"]), sensor_count)

            # File paths match naming convention
            self.assertEqual(data["frames"][0]["file_path"], "images/ref_00000.exr")

    def test_render_and_save_reference_set_creates_images_and_json(self):
        """
        Check that rendering/export creates transforms.json and the expected image
        files.
        """
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)

            repo_root = Path(__file__).resolve().parents[1]
            mesh_path = repo_root / "tests" / "data" / "meshes" / "test_cube.ply"
            self.assertTrue(mesh_path.exists(), f"Missing mesh: {mesh_path}")

            sensor_count = 2
            image_resolution = 16
            reference_samples = 1
            prefix = "ref"

            render_and_save_reference_set(
                sensor_count=sensor_count,
                spp_continuous=1,
                spp_projective=1,
                spp=reference_samples,
                image_resolution=image_resolution,
                save_dir=out_dir,
                mesh_path=mesh_path,
                image_prefix=prefix,
                save_format=SaveFormats.EXR,
            )

            # transforms.json exists
            self.assertTrue((out_dir / "transforms.json").exists())

            data = json.loads((out_dir / "transforms.json").read_text())
            self.assertEqual(len(data["frames"]), sensor_count)
            for fr in data["frames"]:
                p = (
                    out_dir / fr["file_path"]
                )  # file_path is like "images/ref_00000.exr"
                self.assertTrue(
                    p.exists(), f"Missing image for frame: {fr['file_path']}"
                )
