import json
import tempfile
import unittest
from pathlib import Path

import mitsuba as mi

from forge_sim.data_processing.vggt_pose_scene_loader import VGGTPoseSceneLoader


class TestVGGTPoseSceneLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mi.set_variant("llvm_ad_rgb")

    def _write_vggt_transforms(
        self, scene_dir: Path, frame0: list, frame1: list
    ) -> None:
        transforms = {
            "frames": [
                {
                    "rgb": {
                        "file_path": "images/train_000.png",
                        "transform_matrix": frame0,
                        "w": 640,
                        "h": 480,
                        "fovx": 60.0,
                    }
                },
                {
                    "rgb": {
                        "file_path": "images/eval_000.png",
                        "transform_matrix": frame1,
                        "w": 640,
                        "h": 480,
                        "fovx": 60.0,
                    }
                },
            ]
        }
        (scene_dir / "transforms.json").write_text(json.dumps(transforms))

    def test_world_to_camera_3x4_is_inverted(self):
        with tempfile.TemporaryDirectory() as td:
            scene_dir = Path(td)
            # OpenCV world-to-camera extrinsics.
            frame0_w2c = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
            frame1_w2c = [
                [1.0, 0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
            self._write_vggt_transforms(scene_dir, frame0_w2c, frame1_w2c)

            loader = VGGTPoseSceneLoader(scene_dir, align_cameras_xz=False)
            eval_pose = loader.eval_cameras[0].tf_pose_matrix.matrix.numpy()

            # After reference-frame normalization and OpenCV->Mitsuba conversion,
            # translation should be negative x.
            self.assertAlmostEqual(float(eval_pose[0, 3]), -1.0, places=6)
            self.assertAlmostEqual(float(eval_pose[1, 3]), 0.0, places=6)
            self.assertAlmostEqual(float(eval_pose[2, 3]), 0.0, places=6)

    def test_camera_to_world_4x4_is_not_inverted(self):
        with tempfile.TemporaryDirectory() as td:
            scene_dir = Path(td)
            frame0_c2w = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            frame1_c2w = [
                [1.0, 0.0, 0.0, 2.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            self._write_vggt_transforms(scene_dir, frame0_c2w, frame1_c2w)

            loader = VGGTPoseSceneLoader(scene_dir, align_cameras_xz=False)
            eval_pose = loader.eval_cameras[0].tf_pose_matrix.matrix.numpy()

            self.assertAlmostEqual(float(eval_pose[0, 3]), -2.0, places=6)
            self.assertAlmostEqual(float(eval_pose[1, 3]), 0.0, places=6)
            self.assertAlmostEqual(float(eval_pose[2, 3]), 0.0, places=6)

    def test_invalid_transform_shape_raises(self):
        with tempfile.TemporaryDirectory() as td:
            scene_dir = Path(td)
            bad_frame = [[1.0, 0.0], [0.0, 1.0]]
            self._write_vggt_transforms(scene_dir, bad_frame, bad_frame)

            with self.assertRaises(ValueError):
                VGGTPoseSceneLoader(scene_dir, align_cameras_xz=False)
