import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

TORCH_AVAILABLE = True
try:
    import torch
    from PIL import Image

    from rebel_shape.scripts.segmentation.mask_with_sam3 import (
        Sam3ThermoScenesParams,
        choose_mask,
        pil_to_rgba_with_alpha,
        run,
    )
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch/Pillow required for full run flow test.")
class TestSam3ThermoScenes(unittest.TestCase):
    def setUp(self):
        """
        Sets up a temporary directory with a minimal ThermoScenes structure.
        This mimics the input expected by the SAM3 automation script.
        """
        self.test_dir = Path("temp_test_scene")
        self.test_dir.mkdir(parents=True, exist_ok=True)

        self.img_dir = self.test_dir / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.dummy_img_path = self.img_dir / "frame_001.jpg"

        if TORCH_AVAILABLE:
            Image.new("RGB", (100, 100), color="red").save(self.dummy_img_path)
        else:
            self.dummy_img_path.touch()

        self.transforms_data = {
            "camera_angle_x": 0.8,
            "frames": [
                {
                    "file_path": "images/frame_001.jpg",
                    "transform_matrix": [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ],
                }
            ],
        }
        with (self.test_dir / "transforms.json").open("w") as f:
            json.dump(self.transforms_data, f)

    def tearDown(self) -> None:
        """Cleans up the temporary directories created during tests."""
        for d in [self.test_dir]:
            if d.exists():
                shutil.rmtree(d)

    def test_choose_mask_best(self) -> None:
        """Tests that 'best' mode correctly selects the mask with the highest score."""
        masks = torch.zeros((2, 10, 10))
        masks[1, 5, 5] = 1.0
        scores = torch.tensor([0.1, 0.9])

        params = Sam3ThermoScenesParams(
            scene_dir=self.test_dir,
            sam3_checkpoint_dir=Path("."),
            mask_mode="best",
            text_prompt="test",
            bpe_path=Path("."),
        )

        mask = choose_mask(masks, scores, params)
        self.assertEqual(mask[5, 5], 1.0)
        self.assertEqual(mask[0, 0], 0.0)

    def test_choose_mask_union(self) -> None:
        """Tests that 'union' mode merges all masks above the confidence threshold."""
        masks = torch.zeros((2, 10, 10))
        masks[0, 2, 2] = 1.0
        masks[1, 8, 8] = 1.0
        scores = torch.tensor([0.8, 0.7])

        params = Sam3ThermoScenesParams(
            scene_dir=self.test_dir,
            sam3_checkpoint_dir=Path("."),
            mask_mode="union",
            confidence_threshold=0.5,
            text_prompt="test",
            bpe_path=Path("."),
        )

        mask = choose_mask(masks, scores, params)
        self.assertEqual(mask[2, 2], 1.0)
        self.assertEqual(mask[8, 8], 1.0)

    def test_pil_to_rgba(self) -> None:
        """Tests the conversion from RGB + mask to an RGBA PIL Image."""
        rgb = Image.new("RGB", (10, 10), color="white")
        alpha = np.zeros((10, 10), dtype=np.float32)
        alpha[0, 0] = 1.0

        rgba = pil_to_rgba_with_alpha(rgb, alpha)
        self.assertEqual(rgba.mode, "RGBA")
        self.assertEqual(rgba.getpixel((0, 0)), (255, 255, 255, 255))
        self.assertEqual(rgba.getpixel((1, 1)), (255, 255, 255, 0))

    @patch("rebel_shape.scripts.segmentation.mask_with_sam3.build_processor")
    def test_full_run_flow(self, mock_build_processor) -> None:
        """
        Integration test: mocks the SAM3 processor to verify the file system
        operations, JSON updates, and RGBA saving logic.
        """
        mock_processor = MagicMock()
        mock_build_processor.return_value = mock_processor
        mock_state = {"masks": torch.ones((1, 100, 100)), "scores": torch.tensor([0.9])}

        mock_processor.set_image.return_value = "dummy_state"
        mock_processor.set_text_prompt.return_value = mock_state

        params = Sam3ThermoScenesParams(
            scene_dir=self.test_dir,
            sam3_checkpoint_dir=Path("fake_ckpt"),
            text_prompt="test",
            bpe_path=Path("."),
            from_vggt=False,
        )

        run(params)

        out_img = self.test_dir / "images_sam3" / "frame_001.png"
        self.assertTrue(out_img.exists(), "Output image should be saved as .png")

        with (self.test_dir / "transforms.json").open() as f:
            updated_json = json.load(f)
            self.assertEqual(
                updated_json["frames"][0]["file_path"], "images_sam3/frame_001.png"
            )
