import unittest
from pathlib import Path

import mitsuba as mi
import numpy as np

from rebel_shape.loss_evaluators.image_loss_l1 import ImageLossL1
from rebel_shape.loss_evaluators.image_loss_l2 import ImageLossL2
from rebel_shape.loss_evaluators.image_loss_msssim import ImageLossMSSSIM
from rebel_shape.loss_evaluators.image_loss_ssim import ImageLossSSIM


class TestImageLossFunctions(unittest.TestCase):
    def setUp(self):
        mi.set_variant("llvm_ad_rgb")

        self.image_1 = mi.TensorXf([[[0.1], [0.2]], [[0.3], [0.4]]])
        self.image_2 = mi.TensorXf([[[0.0], [0.4]], [[0.3], [0.8]]])

        local_dir = Path(__file__).parent.resolve()
        dataset_path = local_dir / "data/thermoscenes_example/buildingA_winter"

        # Resolution reduced for speed, but not too much so MS-SSIM works
        # Remember to update expected values if you change this
        self.ssim_image_1 = mi.TensorXf(
            mi.Bitmap(str(dataset_path / "images/frame_eval_00012.png")).convert(
                component_format=mi.Struct.Type.Float32, srgb_gamma=False
            )
        )[::4, ::4, :]
        self.ssim_image_2 = mi.TensorXf(
            mi.Bitmap(str(dataset_path / "images/frame_train_00021.png")).convert(
                component_format=mi.Struct.Type.Float32, srgb_gamma=False
            )
        )[::4, ::4, :]

    def test_l1_loss(self):
        l1_loss = ImageLossL1()
        computed_l1_loss = l1_loss.evaluate(self.image_1, self.image_2)

        self.assertAlmostEqual(
            computed_l1_loss.numpy().astype(np.float64), 0.25 * 0.7, places=6
        )

        same_image_loss = l1_loss.evaluate(self.image_1, self.image_1)
        self.assertAlmostEqual(
            same_image_loss.numpy().astype(np.float64), 0.0, places=6
        )

    def test_l2_loss(self):
        l2_loss = ImageLossL2()
        computed_l2_loss = l2_loss.evaluate(self.image_1, self.image_2)

        self.assertAlmostEqual(
            computed_l2_loss.numpy().astype(np.float64), 0.25 * 0.21, places=6
        )

        same_image_loss = l2_loss.evaluate(self.image_1, self.image_1)
        self.assertAlmostEqual(
            same_image_loss.numpy().astype(np.float64), 0.0, places=6
        )

    def test_ssim_loss(self):
        """
        Tests the Mitsuba SSIM implementation against the PyTorch SSIM. Note
        that they do not behave the same around the edges of the images so the
        reference value is a little off. The values are hardcoded to avoid depending
        on PyTorch.
        """
        ssim_loss = ImageLossSSIM(k1=0.01, k2=0.03, dynamic_range=1.0)
        computed_ssim = 1.0 - ssim_loss.evaluate(self.ssim_image_1, self.ssim_image_2)

        # PyTorch SSIM is slightly different because the edges of the images are
        # not handled the same way in the two implementations - in the PyTorch
        # implementation, the edges are mirrored, while in the Mitsuba implementation
        # they are just duplicated. The number is hardcoded to avoid pytorch
        expected_ssim = 0.3842

        self.assertAlmostEqual(
            computed_ssim.numpy().astype(np.float64), expected_ssim, places=3
        )

        same_image_loss = ssim_loss.evaluate(self.ssim_image_1, self.ssim_image_1)
        self.assertAlmostEqual(
            same_image_loss.numpy().astype(np.float64), 0.0, places=6
        )

    def test_msssim_loss(self):
        """
        Tests the Mitsuba MS-SSIM implementation against the PyTorch MS-SSIM. Note
        that they do not behave the same around the edges of the images so the
        reference value is a little off. The values are hardcoded to avoid depending
        on PyTorch.
        """
        msssim_loss = ImageLossMSSSIM(
            k1=0.01, k2=0.03, dynamic_range=1.0, weights=[0.25, 0.5, 0.25]
        )
        computed_msssim = 1.0 - msssim_loss.evaluate(
            self.ssim_image_1, self.ssim_image_2
        )

        # Expected MSSSIM is slightly different because the edges of the images are
        # not handled the same way in the two implementations - in the PyTorch
        # implementation, the edges are mirrored, while in the Mitsuba implementation
        # they are just duplicated.
        expected_msssim = 0.4677

        # Differences accumulate over the scales, so we allow a larger, but still
        # reasonable tolerance
        equal = np.isclose(
            computed_msssim.numpy().astype(np.float64), expected_msssim, atol=0.02
        )

        self.assertTrue(
            equal, f"Computed MSSSIM: {computed_msssim}, Expected: {expected_msssim}"
        )

        same_image_loss = msssim_loss.evaluate(self.ssim_image_1, self.ssim_image_1)
        self.assertAlmostEqual(
            same_image_loss.numpy().astype(np.float64), 0.0, places=6
        )
