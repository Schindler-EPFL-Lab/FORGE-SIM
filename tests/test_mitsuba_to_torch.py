try:
    PYTORCH_IMPORTED = True
    import torch

    from rebel_shape.metrics.novel_view_synthesis import mitsuba_to_torch
except ImportError:
    PYTORCH_IMPORTED = False

import unittest

import mitsuba as mi
import numpy as np


@unittest.skipUnless(PYTORCH_IMPORTED, "pytorch not installed.")
class TestMitsubaToTorch(unittest.TestCase):
    """Tests conversion from mitsuba to torch function"""

    @classmethod
    def setUpClass(cls):
        mi.set_variant("llvm_ad_rgb")

    def test_mitsuba_to_torch_conversion_raise(self):
        """
        Tests that mitsuba to torch function raises an error if the input has
        incorrect shape.
        """
        mitsuba_image_1 = mi.TensorXf(np.zeros((1, 12, 13, 3)))
        with self.assertRaises(RuntimeError):
            mitsuba_to_torch(mitsuba_image_1)

        mitsuba_image_2 = mi.TensorXf(np.zeros((17, 29)))
        with self.assertRaises(RuntimeError):
            mitsuba_to_torch(mitsuba_image_2)

    def test_mitsuba_to_torch_conversion(self):
        """
        Tests that mitsuba to torch function works correct.
        """
        mitsuba_image = mi.TensorXf(
            [[[0.4, 0.2, 0.7], [0.5, 0.7, 0.6]], [[0.8, 0.8, 0.5], [0.9, 0.2, 0.6]]]
        )
        torch_image_real = torch.tensor(
            [
                [[0.4, 0.5], [0.8, 0.9]],
                [[0.2, 0.7], [0.8, 0.2]],
                [[0.7, 0.6], [0.5, 0.6]],
            ]
        )

        torch_image_converted = mitsuba_to_torch(mitsuba_image)

        self.assertTrue(torch.allclose(torch_image_real, torch_image_converted))
