import unittest

import drjit as dr
import mitsuba as mi
import numpy as np

from rebel_shape.data_processing.masked_pose_image import MaskedPoseImage
from rebel_shape.plugin_wrappers.perspective_camera import PerspectiveCamera


class TestMaskedPoseImage(unittest.TestCase):
    """Unit tests for MaskedPoseImage class."""

    @classmethod
    def setUpClass(cls):
        """Set up a mi_variant for testing."""
        mi.set_variant("llvm_ad_rgb")

        cls.full_0_mask_5x5 = mi.TensorXf(np.zeros((5, 5)))
        cls.full_1_mask_5x5 = mi.TensorXf(np.ones((5, 5)))
        cls.random_mask_5x5 = mi.TensorXf(
            [
                [0, 1, 0, 0, 1],
                [1, 1, 1, 1, 1],
                [0, 1, 0, 0, 1],
                [0, 0, 0, 1, 1],
                [0, 1, 1, 0, 1],
            ]
        )

        cls.resolution = (5, 5)

        cls.image_5x5x3 = mi.TensorXf(
            [
                [
                    [0.37454012, 0.95071431, 0.73199394],
                    [0.59865848, 0.15601864, 0.15599452],
                    [0.05808361, 0.86617615, 0.60111501],
                    [0.70807258, 0.02058449, 0.96990985],
                    [0.83244264, 0.21233911, 0.18182497],
                ],
                [
                    [0.18340451, 0.30424224, 0.52475643],
                    [0.43194502, 0.29122914, 0.61185289],
                    [0.13949386, 0.29214465, 0.36636184],
                    [0.45606998, 0.78517596, 0.19967378],
                    [0.51423444, 0.59241457, 0.04645041],
                ],
                [
                    [0.60754485, 0.17052412, 0.06505159],
                    [0.94888554, 0.96563203, 0.80839735],
                    [0.30461377, 0.09767211, 0.68423303],
                    [0.44015249, 0.12203823, 0.49517691],
                    [0.03438852, 0.9093204, 0.25877998],
                ],
                [
                    [0.66252228, 0.31171108, 0.52006802],
                    [0.54671028, 0.18485446, 0.96958463],
                    [0.77513282, 0.93949894, 0.89482735],
                    [0.59789998, 0.92187424, 0.0884925],
                    [0.19598286, 0.04522729, 0.32533033],
                ],
                [
                    [0.38867729, 0.27134903, 0.82873751],
                    [0.35675333, 0.28093451, 0.54269608],
                    [0.14092422, 0.80219698, 0.07455064],
                    [0.98688694, 0.77224477, 0.19871568],
                    [0.00552212, 0.81546143, 0.70685734],
                ],
            ]
        )

        cls.image_5x5x3_masked_by_random_mask_5x5 = mi.TensorXf(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.59865848, 0.15601864, 0.15599452],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.83244264, 0.21233911, 0.18182497],
                ],
                [
                    [0.18340451, 0.30424224, 0.52475643],
                    [0.43194502, 0.29122914, 0.61185289],
                    [0.13949386, 0.29214465, 0.36636184],
                    [0.45606998, 0.78517596, 0.19967378],
                    [0.51423444, 0.59241457, 0.04645041],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [0.94888554, 0.96563203, 0.80839735],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.03438852, 0.9093204, 0.25877998],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.59789998, 0.92187424, 0.0884925],
                    [0.19598286, 0.04522729, 0.32533033],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [0.35675333, 0.28093451, 0.54269608],
                    [0.14092422, 0.80219698, 0.07455064],
                    [0.0, 0.0, 0.0],
                    [0.00552212, 0.81546143, 0.70685734],
                ],
            ]
        )

        cls.camera = PerspectiveCamera.from_origin_and_target(
            origin=(1.0, 1.0, 1.0), target=(0.0, 0.0, 0.0)
        )

    def test_invalid_mask_ndims(self):
        """
        Test that MaskedPoseImage raises ValueError for invalid mask dimensions.
        """

        with self.assertRaises(ValueError) as context:
            MaskedPoseImage(
                image=self.image_5x5x3,
                camera=self.camera,
                mask=mi.TensorXf(np.full((5, 5, 5, 5, 5), fill_value=0.5)),
            )

        self.assertIn("ndim", str(context.exception))

        with self.assertRaises(ValueError):
            MaskedPoseImage(
                image=self.image_5x5x3,
                camera=self.camera,
                mask=mi.TensorXf(np.full((5,), fill_value=0.5)),
            )

        self.assertIn("ndim", str(context.exception))

    def test_invalid_mask_resolution(self):
        """
        Test that MaskedPoseImage raises a ValueError when the
        mask resolution does not match the image resolution.
        """
        with self.assertRaises(ValueError) as context:
            MaskedPoseImage(
                image=self.image_5x5x3,
                camera=self.camera,
                mask=mi.TensorXf(np.full((3, 2), fill_value=1.0)),
            )
        self.assertIn("resolution", str(context.exception))

    def test_invalid_mask_boundaries(self):
        """
        Test MaskedPoseImage behaviour when mask.min < 0 or mask.max > 1
        """
        with self.assertRaises(ValueError) as context:
            MaskedPoseImage(
                image=self.image_5x5x3,
                camera=self.camera,
                mask=mi.TensorXf(np.full((5, 5), fill_value=-1.0)),
            )
        self.assertIn("range [0, 1]", str(context.exception))

        with self.assertRaises(ValueError) as context:
            MaskedPoseImage(
                image=self.image_5x5x3,
                camera=self.camera,
                mask=mi.TensorXf(np.full((5, 5), fill_value=2.0)),
            )
        self.assertIn("range [0, 1]", str(context.exception))

    def test_invalid_mask_non_binary_values(self):
        """
        Tests that MaskedPoseImage raises a ValueError when the mask contains
        non-binary values (e.g., 0.5).
        """
        with self.assertRaises(ValueError) as context:
            MaskedPoseImage(
                image=self.image_5x5x3,
                camera=self.camera,
                mask=mi.TensorXf(np.full((5, 5), fill_value=0.5)),
            )
        self.assertIn("Mask values should be either 0 or 1", str(context.exception))

    def test_mask_not_presented(self):
        """
        Tests properties .image and .original_image when mask is not presented.
        """
        image = MaskedPoseImage(
            image=self.image_5x5x3,
            camera=self.camera,
            mask=None,
        )

        self.assertTrue(dr.allclose(image.image, self.image_5x5x3))
        self.assertTrue(dr.allclose(image.original_image, self.image_5x5x3))
        self.assertTrue(
            dr.allclose(image.apply_mask(self.image_5x5x3), self.image_5x5x3)
        )
        self.assertIsNone(image.mask)

    def test_invalid_apply_mask_ndims(self):
        """
        Test that MaskedPoseImage.apply_mask raises
        ValueError for invalid input dimensions.
        """
        image = MaskedPoseImage(
            image=self.image_5x5x3,
            camera=self.camera,
            mask=self.random_mask_5x5,
        )

        with self.assertRaises(ValueError) as context:
            image.apply_mask(mi.TensorXf(np.full((5, 5, 5, 5), fill_value=0.5)))

        self.assertIn("dim", str(context.exception))

        with self.assertRaises(ValueError) as context:
            image.apply_mask(mi.TensorXf(np.full((5,), fill_value=0.5)))

        self.assertIn("dim", str(context.exception))

    def test_invalid_apply_mask_resolution(self):
        """
        Test that MaskedPoseImage.apply_mask raises RuntimeError when the input tensor
        resolution does not match the image resolution.
        """
        image = MaskedPoseImage(
            image=self.image_5x5x3,
            camera=self.camera,
            mask=self.full_1_mask_5x5,
        )

        with self.assertRaises(ValueError) as context:
            image.apply_mask(mi.TensorXf(np.full((3, 3, 3), fill_value=0.5)))

        self.assertIn("Resolution of the input tensor", str(context.exception))

    def test_mask_is_presented_3d_images(self):
        """
        Tests that the MaskedPoseImage correctly applies masks to 3d images.
        """
        test_cases = [
            # (mask, masked_image_result)
            (self.full_0_mask_5x5[..., None], mi.TensorXf(np.zeros((5, 5, 3)))),
            (self.full_1_mask_5x5, self.image_5x5x3),
            (self.random_mask_5x5, self.image_5x5x3_masked_by_random_mask_5x5),
        ]

        for i, (mask, masked_image_result) in enumerate(test_cases):
            with self.subTest(i=i):
                image_mask = MaskedPoseImage(
                    image=self.image_5x5x3,
                    camera=self.camera,
                    mask=mask,
                )

                self.assertTrue(dr.allclose(image_mask.image, masked_image_result))
                self.assertTrue(
                    dr.allclose(image_mask.original_image, self.image_5x5x3)
                )
                self.assertTrue(
                    dr.allclose(
                        image_mask.apply_mask(self.image_5x5x3),
                        masked_image_result,
                    )
                )
                returned_mask = image_mask.mask
                assert returned_mask is not None
                if returned_mask.ndim == 2 and mask.ndim == 3:
                    returned_mask = returned_mask[..., None]
                if returned_mask.ndim == 3 and mask.ndim == 2:
                    returned_mask = returned_mask[..., 0]
                self.assertTrue(dr.allclose(returned_mask, mask))

    def test_mask_is_presented_2d_images(self):
        """
        Tests that the MaskedPoseImage correctly applies masks to 2d images.
        """
        test_cases = [
            # (mask, masked_image_result)
            (self.full_0_mask_5x5, mi.TensorXf(np.zeros((5, 5)))),
            (self.full_1_mask_5x5[..., None], self.image_5x5x3[..., 0]),
            (self.random_mask_5x5, self.image_5x5x3_masked_by_random_mask_5x5[..., 0]),
        ]

        for i, (mask, masked_image_result) in enumerate(test_cases):
            with self.subTest(i=i):
                image_mask = MaskedPoseImage(
                    image=self.image_5x5x3[..., 0],
                    camera=self.camera,
                    mask=mask,
                )

                self.assertTrue(dr.allclose(image_mask.image, masked_image_result))
                self.assertTrue(
                    dr.allclose(image_mask.original_image, self.image_5x5x3[..., 0])
                )
                self.assertTrue(
                    dr.allclose(
                        image_mask.apply_mask(self.image_5x5x3[..., 0]),
                        masked_image_result,
                    )
                )
                returned_mask = image_mask.mask
                assert returned_mask is not None
                if returned_mask.ndim == 2 and mask.ndim == 3:
                    returned_mask = returned_mask[..., None]
                if returned_mask.ndim == 3 and mask.ndim == 2:
                    returned_mask = returned_mask[..., 0]
                self.assertTrue(dr.allclose(returned_mask, mask))
