import unittest

import numpy as np

from rebel_shape.scripts.segmentation.extract_foreground_and_background import (  # noqa: E501
    _not_masked_in_convex_hull,
)


class TestNotMaskedInConvexHull(unittest.TestCase):
    """Unit tests for _not_masked_in_convex_hull"""

    def test_invalid_input(self) -> None:
        """
        Test that _not_masked_in_convex_hull raises ValueError for invalid inputs.
        """
        tensor_invalid_cases = [
            np.zeros((481, 641, 3), dtype=np.bool),
            np.zeros((481,), dtype=np.bool),
        ]
        for i, tensor_invalid in enumerate(tensor_invalid_cases):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    _not_masked_in_convex_hull(tensor_invalid)

    def test_object_full_image(self) -> None:
        """
        Tests that not_masked_in_convex_hull produces all False if input image mask has
        the shape of the whole image, i.e. the convex hull(object mask) = object mask
        """
        mask = np.ones((481, 641), dtype=np.bool)

        expected_result = np.zeros(mask.shape[:2], dtype=bool)
        result = _not_masked_in_convex_hull(mask)
        self.assertTrue(np.all(expected_result == result))

    def test_big_image(self) -> None:
        """
        Tests that not_masked_in_convex_hull gives a correct answer for a rectangle
        with a rectangular hole.
        """
        mask = np.zeros((480, 640), dtype=np.bool)
        mask[100:500, 100:400] = True
        # create a hole
        mask[200:300, 150:400] = False

        expected_result = np.zeros((480, 640), dtype=np.bool)
        expected_result[200:300, 150:400] = True

        result = _not_masked_in_convex_hull(mask)
        self.assertTrue(np.all(expected_result == result))

    def test_multiple_holes(self) -> None:
        """
        Tests that not_masked_in_convex_hull gives a correct answer on a rectangle with
        multiple rectangular holes
        """
        mask = np.zeros((480, 640), dtype=np.bool)
        expected_result = np.zeros((480, 640), dtype=np.bool)

        mask[100:400, 100:500] = True

        holes = [
            [(110, 100), (200, 200)],
            [(410, 110), (490, 210)],
            [(160, 150), (250, 250)],
            [(300, 260), (400, 380)],
        ]

        for hole in holes:
            mask[hole[0][1] : hole[1][1], hole[0][0] : hole[0][1]] = False
            expected_result[hole[0][1] : hole[1][1], hole[0][0] : hole[0][1]] = True

        expected_result = expected_result

        result = _not_masked_in_convex_hull(mask)
        self.assertTrue(np.all(expected_result == result))
