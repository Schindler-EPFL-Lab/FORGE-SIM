import unittest
from itertools import product

import numpy as np

SCIKIT_IMAGE = True
try:
    from rebel_shape.scripts.segmentation.extract_foreground_and_background import (  # noqa: E501
        extract_foreground_and_background,
    )
except ImportError:
    SCIKIT_IMAGE = False


@unittest.skipUnless(SCIKIT_IMAGE, "Skimage and moge required for full run of tests.")
class TestExtractForegroundAndBackground(unittest.TestCase):
    """Unit tests for extract_foreground_and_background"""

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def test_invalid_input(self) -> None:
        """
        Test that extract_foreground_and_background raises ValueError for invalid inputs
        """
        image_invalid_cases_cross = [
            np.full((481, 641), fill_value=127, dtype=np.uint8),
            np.full((1, 482, 642, 4), fill_value=127, dtype=np.uint8),
            np.full((481, 641, 3), fill_value=127, dtype=np.uint8),
        ]

        depth_invalid_cases_cross = [
            np.ones((481,), dtype=np.float32),
            np.ones((1, 482, 642), dtype=np.float32),
        ]

        image_depth_invalid_cases = [
            (
                np.zeros((481, 641, 4), dtype=np.uint8),
                np.ones((123, 602), dtype=np.float32),
            ),
            (
                np.full((481, 641, 4), fill_value=127, dtype=np.uint8),
                np.ones((481, 641), dtype=np.float32),
            ),
        ]

        image_depth_invalid_cases = image_depth_invalid_cases + list(
            product(image_invalid_cases_cross, depth_invalid_cases_cross)
        )

        for i, (image, depth) in enumerate(image_depth_invalid_cases):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    extract_foreground_and_background(image=image, depth=depth)

    def test_object_full_image(self) -> None:
        """
        Tests that extract_foreground_and_background produces correct answers if the
        object mask is the whole image
        """
        image = np.full((481, 641, 4), fill_value=255, dtype=np.uint8)
        depth = np.ones(image.shape[:2], dtype=np.float32)

        expected_foreground = np.zeros(image.shape[:2], dtype=bool)
        expected_background = np.zeros(image.shape[:2], dtype=bool)

        foreground, background = extract_foreground_and_background(
            image=image, depth=depth
        )
        self.assertTrue(np.all(foreground == expected_foreground))
        self.assertTrue(np.all(background == expected_background))

    def test_simple_rectangles(self) -> None:
        """
        Tests that extract_foreground_and_background gives a correct answer when we have
        three nesterd rectangles as below:

        .---------1---------.
        |   .-----2-----.   |
        |   |   .-3-.   |   |
        |   |   |   |   |   |
        |   |   '---'   |   |
        |   '-----------'   |
        '-------------------'

        where:
        - 1 represents a background
        - 2 represents an object
        - 3 represents a foreground
        """
        image = np.full((480, 640, 3), fill_value=127, dtype=np.uint8)
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:400, 100:600] = 255
        # create a hole
        mask[200:300, 200:500] = 0
        image = np.concatenate([image, mask[:, :, None]], axis=-1)

        depth = np.ones((480, 640), dtype=np.float32) * 16.0
        depth[100:400, 100:600] = 8.0
        depth[200:300, 200:500] = 4.0

        expected_foreground = np.zeros((480, 640), dtype=np.bool_)
        expected_foreground[200:300, 200:500] = True

        expected_background = np.ones((480, 640), dtype=np.bool_)
        expected_background[100:400, 100:600] = False

        foreground, background = extract_foreground_and_background(
            image=image, depth=depth
        )

        self.assertTrue(np.all(foreground == expected_foreground))
        self.assertTrue(np.all(background == expected_background))

    def test_two_rectangles(self) -> None:
        """
        Tests that extract_foreground_and_background gives a correct answer when we have
        four nesterd rectangles as below:

        .-------------1------------.
        |   .---------2--------.   |
        |   |   .-3-.  .-4-.   |   |
        |   |   |   |  |   |   |   |
        |   |   '---'  '---'   |   |
        |   '------------------'   |
        '--------------------------'

        where:
        - 1 represents a background
        - 2 represents an object
        - 3 represents a foreground
        - 4 represents a background
        """

        image = np.full((480, 640, 3), fill_value=127, dtype=np.uint8)
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:400, 100:600] = 255
        # create holes
        mask[200:300, 200:300] = 0
        mask[200:300, 400:500] = 0
        image = np.concatenate([image, mask[:, :, None]], axis=-1)

        depth = np.ones((480, 640), dtype=np.float32) * 16.0
        depth[100:400, 100:600] = 8.0
        depth[200:300, 200:300] = 4.0  # foreground
        depth[200:300, 400:500] = 12.0  # background

        expected_foreground = np.zeros((480, 640), dtype=np.bool_)
        expected_foreground[200:300, 200:300] = True

        expected_background = np.ones((480, 640), dtype=np.bool_)
        expected_background[100:400, 100:600] = False
        expected_background[200:300, 400:500] = True

        foreground, background = extract_foreground_and_background(
            image=image, depth=depth
        )

        self.assertTrue(np.all(foreground == expected_foreground))
        self.assertTrue(np.all(background == expected_background))


if __name__ == "__main__":
    unittest.main()
