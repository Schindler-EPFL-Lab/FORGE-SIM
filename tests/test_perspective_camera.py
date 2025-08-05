import unittest

import mitsuba as mi

from rebel_shape.plugin_wrappers.perspective_camera import PerspectiveCamera
from rebel_shape.plugin_wrappers.sampler import Sampler


class TestPerspectiveCamera(unittest.TestCase):
    def setUp(self):
        """Initialize Mitsuba variant"""
        mi.set_variant("llvm_ad_rgb")

    def test_invalid_origin(self):
        """Test that an invalid origin raises ValueError."""
        with self.assertRaises(ValueError) as context:
            PerspectiveCamera.from_origin_and_target(
                origin=(1.0, 2.0), target=(1.0, 2.0, 3.0)
            )

        self.assertIn("3", str(context.exception))

    def test_invalid_target(self):
        """Test that an invalid target raises ValueError."""
        with self.assertRaises(ValueError) as context:
            PerspectiveCamera.from_origin_and_target(
                origin=(1.0, 2.0, 3.0), target=(1.0, 2.0)
            )

        self.assertIn("3", str(context.exception))

    def test_invalid_camera_resolution(self):
        """Test that invalid camera_resolution values raise ValueError."""
        with self.assertRaises(ValueError) as context:
            PerspectiveCamera.from_origin_and_target(
                origin=(1, 1, 1), target=(2, 2, 2), camera_resolution=(0, 256)
            )

        self.assertIn("sensor_resolution", str(context.exception))

        with self.assertRaises(ValueError) as context:
            PerspectiveCamera.from_origin_and_target(
                origin=(1, 1, 1), target=(2, 2, 2), camera_resolution=(256, 0)
            )

        self.assertIn("sensor_resolution", str(context.exception))

    def test_mitsuba_sample_overflow(self):
        """Test that less than 2**32 samples are generated."""
        with self.assertRaises(ValueError) as context:
            # Generate exactly 2**32 samples
            PerspectiveCamera.from_origin_and_target(
                origin=(1, 1, 1),
                target=(2, 2, 2),
                camera_resolution=(1024, 1024),
                sampler=Sampler(sample_count=4096),
            )

        self.assertIn("2**32", str(context.exception))

    def test_invalid_fov(self):
        """Test that invalid field of view (fov) raises ValueError."""
        with self.assertRaises(ValueError) as context:
            PerspectiveCamera.from_origin_and_target(
                origin=(1, 1, 1), target=(2, 2, 2), fov=0
            )

        self.assertIn("0 and 180", str(context.exception))

        with self.assertRaises(ValueError) as context:
            PerspectiveCamera.from_origin_and_target(
                origin=(1, 1, 1), target=(2, 2, 2), fov=-30
            )

        self.assertIn("0 and 180", str(context.exception))

    def test_mitsuba_dict(self):
        """Test mitsuba_dict returns the correct dictionary representation."""
        origin = (0.0, 1.0, 2.0)
        target = (1.0, 2.0, 3.0)
        camera_resolution = (256, 256)
        fov = 45

        sampler = Sampler(type=Sampler.Type.INDEPENDENT, sample_count=16, random_seed=0)

        camera = PerspectiveCamera.from_origin_and_target(
            origin=origin,
            target=target,
            camera_resolution=camera_resolution,
            fov=fov,
            sampler=sampler,
        )

        expected_dict = {
            "type": "perspective",
            "fov": 45,
            "fov_axis": "x",
            "to_world": mi.ScalarTransform4f().look_at(
                target=mi.ScalarPoint3f(target),
                origin=mi.ScalarPoint3f(origin),
                up=mi.ScalarPoint3f([0, 1, 0]),
            ),
            "film": {
                "type": "hdrfilm",
                "width": camera_resolution[0],
                "height": camera_resolution[1],
                "pixel_format": "rgb",
                "filter": {"type": "gaussian"},
                "sample_border": True,
            },
            "sampler": {"type": "independent", "sample_count": 16, "seed": 0},
        }

        self.assertEqual(camera.mitsuba_dict, expected_dict)
