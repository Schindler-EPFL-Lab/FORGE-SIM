import unittest

from rebel_shape.plugin_wrappers.sampler import Sampler


class TestSampler(unittest.TestCase):
    def test_invalid_sample_count(self):
        """Test that negative or zero sample_count raises ValueError."""
        with self.assertRaises(ValueError) as context:
            Sampler(sample_count=0)

        self.assertIn("sample_count", str(context.exception))

        with self.assertRaises(ValueError) as context:
            Sampler(sample_count=-5)

        self.assertIn("sample_count", str(context.exception))

    def test_invalid_random_seed(self):
        """Test that negative random_seed raises ValueError."""
        with self.assertRaises(ValueError) as context:
            Sampler(random_seed=-1)

        self.assertIn("random_seed", str(context.exception))

    def test_mitsuba_dict(self):
        """
        Test mitsuba_dict returns the correct dictionary representation for
        all sampler types.
        """
        for sampler_string, sampler_type in [
            ("independent", Sampler.Type.INDEPENDENT),
            ("stratified", Sampler.Type.STRATIFIED),
            ("multijitter", Sampler.Type.MULTIJITTER),
            ("orthogonal", Sampler.Type.ORTHOGONAL),
            ("ldsampler", Sampler.Type.LDSAMPLER),
        ]:
            with self.subTest(sampler_type=sampler_type, sampler_string=sampler_string):
                sampler = Sampler(sampler_type, 32, 123)  # Example fixed values
                expected_dict = {
                    "type": sampler_string,
                    "sample_count": 32,
                    "seed": 123,
                }
                self.assertEqual(sampler.mitsuba_dict, expected_dict)
