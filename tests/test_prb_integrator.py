import unittest

from rebel_shape.plugin_wrappers.prb_integrator import GuidingOptions, PRBIntegrator


class TestPRBIntegrator(unittest.TestCase):
    def test_invalid_max_bounces(self):
        """Test that invalid max_bounces values raise ValueError."""
        with self.assertRaises(ValueError) as context:
            PRBIntegrator(max_bounces=0)

        self.assertIn("at least one bounce", str(context.exception))

    def test_invalid_bounces_without_rr(self):
        """Test that invalid bounces_without_rr values raise ValueError."""
        with self.assertRaises(ValueError) as context:
            PRBIntegrator(bounces_without_russian_roulette=0)

        self.assertIn("at least one bounce", str(context.exception))

    def test_invalid_spp_values(self):
        """Test that spp_continuous, spp_projective, spp_indirect must be positive."""
        with self.assertRaises(ValueError) as context:
            PRBIntegrator(spp_continuous=0)

        self.assertIn("spp_continuous", str(context.exception))

        with self.assertRaises(ValueError) as context:
            PRBIntegrator(spp_projective=-5)

        self.assertIn("spp_projective", str(context.exception))

        with self.assertRaises(ValueError) as context:
            PRBIntegrator(spp_indirect=0)

        self.assertIn("spp_indirect", str(context.exception))

    def test_invalid_guiding_rounds(self):
        """Test that guiding_rounds must be at least 1 if guiding is enabled."""
        with self.assertRaises(ValueError) as context:
            PRBIntegrator(guiding=GuidingOptions.GRID, guiding_rounds=0)

        self.assertIn("at least one guiding round", str(context.exception))

    def test_mitsuba_dict(self):
        """Test mitsuba_dict returns the correct dictionary representation."""
        for guiding_string, guiding_type in [
            ("none", GuidingOptions.NONE),
            ("grid", GuidingOptions.GRID),
            ("octree", GuidingOptions.OCTREE),
        ]:
            with self.subTest(guiding_type=guiding_type, guiding_string=guiding_string):
                integrator = PRBIntegrator(
                    max_bounces=10,
                    bounces_without_russian_roulette=5,
                    spp_continuous=16,
                    spp_projective=16,
                    spp_indirect=16,
                    guiding=guiding_type,
                    guiding_rounds=2,
                )

                expected_dict = {
                    "type": "prb_projective",
                    "max_depth": 10,
                    "rr_depth": 5,
                    "sppc": 16,
                    "sppp": 16,
                    "sppi": 16,
                    "guiding": guiding_string,
                    "guiding_rounds": 2,
                }

                self.assertEqual(integrator.mitsuba_dict, expected_dict)
