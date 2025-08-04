import unittest

import mitsuba as mi
import numpy as np

from rebel_shape.property_optimizers.sigmoid_parametrization import (
    SigmoidParametrization,
)


class TestSigmoidParametrization(unittest.TestCase):
    def setUp(self):
        mi.set_variant("scalar_rgb")
        self.sigmoid_epsilon = SigmoidParametrization.SIGMOID_EPSILON

    def test_initialization(self):
        """Test valid and invalid initializations of SigmoidParametrization."""
        # Default initialization
        sigmoid_default = SigmoidParametrization()
        self.assertEqual(sigmoid_default._projection_steepness, 1.0)
        self.assertEqual(
            sigmoid_default._approximate_clip, -np.log(self.sigmoid_epsilon)
        )

        # Invalid steepness values
        with self.assertRaises(ValueError):
            SigmoidParametrization(projection_steepness=0.0)

        with self.assertRaises(ValueError):
            SigmoidParametrization(projection_steepness=-1.0)

    def test_from_parametric(self):
        """Test transformation from parametric to [0, 1] range."""
        sigmoid = SigmoidParametrization()
        result = sigmoid.from_parametric(mi.TensorXf([-100.0, 0.0, 100.0, 1.0]))

        expected = np.array(
            [self.sigmoid_epsilon, 0.5, 1.0 - self.sigmoid_epsilon, 0.731012367]
        )
        np.testing.assert_array_almost_equal(result.numpy(), expected, decimal=6)

    def test_to_parametric(self):
        """Test transformation from [0, 1] range to real line."""
        sigmoid = SigmoidParametrization()
        result = sigmoid.to_parametric(mi.TensorXf([0.0, 0.5, 1.0, 0.731053957]))

        expected = np.array([-9.210040327, 0.0, 9.210040327, 1.0])
        # There is small drift in the expected value due to numerical precision
        # for the 1.0 case, so we use decimal=3
        np.testing.assert_array_almost_equal(result.numpy(), expected, decimal=3)

    def test_roundtrip_transformation(self):
        """Test that from_parametric and to_parametric are inverses."""

        sigmoid = SigmoidParametrization()
        original_values = mi.TensorXf([0.0, 1.0, -4.0, 5.0])
        transformed = sigmoid.from_parametric(original_values)
        restored = sigmoid.to_parametric(transformed)

        np.testing.assert_array_almost_equal(
            restored.numpy(), original_values.numpy(), decimal=5
        )

    def test_projection_steepness_effect(self):
        """Test different projection steepness values."""
        sigmoid1 = SigmoidParametrization(projection_steepness=1.0)
        sigmoid2 = SigmoidParametrization(projection_steepness=2.0)

        result1 = sigmoid1.from_parametric(mi.TensorXf(1.0))
        np.testing.assert_array_almost_equal(result1.numpy(), [0.731012367], decimal=6)

        result2 = sigmoid2.from_parametric(mi.TensorXf(1.0))
        np.testing.assert_array_almost_equal(result2.numpy(), [0.880720919], decimal=6)
