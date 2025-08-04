import unittest

import drjit as dr
import mitsuba as mi
import numpy as np

from rebel_shape.splines.bivariate_bspline import BivariateBSpline
from rebel_shape.splines.bspline import BSpline


class TestBivariateBSpline(unittest.TestCase):
    """
    Unit tests for the `BivariateBSpline` class.

    Note that the `interpolate` method is not tested here as B-spline curves and
    surfaces use it to interpolate values. Thus, see `test_bspline_patch.py` for
    tests that cover interpolation of values. Same applies to knot insertion with
    updates to the values tensor.
    """

    @classmethod
    def setUpClass(cls):
        """Set up an example `BivariateBSpline` for testing."""
        mi.set_variant("scalar_rgb")
        # Create simple uniform B-splines for testing
        cls.knots_u = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0]
        cls.knots_v = [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0]

        cls.u_spline = BSpline(cls.knots_u, degree=2)
        cls.v_spline = BSpline(cls.knots_v, degree=2)
        cls.bivariate_bspline = BivariateBSpline(cls.u_spline, cls.v_spline)

        cls.evaluation_samples_per_direction = 13
        """
        Number of samples per each direction when evaluating the surface.
        Should be kept small to avoid long test times, but large enough
        to ensure that the surface is well sampled.
        """

    def test_num_basis_functions(self):
        """Test property for number of basis functions"""
        self.assertEqual(self.bivariate_bspline.num_basis_functions, 20)
        self.assertEqual(self.bivariate_bspline.basis_functions_per_span, 9)

    def test_degree(self):
        """Test degree property"""
        self.assertEqual(self.bivariate_bspline.degree, 2)

        # Test with different degrees
        u_spline = BSpline(self.knots_u, degree=3)
        v_spline = BSpline(self.knots_v, degree=2)
        third_degree_spline = BivariateBSpline(u_spline, v_spline)

        self.assertEqual(third_degree_spline.degree, 3)

    def test_validate_parametric_coords(self):
        """
        Test validation of valid parametric coordinates.

        Multiple invalid cases are tested to ensure that the validation works correctly.
        """
        valid_params = mi.TensorXf([[1.0, 1.0], [2.0, 1.5]])
        try:
            self.bivariate_bspline.validate_parametric_coords(valid_params)
        except Exception as e:
            self.fail(f"validate_parametric_coords() raised {e} unexpectedly!")

        invalid_test_cases = [
            # (description, test_input)
            ("not a 2D tensor", mi.TensorXf([1.0, 1.0])),
            (
                "3 parametric coordinates",
                mi.TensorXf([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]),
            ),
            (
                "parametric direction u out of range (lower bound)",
                mi.TensorXf([[1.0, 1.0], [-2.0, 1.5]]),
            ),
            (
                "parametric direction u out of range (upper bound)",
                mi.TensorXf([[1.0, 1.0], [20.0, 1.5]]),
            ),
            (
                "parametric direction v out of range (lower bound)",
                mi.TensorXf([[1.0, 1.0], [1.5, -2.0]]),
            ),
            (
                "parametric direction v out of range (upper bound)",
                mi.TensorXf([[1.0, 1.0], [1.5, 20.0]]),
            ),
        ]

        for description, invalid_input in invalid_test_cases:
            with self.subTest(description=description):
                with self.assertRaises(ValueError):
                    self.bivariate_bspline.validate_parametric_coords(invalid_input)

    def test_insert_knot(self):
        """Test knot insertion in the u direction."""
        new_spline = BivariateBSpline(
            u_spline=BSpline(knot_vector=self.knots_u, degree=2),
            v_spline=BSpline(knot_vector=self.knots_v, degree=2),
        )

        new_spline.insert_knot_u(knot_position=2.5)
        self.assertEqual(
            new_spline.u_spline.knots.tolist(),
            [0.0, 0.0, 0.0, 1.0, 2.0, 2.5, 3.0, 3.0, 3.0],
        )

        new_spline.insert_knot_v(knot_position=1.5)
        self.assertEqual(
            new_spline.v_spline.knots.tolist(), [0.0, 0.0, 0.0, 1.0, 1.5, 2.0, 2.0, 2.0]
        )

    def test_knot_spans(self):
        """Tests the knot spans computation."""
        knot_spans = self.bivariate_bspline.knot_spans

        # The knot spans should be the intervals between the knots
        expected_spans = np.array(
            [
                [[0.0, 1.0], [0.0, 1.0]],
                [[1.0, 2.0], [0.0, 1.0]],
                [[2.0, 3.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 2.0]],
                [[1.0, 2.0], [1.0, 2.0]],
                [[2.0, 3.0], [1.0, 2.0]],
            ]
        )

        np.testing.assert_array_almost_equal(knot_spans.numpy(), expected_spans)

    def test_has_open_knot_vector(self):
        """Test if the B-spline has an open knot vector."""
        self.assertTrue(self.bivariate_bspline.has_open_knot_vectors)

        # Test with a closed knot vector
        closed_u_spline = BSpline([0.0, 1.0, 2.0, 3.0], degree=2)
        closed_v_spline = BSpline([0.0, 1.0, 2.0], degree=2)
        closed_bivariate_bspline = BivariateBSpline(closed_u_spline, closed_v_spline)

        self.assertFalse(closed_bivariate_bspline.has_open_knot_vectors)

    def test_first_derivative_finite_difference(self):
        test_cases = [
            {
                "u_knots": [0.0, 0.0, 0.49, 1.0, 1.0],
                "v_knots": [0.0, 0.0, 0.51, 1.0, 1.0],
                "u_degree": 1,
                "v_degree": 1,
            },
            {
                "u_knots": [0.0, 0.0, 0.0, 0.55, 1.0, 1.0, 1.0],
                "v_knots": [0.0, 0.0, 0.47, 1.0, 1.0],
                "u_degree": 2,
                "v_degree": 1,
            },
            {
                "u_knots": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                "v_knots": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                "u_degree": 2,
                "v_degree": 2,
            },
        ]

        for i, case in enumerate(test_cases):
            with self.subTest(i=i):
                u_spline = BSpline(knot_vector=case["u_knots"], degree=case["u_degree"])
                v_spline = BSpline(knot_vector=case["v_knots"], degree=case["v_degree"])
                surface = BivariateBSpline(
                    u_spline=u_spline,
                    v_spline=v_spline,
                )

                n_points = self.evaluation_samples_per_direction
                h = 1.0e-3

                # Create a grid of parametric coordinates
                u_vals = np.linspace(0.1, 0.9, n_points)
                v_vals = np.linspace(0.1, 0.9, n_points)
                u_grid, v_grid = np.meshgrid(u_vals, v_vals)
                params = np.column_stack((u_grid.flatten(), v_grid.flatten()))

                # Create offset parametric coordinates for finite difference
                params_u_low = params.copy()
                params_u_high = params.copy()
                params_v_low = params.copy()
                params_v_high = params.copy()

                params_u_low[:, 0] -= 0.5 * h
                params_u_high[:, 0] += 0.5 * h
                params_v_low[:, 1] -= 0.5 * h
                params_v_high[:, 1] += 0.5 * h

                # Evaluate surface at offset coordinates
                eval_u_low = surface.evaluate_basis_functions(
                    parametric_coords=mi.TensorXf(params_u_low)
                )
                eval_u_high = surface.evaluate_basis_functions(
                    parametric_coords=mi.TensorXf(params_u_high)
                )
                eval_v_low = surface.evaluate_basis_functions(
                    parametric_coords=mi.TensorXf(params_v_low)
                )
                eval_v_high = surface.evaluate_basis_functions(
                    parametric_coords=mi.TensorXf(params_v_high)
                )

                # Calculate finite differences
                u_finite_diff = (eval_u_high - eval_u_low) / h
                v_finite_diff = (eval_v_high - eval_v_low) / h

                # Calculate analytical derivatives
                first_derivative = surface.evaluate_basis_functions_first_derivative(
                    parametric_coords=mi.TensorXf(params)
                )

                # Compare analytical derivatives with finite differences
                np.testing.assert_array_almost_equal(
                    first_derivative[:, :, 0].numpy(), u_finite_diff.numpy(), decimal=3
                )
                np.testing.assert_array_almost_equal(
                    first_derivative[:, :, 1], v_finite_diff.numpy(), decimal=3
                )

    def test_active_basis_function_indices(self):
        """Test the active basis function indices computation."""
        params = mi.TensorXf([[0.5, 0.5], [1.5, 1.5], [2.5, 1.5]])
        indices = self.bivariate_bspline.active_basis_function_indices(params)

        # For the given parameters, we expect the indices of the active basis functions
        expected_indices = np.array(
            [
                [0, 1, 2, 5, 6, 7, 10, 11, 12],
                [6, 7, 8, 11, 12, 13, 16, 17, 18],
                [7, 8, 9, 12, 13, 14, 17, 18, 19],
            ]
        )

        np.testing.assert_array_equal(indices.numpy(), expected_indices)

    def test_project_function_constant_linear(self):
        """
        Constant and linear functions must always be able to be projected onto
        the bivariate spline exactly.
        """
        spline = self.bivariate_bspline

        projected_values = spline.project_function(
            func=lambda coords: dr.ones(dtype=mi.TensorXf, shape=(coords.shape[0], 1))
        )
        expected_values = np.ones((spline.num_basis_functions, 1))
        np.testing.assert_array_almost_equal(projected_values.numpy(), expected_values)

        # Linear function projection
        projected_values = spline.project_function(
            func=lambda coords: (coords[:, 0] + coords[:, 1])[:, None]
        )
        samples = self.evaluation_samples_per_direction

        u = np.linspace(spline.u_spline.knots[0], spline.u_spline.knots[-1], samples)
        v = np.linspace(spline.v_spline.knots[0], spline.v_spline.knots[-1], samples)
        uu, vv = np.meshgrid(u, v)
        sample_points = mi.TensorXf(np.column_stack([uu.flatten(), vv.flatten()]))

        interpolated_values = spline.interpolate(
            values=projected_values, parametric_coords=sample_points
        )
        expected = sample_points.numpy().sum(axis=1)

        np.testing.assert_array_almost_equal(
            interpolated_values.numpy().flatten(),
            expected,
        )

    def test_project_function_bezier(self):
        """
        Analytical projection for a quadratic bivariate function.

        Since the bezier surface can analytically represent degree 2 polynomials,
        the projection should be mapped to the control points in an exact manner
        for function f(u, v) = 4 * u * (1-u) * 4 * v * (1-v).
        """

        def quadratic_function(parametric_coords: mi.TensorXf) -> mi.TensorXf:
            # 4 * u * (1-u) * 4 * v * (1-v) for each (u, v)
            u = parametric_coords[:, 0]
            v = parametric_coords[:, 1]
            return (4.0 * u * (1.0 - u) * 4.0 * v * (1.0 - v))[:, None]

        # Create a bivariate Bezier surface (degree 2, open knot vectors)
        bezier_u = BSpline([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], degree=2)
        bezier_v = BSpline([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], degree=2)
        bezier_surface = BivariateBSpline(bezier_u, bezier_v)

        projected_values = bezier_surface.project_function(func=quadratic_function)
        # Control points for this function are 0 except at the center
        expected_values = np.zeros((bezier_surface.num_basis_functions, 1))
        expected_values[4, 0] = 4.0

        np.testing.assert_array_almost_equal(projected_values.numpy(), expected_values)

    def test_project_function_spline(self):
        """
        Sine function projection onto a refined B-spline surface.

        Since the resolution is not overly high to keep down the test time,
        only 4 decimal places are recovered at the selected refinement factor.
        """

        def sine_function(parametric_coords: mi.TensorXf) -> mi.TensorXf:
            return mi.TensorXf(
                np.sin(np.pi * parametric_coords[:, 0])
                * np.sin(np.pi * parametric_coords[:, 1])
            )[:, None]

        # Refine the surface for better approximation
        surface = BivariateBSpline(self.u_spline, self.v_spline)
        surface.h_refinement(factor=12, values_list=[])

        projected_values = surface.project_function(func=sine_function)
        samples = self.evaluation_samples_per_direction

        u = np.linspace(surface.u_spline.knots[0], surface.u_spline.knots[-1], samples)
        v = np.linspace(surface.v_spline.knots[0], surface.v_spline.knots[-1], samples)
        uu, vv = np.meshgrid(u, v)
        sample_points = mi.TensorXf(np.column_stack([uu.flatten(), vv.flatten()]))

        interpolated_values = surface.interpolate(
            values=projected_values, parametric_coords=sample_points
        )
        expected_values = sine_function(sample_points)

        np.testing.assert_array_almost_equal(
            interpolated_values.numpy().flatten(),
            expected_values.numpy().flatten(),
            decimal=4,
        )
