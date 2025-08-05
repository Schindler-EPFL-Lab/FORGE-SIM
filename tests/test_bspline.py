import unittest

import mitsuba as mi
import numpy as np

from rebel_shape.splines.bspline import BSpline


class TestBSpline(unittest.TestCase):
    """
    Unit tests for the `BSpline` class.

    Note that the `interpolate` method is not tested here as B-spline curves use it
    to interpolate values. Thus, see `test_bspline_curve.py` for tests that cover
    interpolation of values. Same applies to knot insertion with updates to the
    values tensor.
    """

    @classmethod
    def setUpClass(cls) -> None:
        mi.set_variant("llvm_ad_rgb")

        # Typical example of a B-spline curve, taken from the NURBS book
        cls.knots_1 = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 5.0]
        cls.degree_1 = 2
        cls.spline_1 = BSpline(knot_vector=cls.knots_1, degree=cls.degree_1)

        # Simpler curve than the previous one, used for ghost-knot testing
        cls.knots_2 = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0]
        cls.degree_2 = 2
        cls.spline_2 = BSpline(knot_vector=cls.knots_2, degree=cls.degree_2)

        # Simple Bezier curve
        cls.knots_bezier = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        cls.degree_bezier = 2
        cls.spline_bezier = BSpline(
            knot_vector=cls.knots_bezier, degree=cls.degree_bezier
        )

        cls.evaluation_samples = 99
        """
        Number of samples used for evaluating the basis functions and interpolating
        values. This is used to ensure that the tests cover a range of parametric
        coordinates. Should not be too large to avoid long test execution times.
        """

    def test_basic_properties(self):
        """Tests whether the basic initialization of the three splines is correct."""
        self.assertEqual(self.spline_1.degree, self.degree_1)
        self.assertEqual(len(self.spline_1._unique_knots), 6)
        self.assertEqual(self.spline_1._knots.shape, (11,))
        self.assertEqual(self.spline_1._knot_multiplicities.shape, (6,))

        self.assertEqual(self.spline_2.degree, self.degree_2)
        self.assertEqual(len(self.spline_2._unique_knots), 4)
        self.assertEqual(self.spline_2._knots.shape, (8,))
        self.assertEqual(self.spline_2._knot_multiplicities.shape, (4,))

        self.assertEqual(self.spline_bezier.degree, self.degree_bezier)
        self.assertEqual(len(self.spline_bezier._unique_knots), 2)
        self.assertEqual(self.spline_bezier._knots.shape, (6,))
        self.assertEqual(self.spline_bezier._knot_multiplicities.shape, (2,))

    def test_invalid_degree(self):
        """Tests whether a negative spline degree raises a `ValueError`"""
        with self.assertRaisesRegex(ValueError, "positive"):
            BSpline(knot_vector=self.knots_1, degree=0)

    def test_invalid_knot_vector_order(self):
        """Tests whether a non ascending knot vector raises a `ValueError`"""
        knots = [0.0, 0.0, 0.0, 2.0, 1.0, 3.0, 4.0, 4.0, 5.0, 5.0, 5.0]

        with self.assertRaisesRegex(ValueError, "ascending"):
            BSpline(knot_vector=knots, degree=self.degree_1)

    def test_invalid_knot_multiplicity(self):
        """
        Tests whether the first and last knot having a different multiplicity
        raises a `ValueError`
        """
        knots = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 5.0]

        with self.assertRaisesRegex(ValueError, "same multiplicity"):
            BSpline(knot_vector=knots, degree=self.degree_1)

        knots = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0]

        with self.assertRaisesRegex(ValueError, "greater than the degree"):
            BSpline(knot_vector=knots, degree=self.degree_1)

        knots = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0]

        with self.assertRaisesRegex(ValueError, "edge knot"):
            BSpline(knot_vector=knots, degree=self.degree_1)

    def test_process_knot_vector(self):
        """T
        ests the function that returns the number of elements, unique knots and
        their multiplicities from the input knot vector.
        """
        num_elements, unique_knots, knot_multiplicities = BSpline.process_knot_vector(
            knot_vector=self.knots_1, degree=self.degree_1
        )

        self.assertEqual(num_elements, 5)
        self.assertEqual(len(unique_knots), 6)

        np.testing.assert_array_almost_equal(
            unique_knots[:], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        )
        np.testing.assert_array_equal(knot_multiplicities[:], [3, 1, 1, 1, 2, 3])

    def test_ghost_knots_computation(self):
        """Tests whether ghost knots are computed correctly."""
        ghost_knots = self.spline_2.compute_extended_knot_vector(degree=self.degree_2)

        # For degree 2, we should have 3 ghost knots on each side
        expected_ghost_knots = np.array(
            [-3.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0]
        )

        self.assertEqual(len(ghost_knots), len(expected_ghost_knots))
        np.testing.assert_array_almost_equal(ghost_knots, expected_ghost_knots)

    def test_get_knot(self):
        """Tests the get_knot method for both regular and ghost knots."""
        # Test regular indices
        self.assertEqual(self.spline_2.get_knot(0), 0.0)
        self.assertEqual(self.spline_2.get_knot(2), 0.0)
        self.assertEqual(self.spline_2.get_knot(3), 1.0)
        self.assertEqual(self.spline_2.get_knot(7), 3.0)

        # Test ghost knots to the left
        self.assertEqual(self.spline_2.get_knot(-1), -1.0)
        self.assertEqual(self.spline_2.get_knot(-2), -2.0)
        self.assertEqual(self.spline_2.get_knot(-3), -3.0)

        # Test ghost knots to the right
        self.assertEqual(self.spline_2.get_knot(8), 4.0)
        self.assertEqual(self.spline_2.get_knot(9), 5.0)
        self.assertEqual(self.spline_2.get_knot(10), 6.0)

    def test_num_basis_functions(self):
        """Tests the number of basis functions for the spline."""
        self.assertEqual(self.spline_1.num_basis_functions, 8)
        self.assertEqual(self.spline_2.num_basis_functions, 5)
        self.assertEqual(self.spline_bezier.num_basis_functions, 3)

        self.assertEqual(self.spline_1.basis_functions_per_span, 3)
        self.assertEqual(self.spline_2.basis_functions_per_span, 3)
        self.assertEqual(self.spline_bezier.basis_functions_per_span, 3)

    def test_parametric_span_data(self):
        """Tests the ParametricSpanData class."""
        parametric_coords = mi.TensorXf([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

        span_data = self.spline_1.compute_span_data(
            parametric_coords=parametric_coords,
        )

        self.assertEqual(span_data._parametric_coords.shape[0], 6)
        self.assertEqual(span_data._knot_spans.shape[0], 6)

        index_0 = span_data.global_basis_function_index(local_index=0)
        index_1 = span_data.global_basis_function_index(local_index=1)
        index_2 = span_data.global_basis_function_index(local_index=2)

        np.testing.assert_array_equal(index_0, np.array([0, 1, 2, 3, 5, 5]))
        np.testing.assert_array_equal(index_1, np.array([1, 2, 3, 4, 6, 6]))
        np.testing.assert_array_equal(index_2, np.array([2, 3, 4, 5, 7, 7]))

    def test_parametric_span_data_periodic(self):
        """Tests the ParametricSpanData class."""
        knots = [0.0, 1.0, 2.0, 3.0, 4.0]
        degree = 2
        spline = BSpline(knot_vector=knots, degree=degree)

        parametric_coords = mi.TensorXf([0.1, 1.1, 2.1, 3.1])

        span_data = spline.compute_span_data(
            parametric_coords=parametric_coords,
        )

        self.assertEqual(span_data._parametric_coords.shape[0], 4)
        self.assertEqual(span_data._knot_spans.shape[0], 4)

        index_0 = span_data.global_basis_function_index(local_index=0)
        index_1 = span_data.global_basis_function_index(local_index=1)
        index_2 = span_data.global_basis_function_index(local_index=2)

        np.testing.assert_array_equal(index_0, np.array([2, 3, 0, 1]))
        np.testing.assert_array_equal(index_1, np.array([3, 0, 1, 2]))
        np.testing.assert_array_equal(index_2, np.array([0, 1, 2, 3]))

    def test_invalid_parametric_span_local_index(self):
        """Tests whether an invalid local index raises a `ValueError`."""
        parametric_coords = mi.TensorXf([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        span_data = self.spline_1.compute_span_data(
            parametric_coords=parametric_coords,
        )

        with self.assertRaisesRegex(ValueError, "local_index"):
            span_data.global_basis_function_index(local_index=-1)

        with self.assertRaisesRegex(ValueError, "local_index"):
            span_data.global_basis_function_index(local_index=3)

    def test_basis_funcs_evaluation(self):
        """
        Tests the evaluation of the basis functions.

        The test is done for a Bezier curve as we know that its basis functions are
        B_{0,2} = (1-t)^2, B_{1,2} = 2t(1-t), B_{2,2} = = t^2.
        """
        basis_funcs = self.spline_bezier.evaluate_basis_functions(
            parametric_coords=mi.TensorXf([[0.0], [0.5], [1.0]])
        )

        self.assertEqual(basis_funcs.shape, (3, 3))

        np.testing.assert_array_almost_equal(basis_funcs[0], [1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(basis_funcs[1], [0.25, 0.5, 0.25])
        np.testing.assert_array_almost_equal(basis_funcs[2], [0.0, 0.0, 1.0])

    def test_validate_parametric_coords(self):
        """
        Test validation of parametric coordinates.

        Multiple invalid cases are tested, including invalid shapes and out-of-range
        parametric coordinates.
        """
        valid_params = mi.TensorXf([[0.25], [0.5]])
        try:
            self.spline_bezier.validate_parametric_coords(valid_params)
        except Exception as e:
            self.fail(f"validate_parametric_coords() raised {e} unexpectedly!")

        invalid_test_cases = [
            # (description, test_input)
            ("not a 2D tensor", mi.TensorXf([0.25, 0.5])),
            (
                "2 parametric coordinates",
                mi.TensorXf([[1.0, 1.0], [2.0, 1.0]]),
            ),
            (
                "parametric direction out of range (lower bound)",
                mi.TensorXf([[0.5], [-0.5]]),
            ),
            (
                "parametric direction out of range (upper bound)",
                mi.TensorXf([[0.5], [1.5]]),
            ),
        ]

        for description, invalid_input in invalid_test_cases:
            with self.subTest(description=description):
                with self.assertRaises(ValueError):
                    self.spline_bezier.validate_parametric_coords(invalid_input)

    def test_validate_interpolating_values(self):
        """
        Test validation of valid interpolating values

        Nominally, this method is part of the `InterpolatingSpline` interface, but it
        is tested here as BSpline does not override it.
        """
        # 20 control points, 3 dimensions
        values = mi.TensorXf(np.random.rand(3, 3))
        try:
            self.spline_bezier.validate_interpolating_values(values)
        except Exception as e:
            self.fail(f"validate_interpolating_values() raised {e} unexpectedly!")

        # Must be a 2D tensor
        invalid_shape = mi.TensorXf(np.random.rand(3))
        with self.assertRaises(ValueError):
            self.spline_bezier.validate_interpolating_values(invalid_shape)

        # Number of control points must match the number of the basis functions
        wrong_size = mi.TensorXf(np.random.rand(2, 3))
        with self.assertRaises(ValueError):
            self.spline_bezier.validate_interpolating_values(wrong_size)

    def test_insert_knot(self):
        """Tests the knot insertion method."""
        # Insert a knot in the middle of the spline
        new_spline = BSpline(knot_vector=self.knots_2, degree=self.degree_2)
        new_spline.insert_knot(knot_position=2.5)

        # The new knot vector should have one more element
        self.assertEqual(len(new_spline.knots), len(self.knots_2) + 1)
        self.assertEqual(new_spline.knots[5], 2.5)

        # The multiplicity of the inserted knot should be 1
        self.assertEqual(new_spline.knots[5] - new_spline.knots[4], 0.5)
        self.assertEqual(new_spline.knots[6] - new_spline.knots[5], 0.5)

        self.assertEqual(
            len(new_spline._unique_knots), len(self.spline_2._unique_knots) + 1
        )
        self.assertEqual(new_spline._unique_knots[3], 2.5)
        self.assertEqual(new_spline._num_elements, self.spline_2._num_elements + 1)

        new_spline_2 = BSpline(knot_vector=self.knots_2, degree=self.degree_2)
        new_spline_2.insert_knot(knot_position=2.0)

        # The new knot vector should have one more element
        self.assertEqual(len(new_spline_2.knots), len(self.knots_2) + 1)
        self.assertEqual(new_spline_2.knots[5], 2.0)

        # The multiplicity of the inserted knot should be 1
        self.assertEqual(new_spline_2.knots[5] - new_spline_2.knots[4], 0.0)
        self.assertEqual(new_spline_2.knots[6] - new_spline_2.knots[5], 1.0)

        self.assertEqual(
            len(new_spline_2._unique_knots), len(self.spline_2._unique_knots)
        )
        self.assertEqual(new_spline_2._unique_knots[2], 2.0)
        self.assertEqual(new_spline_2._num_elements, self.spline_2._num_elements)

    def test_insert_knot_edge(self):
        """
        Check that inserting a knot at the end is mirrored by inclusion at the start
        to maintain that the first and last knots have the same multiplicity
        """
        new_spline_right = BSpline(knot_vector=self.knots_2[1:-1], degree=self.degree_2)
        new_spline_right.insert_knot(knot_position=3.0)

        self.assertEqual(len(new_spline_right.knots), len(self.knots_2))
        np.testing.assert_array_almost_equal(
            self.spline_2.knots, new_spline_right.knots
        )

        new_spline_left = BSpline(knot_vector=self.knots_2[1:-1], degree=self.degree_2)
        new_spline_left.insert_knot(knot_position=0.0)
        self.assertEqual(len(new_spline_left.knots), len(self.knots_2))
        np.testing.assert_array_almost_equal(self.spline_2.knots, new_spline_left.knots)

    def test_basis_func_derivatives_evaluation(self):
        """
        Tests the evaluation of the basis functions.

        The test is done for a Bezier curve as we know that its basis functions are
        B_{0,2} = (1-t)^2, B_{1,2} = 2t(1-t), B_{2,2} = = t^2.
        """
        parametric_coords = mi.TensorXf([0.0, 0.25, 0.5, 0.75, 1.0])
        span_data = self.spline_bezier.compute_span_data(
            parametric_coords=parametric_coords,
        )

        basis_funcs_derivatives = (
            self.spline_bezier.evaluate_basis_function_derivatives(
                parametric_span_data=span_data, derivative_order=2
            )
        )
        self.assertEqual(basis_funcs_derivatives.shape, (5, 3, 3))

        expected_basis_funcs = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.5625, 0.375, 0.0625],
                [0.25, 0.5, 0.25],
                [0.0625, 0.375, 0.5625],
                [0.0, 0.0, 1.0],
            ]
        )
        expected_basis_func_first_derivatives = np.array(
            [
                [-2.0, 2.0, 0.0],
                [-1.5, 1.0, 0.5],
                [-1.0, 0.0, 1.0],
                [-0.5, -1.0, 1.5],
                [0.0, -2.0, 2.0],
            ]
        )
        expected_basis_func_second_derivatives = np.array(
            [[2.0, -4.0, 2.0] for _ in range(5)]
        )

        np.testing.assert_array_almost_equal(
            basis_funcs_derivatives[:, 0, :].numpy(),
            expected_basis_funcs,
            decimal=6,
        )
        np.testing.assert_array_almost_equal(
            basis_funcs_derivatives[:, 1, :].numpy(),
            expected_basis_func_first_derivatives,
            decimal=6,
        )
        np.testing.assert_array_almost_equal(
            basis_funcs_derivatives[:, 2, :].numpy(),
            expected_basis_func_second_derivatives,
            decimal=6,
        )

    def test_basis_function_first_derivative_finite_difference(self):
        """
        Tests the evaluation of the basis function derivatives using finite differences.
        """

        test_cases = [
            (1, [0.0, 0.0, 0.24, 0.49, 0.74, 1.0, 1.0]),
            (2, [0.0, 0.0, 0.0, 0.26, 0.51, 0.76, 1.0, 1.0, 1.0]),
            (3, [0.0, 0.0, 0.0, 0.0, 0.21, 0.47, 0.79, 1.0, 1.0, 1.0, 1.0]),
            (3, [0.0, 0.21, 0.47, 0.79, 1.0]),
        ]

        for i, (degree, knots) in enumerate(test_cases):
            with self.subTest(i=i):
                spline = BSpline(knot_vector=knots, degree=degree)

                n_points = self.evaluation_samples
                h = 1.0e-3

                sample_pts = mi.TensorXf(np.arange(1, n_points) / n_points)
                sample_pts_low = mi.TensorXf(sample_pts - 0.5 * h)
                sample_pts_high = mi.TensorXf(sample_pts + 0.5 * h)

                eval_low = spline.evaluate_basis_functions(
                    parametric_coords=sample_pts_low[:, None]
                )
                eval_high = spline.evaluate_basis_functions(
                    parametric_coords=sample_pts_high[:, None]
                )

                derivative = spline.evaluate_basis_functions_first_derivative(
                    parametric_coords=sample_pts[:, None]
                )
                finite_diff = (eval_high - eval_low) / h

                np.testing.assert_array_almost_equal(
                    derivative.numpy(), finite_diff.numpy(), decimal=3
                )

    def test_active_basis_function_indices(self):
        """Tests the active basis function indices computation."""
        parametric_coords = mi.TensorXf([[0.5], [1.5], [2.5], [3.5], [4.5]])

        active_indices = self.spline_1.active_basis_function_indices(
            parametric_coords=parametric_coords,
        )

        # The active indices should be the indices of the basis functions that are
        # non-zero at the given parametric coordinates
        expected_indices = np.array(
            [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [5, 6, 7]]
        )

        np.testing.assert_array_equal(active_indices.numpy(), expected_indices)

    def test_project_function_constant_linear(self):
        """
        Constant and linear functions must always be able to be projected onto
        the spline without any loss of precision.
        """
        for spline in [self.spline_1, self.spline_2, self.spline_bezier]:
            with self.subTest(spline=spline):
                projected_values = spline.project_function(
                    func=lambda coords: mi.TensorXf(np.ones_like(coords))
                )

                expected_values = np.ones((spline.num_basis_functions, 1))
                np.testing.assert_array_almost_equal(
                    projected_values.numpy(), expected_values
                )

                projected_values = spline.project_function(func=lambda coords: coords)

                samples = self.evaluation_samples
                sample_points = mi.TensorXf(
                    np.linspace(spline.knots[0], spline.knots[-1], samples)
                )
                interpolated_values = spline.interpolate(
                    values=projected_values, parametric_coords=sample_points[:, None]
                )

                np.testing.assert_array_almost_equal(
                    interpolated_values.numpy().flatten(),
                    sample_points.numpy(),
                )

    def test_project_function_bezier(self):
        """
        A second degree Bézier curve can exactly represent the quadratic function
            f(t) = 4t(1-t) for t in [0, 1].
        This is a special case where the projection should yield the exact values.
        """
        projected_values = self.spline_bezier.project_function(
            func=lambda coords: 4.0 * coords * (1.0 - coords)
        )

        expected_values = np.array([[0.0], [2.0], [0.0]])
        np.testing.assert_array_almost_equal(projected_values.numpy(), expected_values)

    def test_project_function_spline(self):
        """Projects a sine function onto a B-spline and checks the interpolation."""

        def sine_function(parametric_coords: mi.TensorXf) -> mi.TensorXf:
            return mi.TensorXf(np.sin(np.pi * parametric_coords))

        local_bezier_spline = BSpline(
            knot_vector=self.knots_bezier, degree=self.degree_bezier
        )
        local_bezier_spline.h_refinement(factor=32, values_list=[])

        projected_values = local_bezier_spline.project_function(func=sine_function)

        samples = self.evaluation_samples
        sample_points = mi.TensorXf(
            np.linspace(
                local_bezier_spline.knots[0], local_bezier_spline.knots[-1], samples
            )
        )
        interpolated_values = local_bezier_spline.interpolate(
            values=projected_values, parametric_coords=sample_points[:, None]
        )

        # The interpolated values should be close to the quadratic function
        expected_values = sine_function(sample_points[:, None])
        np.testing.assert_array_almost_equal(
            interpolated_values.numpy().flatten(),
            expected_values.numpy().flatten(),
            decimal=5,
        )

    def test_knot_spans(self):
        """Tests the knot spans computation."""
        knot_spans = self.spline_1.knot_spans

        # The knot spans should be the intervals between the knots
        expected_spans = np.array(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
        )

        np.testing.assert_array_almost_equal(knot_spans.numpy(), expected_spans)

    def test_has_open_knot_vector(self):
        """Tests the open knot vector computation."""
        self.assertEqual(self.spline_1.has_open_knot_vector, True)
        self.assertEqual(self.spline_2.has_open_knot_vector, True)
        self.assertEqual(self.spline_bezier.has_open_knot_vector, True)

        # Create a periodic spline (not open)
        new_spline = BSpline(knot_vector=[0.0, 1.0, 2.0, 3.0, 4.0], degree=2)
        self.assertEqual(new_spline.has_open_knot_vector, False)
