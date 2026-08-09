import unittest

import drjit as dr
import mitsuba as mi
import numpy as np

from rebel_shape.splines.bspline import BSpline
from rebel_shape.splines.bspline_curve import BSplineCurve


class TestBSplineCurve(unittest.TestCase):
    # Control-point coordinates for the linear and Bezier reference curves,
    # held as plain tuples rather than `mi.TensorXf` on the class: a Mitsuba
    # object that outlives the test which built it aborts a *later,
    # unrelated* test inside Dr.Jit's thread pool ("Assertion failed ...
    # queue.cpp: remain == 1"), so the tensors (and the curves built from
    # them) are constructed fresh in `setUp` instead.
    _CONTROL_POINTS_LINEAR = ((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    _CONTROL_POINTS_BEZIER = ((0.0, 0.0, 0.0), (1.0, 2.0, 0.0), (2.0, 0.0, 0.0))

    @classmethod
    def setUpClass(cls) -> None:
        mi.set_variant("llvm_ad_rgb")

        cls.knots_linear = [0.0, 0.0, 1.0, 1.0]
        cls.degree_linear = 1
        cls.spline_linear = BSpline(
            knot_vector=cls.knots_linear, degree=cls.degree_linear
        )

        # Simple Bezier curve example
        cls.knots_bezier = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        cls.degree_bezier = 2
        cls.spline_bezier = BSpline(
            knot_vector=cls.knots_bezier, degree=cls.degree_bezier
        )

        cls.samples = 99
        """
        Number of samples along the curve when evaluating it.
        Should be kept small to avoid long test times, but large enough
        to ensure that the curve is sampled sufficiently.
        """

        cls.autodiff_test_cases = [
            {
                "knots": [0.0, 0.0, 0.21, 0.49, 0.76, 1.0, 1.0],
                "degree": 1,
                "control_points": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [0.5, 0.5, 1.0],
                    ]
                ),
            },
            {
                "knots": [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 5.0],
                "degree": 2,
                "control_points": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [0.5, 0.5, 1.0],
                        [0.0, 0.0, 0.5],
                        [-0.5, 0.0, 0.5],
                        [-0.5, 0.0, 0.0],
                    ]
                ),
            },
            {
                "knots": [0.0, 0.0, 0.0, 0.21, 0.49, 0.76, 1.0, 1.0, 1.0],
                "degree": 3,
                "control_points": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [0.5, 0.5, 1.0],
                        [0.0, 0.0, 0.5],
                    ]
                ),
            },
            {
                "knots": [0.0, 0.0, 0.21, 0.49, 0.76, 1.0, 1.0],
                "degree": 2,
                "control_points": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [0.5, 0.5, 1.0],
                    ]
                ),
            },
            {
                "knots": [0.0, 0.21, 0.49, 0.76, 1.0],
                "degree": 2,
                "control_points": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [1.0, 1.0, 1.0],
                    ]
                ),
            },
        ]

    def setUp(self) -> None:
        self.control_points_linear = mi.TensorXf(np.array(self._CONTROL_POINTS_LINEAR))
        self.curve_linear = BSplineCurve(
            spline=self.spline_linear, control_points=self.control_points_linear
        )

        self.control_points_bezier = mi.TensorXf(np.array(self._CONTROL_POINTS_BEZIER))
        self.curve_bezier = BSplineCurve(
            spline=self.spline_bezier, control_points=self.control_points_bezier
        )

    def test_invalid_control_points_shape(self):
        """
        Tests whether passing control points with an incorrect shape or with incorrect
        dimension raises a `ValueError` as well as whether passing an incorrect number
        of points raises a `ValueError`.
        """
        control_points = mi.TensorXf(np.array([0.0, 1.0, 2.0]))

        with self.assertRaisesRegex(ValueError, "2D tensor"):
            BSplineCurve(spline=self.spline_bezier, control_points=control_points)

        control_points = mi.TensorXf(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]))

        with self.assertRaisesRegex(ValueError, "shape"):
            BSplineCurve(spline=self.spline_bezier, control_points=control_points)

        control_points = mi.TensorXf(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]))

        with self.assertRaisesRegex(ValueError, "control points"):
            BSplineCurve(spline=self.spline_bezier, control_points=control_points)

    def test_control_points_setter(self):
        """
        Tests whether the setter for control points works correctly.
        """
        new_control_points = mi.TensorXf([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        self.curve_linear.control_points = new_control_points

        self.assertEqual(self.curve_linear.control_points.shape, (2, 3))
        np.testing.assert_array_almost_equal(
            self.curve_linear.control_points.numpy(), new_control_points.numpy()
        )

        invalid_shape = mi.TensorXf([[0.0, 0.0], [1.0, 2.0]])
        with self.assertRaises(ValueError):
            self.curve_linear.control_points = invalid_shape

        invalid_number = mi.TensorXf(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )
        with self.assertRaises(ValueError):
            self.curve_linear.control_points = invalid_number

    def test_evaluate_simple(self):
        """
        Test the B-spline in the regime where it behaves like a linear curve between
        control points (0, 0, 0) and (1, 2, 3).
        """
        params = mi.TensorXf(np.array([[0.0], [0.5], [1.0]]))
        points = self.curve_linear.evaluate_coordinates(params)

        expected = np.array(
            [
                [0.0, 0.0, 0.0],  # t = 0.0
                [0.5, 1.0, 1.5],  # t = 0.5
                [1.0, 2.0, 3.0],  # t = 1.0
            ]
        )

        self.assertEqual(points.shape, (3, 3))
        np.testing.assert_array_almost_equal(points.numpy(), expected)

    def test_evaluate_quadratic_bezier(self):
        """
        Test the B-spline in the regime where it behaves like a Bézier curve of
        degree 2 which can be evaluated nicely as:
        P(t) = (1 - t)^2 * P_0 + 2 * (1 - t) * t * P_1 + t^2 * P_2
        """
        params = mi.TensorXf(np.array([[0.0], [0.25], [0.5], [0.75], [1.0]]))
        points = self.curve_bezier.evaluate_coordinates(params)

        expected = np.array(
            [
                [0.0, 0.0, 0.0],  # t = 0.0
                [0.5, 0.75, 0.0],  # t = 0.25
                [1.0, 1.0, 0.0],  # t = 0.5
                [1.5, 0.75, 0.0],  # t = 0.75
                [2.0, 0.0, 0.0],  # t = 1.0
            ]
        )

        self.assertEqual(points.shape, (5, 3))
        np.testing.assert_array_almost_equal(points.numpy(), expected)

    def test_evaluate_periodic(self):
        """Tests that the periodic B-spline curve is a closed loop."""
        knots = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        degree = 2
        control_points = mi.TensorXf(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 1.0, 0.0],
                    [4.0, 0.0, 0.0],
                ]
            )
        )
        spline = BSpline(knot_vector=knots, degree=degree)
        curve = BSplineCurve(spline=spline, control_points=control_points)

        # Take two points close to the start and end of the curve - they should match
        params = mi.TensorXf(np.array([[knots[0] + 1.0e-5], [knots[-1] - 1.0e-5]]))
        points = curve.evaluate_coordinates(params).numpy()

        self.assertAlmostEqual(
            first=np.linalg.norm(points[0] - points[1]), second=0.0, places=4
        )

    def test_evaluate_periodic_repeat_knot(self):
        """
        Tests that the periodic B-spline curve is a closed loop and has one
        repeated knot.
        """
        knots = [0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0]
        degree = 2
        control_points = mi.TensorXf(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 2.0, 0.0],
                    [3.0, 1.0, 0.0],
                    [4.0, 0.0, 0.0],
                ]
            )
        )
        spline = BSpline(knot_vector=knots, degree=degree)
        curve = BSplineCurve(spline=spline, control_points=control_points)

        # Take two points close to the start and end of the curve - they should match
        params = mi.TensorXf(np.array([[knots[0] + 1.0e-5], [knots[-1] - 1.0e-5]]))
        points = curve.evaluate_coordinates(params).numpy()

        self.assertAlmostEqual(
            first=np.linalg.norm(points[0] - points[1]), second=0.0, places=4
        )

    def test_evaluate_first_derivative_linear(self):
        """
        Tests whether the first derivative of the B-spline curve is computed correctly
        in the linear case.
        """
        params = mi.TensorXf(np.array([[0.0], [0.5], [1.0]]))
        first_derivative = self.curve_linear.evaluate_first_derivatives(params)

        expected = np.array(
            [
                [1.0, 2.0, 3.0],  # t = 0.0
                [1.0, 2.0, 3.0],  # t = 0.5
                [1.0, 2.0, 3.0],  # t = 1.0
            ]
        )

        self.assertEqual(first_derivative.shape, (3, 3))
        np.testing.assert_array_almost_equal(first_derivative.numpy(), expected)

    def test_evaluate_first_derivative_bezier(self):
        """
        Tests whether the first derivative of the B-spline curve is computed correctly.
        """
        params = mi.TensorXf(np.array([[0.0], [0.5], [1.0]]))
        first_derivative = self.curve_bezier.evaluate_first_derivatives(params)
        expected = np.array(
            [
                [2.0, 4.0, 0.0],  # t = 0.0
                [2.0, 0.0, 0.0],  # t = 0.5
                [2.0, -4.0, 0.0],  # t = 1.0
            ]
        )

        self.assertEqual(first_derivative.shape, (3, 3))
        np.testing.assert_array_almost_equal(first_derivative.numpy(), expected)

    def test_first_derivative_autodiff(self):
        for i, case in enumerate(self.autodiff_test_cases):
            with self.subTest(i=i):
                spline = BSpline(knot_vector=case["knots"], degree=case["degree"])
                curve = BSplineCurve(
                    spline=spline, control_points=mi.TensorXf(case["control_points"])
                )

                params = dr.linspace(mi.TensorXf, 0.0, 1.0, self.samples)
                params = dr.reshape(mi.TensorXf, params, (-1, 1))

                dr.enable_grad(params)
                evaluation = curve.evaluate_coordinates(parametric_coords=params)

                dr.forward(params)
                derivative_ad = mi.TensorXf(evaluation.grad)

                derivative = curve.evaluate_first_derivatives(parametric_coords=params)

                np.testing.assert_array_almost_equal(
                    derivative.numpy(), derivative_ad.numpy(), decimal=3
                )

    def test_second_derivative_autodiff(self):
        for i, case in enumerate(self.autodiff_test_cases[2:]):
            with self.subTest(i=i):
                spline = BSpline(knot_vector=case["knots"], degree=case["degree"])
                curve = BSplineCurve(
                    spline=spline, control_points=mi.TensorXf(case["control_points"])
                )

                params = dr.linspace(mi.TensorXf, 0.0, 1.0, self.samples)
                params = dr.reshape(mi.TensorXf, params, (-1, 1))

                dr.enable_grad(params)
                evaluation = curve.evaluate_first_derivatives(parametric_coords=params)

                dr.forward(params)
                derivative_ad = mi.TensorXf(evaluation.grad)

                derivative = curve.evaluate_second_derivatives(parametric_coords=params)

                np.testing.assert_array_almost_equal(
                    derivative.numpy(), derivative_ad.numpy(), decimal=3
                )

    def test_knot_insertion(self):
        """
        Tests whether knot insertion works correctly for different cases and updates
        the control points such that the geometry is preserved.

        Implicitly tests if `BSpline.insert_knot_and_update_values` works correctly.
        """
        test_cases = [
            {
                "knots": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                "degree": 2,
                "control_points": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 2.0, 0.0],
                        [2.0, 0.0, 0.0],
                    ]
                ),
                "insertion_knots": [0.5],
            },
            {
                "knots": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                "degree": 2,
                "control_points": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 2.0, 0.0],
                        [2.0, 0.0, 0.0],
                    ]
                ),
                "insertion_knots": [0.25, 0.5, 0.75],
            },
            {
                "knots": [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
                "degree": 2,
                "control_points": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 2.0, 0.0],
                        [1.5, 1.0, 0.0],
                        [2.0, 0.0, 0.0],
                    ]
                ),
                "insertion_knots": [0.25, 0.75],
            },
            {
                "knots": [0.0, 0.0, 0.0, 0.21, 0.49, 0.76, 1.0, 1.0, 1.0],
                "degree": 2,
                "control_points": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [0.5, 0.5, 1.0],
                        [0.0, 0.0, 0.5],
                    ]
                ),
                "insertion_knots": [0.35, 0.6],
            },
        ]

        for i, case in enumerate(test_cases):
            with self.subTest(i=i):
                spline = BSpline(knot_vector=case["knots"], degree=case["degree"])
                curve = BSplineCurve(
                    spline=spline, control_points=mi.TensorXf(case["control_points"])
                )

                sample_points = np.linspace(0.0, 1.0, 99).reshape(-1, 1)

                evaluations_before = curve.evaluate_coordinates(
                    parametric_coords=mi.TensorXf(sample_points)
                )

                for knot in case["insertion_knots"]:
                    curve.insert_knot(knot=knot)

                evaluations_after = curve.evaluate_coordinates(
                    parametric_coords=mi.TensorXf(sample_points)
                )

                np.testing.assert_array_almost_equal(
                    evaluations_before.numpy(), evaluations_after.numpy()
                )

    def test_h_refinement(self):
        """
        Tests whether h-refinement works correctly.
        """
        sample_points = np.linspace(0.0, 1.0, 99).reshape(-1, 1)

        for i in range(4):
            with self.subTest(i=i):
                spline = BSpline(
                    knot_vector=self.knots_bezier, degree=self.degree_bezier
                )
                curve = BSplineCurve(
                    spline=spline, control_points=self.control_points_bezier
                )

                evaluations_before = curve.evaluate_coordinates(
                    parametric_coords=mi.TensorXf(sample_points)
                )

                curve.h_refinement(factor=i + 1)
                # Had one knot span before refinement, now should have i + 1
                self.assertEqual(curve.interpolating_spline.knot_spans.shape[0], i + 1)

                evaluations_after = curve.evaluate_coordinates(
                    parametric_coords=mi.TensorXf(sample_points)
                )

                np.testing.assert_array_almost_equal(
                    evaluations_before.numpy(), evaluations_after.numpy()
                )
