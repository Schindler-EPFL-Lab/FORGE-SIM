import unittest

import jsonpickle
import mitsuba as mi
import numpy as np

from rebel_shape.splines.bspline import BSpline
from rebel_shape.splines.bspline_patch import BSplinePatch


class TestBSplinePatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mi.set_variant("llvm_ad_rgb")

        # Simple Bezier surface for a unit square
        cls.bezier_1_knots = [0.0, 0.0, 1.0, 1.0]
        cls.bezier_1_degree = 1
        cls.bezier_1_spline = BSpline(
            knot_vector=cls.bezier_1_knots, degree=cls.bezier_1_degree
        )
        cls.bezier_1_control_points = mi.TensorXf(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                ]
            )
        )
        cls.bezier_1_surface = BSplinePatch(
            name="bezier_1_surface",
            u_spline=cls.bezier_1_spline,
            v_spline=cls.bezier_1_spline,
            control_points=cls.bezier_1_control_points,
        )

        cls.parametric_coords = mi.TensorXf(
            np.array(
                [
                    [0.0, 0.0],  # Bottom-left
                    [1.0, 0.0],  # Bottom-right
                    [0.0, 1.0],  # Top-left
                    [1.0, 1.0],  # Top-right
                    [0.5, 0.5],  # Center
                ]
            )
        )

        cls.bezier_2_u_knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        cls.bezier_2_v_knots = [0.0, 0.0, 1.0, 1.0]
        cls.bezier_2_control_points = mi.TensorXf(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 1.0, 0.0],
                ]
            )
        )
        cls.bezier_2_u_spline = BSpline(knot_vector=cls.bezier_2_u_knots, degree=2)
        cls.bezier_2_v_spline = BSpline(knot_vector=cls.bezier_2_v_knots, degree=1)
        cls.bezier_2_surface = BSplinePatch(
            name="bezier_2_surface",
            u_spline=cls.bezier_2_u_spline,
            v_spline=cls.bezier_2_v_spline,
            control_points=cls.bezier_2_control_points,
        )

    def test_basic_properties(self):
        """Tests whether the basic initialization is correct."""
        self.assertEqual(self.bezier_1_surface.degree, 1)
        self.assertEqual(self.bezier_1_surface.control_points.shape, (4, 3))
        self.assertEqual(self.bezier_1_surface.control_points_grid.shape, (2, 2, 3))
        self.assertEqual(self.bezier_1_surface._num_cps_u, 2)
        self.assertEqual(self.bezier_1_surface._num_cps_v, 2)

        self.assertEqual(self.bezier_2_surface.degree, 2)
        self.assertEqual(self.bezier_2_surface.control_points.shape, (6, 3))
        self.assertEqual(self.bezier_2_surface.control_points_grid.shape, (2, 3, 3))
        self.assertEqual(self.bezier_2_surface._num_cps_u, 3)
        self.assertEqual(self.bezier_2_surface._num_cps_v, 2)

    def test_quads(self):
        """Tests whether the quads property returns the correct indices."""
        expected_quads_1 = np.array([[0, 1, 3, 2]], dtype=np.uint32)
        np.testing.assert_array_equal(
            self.bezier_1_surface.quads.numpy(), expected_quads_1
        )

        expected_quads_2 = np.array([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=np.uint32)
        np.testing.assert_array_equal(
            self.bezier_2_surface.quads.numpy(), expected_quads_2
        )

    def test_invalid_control_points(self):
        """Tests whether invalid control point inputs raise `ValueError`"""
        # Test invalid control points shape
        invalid_points_1 = mi.TensorXf(np.ones((4,)))
        with self.assertRaisesRegex(ValueError, "2D tensor"):
            BSplinePatch(
                name="invalid_surface",
                u_spline=self.bezier_1_spline,
                v_spline=self.bezier_1_spline,
                control_points=invalid_points_1,
            )

        invalid_points_2 = mi.TensorXf(np.ones((4, 2)))
        with self.assertRaisesRegex(ValueError, "shape"):
            BSplinePatch(
                name="invalid_surface",
                u_spline=self.bezier_1_spline,
                v_spline=self.bezier_1_spline,
                control_points=invalid_points_2,
            )

        # Test invalid number of control points
        invalid_points_3 = mi.TensorXf(np.ones((5, 3)))
        with self.assertRaisesRegex(ValueError, "Number of control"):
            BSplinePatch(
                name="invalid_surface",
                u_spline=self.bezier_1_spline,
                v_spline=self.bezier_1_spline,
                control_points=invalid_points_3,
            )

    def test_control_points_setter(self):
        """
        Tests whether the setter for control points works correctly.
        """
        linear_surface = BSplinePatch(
            name="bezier_1_surface",
            u_spline=self.bezier_1_spline,
            v_spline=self.bezier_1_spline,
            control_points=self.bezier_1_control_points,
        )
        new_control_points = mi.TensorXf(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        )
        linear_surface.control_points = new_control_points

        self.assertEqual(linear_surface.control_points.shape, (4, 3))
        np.testing.assert_array_almost_equal(
            linear_surface.control_points.numpy(), new_control_points.numpy()
        )

        invalid_shape = mi.TensorXf([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        with self.assertRaises(ValueError):
            linear_surface.control_points = invalid_shape

        invalid_number = mi.TensorXf(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )
        with self.assertRaises(ValueError):
            linear_surface.control_points = invalid_number

    def test_evaluate_linear(self):
        """Tests that evaluating a simple planar surface yields correct results."""
        points = self.bezier_1_surface.evaluate_coordinates(
            parametric_coords=self.parametric_coords
        )

        # Check that corners match control points
        np.testing.assert_array_almost_equal(points[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(points[1], [1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(points[2], [0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(points[3], [1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(points[4], [0.5, 0.5, 0.25])

    def test_cubic_bezier_surface(self):
        """
        Tests whether a cubic bezier surface is evaluated correctly.
        Explicit computation can be applied to retrieve the point in the middle of the
        Bezier patch without too much effort (exploiting the tensor product nature).
        """
        # Create a cubic Bezier surface (degree 3 in both directions)
        u_knots = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        v_knots = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        u_spline = BSpline(knot_vector=u_knots, degree=3)
        v_spline = BSpline(knot_vector=v_knots, degree=3)

        # 4x4 control grid for cubic surface
        control_points = mi.TensorXf(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 3.0],
                    [2.0, 1.0, 3.0],
                    [3.0, 1.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [1.0, 2.0, 3.0],
                    [2.0, 2.0, 3.0],
                    [3.0, 2.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [1.0, 3.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [3.0, 3.0, 0.0],
                ]
            )
        )
        surface = BSplinePatch(
            name="cubic_bezier_surface",
            u_spline=u_spline,
            v_spline=v_spline,
            control_points=control_points,
        )

        points = surface.evaluate_coordinates(parametric_coords=self.parametric_coords)
        expected_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [3.0, 3.0, 0.0],
                [1.5, 1.5, 1.6875],
            ]
        )

        np.testing.assert_array_almost_equal(points.numpy(), expected_points)

    def test_evaluate_different_degrees(self):
        """
        Tests whether using a different degree for u and v splines behaves as expected.
        """
        points = self.bezier_2_surface.evaluate_coordinates(
            parametric_coords=self.parametric_coords
        )

        expected_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [1.0, 0.5, 0.5],
            ]
        )

        np.testing.assert_array_almost_equal(points.numpy(), expected_points)

    def test_evaluate_first_derivatives_linear(self):
        """
        Tests whether the first derivatives of the surface are evaluated correctly.
        """
        u_derivative, v_derivative = self.bezier_1_surface.evaluate_first_derivatives(
            parametric_coords=self.parametric_coords
        )

        expected_du = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 0.5],
            ]
        )
        expected_dv = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 0.5],
            ]
        )

        np.testing.assert_array_almost_equal(u_derivative.numpy(), expected_du)
        np.testing.assert_array_almost_equal(v_derivative.numpy(), expected_dv)

    def test_evaluate_first_derivatives_different_degrees(self):
        """
        Tests whether the first derivatives of the surface are evaluated correctly
        with different degrees for u and v splines.
        """
        u_derivative, v_derivative = self.bezier_2_surface.evaluate_first_derivatives(
            parametric_coords=self.parametric_coords
        )

        expected_du = np.array(
            [
                [2.0, 0.0, 2.0],
                [2.0, 0.0, -2.0],
                [2.0, 0.0, 2.0],
                [2.0, 0.0, -2.0],
                [2.0, 0.0, 0.0],
            ]
        )
        # Derivative of degree 1 spline is constant
        expected_dv = np.repeat(np.array([[0.0, 1.0, 0.0]]), 5, axis=0)

        np.testing.assert_array_almost_equal(u_derivative.numpy(), expected_du)
        np.testing.assert_array_almost_equal(v_derivative.numpy(), expected_dv)

    def test_evaluate_normals_linear(self):
        """
        Tests whether the normals of the surface are computed correctly in the
        linear case.
        """
        normals = self.bezier_1_surface.evaluate_normals(
            parametric_coords=self.parametric_coords
        )

        expected_normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 1.0] / np.sqrt(2),
                [-1.0, 0.0, 1.0] / np.sqrt(2),
                [-1.0, -1.0, 1.0] / np.sqrt(3),
                [-1.0, -1.0, 2.0] / np.sqrt(6),
            ]
        )

        np.testing.assert_array_almost_equal(normals.numpy(), expected_normals)

    def test_evaluate_normals_different_degrees(self):
        """
        Tests whether the normals of the surface are computed correctly with different
        degrees for u and v splines.
        """
        normals = self.bezier_2_surface.evaluate_normals(
            parametric_coords=self.parametric_coords
        )

        expected_normals = np.array(
            [
                [-1.0, 0.0, 1.0] / np.sqrt(2),
                [1.0, 0.0, 1.0] / np.sqrt(2),
                [-1.0, 0.0, 1.0] / np.sqrt(2),
                [1.0, 0.0, 1.0] / np.sqrt(2),
                [0.0, 0.0, 1.0],
            ]
        )

        np.testing.assert_array_almost_equal(normals.numpy(), expected_normals)

    def test_first_derivative_finite_difference(self):
        test_cases = [
            {
                "u_knots": [0.0, 0.0, 0.49, 1.0, 1.0],
                "v_knots": [0.0, 0.0, 0.51, 1.0, 1.0],
                "u_degree": 1,
                "v_degree": 1,
                "control_points": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 1.0, 2.0],
                        [2.0, 1.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [1.0, 2.0, 0.0],
                        [2.0, 2.0, 0.0],
                    ]
                ),
            },
            {
                "u_knots": [0.0, 0.0, 0.0, 0.55, 1.0, 1.0, 1.0],
                "v_knots": [0.0, 0.0, 0.47, 1.0, 1.0],
                "u_degree": 2,
                "v_degree": 1,
                "control_points": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 1.0],
                        [2.0, 0.0, 0.6],
                        [3.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [2.0, 1.0, 0.9],
                        [3.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 1.32],
                        [2.0, 0.0, 0.34],
                        [3.0, 0.0, 0.0],
                    ]
                ),
            },
            {
                "u_knots": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                "v_knots": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                "u_degree": 2,
                "v_degree": 2,
                "control_points": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 1.0, 2.0],
                        [2.0, 1.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [1.0, 2.0, 0.0],
                        [2.0, 2.0, 0.0],
                    ]
                ),
            },
        ]

        for i, case in enumerate(test_cases):
            with self.subTest(i=i):
                u_spline = BSpline(knot_vector=case["u_knots"], degree=case["u_degree"])
                v_spline = BSpline(knot_vector=case["v_knots"], degree=case["v_degree"])
                surface = BSplinePatch(
                    name=f"test_surface_{i}",
                    u_spline=u_spline,
                    v_spline=v_spline,
                    control_points=mi.TensorXf(case["control_points"]),
                )

                n_points = 19
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
                eval_u_low = surface.evaluate_coordinates(
                    parametric_coords=mi.TensorXf(params_u_low)
                )
                eval_u_high = surface.evaluate_coordinates(
                    parametric_coords=mi.TensorXf(params_u_high)
                )
                eval_v_low = surface.evaluate_coordinates(
                    parametric_coords=mi.TensorXf(params_v_low)
                )
                eval_v_high = surface.evaluate_coordinates(
                    parametric_coords=mi.TensorXf(params_v_high)
                )

                # Calculate finite differences
                u_finite_diff = (eval_u_high - eval_u_low) / h
                v_finite_diff = (eval_v_high - eval_v_low) / h

                # Calculate analytical derivatives
                u_derivative, v_derivative = surface.evaluate_first_derivatives(
                    parametric_coords=mi.TensorXf(params)
                )

                # Compare analytical derivatives with finite differences
                np.testing.assert_array_almost_equal(
                    u_derivative.numpy(), u_finite_diff.numpy(), decimal=3
                )
                np.testing.assert_array_almost_equal(
                    v_derivative.numpy(), v_finite_diff.numpy(), decimal=3
                )

    def test_evaluate_jacobian(self):
        """
        Tests whether the Jacobian of the Bezier surface with different degrees
        in the u and v directions is computed correctly.
        """
        jacobian = self.bezier_2_surface.evaluate_jacobian(
            parametric_coords=self.parametric_coords
        )

        expected_jac = np.array(
            [
                [[2.0, 0.0, 2.0], [0.0, 1.0, 0.0]],
                [[2.0, 0.0, -2.0], [0.0, 1.0, 0.0]],
                [[2.0, 0.0, 2.0], [0.0, 1.0, 0.0]],
                [[2.0, 0.0, -2.0], [0.0, 1.0, 0.0]],
                [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ]
        )
        expected_jac = expected_jac.transpose(0, 2, 1)

        np.testing.assert_array_almost_equal(jacobian.numpy(), expected_jac)

    def test_metric_tensor(self):
        """
        Tests whether the metric tensor is computed correctly for the Bezier surface.
        """
        jacobian = self.bezier_2_surface.evaluate_jacobian(
            parametric_coords=self.parametric_coords
        )

        metric_tensor = self.bezier_2_surface.metric_tensor(jacobian=jacobian)

        expected_metric_tensor = np.array(
            [
                [[8.0, 0.0], [0.0, 1.0]],
                [[8.0, 0.0], [0.0, 1.0]],
                [[8.0, 0.0], [0.0, 1.0]],
                [[8.0, 0.0], [0.0, 1.0]],
                [[4.0, 0.0], [0.0, 1.0]],
            ]
        )

        np.testing.assert_array_almost_equal(
            metric_tensor.numpy(), expected_metric_tensor
        )

    def test_metric_tensor_invalid_input(self):
        """
        Tests whether the metric tensor raises an error when an invalid jacobian is
        provided.
        """
        invalid_jacobian = mi.TensorXf(np.ones((5, 3, 3)))
        with self.assertRaises(ValueError):
            self.bezier_2_surface.metric_tensor(jacobian=invalid_jacobian)

        invalid_jacobian = mi.TensorXf(np.ones((5, 2, 3)))
        with self.assertRaises(ValueError):
            self.bezier_2_surface.metric_tensor(jacobian=invalid_jacobian)

        invalid_jacobian = mi.TensorXf(np.ones((3, 2)))
        with self.assertRaises(ValueError):
            self.bezier_2_surface.metric_tensor(jacobian=invalid_jacobian)

        invalid_jacobian = mi.TensorXf(np.ones((5, 5, 3, 2)))
        with self.assertRaises(ValueError):
            self.bezier_2_surface.metric_tensor(jacobian=invalid_jacobian)

    def test_inv_metric_tensor(self):
        """
        Tests whether the inverse metric tensor and its determinant are computed
        correctly.
        """
        jacobian = self.bezier_2_surface.evaluate_jacobian(
            parametric_coords=self.parametric_coords
        )

        metric_tensor = self.bezier_2_surface.metric_tensor(jacobian=jacobian)
        inv_metric_tensor, det_metric_tensor = self.bezier_2_surface.inv_metric_tensor(
            metric=metric_tensor
        )

        expected_inv_metric_tensor = np.array(
            [
                [[0.125, 0.0], [0.0, 1.0]],
                [[0.125, 0.0], [0.0, 1.0]],
                [[0.125, 0.0], [0.0, 1.0]],
                [[0.125, 0.0], [0.0, 1.0]],
                [[0.25, 0.0], [0.0, 1.0]],
            ]
        )
        expected_det_metric_tensor = np.array([8.0, 8.0, 8.0, 8.0, 4.0])
        np.testing.assert_array_almost_equal(
            inv_metric_tensor.numpy(), expected_inv_metric_tensor
        )
        np.testing.assert_array_almost_equal(
            det_metric_tensor.numpy(), expected_det_metric_tensor
        )

    def test_inv_metric_tensor_invalid_input(self):
        """
        Tests whether the inverse metric tensor raises an error when an invalid metric
        is provided.
        """
        invalid_metric = mi.TensorXf(np.ones((5, 3, 2)))
        with self.assertRaises(ValueError):
            self.bezier_2_surface.inv_metric_tensor(metric=invalid_metric)

        invalid_metric = mi.TensorXf(np.ones((5, 2, 3)))
        with self.assertRaises(ValueError):
            self.bezier_2_surface.inv_metric_tensor(metric=invalid_metric)

        invalid_metric = mi.TensorXf(np.ones((2, 2)))
        with self.assertRaises(ValueError):
            self.bezier_2_surface.inv_metric_tensor(metric=invalid_metric)

        invalid_metric = mi.TensorXf(np.ones((5, 5, 2, 2)))
        with self.assertRaises(ValueError):
            self.bezier_2_surface.inv_metric_tensor(metric=invalid_metric)

    def test_knot_insertion(self):
        """
        Tests whether knot insertion works correctly for different cases and updates
        the control points such that the geometry is preserved.

        Implicitly tests whether `BivariateBSpline.insert_knot_and_update_values_u`
        and `BivariateBSpline.insert_knot_and_update_values_v` work correctly.
        """
        test_cases = [
            {
                "u_knots": [0.0, 0.0, 0.49, 1.0, 1.0],
                "v_knots": [0.0, 0.0, 0.51, 1.0, 1.0],
                "u_degree": 1,
                "v_degree": 1,
                "control_points": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 1.0, 2.0],
                        [2.0, 1.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [1.0, 2.0, 0.0],
                        [2.0, 2.0, 0.0],
                    ]
                ),
                "u_insertion_knots": [0.33, 0.66],
                "v_insertion_knots": [0.3, 0.7],
            },
            {
                "u_knots": [0.0, 0.0, 0.0, 0.55, 1.0, 1.0, 1.0],
                "v_knots": [0.0, 0.0, 0.47, 1.0, 1.0],
                "u_degree": 2,
                "v_degree": 1,
                "control_points": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 1.0],
                        [2.0, 0.0, 0.6],
                        [3.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [2.0, 1.0, 0.9],
                        [3.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 1.32],
                        [2.0, 0.0, 0.34],
                        [3.0, 0.0, 0.0],
                    ]
                ),
                "u_insertion_knots": [0.25, 0.75],
                "v_insertion_knots": [0.2, 0.8],
            },
            {
                "u_knots": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                "v_knots": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                "u_degree": 2,
                "v_degree": 2,
                "control_points": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 1.0, 2.0],
                        [2.0, 1.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [1.0, 2.0, 0.0],
                        [2.0, 2.0, 0.0],
                    ]
                ),
                "u_insertion_knots": [0.25, 0.4, 0.75],
                "v_insertion_knots": [0.3, 0.5, 0.7],
            },
        ]
        for i, case in enumerate(test_cases):
            with self.subTest(i=i):
                u_spline = BSpline(knot_vector=case["u_knots"], degree=case["u_degree"])
                v_spline = BSpline(knot_vector=case["v_knots"], degree=case["v_degree"])
                surface = BSplinePatch(
                    name=f"test_surface_{i}",
                    u_spline=u_spline,
                    v_spline=v_spline,
                    control_points=mi.TensorXf(case["control_points"]),
                )

                u_sample_points = np.linspace(0.0, 1.0, 19)
                v_sample_points = np.linspace(0.0, 1.0, 19)
                parametric_coords = np.array(
                    np.meshgrid(u_sample_points, v_sample_points, indexing="xy")
                ).reshape(-1, 2)

                original_points = surface.evaluate_coordinates(
                    parametric_coords=mi.TensorXf(parametric_coords)
                )

                original_num_cps_v = surface._num_cps_v
                original_num_cps_u = surface._num_cps_u

                for u_knot in case["u_insertion_knots"]:
                    surface.insert_knot_u(knot_position=u_knot)

                for v_knot in case["v_insertion_knots"]:
                    surface.insert_knot_v(knot_position=v_knot)

                expected_num_cps_u = original_num_cps_u + len(case["u_insertion_knots"])
                expected_num_cps_v = original_num_cps_v + len(case["v_insertion_knots"])

                self.assertEqual(surface._num_cps_u, expected_num_cps_u)
                self.assertEqual(surface._num_cps_v, expected_num_cps_v)
                self.assertEqual(
                    surface.quads.shape[0],
                    (expected_num_cps_u - 1) * (expected_num_cps_v - 1),
                )

                new_points = surface.evaluate_coordinates(
                    parametric_coords=mi.TensorXf(parametric_coords)
                )

                np.testing.assert_array_almost_equal(
                    original_points.numpy(), new_points.numpy()
                )

    def test_h_refinement(self):
        """
        Tests whether the h-refinement of the surface works correctly.

        Implicitly tests the implementation of `BSpline.h_refinement` method with
        control point positions passed as values.
        """
        u_sample_points = np.linspace(0.0, 1.0, 19)
        v_sample_points = np.linspace(0.0, 1.0, 19)
        parametric_coords = np.array(
            np.meshgrid(u_sample_points, v_sample_points, indexing="xy")
        ).reshape(-1, 2)

        for i in range(4):
            with self.subTest(i=i):
                surface = BSplinePatch(
                    name=f"test_surface_{i}",
                    u_spline=BSpline(knot_vector=self.bezier_2_u_knots, degree=2),
                    v_spline=BSpline(knot_vector=self.bezier_2_v_knots, degree=1),
                    control_points=self.bezier_2_surface.control_points,
                )
                original_points = surface.evaluate_coordinates(
                    parametric_coords=mi.TensorXf(parametric_coords)
                )

                original_num_cps_v = surface._num_cps_v
                original_num_cps_u = surface._num_cps_u

                surface.h_refinement(factor=i + 1)

                expected_num_cps_u = original_num_cps_u + i
                expected_num_cps_v = original_num_cps_v + i

                self.assertEqual(surface._num_cps_u, expected_num_cps_u)
                self.assertEqual(surface._num_cps_v, expected_num_cps_v)
                self.assertEqual(
                    surface.quads.shape[0],
                    (expected_num_cps_u - 1) * (expected_num_cps_v - 1),
                )

                self.assertEqual(
                    surface.interpolating_spline.u_spline.knot_spans.shape[0], i + 1
                )
                self.assertEqual(
                    surface.interpolating_spline.v_spline.knot_spans.shape[0], i + 1
                )

                new_points = surface.evaluate_coordinates(
                    parametric_coords=mi.TensorXf(parametric_coords)
                )

                np.testing.assert_array_almost_equal(
                    original_points.numpy(), new_points.numpy()
                )

    def test_get_node_indices_for(self):
        """
        Tests whether the get_node_indices_for method returns the correct indices
        for each side of the surface.
        """
        # Get node indices for each side
        left = self.bezier_1_surface.get_node_indices_for(BSplinePatch.Side.LEFT)
        right = self.bezier_1_surface.get_node_indices_for(BSplinePatch.Side.RIGHT)
        top = self.bezier_1_surface.get_node_indices_for(BSplinePatch.Side.TOP)
        bottom = self.bezier_1_surface.get_node_indices_for(BSplinePatch.Side.BOTTOM)

        # Check correct vertices are returned for each side
        np.testing.assert_array_almost_equal(
            left.numpy(), np.array([0, 2], dtype=np.uint32)
        )
        np.testing.assert_array_almost_equal(
            right.numpy(), np.array([1, 3], dtype=np.uint32)
        )
        np.testing.assert_array_almost_equal(
            top.numpy(), np.array([2, 3], dtype=np.uint32)
        )

        np.testing.assert_array_almost_equal(
            bottom.numpy(), np.array([0, 1], dtype=np.uint32)
        )

    def test_unit_square_bezier(self):
        """
        Tests whether the unit square mesh is created correctly from a transformed
        unit square.
        """
        patch_square = BSplinePatch.unit_square(
            patch_name="test_unit_square", degree=2, u_subdivisions=1, v_subdivisions=1
        )

        # Check mesh has correct dimensions
        self.assertEqual(patch_square.control_points.shape, (9, 3))
        self.assertEqual(patch_square._spline.u_spline.degree, 2)
        self.assertEqual(patch_square._spline.v_spline.degree, 2)

        expected_vertices = np.array(
            [
                [-0.5, -0.5, 0.0],
                [0.0, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [-0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [-0.5, 0.5, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.5, 0.0],
            ]
        )
        np.testing.assert_array_almost_equal(
            patch_square.control_points.numpy(), expected_vertices
        )

    def test_unit_square_subdivision(self):
        """
        Tests whether the unit square mesh is created correctly from a transformed
        unit square.
        """
        patch_square = BSplinePatch.unit_square(
            patch_name="test_unit_square", degree=2, u_subdivisions=2, v_subdivisions=2
        )

        # Check mesh has correct dimensions
        self.assertEqual(patch_square.control_points.shape, (16, 3))
        self.assertEqual(patch_square._spline.u_spline.degree, 2)
        self.assertEqual(patch_square._spline.v_spline.degree, 2)

        expected_vertices = np.array(
            [
                [-0.5, -0.5, 0.0],
                [-0.25, -0.5, 0.0],
                [0.25, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [-0.5, -0.25, 0.0],
                [-0.25, -0.25, 0.0],
                [0.25, -0.25, 0.0],
                [0.5, -0.25, 0.0],
                [-0.5, 0.25, 0.0],
                [-0.25, 0.25, 0.0],
                [0.25, 0.25, 0.0],
                [0.5, 0.25, 0.0],
                [-0.5, 0.5, 0.0],
                [-0.25, 0.5, 0.0],
                [0.25, 0.5, 0.0],
                [0.5, 0.5, 0.0],
            ]
        )
        np.testing.assert_array_almost_equal(
            patch_square.control_points.numpy(), expected_vertices
        )

    def test_serialization(self):
        """
        Tests whether the BSplinePatch can be serialized and deserialized correctly.

        Only the global control points and quads require special handling, the rest are
        handled by the default serialization mechanism.
        """
        for spline in [self.bezier_1_surface, self.bezier_2_surface]:
            # Serialize the surface
            serialized = jsonpickle.encode(spline)
            deserialized = jsonpickle.decode(serialized)

            # Check that the basic properties are preserved
            np.testing.assert_array_equal(
                deserialized.control_points.numpy(),
                spline.control_points.numpy(),
            )
            np.testing.assert_array_equal(
                deserialized.quads.numpy(), spline.quads.numpy()
            )

    def test_tessellate(self):
        """
        Tests whether the tessellation of the surface is correct

        Checks vertices, faces, normals, and texcoords of the tessellated mesh.
        """
        mesh = self.bezier_1_surface.tessellate(2, 2)

        # Check that the mesh has the expected number of vertices and faces
        self.assertEqual(mesh.vertices.shape[0], 9)
        self.assertEqual(mesh.faces.shape[0], 8)

        # Check that the vertex positions are correct
        vertices = mesh.vertices.numpy()
        faces = mesh.faces.numpy()
        normals = mesh.vertex_normals.numpy()
        texcoords = mesh.texcoords.numpy()

        expected_vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.5, 0.25],
                [1.0, 0.5, 0.5],
                [0.0, 1.0, 0.0],
                [0.5, 1.0, 0.5],
                [1.0, 1.0, 1.0],
            ]
        )
        np.testing.assert_array_almost_equal(vertices, expected_vertices)

        expected_faces = np.array(
            [
                [0, 1, 4],
                [0, 4, 3],
                [1, 2, 5],
                [1, 5, 4],
                [3, 4, 7],
                [3, 7, 6],
                [4, 5, 8],
                [4, 8, 7],
            ]
        )
        np.testing.assert_array_equal(faces[:, :3], expected_faces)

        # Avoids indices 1 and 3 because it is hard to compute the normals by hand
        np.testing.assert_array_almost_equal(normals[0], [0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(normals[2], [0.0, -1.0, 1.0] / np.sqrt(2))
        np.testing.assert_array_almost_equal(normals[4], [-1.0, -1.0, 2.0] / np.sqrt(6))
        np.testing.assert_array_almost_equal(normals[5], [-1.0, -2.0, 2.0] / np.sqrt(9))
        np.testing.assert_array_almost_equal(normals[6], [-1.0, 0.0, 1.0] / np.sqrt(2))
        np.testing.assert_array_almost_equal(normals[7], [-2.0, -1.0, 2.0] / np.sqrt(9))
        np.testing.assert_array_almost_equal(normals[8], [-1.0, -1.0, 1.0] / np.sqrt(3))

        expected_texcoords = np.array(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [0.0, 0.5],
                [0.5, 0.5],
                [1.0, 0.5],
                [0.0, 1.0],
                [0.5, 1.0],
                [1.0, 1.0],
            ]
        )
        np.testing.assert_array_almost_equal(texcoords, expected_texcoords)
