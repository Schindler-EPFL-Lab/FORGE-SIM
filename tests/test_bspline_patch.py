import unittest

import drjit as dr
import jsonpickle
import mitsuba as mi
import numpy as np
from OCP.Geom import Geom_BSplineSurface

from rebel_shape import logger as rebel_shape_logger
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
        cls._bezier_1_control_points_data = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]

        cls._parametric_coords_data = [
            [0.0, 0.0],  # Bottom-left
            [1.0, 0.0],  # Bottom-right
            [0.0, 1.0],  # Top-left
            [1.0, 1.0],  # Top-right
            [0.5, 0.5],  # Center
        ]

        cls.bezier_2_u_knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        cls.bezier_2_v_knots = [0.0, 0.0, 1.0, 1.0]
        cls._bezier_2_control_points_data = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
        cls.bezier_2_u_spline = BSpline(knot_vector=cls.bezier_2_u_knots, degree=2)
        cls.bezier_2_v_spline = BSpline(knot_vector=cls.bezier_2_v_knots, degree=1)

        cls.samples_per_dir = 19
        """
        Number of samples per each direction when evaluating the surface.
        Should be kept small to avoid long test times, but large enough
        to ensure that the surface is well sampled.
        """

        cls.bezier_3_u_knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        cls.bezier_3_v_knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        cls._bezier_3_control_points_data = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.5, 0.0],
            [2.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
        cls.bezier_3_u_spline = BSpline(knot_vector=cls.bezier_3_u_knots, degree=2)
        cls.bezier_3_v_spline = BSpline(knot_vector=cls.bezier_3_v_knots, degree=2)

        cls.autodiff_test_cases_dict = [
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
            {
                "u_knots": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                "v_knots": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                "u_degree": 3,
                "v_degree": 2,
                "control_points": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 1.0, 2.0],
                        [2.0, 1.0, 0.0],
                        [3.0, 1.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [1.0, 2.0, 0.0],
                        [2.0, 2.0, 0.0],
                        [3.0, 2.0, 0.0],
                    ]
                ),
                "u_insertion_knots": [0.27, 0.39, 0.58, 0.83],
                "v_insertion_knots": [0.32, 0.51, 0.77],
            },
            {
                "u_knots": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                "v_knots": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                "u_degree": 3,
                "v_degree": 3,
                "control_points": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 1.0, 2.0],
                        [2.0, 1.0, 0.0],
                        [3.0, 1.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [1.0, 2.0, 0.0],
                        [2.0, 2.0, 0.0],
                        [3.0, 2.0, 0.0],
                        [0.0, 3.0, 0.0],
                        [1.0, 3.0, 0.0],
                        [2.0, 3.0, 0.0],
                        [3.0, 3.0, 0.0],
                    ]
                ),
                "u_insertion_knots": [0.24, 0.4, 0.6, 0.81],
                "v_insertion_knots": [0.3, 0.5, 0.7],
            },
        ]

    def setUp(self) -> None:
        # Control-point tensors and the surfaces built from them are
        # constructed per test rather than cached on the class: a Mitsuba
        # object that outlives the test which built it aborts a *later,
        # unrelated* test inside Dr.Jit's thread pool ("Assertion failed ...
        # queue.cpp: remain == 1").
        self.bezier_1_control_points = mi.TensorXf(
            np.array(self._bezier_1_control_points_data)
        )
        self.bezier_1_surface = BSplinePatch(
            name="bezier_1_surface",
            u_spline=self.bezier_1_spline,
            v_spline=self.bezier_1_spline,
            control_points=self.bezier_1_control_points,
        )

        self.parametric_coords = mi.TensorXf(np.array(self._parametric_coords_data))

        self.bezier_2_control_points = mi.TensorXf(
            np.array(self._bezier_2_control_points_data)
        )
        self.bezier_2_surface = BSplinePatch(
            name="bezier_2_surface",
            u_spline=self.bezier_2_u_spline,
            v_spline=self.bezier_2_v_spline,
            control_points=self.bezier_2_control_points,
        )

        self.bezier_3_control_points = mi.TensorXf(
            np.array(self._bezier_3_control_points_data)
        )
        self.bezier_3_surface = BSplinePatch(
            name="bezier_3_surface",
            u_spline=self.bezier_3_u_spline,
            v_spline=self.bezier_3_v_spline,
            control_points=self.bezier_3_control_points,
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

    @staticmethod
    def _tilted_plane_patch(scale: float) -> BSplinePatch:
        """
        A degree-1 (bilinear) patch whose control points describe a tilted flat
        plane, uniformly scaled by `scale`. `|du x dv|` at the domain center is
        `scale ** 2 * sqrt(2)`, so `scale` alone controls how close the patch
        sits to `evaluate_normals`'s degeneracy threshold. The plane is tilted
        (not axis-aligned) so its true normal, `(0, -1, 1) / sqrt(2)`, cannot be
        mistaken for the `(0, 0, 1)` degenerate fallback.
        """
        knots = [0.0, 0.0, 1.0, 1.0]
        spline = BSpline(knot_vector=knots, degree=1)
        control_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        return BSplinePatch(
            name="tilted_plane",
            u_spline=spline,
            v_spline=spline,
            control_points=mi.TensorXf(control_points * scale),
        )

    def test_evaluate_normals_below_tolerance_uses_unit_z_fallback(self):
        """
        Regression test: `evaluate_normals` treats `|du x dv| <= ZERO_TOL ** 2`
        as numerically degenerate (colinear or vanishing derivatives). Below
        that threshold it must return the documented `(0, 0, 1)` fallback
        instead of dividing a near-zero cross product by its own near-zero
        magnitude, and it must warn so the caller can see their
        parameterization or scale collapsed.
        """
        # scale ** 2 * sqrt(2) ~= 1.4e-14, far below ZERO_TOL ** 2 = 1e-12.
        patch = self._tilted_plane_patch(scale=1e-7)
        uv = mi.TensorXf(np.array([[0.5, 0.5]]))

        with self.assertLogs(rebel_shape_logger, level="WARNING") as captured:
            normals = patch.evaluate_normals(parametric_coords=uv)

        np.testing.assert_array_equal(normals.numpy(), np.array([[0.0, 0.0, 1.0]]))
        self.assertTrue(
            any("fallback" in message for message in captured.output),
            captured.output,
        )

    def test_evaluate_normals_just_above_tolerance_is_unit_length_not_fallback(self):
        """
        A cross product a factor of two above the `ZERO_TOL ** 2` threshold must
        be treated as valid: normalized to unit length from the real geometry
        (not the `(0, 0, 1)` fallback), and without a degeneracy warning.

        Together with `test_evaluate_normals_below_tolerance_uses_unit_z_fallback`
        this brackets the threshold's *magnitude*: `2 * ZERO_TOL ** 2` is six
        orders of magnitude below `ZERO_TOL`, so comparing against `ZERO_TOL`
        itself -- the natural mix-up, since the cross product has units of
        (length)^2 while `ZERO_TOL` is a length -- sends this case to the
        fallback and fails here. It does not pin the comparison as `>` rather
        than `>=`: those differ only for a magnitude bitwise equal to the
        threshold, which no float32 evaluation chain can be steered onto
        reproducibly, and which the threshold's meaning makes immaterial anyway.
        """
        # scale chosen so |du x dv| = 2 * ZERO_TOL ** 2, i.e. just inside the
        # valid region.
        scale = float(np.sqrt(2.0 * BSpline.ZERO_TOL**2 / np.sqrt(2.0)))
        patch = self._tilted_plane_patch(scale=scale)
        uv = mi.TensorXf(np.array([[0.5, 0.5]]))

        with self.assertNoLogs(rebel_shape_logger, level="WARNING"):
            normals = patch.evaluate_normals(parametric_coords=uv)

        normal = normals.numpy()[0]
        self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0, places=5)
        expected_direction = np.array([0.0, -1.0, 1.0]) / np.sqrt(2.0)
        np.testing.assert_allclose(normal, expected_direction, rtol=0, atol=1e-4)

    def test_first_derivative_autodiff(self):
        for i, case in enumerate(self.autodiff_test_cases_dict):
            with self.subTest(i=i):
                u_spline = BSpline(knot_vector=case["u_knots"], degree=case["u_degree"])
                v_spline = BSpline(knot_vector=case["v_knots"], degree=case["v_degree"])
                surface = BSplinePatch(
                    name=f"test_surface_{i}",
                    u_spline=u_spline,
                    v_spline=v_spline,
                    control_points=mi.TensorXf(case["control_points"]),
                )

                # Create a grid of parametric coordinates
                u_vals = dr.linspace(mi.Float, 0.0, 1.0, self.samples_per_dir)
                v_vals = dr.linspace(mi.Float, 0.0, 1.0, self.samples_per_dir)
                u_grid, v_grid = dr.meshgrid(u_vals, v_vals, indexing="xy")

                params = dr.empty(dtype=mi.TensorXf, shape=(self.samples_per_dir**2, 2))
                params[:, 0] = u_grid
                params[:, 1] = v_grid

                dr.enable_grad(params)
                evaluation = surface.evaluate_coordinates(parametric_coords=params)

                dr.forward(params)
                derivative_ad_u_plus_v = mi.TensorXf(evaluation.grad)

                # Calculate analytical derivatives
                u_derivative, v_derivative = surface.evaluate_first_derivatives(
                    parametric_coords=params
                )

                # Compare analytical derivatives with automatic differentiation
                np.testing.assert_array_almost_equal(
                    (u_derivative + v_derivative).numpy(),
                    derivative_ad_u_plus_v,
                    decimal=5,
                )

    def test_second_derivative_autodiff(self):
        for i, case in enumerate(self.autodiff_test_cases_dict[2:]):
            with self.subTest(i=i):
                u_spline = BSpline(knot_vector=case["u_knots"], degree=case["u_degree"])
                v_spline = BSpline(knot_vector=case["v_knots"], degree=case["v_degree"])
                surface = BSplinePatch(
                    name=f"test_surface_{i}",
                    u_spline=u_spline,
                    v_spline=v_spline,
                    control_points=mi.TensorXf(case["control_points"]),
                )

                # Create a grid of parametric coordinates
                u_vals = dr.linspace(mi.Float, 0.0, 1.0, self.samples_per_dir)
                v_vals = dr.linspace(mi.Float, 0.0, 1.0, self.samples_per_dir)
                u_grid, v_grid = dr.meshgrid(u_vals, v_vals, indexing="xy")

                params = dr.empty(dtype=mi.TensorXf, shape=(self.samples_per_dir**2, 2))
                params[:, 0] = u_grid
                params[:, 1] = v_grid

                dr.enable_grad(params)
                du, dv = surface.evaluate_first_derivatives(parametric_coords=params)

                dr.forward(params)
                derivative_ad_uu_plus_2uv_plus_vv = mi.TensorXf(du.grad + dv.grad)

                # Calculate analytical derivatives
                uu_derivative, uv_derivative, vv_derivative = (
                    surface.evaluate_second_derivatives(parametric_coords=params)
                )

                # Compare analytical second derivatives with autodiff values
                np.testing.assert_array_almost_equal(
                    (uu_derivative + 2 * uv_derivative + vv_derivative).numpy(),
                    derivative_ad_uu_plus_2uv_plus_vv,
                    decimal=5,
                )

    def test_normal_derivative_autodiff(self):
        for i, case in enumerate(self.autodiff_test_cases_dict[2:]):
            with self.subTest(i=i):
                u_spline = BSpline(knot_vector=case["u_knots"], degree=case["u_degree"])
                v_spline = BSpline(knot_vector=case["v_knots"], degree=case["v_degree"])
                surface = BSplinePatch(
                    name=f"test_surface_{i}",
                    u_spline=u_spline,
                    v_spline=v_spline,
                    control_points=mi.TensorXf(case["control_points"]),
                )

                u_vals = dr.linspace(mi.Float, 0.0, 1.0, self.samples_per_dir)
                v_vals = dr.linspace(mi.Float, 0.0, 1.0, self.samples_per_dir)
                u_grid, v_grid = dr.meshgrid(u_vals, v_vals, indexing="xy")

                params = dr.empty(dtype=mi.TensorXf, shape=(self.samples_per_dir**2, 2))
                params[:, 0] = u_grid
                params[:, 1] = v_grid

                dr.enable_grad(params)
                normals = surface.evaluate_normals(parametric_coords=params)

                dr.forward(params)
                derivative_ad_u_plus_v = mi.TensorXf(normals.grad)

                jacobian = surface.evaluate_jacobian(parametric_coords=params)
                hessian = surface.evaluate_hessian(parametric_coords=params)
                normal_derivative = surface.normal_derivative(
                    jacobian=jacobian, hessian=hessian, normal=normals
                )

                np.testing.assert_array_almost_equal(
                    (normal_derivative[:, :, 0] + normal_derivative[:, :, 1]).numpy(),
                    derivative_ad_u_plus_v,
                    decimal=5,
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

    def test_evaluate_hessian(self):
        """
        Tests whether the Hessian of the Bezier surface with degrees 2
        in the u and v directions is computed correctly.


        The surface of the Bezier curve defined as
        S(u, v) = (1 - u)^2 (1 - v)^2 P_{0, 0} + \
            + (1 - u)^2 2(1 - v)v P_{0, 1} + \
            + (1 - u)^2 v^2 P_{0, 2} + \
            + 2(1 - u)u (1 - v)^2 P_{1, 0} + \
            + 2(1 - u)u 2(1 - v)v P_{1, 1} + \
            + 2(1 - u)u v^2 P_{1, 2} + \
            + u^2 (1 - v)^2 P_{2, 0} + \
            + u^2 2(1 - v)v P_{2, 1} + \
            + u^2 v^2 P_{2, 2}

        One can manually derive _uu, _uv, _vv
        """
        hessian = self.bezier_3_surface.evaluate_hessian(
            parametric_coords=self.parametric_coords
        )

        second_derivatives = np.array(
            [
                [[0, 0, -4], [0, 0, -4], [0, 0, -4], [0, 0, -4], [0, 0, 0]],  # uu
                [[0, 2, -8], [0, 2, 8], [0, -2, 8], [0, -2, -8], [0, 0, 0]],  # uv
                [[0, 2, -4], [0, -2, -4], [0, 2, -4], [0, -2, -4], [0, 0, 0]],  # vv
            ]
        )

        expected_hessian = np.empty((5, 3, 2, 2), dtype=np.float32)
        for point_idx in range(5):
            for coord_idx in range(3):
                expected_hessian[point_idx, coord_idx, :, :] = [
                    [
                        second_derivatives[0, point_idx, coord_idx],
                        second_derivatives[1, point_idx, coord_idx],
                    ],
                    [
                        second_derivatives[1, point_idx, coord_idx],
                        second_derivatives[2, point_idx, coord_idx],
                    ],
                ]

        np.testing.assert_array_almost_equal(hessian.numpy(), expected_hessian)

    def test_knot_insertion(self):
        """
        Tests whether knot insertion works correctly for different cases and updates
        the control points such that the geometry is preserved.

        Implicitly tests whether `BivariateBSpline.insert_knot_and_update_values_u`
        and `BivariateBSpline.insert_knot_and_update_values_v` work correctly.
        """
        for i, case in enumerate(self.autodiff_test_cases_dict):
            with self.subTest(i=i):
                u_spline = BSpline(knot_vector=case["u_knots"], degree=case["u_degree"])
                v_spline = BSpline(knot_vector=case["v_knots"], degree=case["v_degree"])
                surface = BSplinePatch(
                    name=f"test_surface_{i}",
                    u_spline=u_spline,
                    v_spline=v_spline,
                    control_points=mi.TensorXf(case["control_points"]),
                )

                u_sample_points = np.linspace(0.0, 1.0, self.samples_per_dir)
                v_sample_points = np.linspace(0.0, 1.0, self.samples_per_dir)
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
        u_sample_points = np.linspace(0.0, 1.0, self.samples_per_dir)
        v_sample_points = np.linspace(0.0, 1.0, self.samples_per_dir)

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

    def test_side_knot_spans(self):
        """
        Tests whether the side_knot_spans method returns the correct knot spans
        for each side of the B-spline patch.
        """
        knots_u = [0.0, 0.0, 0.0, 0.24, 0.5, 0.72, 1.0, 1.0, 1.0]
        knots_v = [0.0, 0.0, 0.32, 0.66, 1.0, 1.0]
        control_points = mi.TensorXf(np.zeros((((2 + 4) * (1 + 3)), 3)))

        spline_u = BSpline(knot_vector=knots_u, degree=2)
        spline_v = BSpline(knot_vector=knots_v, degree=1)
        spline_patch = BSplinePatch(
            name="test_patch",
            u_spline=spline_u,
            v_spline=spline_v,
            control_points=control_points,
        )

        # Test with bezier_1_surface (linear, uniform knot vectors)
        left_spans = spline_patch.side_knot_spans(BSplinePatch.Side.LEFT)
        right_spans = spline_patch.side_knot_spans(BSplinePatch.Side.RIGHT)
        top_spans = spline_patch.side_knot_spans(BSplinePatch.Side.TOP)
        bottom_spans = spline_patch.side_knot_spans(BSplinePatch.Side.BOTTOM)

        # For LEFT/RIGHT sides, should return v_spline knot spans
        self.assertTrue(dr.allclose(left_spans, spline_v.knot_spans))
        self.assertTrue(dr.allclose(right_spans, spline_v.knot_spans))

        # For TOP/BOTTOM sides, should return u_spline knot spans
        self.assertTrue(dr.allclose(top_spans, spline_u.knot_spans))
        self.assertTrue(dr.allclose(bottom_spans, spline_u.knot_spans))

    def test_side_parametric_coords(self):
        """
        Tests whether the side_parametric_coords method returns correct full 2D
        parametric coordinates for points along each side of the patch.
        """
        # Test coordinates along the side
        side_coords = mi.TensorXf(np.array([0.0, 0.5, 1.0]))

        expected_full_coords = [
            mi.TensorXf([[0.0, v] for v in side_coords.numpy()]),  # LEFT
            mi.TensorXf([[1.0, v] for v in side_coords.numpy()]),  # RIGHT
            mi.TensorXf([[u, 0.0] for u in side_coords.numpy()]),  # BOTTOM
            mi.TensorXf([[u, 1.0] for u in side_coords.numpy()]),  # TOP
        ]

        for side, coords in zip(BSplinePatch.Side.all_sides(), expected_full_coords):
            with self.subTest(side=side):
                result = self.bezier_1_surface.side_parametric_coords(side, side_coords)
                self.assertTrue(dr.allclose(result, coords))

    def test_side_parametric_coords_flip(self):
        """
        Tests whether the flip parameter correctly reverses the order of
        side coordinates in side_parametric_coords method.
        """
        side_coords = mi.TensorXf(np.array([0.0, 0.5, 1.0]))

        # Test LEFT side with flip=True
        left_coords_flipped = self.bezier_1_surface.side_parametric_coords(
            BSplinePatch.Side.LEFT, side_coords, flip=True
        )
        expected_left_flipped = mi.TensorXf(
            [[0.0, v] for v in side_coords.numpy()[::-1]]
        )
        self.assertTrue(dr.allclose(left_coords_flipped, expected_left_flipped))

    def test_side_parametric_coords_validation(self):
        """
        Tests that side_parametric_coords properly validates input coordinates
        and raises appropriate errors for invalid values.
        """
        # Test with coordinates outside valid range
        invalid_coords = mi.TensorXf(np.array([-0.1, 0.5, 1.1]))

        # This should raise ValueError due to validation in the underlying splines
        for side in BSplinePatch.Side.all_sides():
            with self.subTest(side=side):
                with self.assertRaises(ValueError):
                    self.bezier_1_surface.side_parametric_coords(side, invalid_coords)

    def test_brep_export(self):
        """
        Tests whether the B-Rep export functionality produces valid OCP B-Spline
        surfaces.
        """
        # Test with bezier_1_surface
        brep_surface = self.bezier_1_surface.to_brep()

        # Check that the B-Rep surface is valid
        self.assertIsInstance(brep_surface, Geom_BSplineSurface)

        # Check that the degrees match
        self.assertEqual(
            brep_surface.UDegree(), self.bezier_1_surface._spline.u_spline.degree
        )
        self.assertEqual(
            brep_surface.VDegree(), self.bezier_1_surface._spline.v_spline.degree
        )

        # Check number of control points
        self.assertEqual(brep_surface.NbUPoles(), self.bezier_1_surface._num_cps_u)
        self.assertEqual(brep_surface.NbVPoles(), self.bezier_1_surface._num_cps_v)

        # Test with more complex surface
        brep_surface_2 = self.bezier_2_surface.to_brep()
        self.assertIsInstance(brep_surface_2, Geom_BSplineSurface)
        self.assertEqual(brep_surface_2.UDegree(), 2)
        self.assertEqual(brep_surface_2.VDegree(), 1)
        self.assertEqual(brep_surface_2.NbUPoles(), 3)
        self.assertEqual(brep_surface_2.NbVPoles(), 2)

        # Test with cubic surface
        brep_surface_3 = self.bezier_3_surface.to_brep()
        self.assertIsInstance(brep_surface_3, Geom_BSplineSurface)
        self.assertEqual(brep_surface_3.UDegree(), 2)
        self.assertEqual(brep_surface_3.VDegree(), 2)
        self.assertEqual(brep_surface_3.NbUPoles(), 3)
        self.assertEqual(brep_surface_3.NbVPoles(), 3)
