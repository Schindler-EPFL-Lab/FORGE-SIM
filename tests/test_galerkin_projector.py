import unittest

import drjit as dr
import mitsuba as mi
import numpy as np

from rebel_shape.isogeometric_analysis.galerkin_projector import GalerkinProjector
from rebel_shape.isogeometric_analysis.galerkin_projector_smooth import (
    GalerkinProjectorSmooth,
)
from rebel_shape.splines.bspline_patch import BSplinePatch
from rebel_shape.splines.iga_spline_patch import IGASplinePatch


class TestGalerkinProjector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mi.set_variant("llvm_ad_rgb")

    def setUp(self) -> None:
        self.bezier_2_surface = BSplinePatch.unit_square(
            patch_name="Bezier_2", degree=2, u_subdivisions=1, v_subdivisions=1
        )

        self.bezier_2_field_values = dr.zeros(
            dtype=mi.TensorXf,
            shape=(self.bezier_2_surface.interpolating_spline.num_basis_functions, 1),
        )
        self.bezier_2_iga_surface = IGASplinePatch(
            spline_patch=self.bezier_2_surface, field_values=self.bezier_2_field_values
        )

        self.evaluation_samples_per_direction = 23

    def test_project_linear_function(self):
        """
        The linear function f(u, v) = u + v must be exactly projected onto a degree
        2 Bezier surface defined on the unit square.
        """

        def linear_function(parametric_coords: "mi.TensorXf") -> "mi.TensorXf":
            u = parametric_coords[:, 0].numpy()
            v = parametric_coords[:, 1].numpy()
            return mi.TensorXf(u + v)[:, None]

        galerkin_projector = GalerkinProjector(spline_surface=self.bezier_2_iga_surface)

        projected_values = galerkin_projector.galerkin_project_l2(
            rhs_term_function=linear_function
        )

        projected_values_np = projected_values.numpy().reshape(3, 3)
        expected_values = np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 1.5], [1.0, 1.5, 2.0]])
        np.testing.assert_array_almost_equal(projected_values_np, expected_values)

    def test_l2_error_quadratic(self):
        """
        The l2 error of projecting a quadratic function onto a degree 2 Bezier surface
        should be zero, as the surface can exactly represent quadratic functions.
        """

        def quadratic_function(parametric_coords: "mi.TensorXf") -> "mi.TensorXf":
            u = parametric_coords[:, 0].numpy()
            v = parametric_coords[:, 1].numpy()
            return mi.TensorXf(u**2 + v**2)[:, None]

        galerkin_projector = GalerkinProjector(spline_surface=self.bezier_2_iga_surface)

        projected_values = galerkin_projector.galerkin_project_l2(
            rhs_term_function=quadratic_function
        )

        l2_error = galerkin_projector.compute_l2_error(
            projected_values=projected_values, rhs_term_function=quadratic_function
        )
        np.testing.assert_almost_equal(l2_error, 0.0, decimal=6)

    def test_l2_error_cubic(self):
        """
        The l2 error of projecting a cubic function onto a degree 2 Bezier surface
        should be non-zero, as the surface cannot exactly represent cubic functions.
        The exact solution for the best fit of function
            f(u, v) = 8 * (u - 0.5)^3 + 8 * (v - 0.5)^3
        can be computed as f_cap (u, v) = -6/5 + 6/5 (u + v) so the least squares
        error should be sqrt(8/175).
        """

        def cubic_function(parametric_coords: "mi.TensorXf") -> "mi.TensorXf":
            u = parametric_coords[:, 0].numpy()
            v = parametric_coords[:, 1].numpy()
            return mi.TensorXf(8 * (u - 0.5) ** 3 + 8 * (v - 0.5) ** 3)[:, None]

        galerkin_projector = GalerkinProjector(spline_surface=self.bezier_2_iga_surface)

        projected_values = galerkin_projector.galerkin_project_l2(
            rhs_term_function=cubic_function
        )

        l2_error = galerkin_projector.compute_l2_error(
            projected_values=projected_values, rhs_term_function=cubic_function
        )

        np.testing.assert_almost_equal(
            l2_error.numpy(), np.sqrt(8.0 / 175.0), decimal=6
        )

    def test_project_sine_function(self):
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
        self.bezier_2_iga_surface.h_refinement(factor=16)

        galerkin_projector = GalerkinProjector(spline_surface=self.bezier_2_iga_surface)

        projected_values = galerkin_projector.galerkin_project_l2(
            rhs_term_function=sine_function
        )

        samples = self.evaluation_samples_per_direction

        (u_min, u_max), (v_min, v_max) = (
            self.bezier_2_iga_surface.parametric_domain_size
        )
        u = np.linspace(u_min, u_max, samples)
        v = np.linspace(v_min, v_max, samples)
        uu, vv = np.meshgrid(u, v)
        sample_points = mi.TensorXf(np.column_stack([uu.flatten(), vv.flatten()]))

        interpolated_values = (
            self.bezier_2_iga_surface.interpolating_spline.interpolate(
                values=projected_values, parametric_coords=sample_points
            )
        )
        expected_values = sine_function(sample_points)

        # Ensure absolute point-wise errors are less than 1e-4
        np.testing.assert_array_almost_equal(
            interpolated_values.numpy().flatten(),
            expected_values.numpy().flatten(),
            decimal=4,
        )

        l2_error = galerkin_projector.compute_l2_error(
            projected_values=projected_values, rhs_term_function=sine_function
        )
        # For smooth functions the error scales as O(h^(p+1)), where h is the
        # biggest knot span size and p is the degree of the spline.
        self.assertLess(l2_error, 1e-4, "L2 error is too high")

    def test_project_window_l2_smooth(self):
        """
        Projection of a window function onto the B-spline surface.

        The window function is defined as a rectangular window in the parametric space.
        The projection should yield non-zero values only within the window area.
        """

        def window_function(parametric_coords: "mi.TensorXf") -> "mi.TensorXf":
            u = parametric_coords[:, 0].numpy()
            v = parametric_coords[:, 1].numpy()
            return mi.TensorXf((u >= 0.4) & (u <= 0.6) & (v >= 0.4) & (v <= 0.6))[
                :, None
            ]

        # Refine the surface for better approximation
        self.bezier_2_iga_surface.h_refinement(factor=16)

        galerkin_projector = GalerkinProjectorSmooth(
            spline_surface=self.bezier_2_iga_surface
        )

        projected_values = galerkin_projector.galerkin_project_l2(
            rhs_term_function=window_function
        )

        l2_error = galerkin_projector.compute_l2_error(
            projected_values=projected_values, rhs_term_function=window_function
        )

        # The allowed error is still very high but this is because smooth basis
        # functions fundamentally cannot represent non-smooth functions well. The
        # error estimate degrades to O(h^0.5), where h is the biggest knot span size.
        # And thus for the sake of a quick test a high error threshold is set.
        self.assertLess(l2_error, 0.1, "L2 error is too high")
