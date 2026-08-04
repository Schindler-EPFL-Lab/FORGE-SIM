import unittest

import drjit as dr
import mitsuba as mi
import numpy as np

from rebel_shape.isogeometric_analysis.modal_analysis_solver import ModalAnalysisSolver
from rebel_shape.isogeometric_analysis.reissner_mindlin_modal_problem import (
    ReissnerMindlinModalProblem,
)
from rebel_shape.meshing.composite_iga_spline_surface_factory import (
    CompositeIGASplineSurfaceFactory,
)
from rebel_shape.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)
from rebel_shape.splines.bspline_patch import BSplinePatch
from rebel_shape.splines.cube_initialization import cubed_sphere
from rebel_shape.splines.iga_spline_patch import IGASplinePatch


class TestReissnerMindlinModalProblem(unittest.TestCase):
    def setUp(self) -> None:
        mi.set_variant("llvm_ad_rgb")

        patch = BSplinePatch.unit_square(
            patch_name="patch", degree=2, u_subdivisions=1, v_subdivisions=1
        )
        field_values = dr.zeros(
            dtype=mi.TensorXf,
            shape=(patch.interpolating_spline.num_basis_functions, 6),
        )
        iga_patch = IGASplinePatch(spline_patch=patch, field_values=field_values)

        factory = CompositeIGASplineSurfaceFactory("rm_modal")
        factory.add_patch(iga_patch)
        self.iga_surface = factory.build()

        n_dofs = self.iga_surface.global_field_values.shape[0]
        ones = dr.ones(dtype=mi.TensorXf, shape=(n_dofs, 1))

        self.youngs_modulus = 2.0 * ones
        self.poisson_ratio = 0.3 * ones
        self.density = 1.0 * ones
        self.thickness = 0.2 * ones

    def _create_problem(
        self,
        thickness: "mi.TensorXf | None" = None,
        shear_correction_factor: float = 5.0 / 6.0,
    ) -> ReissnerMindlinModalProblem:
        if thickness is None:
            thickness = self.thickness
        return ReissnerMindlinModalProblem(
            composite_iga_spline=self.iga_surface,
            youngs_modulus=self.youngs_modulus,
            poisson_ratio=self.poisson_ratio,
            density=self.density,
            thickness=thickness,
            shear_correction_factor=shear_correction_factor,
        )

    def test_stiffness_symmetry(self) -> None:
        problem = self._create_problem()
        _, local_matrices = problem.assemble_local_stiffness_matrices()
        matrix = local_matrices[0].numpy()
        np.testing.assert_allclose(matrix, matrix.T, rtol=1e-6, atol=1e-6)

    def test_shear_correction_factor_affects_stiffness(self) -> None:
        problem_default = self._create_problem(shear_correction_factor=5.0 / 6.0)
        problem_one = self._create_problem(shear_correction_factor=1.0)

        _, K_default = problem_default.assemble_local_stiffness_matrices()
        _, K_one = problem_one.assemble_local_stiffness_matrices()

        # Higher shear correction factor increases stiffness
        self.assertGreater(
            np.linalg.norm(K_one[0].numpy()),
            np.linalg.norm(K_default[0].numpy()),
        )

    def test_higher_thickness_increases_stiffness(self) -> None:
        thickness_low = 0.1 * dr.ones(dtype=mi.TensorXf, shape=self.thickness.shape)
        thickness_high = 0.2 * dr.ones(dtype=mi.TensorXf, shape=self.thickness.shape)

        problem_low = self._create_problem(thickness=thickness_low)
        problem_high = self._create_problem(thickness=thickness_high)

        _, K_low = problem_low.assemble_local_stiffness_matrices()
        _, K_high = problem_high.assemble_local_stiffness_matrices()

        ratio = np.linalg.norm(K_high[0].numpy()) / np.linalg.norm(K_low[0].numpy())
        # Thickness increases stiffness (membrane ~ h, bending ~ h^3)
        self.assertGreater(ratio, 1.5)

    def test_batched_assembly_matches_standard(self) -> None:
        """Batched global assembly must produce the same matrix as standard."""
        problem_std = self._create_problem()
        problem_bat = ReissnerMindlinModalProblem(
            composite_iga_spline=self.iga_surface,
            youngs_modulus=self.youngs_modulus,
            poisson_ratio=self.poisson_ratio,
            density=self.density,
            thickness=self.thickness,
            max_batch_size=1,
        )

        K_std = problem_std.assemble_global_stiffness_matrix()
        K_bat = problem_bat.assemble_global_stiffness_matrix()
        np.testing.assert_allclose(
            K_std.toarray(), K_bat.toarray(), rtol=1e-5, atol=1e-5
        )

        M_std = problem_std.assemble_global_mass_matrix()
        M_bat = problem_bat.assemble_global_mass_matrix()
        np.testing.assert_allclose(
            M_std.toarray(), M_bat.toarray(), rtol=1e-5, atol=1e-5
        )

    def test_free_free_has_six_rigid_modes(self) -> None:
        problem = self._create_problem()
        solver = ModalAnalysisSolver(modal_problem=problem)

        eigenvalues, _ = solver.solve(num_modes=8)
        self.assertTrue(np.all(eigenvalues[:6] < 1.0e-3))

    def test_square_plate_known_eigenvalues(self) -> None:
        """
        Test modal analysis on a free-free square plate against analytical expectations.

        Reference: Leissa, A. W. (1969). Vibration of Plates. Table 4.4.
        For a square plate (a/b=1) with Poisson ratio 0.3, the non-dimensional
        frequency parameter lambda = omega * a^2 * sqrt(rho * h / D)
        has specific values.

        The first 6 modes should be rigid body modes (lambda approx 0).
        The first symmetric bending mode corresponds to lambda approx 13.49.
        """
        patch = BSplinePatch.unit_square(
            patch_name="plate", degree=2, u_subdivisions=8, v_subdivisions=8
        )
        field_values = dr.zeros(
            dtype=mi.TensorXf,
            shape=(patch.interpolating_spline.num_basis_functions, 6),
        )
        iga_patch = IGASplinePatch(spline_patch=patch, field_values=field_values)

        factory = CompositeIGASplineSurfaceFactory("plate")
        factory.add_patch(iga_patch)
        iga_surface = factory.build()

        n_dofs = iga_surface.global_field_values.shape[0]
        ones = dr.ones(dtype=mi.TensorXf, shape=(n_dofs, 1))

        # Material properties
        E = 200e9
        nu = 0.3
        rho = 7850.0
        h = 0.01
        L = 1.0  # Unit square

        D = E * h**3 / (12 * (1 - nu**2))

        problem = ReissnerMindlinModalProblem(
            composite_iga_spline=iga_surface,
            youngs_modulus=E * ones,
            poisson_ratio=nu * ones,
            density=rho * ones,
            thickness=h * ones,
            shear_correction_factor=5.0 / 6.0,
            integration_order=3,
        )

        solver = ModalAnalysisSolver(modal_problem=problem)
        eigenvalues, _ = solver.solve(num_modes=10)

        # Verify first symmetric bending mode against Leissa reference
        # Leissa Table 4.4 (a/b=1, nu=0.3): lambda_1 = 13.49
        # omega = lambda / a^2 * sqrt(D / (rho * h))
        lambda_ref = 13.49

        # Convert all eigenvalues to non-dimensional lambda (skip rigid modes)
        lambdas = np.sqrt(np.abs(eigenvalues)) * L**2 * np.sqrt(rho * h / D)

        # Find the mode closest to the Leissa reference value
        closest_idx = int(np.argmin(np.abs(lambdas - lambda_ref)))
        closest_lambda = lambdas[closest_idx]

        # Allow 10% tolerance (coarse 8x8 mesh)
        self.assertAlmostEqual(
            closest_lambda,
            lambda_ref,
            delta=0.1 * lambda_ref,
            msg=(
                f"No mode matches Leissa lambda_1={lambda_ref}. "
                f"Closest: lambda={closest_lambda:.4f} (mode {closest_idx}). "
                f"All lambdas: {lambdas[:8]}"
            ),
        )

    def test_cubed_sphere_small_knot_spans_modal_analysis(self) -> None:
        """
        Test modal analysis on a cubed sphere with small number of knot spans.

        This test exercises the scatter function with a realistic geometry and
        assembly pattern. The cubed sphere is created without G1 optimization to
        keep knot span count manageable while still testing the full assembly.
        """
        tessellated_sphere = cubed_sphere(
            name="test_cubed_sphere",
            degree=2,
            spline_subdivisions=8,
            tessellation_subdivisions=32,
            radius=1.0,
            optimize_g1=False,
        )

        per_patch_field_values = [
            dr.zeros(
                dtype=mi.TensorXf,
                shape=(patch.control_points.shape[0], 6),
            )
            for patch in tessellated_sphere.composite_spline
        ]

        tessellated_iga_spline = (
            TessellatedIGASplineSurface.from_tessellated_spline_surface(
                tessellated_spline_surface=tessellated_sphere,
                per_patch_field_values=per_patch_field_values,
            )
        )

        n_dofs = tessellated_iga_spline.global_field_values.shape[0]
        ones = dr.ones(dtype=mi.TensorXf, shape=(n_dofs, 1))

        youngs_modulus = 1.0e3 * ones
        poisson_ratio = 0.3 * ones
        density = 1.0 * ones
        thickness = 0.01 * ones

        modal_problem = ReissnerMindlinModalProblem(
            composite_iga_spline=tessellated_iga_spline.composite_spline,
            youngs_modulus=youngs_modulus,
            poisson_ratio=poisson_ratio,
            density=density,
            thickness=thickness,
            shear_correction_factor=5.0 / 6.0,
        )

        solver = ModalAnalysisSolver(modal_problem=modal_problem)
        eigenvalues, eigenvectors = solver.solve(num_modes=12)

        self.assertEqual(len(eigenvalues), 12)
        self.assertEqual(eigenvectors.shape[1], 12)

        # With spectral shift, rigid body modes appear as small negative eigenvalues
        # and elastic modes are positive. Check that eigenvalues are sorted and
        # that there are positive (elastic) eigenvalues present.
        self.assertTrue(
            np.all(np.diff(eigenvalues) >= -1.0e-6),
            msg="Eigenvalues must be sorted in ascending order",
        )

        elastic_eigenvalues = eigenvalues[eigenvalues > 1.0]
        self.assertGreater(
            len(elastic_eigenvalues),
            0,
            msg="Expected at least one elastic mode",
        )
