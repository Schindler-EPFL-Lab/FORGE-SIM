"""
Slow integration test for Reissner-Mindlin modal analysis on a steel sphere.

This test loads an optimized G1 cubed sphere from JSON and verifies the
eigenfrequency spectrum against known reference values. It takes ~35 seconds
and is intended to be run separately from the fast unit tests.

Run with:
    uv run python -m unittest tests.test_steel_sphere_modal_analysis
"""

import unittest
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np

from forge_sim.isogeometric_analysis.modal_analysis_solver import ModalAnalysisSolver
from forge_sim.isogeometric_analysis.reissner_mindlin_modal_problem import (
    ReissnerMindlinModalProblem,
)
from forge_sim.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)
from forge_sim.meshing.tessellated_spline_surface import TessellatedSplineSurface


class TestSteelSphereModalAnalysis(unittest.TestCase):
    """
    Integration test for the Reissner-Mindlin shell eigenvalue solver on an
    optimized G1 cubed sphere with steel material properties.

    The sphere JSON contains a unit sphere (R = 1.0 m) with 16x16 knot spans
    per patch and degree-3 splines. The G1 continuity across patch boundaries
    was obtained by curvature-aware optimization.

    Reference material: E = 193.05 GPa, nu = 0.28, rho = 8025.9 kg/m^3,
    h = 1.5875 mm.

    For a sphere with R = 0.1135 m, the first elastic mode (l = 2 spherical
    harmonic, 5-fold degenerate) is measured at ~5090 Hz with float32 assembly.
    The mode is membrane-dominated (f ~ 1/R), so for R = 1.0 m the expected
    frequency is 5090 * 0.1135 = 577 Hz.
    """

    SPHERE_PATH = Path("data/splines/spline_sphere_16.json")

    @classmethod
    def setUpClass(cls) -> None:
        mi.set_variant("llvm_ad_rgb")

        if not cls.SPHERE_PATH.exists():
            raise unittest.SkipTest(f"Sphere file not found: {cls.SPHERE_PATH}")

        tessellated_sphere = TessellatedSplineSurface.from_file(
            composite_spline_surface_path=cls.SPHERE_PATH
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

        E = 193.05e9
        nu = 0.28
        rho = 8025.9
        h = 1.5875e-3

        problem = ReissnerMindlinModalProblem(
            composite_iga_spline=tessellated_iga_spline.composite_spline,
            youngs_modulus=E * ones,
            poisson_ratio=nu * ones,
            density=rho * ones,
            thickness=h * ones,
            shear_correction_factor=5.0 / 6.0,
            drilling_stiffness_factor=1.0,
        )

        solver = ModalAnalysisSolver(modal_problem=problem)
        cls.eigenvalues, cls.eigenvectors = solver.solve(num_modes=20)

        cls.frequencies = np.sqrt(np.abs(cls.eigenvalues)) / (2.0 * np.pi)

    def test_six_rigid_body_modes(self) -> None:
        """The first 6 eigenvalues should be near-zero (rigid body modes)."""
        # With spectral shift, rigid modes appear as small negative values.
        # The gap between mode 5 and mode 6 should be orders of magnitude.
        gap = abs(self.eigenvalues[6]) / max(abs(self.eigenvalues[5]), 1e-30)
        self.assertGreater(
            gap,
            100.0,
            msg=(
                f"Expected large gap between mode 5 and 6. "
                f"eigenvalue[5]={self.eigenvalues[5]:.4e}, "
                f"eigenvalue[6]={self.eigenvalues[6]:.4e}, "
                f"gap ratio={gap:.1f}"
            ),
        )

    def test_first_elastic_mode_frequency(self) -> None:
        """First elastic mode (l=2) at ~577 Hz for R=1.0 unit sphere."""
        elastic_freqs = self.frequencies[6:11]
        f_ref = 577.0

        for i, f in enumerate(elastic_freqs):
            self.assertAlmostEqual(
                f,
                f_ref,
                delta=0.02 * f_ref,
                msg=(
                    f"Mode {6 + i} frequency {f:.1f} Hz deviates from "
                    f"reference {f_ref:.1f} Hz by more than 2%. "
                    f"Elastic frequencies: {elastic_freqs}"
                ),
            )

    def test_first_elastic_mode_degeneracy(self) -> None:
        """The l=2 mode should be 5-fold degenerate (spread < 1 Hz)."""
        elastic_freqs = self.frequencies[6:11]
        spread = elastic_freqs.max() - elastic_freqs.min()
        self.assertLess(
            spread,
            1.0,
            msg=(
                f"First elastic mode cluster spread {spread:.2f} Hz exceeds "
                f"1 Hz. Frequencies: {elastic_freqs}"
            ),
        )

    def test_eigenvalues_sorted(self) -> None:
        self.assertTrue(
            np.all(np.diff(self.eigenvalues) >= -1.0e-6),
            msg="Eigenvalues must be sorted in ascending order",
        )
