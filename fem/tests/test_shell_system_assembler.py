import unittest

import numpy as np
from _fem_fixtures import (
    IGA_FIRST_ELASTIC_HZ,
    NUM_RIGID_BODY_MODES,
    relative_asymmetry,
    steel_sphere_solver,
)
from dolfinx import fem

from fem.shells.shell_element import ShellElement
from fem.shells.shell_system_assembler import ShellSystemAssembler


class TestShellSystemAssembler(unittest.TestCase):
    """
    Structural properties of the assembled `ShellSystemAssembler.ScipySystem`
    that the two eigensolvers rely on: `scipy.eigsh` with `sigma=0` and SLEPc's
    `GHEP` problem type both assume a symmetric pair with an SPD mass matrix and
    a six-dimensional rigid-body nullspace in `K`. The SLEPc path's combined
    stiffness (the UFL form plus the B-bar blocks) is checked against the scipy
    path's split blocks summed the other way, since nothing else forces the two
    to agree.

    Also the drilling stabilization block: objective on the six analytic
    rigid-body fields (it may not do work on a rigid motion), load-bearing for
    the nullspace (removing it leaves dozens of spurious null directions where
    `project_rotary_inertia` alone leaves only their mass), and a plain linear
    scale under `drilling_stiffness_factor`.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.solver = steel_sphere_solver(
            0.8, 2, locking_cure=ShellElement.LockingCure.BBAR
        )
        cls.system = cls.solver._problem.assemble_scipy_system()

    def test_every_stiffness_block_and_the_mass_are_symmetric(self) -> None:
        """
        Both eigensolvers assume a symmetric pair, so every block must be
        symmetric to round-off. 1e-12 is far above the measured ~1e-16 and far
        below any real defect.

        `K_mem` is in this list on purpose. Under B-bar it is not assembled from
        a bilinear form but as the triple product `Pi^T G Pi`, whose symmetry
        rests on `G` being self-adjoint, which holds only because the membrane
        assumed-strain space is the symmetric tangential tensors. Carrying it as
        unconstrained 3x3 tensors, a strictly larger space, leaves this block
        asymmetric at ~1e-3 (see
        `TestGeometricallyExactShellProblem.test_membrane_material_operator_is_self_adjoint_on_the_reduced_space`).
        """
        for name in ("mass", "stiffness", "membrane", "bending", "shear", "drilling"):
            with self.subTest(block=name):
                self.assertLess(
                    relative_asymmetry(getattr(self.system, name)), 1.0e-12
                )

    def test_both_solve_paths_use_the_same_stiffness(self) -> None:
        """
        The two paths build the stiffness differently: the scipy path sums four
        separately assembled blocks, while the SLEPc path assembles one combined
        UFL form (`_forms.stiffness`) and, under B-bar, adds the two element-local
        blocks to it as matrices. Nothing forces those to agree: a term could be
        double-counted or dropped on one side only, and the symptom would be two
        slightly different spectra rather than an error.

        Reassembled here exactly as `solve_modes_slepc` does. Tolerance 1e-12
        relative to the matrix norm: the two orderings of the same sums differ
        only by floating-point association.
        """
        problem = self.solver._problem
        combined = ShellSystemAssembler._to_scipy(problem._forms.stiffness)
        if problem.el.locking_cure is ShellElement.LockingCure.BBAR:
            membrane, shear = problem.bbar_blocks()
            combined = combined + membrane + shear
        self.assertLess(
            abs(combined - self.system.stiffness).max()
            / abs(self.system.stiffness).max(),
            1.0e-12,
        )

    def test_mass_matrix_is_positive_definite(self) -> None:
        """Required by both eigensolvers. Checked via the smallest eigenvalue of
        a dense symmetric eigendecomposition, affordable at this mesh size."""
        smallest = float(np.linalg.eigvalsh(self.system.mass.toarray()).min())
        self.assertGreater(smallest, 0.0, msg=f"M has eigenvalue {smallest:.3e}.")

    def test_stiffness_nullspace_is_six_dimensional(self) -> None:
        """
        The formulation's central invariant, stated without the eigensolver: K
        must have exactly six near-null directions (the rigid-body motions) and
        the seventh must be far away.

        Taken on the symmetric part, because `eigvalsh` reads only one triangle
        and would otherwise silently analyse a matrix that is not K; both
        eigensolvers likewise see only the symmetric part.

        The threshold separating "null" from "elastic" is placed at 1e-8 of the
        largest stiffness eigenvalue. Measured, the six null eigenvalues are
        ~1e-7 and the seventh is ~1e2 against a largest of 6.7e9, so the gap
        spans nine orders and the classification does not depend on where in it
        the cut is made.
        """
        stiffness = self.system.stiffness.toarray()
        eigenvalues = np.linalg.eigvalsh(0.5 * (stiffness + stiffness.T))
        cutoff = 1.0e-8 * eigenvalues.max()
        self.assertEqual(
            int(np.sum(np.abs(eigenvalues) < cutoff)),
            NUM_RIGID_BODY_MODES,
            msg=f"Smallest stiffness eigenvalues: {eigenvalues[:8]}",
        )

    def test_stiffness_annihilates_the_rigid_body_fields(self) -> None:
        """
        The sharper form of the same statement, and the one that does not depend
        on symmetry: the strain measures vanish pointwise on a rigid motion, so
        `K phi` is zero for each of the six analytic rigid-body fields, not
        merely `phi^T K phi`.

        Bound 1e-12 on `||K phi||_inf / (||K||_max ||phi||_inf)`; measured ~1e-15,
        i.e. accumulated round-off over the assembly, with three orders of slack.
        """
        problem = self.solver._problem
        displacement_space, displacement_map = problem.V.sub(0).collapse()
        rotation_space, rotation_map = problem.V.sub(1).collapse()
        stiffness_scale = abs(self.system.stiffness).max()

        for axis in range(3):
            direction = np.eye(3)[axis]
            for name, displacement_field, rotation in (
                ("translation", lambda x, d=direction: np.tile(
                    d[:, None], (1, x.shape[1])), np.zeros(3)),
                ("rotation", lambda x, d=direction: np.cross(d, x.T).T, direction),
            ):
                with self.subTest(field=f"{name}_{axis}"):
                    phi = np.zeros(problem.ndofs)
                    displacement = fem.Function(displacement_space)
                    displacement.interpolate(displacement_field)
                    rotation_function = fem.Function(rotation_space)
                    rotation_function.interpolate(
                        lambda x, w=rotation: np.tile(w[:, None], (1, x.shape[1]))
                    )
                    phi[displacement_map] = displacement.x.array
                    phi[rotation_map] = rotation_function.x.array
                    residual = np.abs(self.system.stiffness @ phi).max() / (
                        stiffness_scale * np.abs(phi).max()
                    )
                    self.assertLess(residual, 1.0e-12)

    def test_drilling_penalty_is_objective_on_rigid_body_motions(self) -> None:
        """
        A stabilisation may not do work on a rigid motion, or it would stiffen
        the six modes it is supposed to leave alone. It does not, identically:
        `w_dr = 1/2 n.curl(u) - theta.n` gives `n.omega - omega.n = 0` for
        `u = c + omega x x, theta = omega`.

        Measured drilling energy on the six analytic fields is ~1e-11 against an
        elastic-scale reference of ~1e9, i.e. 1e-20 relative. The bound is 1e-15
        relative to that reference, five orders above what is measured, and far
        below the 1e-5 energy fraction the penalty legitimately carries in the
        physical modes (it is a penalty, not an exactly-null term, so its
        objectivity holds only on rigid motions).
        """
        solver = steel_sphere_solver(locking_cure=ShellElement.LockingCure.BBAR)
        system = solver._problem.assemble_scipy_system()
        displacement_space, displacement_map = solver._problem.V.sub(0).collapse()
        rotation_space, rotation_map = solver._problem.V.sub(1).collapse()
        # Energy scale of a physical mode with the same mass normalisation.
        reference = (2.0 * np.pi * IGA_FIRST_ELASTIC_HZ) ** 2

        for axis in range(3):
            direction = np.eye(3)[axis]
            for name, field, rotation in (
                ("translation", lambda x, d=direction: np.tile(
                    d[:, None], (1, x.shape[1])), np.zeros(3)),
                ("rotation", lambda x, d=direction: np.cross(d, x.T).T, direction),
            ):
                with self.subTest(field=f"{name}_{axis}"):
                    phi = np.zeros(solver._problem.ndofs)
                    displacement = fem.Function(displacement_space)
                    displacement.interpolate(field)
                    rotation_function = fem.Function(rotation_space)
                    rotation_function.interpolate(
                        lambda x, w=rotation: np.tile(w[:, None], (1, x.shape[1]))
                    )
                    phi[displacement_map] = displacement.x.array
                    phi[rotation_map] = rotation_function.x.array
                    drilling_energy = abs(float(phi @ (system.drilling @ phi)))
                    mass = float(phi @ (system.mass @ phi))
                    self.assertLess(drilling_energy / (mass * reference), 1.0e-15)

    def test_drilling_penalty_is_what_makes_the_stiffness_non_singular(self) -> None:
        """
        The two guards are not redundant, and this is the difference: the
        drilling penalty gives the unconstrained rotation stiffness, while
        `project_rotary_inertia` only removes its mass. Both clean up a modal
        spectrum, but only the penalty leaves `K` with the six-dimensional
        nullspace it should have.

        Measured with the inertia projection on and the penalty off: the nullity
        of `K` is 65 at h = 0.6 and 91 at h = 0.5 (it grows with the number of
        rotation nodes) against 6 with the penalty on. The extra directions are
        the discrete drilling modes.

        This is why the penalty is load-bearing beyond the modal path: a static
        solve has no `+ epsilon*M` to lift them, and even in the modal path the
        conditioning of `K + epsilon*M` degrades from 4.3e7 to 2.1e11 at h = 0.5
        and keeps worsening under refinement (1.2e12 at h = 0.4), so the margin
        that makes the factorisation survive on these coarse meshes is not one to
        rely on at production resolution.
        """
        for drilling, expected_nullity in ((True, NUM_RIGID_BODY_MODES), (False, None)):
            with self.subTest(drilling=drilling):
                solver = steel_sphere_solver(
                    locking_cure=ShellElement.LockingCure.BBAR,
                    drilling=drilling,
                    project_rotary_inertia=True,
                )
                system = solver._problem.assemble_scipy_system()
                stiffness = system.stiffness.toarray()
                eigenvalues = np.linalg.eigvalsh(0.5 * (stiffness + stiffness.T))
                nullity = int(
                    np.sum(np.abs(eigenvalues) < 1.0e-8 * eigenvalues.max())
                )
                if expected_nullity is not None:
                    self.assertEqual(nullity, expected_nullity)
                else:
                    self.assertGreater(
                        nullity,
                        NUM_RIGID_BODY_MODES,
                        msg=(
                            "Without the drilling penalty the stiffness is "
                            "expected to be rank-deficient beyond the six "
                            "rigid-body modes."
                        ),
                    )

    def test_drilling_stiffness_factor_scales_the_drilling_block(self) -> None:
        """`drilling_stiffness_factor` is a plain linear scale on the drilling
        stiffness (the parameter the entry point exposes), and nothing else."""
        base = steel_sphere_solver(
            0.8, 2, locking_cure=ShellElement.LockingCure.BBAR
        )
        scaled = steel_sphere_solver(
            0.8,
            2,
            locking_cure=ShellElement.LockingCure.BBAR,
            drilling_stiffness_factor=3.0,
        )
        base_system = base._problem.assemble_scipy_system()
        scaled_system = scaled._problem.assemble_scipy_system()
        # The factor multiplies the integrand rather than the assembled matrix, so
        # the quadrature sums are re-evaluated and agreement is to round-off of the
        # block norm, not of each entry: entries span fifteen orders of magnitude
        # and the tiny ones carry no relative accuracy. Hence an absolute tolerance
        # scaled by the block norm (measured residual ~1e-14 on a norm of 2e3).
        drill_scale = abs(base_system.drilling).max()
        np.testing.assert_allclose(
            scaled_system.drilling.toarray(),
            3.0 * base_system.drilling.toarray(),
            rtol=0.0,
            atol=1.0e-12 * 3.0 * drill_scale,
        )
        np.testing.assert_allclose(
            scaled_system.bending.toarray(),
            base_system.bending.toarray(),
            rtol=0.0,
            atol=1.0e-12 * abs(base_system.bending).max(),
        )
