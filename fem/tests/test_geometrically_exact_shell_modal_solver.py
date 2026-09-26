import sys
import unittest
from unittest import mock

import numpy as np
from _fem_fixtures import (
    ANALYTIC_CLUSTERS,
    ANCHOR_MESH_SIZE,
    IGA_FIRST_ELASTIC_HZ,
    NUM_RIGID_BODY_MODES,
    RADIUS,
    RIGID_BODY_LABELS,
    STEEL_SHELL,
    cluster_slices,
    steel_sphere_solver,
)

from fem.petsc_factor_solver import FactorSolver
from fem.shells.curved_sphere_mesh import make_curved_sphere_mesh
from fem.shells.geometrically_exact_shell_modal_solver import (
    GeometricallyExactShellModalSolver,
)
from fem.shells.geometrically_exact_shell_problem import GeometricallyExactShellProblem
from fem.shells.shell_element import ShellElement


class TestGeometricallyExactShellModalSolver(unittest.TestCase):
    """
    The cross-validation anchor: does `solve_modes_slepc` reproduce the steel
    sphere's spectrum, as the IGA solver
    (`tests/test_steel_sphere_modal_analysis.py`) and the analytical thin-shell
    solution give it (576.3 Hz x5, 681.0 Hz x7, 721.8 Hz x9, for n = 2, 3, 4)?
    Two unrelated discretisations (IGA on a six-patch B-spline cubed sphere,
    FEM on an unstructured curved triangle mesh) computing the same physics.
    B-bar and P2/P2 are the production defaults of `shell_modal_analysis_spline`.

    Also covers: `rigid_body_energies`, the Rayleigh-quotient diagnostic that
    isolates the formulation's rigid-body invariance from the eigensolver;
    P3 and the three locking-cure branches still reproducing the reference;
    the two guards `solve_modes_slepc` needs against spurious drilling modes;
    the spectral shift's scaling with geometry; the guard `solve_modes_slepc`
    runs before it spends anything on an unsupported factorization backend;
    and the scipy (`solve_modes`) and SLEPc (`solve_modes_slepc`) paths
    agreeing with each other.

    Everything here is gmsh + DOLFINx only; no Mitsuba, no spline. The
    spline-side pipeline is covered by `test_spline_to_shell_mesh.py` and
    `test_shell_modal_analysis_spline.py`.
    """

    NUM_ANCHOR_MODES = NUM_RIGID_BODY_MODES + sum(d for _, d in ANALYTIC_CLUSTERS)
    """Rigid-body modes plus every mode of the three analytic clusters, the scan
    window the anchor solve requests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.solver = steel_sphere_solver(locking_cure=ShellElement.LockingCure.BBAR)
        cls.result = cls.solver.solve_modes_slepc(num_modes=cls.NUM_ANCHOR_MODES)
        # A second, smaller fixture for the two-solve-path comparison tests, which
        # need both `solve_modes_slepc` and `solve_modes` (the anchor result above
        # only has the SLEPc side) on the same underlying problem.
        cls.num_modes = NUM_RIGID_BODY_MODES + 5
        cls.slepc = cls.solver.solve_modes_slepc(num_modes=cls.num_modes)
        cls.scipy = cls.solver.solve_modes(num_modes=cls.num_modes)

    def test_six_rigid_body_modes_carry_no_energy(self) -> None:
        """
        A closed shell is unsupported, so exactly six rigid-body motions must be
        zero-energy. The geometrically-exact strain measures vanish on them
        identically, so the computed eigenvalues are pure round-off about zero and
        the gap to the first elastic mode is enormous (~1e14 as measured).

        The threshold is 1e8: eight orders below what is observed, but still ten
        orders above what the defect this module exists to cure would produce:
        a flat-facet formulation leaves the three rotational rigid-body modes at
        a finite frequency proportional to curvature, which collapses the gap to
        O(1).
        """
        eigenvalues = np.abs(self.result.eigenvalues)
        last_rigid = eigenvalues[:NUM_RIGID_BODY_MODES].max()
        first_elastic = eigenvalues[NUM_RIGID_BODY_MODES]
        self.assertGreater(
            first_elastic / max(last_rigid, 1.0e-300),
            1.0e8,
            msg=(
                f"Rigid-body modes are not zero-energy: largest of the first "
                f"{NUM_RIGID_BODY_MODES} eigenvalues is {last_rigid:.3e}, first "
                f"elastic is {first_elastic:.3e}."
            ),
        )

    def test_first_elastic_mode_matches_iga_reference(self) -> None:
        """
        The FEM first elastic cluster must land on the IGA's 577 Hz. This is the
        cross-validation proper.

        Tolerance 1 %: the IGA reference is quoted to three significant figures
        and the FEM discretisation error at this mesh size is ~0.06 % (from the
        h-refinement sweep in `_fem_fixtures.ANCHOR_MESH_SIZE`'s docstring), so
        1 % is generous against both while remaining far tighter than any
        formulation defect: flat facets at the same mesh size give 437 Hz, a
        24 % error.
        """
        _, first = next(iter(cluster_slices()))
        mean_frequency = float(self.result.frequencies[first].mean())
        self.assertAlmostEqual(
            mean_frequency,
            IGA_FIRST_ELASTIC_HZ,
            delta=0.01 * IGA_FIRST_ELASTIC_HZ,
            msg=(
                f"First elastic cluster at {mean_frequency:.2f} Hz deviates from "
                f"the IGA reference {IGA_FIRST_ELASTIC_HZ} Hz by more than 1 %. "
                f"Cluster: {self.result.frequencies[first]}"
            ),
        )

    def test_low_spectrum_matches_analytic_thin_shell(self) -> None:
        """
        All three analytic clusters (n = 2, 3, 4), not just the first.

        Tolerance 1.5 %: the measured errors are +0.057 %, +0.223 % and +0.680 %.
        They grow with mode number because a fixed mesh resolves a shorter
        wavelength less well, so 1.5 % covers the n = 4 cluster with a factor
        of two in hand.
        """
        for reference, cluster in cluster_slices():
            with self.subTest(reference_hz=reference):
                mean_frequency = float(self.result.frequencies[cluster].mean())
                self.assertAlmostEqual(
                    mean_frequency,
                    reference,
                    delta=0.015 * reference,
                    msg=(
                        f"Cluster mean {mean_frequency:.2f} Hz deviates from the "
                        f"analytic {reference} Hz by more than 1.5 %."
                    ),
                )

    def test_clusters_are_degenerate(self) -> None:
        """
        Each spheroidal branch of a sphere is (2n+1)-fold degenerate. The
        discrete spread is not zero because an unstructured mesh breaks the
        sphere's exact rotational symmetry; measured spreads are 0.072, 0.354 and
        1.243 Hz for n = 2, 3, 4, i.e. 0.013 %, 0.052 % and 0.172 %.

        The tolerance is 0.5 % of the cluster frequency (2.9, 3.4, 3.6 Hz), i.e.
        about three times the largest measured spread: tight enough that a
        cluster splitting into physically distinct modes (which is what a broken
        formulation does) fails, loose enough not to be a mesh-quality assay.
        """
        for reference, cluster in cluster_slices():
            with self.subTest(reference_hz=reference):
                frequencies = self.result.frequencies[cluster]
                spread = float(frequencies.max() - frequencies.min())
                self.assertLess(
                    spread,
                    0.005 * reference,
                    msg=(
                        f"Cluster around {reference} Hz has spread {spread:.3f} Hz: "
                        f"{frequencies}"
                    ),
                )

    def test_eigenvalues_are_sorted(self) -> None:
        """`solve_modes_slepc` sorts before returning; downstream code (the
        rigid-body report, the XDMF export order) relies on it."""
        self.assertTrue(
            np.all(np.diff(self.result.eigenvalues) >= 0.0),
            msg=f"Eigenvalues not ascending: {self.result.eigenvalues}",
        )

    def test_frequencies_follow_eigenvalues(self) -> None:
        """`f = sqrt(|lambda|) / 2 pi`: the contract every caller assumes."""
        expected = np.sqrt(np.abs(self.result.eigenvalues)) / (2.0 * np.pi)
        np.testing.assert_allclose(self.result.frequencies, expected, rtol=0.0)

    def test_result_shapes(self) -> None:
        self.assertEqual(self.result.ndofs, self.solver._problem.ndofs)
        self.assertEqual(
            self.result.eigenvectors.shape,
            (self.solver._problem.ndofs, self.NUM_ANCHOR_MODES),
        )
        self.assertEqual(self.result.eigenvalues.shape, (self.NUM_ANCHOR_MODES,))

    def test_flat_geometry_gives_zero_rigid_body_energy(self) -> None:
        """
        On an order-1 (flat-facet) mesh the interpolated rigid-body fields are
        represented exactly by the P2 space, so the Rayleigh quotients come out
        as exact zeros: the strongest possible statement of rigid-body
        invariance, and the claim on `GeometricallyExactShellProblem`'s class
        docstring.

        Asserted as a bound rather than as `== 0.0` even so. The exact zero
        depends on `rigid_body_energies` flooring the quotient with
        `max(tot, 0.0)`, so it holds only while the round-off residue lands on
        the non-positive side; a BLAS or basix change could move it to the
        positive side without disturbing the physical claim. 1e-9 Hz against a
        first elastic mode of 576 Hz is that claim, and nothing weaker.
        """
        solver = steel_sphere_solver(
            order=1, locking_cure=ShellElement.LockingCure.BBAR
        )
        energies = solver.rigid_body_energies()
        self.assertEqual(
            tuple(energy.mode.label for energy in energies),
            RIGID_BODY_LABELS,
            msg="rigid_body_energies() must return exactly the six modes of "
                "RigidBodyMode.standard_basis(), in that order.",
        )
        for energy in energies:
            with self.subTest(field=energy.mode.label):
                self.assertLessEqual(
                    energy.frequency,
                    1.0e-9,
                    msg=(
                        f"Rigid-body field {energy.mode.label} has energy: "
                        f"{energy.frequency} Hz."
                    ),
                )

    def test_curved_geometry_rigid_body_energy_is_round_off(self) -> None:
        """
        On a curved (order-2) mesh the quotients are not exact zeros: the
        analytic rotation field is interpolated at the P2 nodes, and the geometric
        normal of the curved element is not exactly the sphere normal in between.
        The residual is nonetheless ~1.5e-5 Hz against a first elastic mode of
        576 Hz, i.e. ten orders of magnitude down in energy.

        The bound is 1e-3 Hz, two orders above what is measured and six orders
        below the first elastic mode.
        """
        solver = steel_sphere_solver(locking_cure=ShellElement.LockingCure.BBAR)
        energies = solver.rigid_body_energies()
        self.assertEqual(
            tuple(energy.mode.label for energy in energies), RIGID_BODY_LABELS
        )
        for energy in energies:
            with self.subTest(field=energy.mode.label):
                self.assertLess(
                    energy.frequency, 1.0e-3, msg=f"field {energy.mode.label}"
                )

    def test_flat_geometry_degrades_accuracy_not_rigid_body_modes(self) -> None:
        """
        Flat facets are an accuracy question, not a correctness one: the O(h^2)
        geometry error puts the first cluster at ~437 Hz with a ~120 Hz spread at
        this mesh size, while the rigid-body modes stay exactly zero-energy
        (previous test). This pins the claim that curved geometry is about
        accuracy, not rigid-body correctness, and guards against someone
        silently defaulting the pipeline to order 1.
        """
        solver = steel_sphere_solver(
            order=1, locking_cure=ShellElement.LockingCure.BBAR
        )
        result = solver.solve_modes_slepc(num_modes=NUM_RIGID_BODY_MODES + 5)
        elastic = result.frequencies[NUM_RIGID_BODY_MODES:]
        self.assertLess(
            elastic.mean(),
            0.9 * IGA_FIRST_ELASTIC_HZ,
            msg=(
                "Flat-facet geometry is expected to be markedly less accurate "
                f"than curved; got {elastic.mean():.1f} Hz."
            ),
        )

    def test_all_integration_schemes_agree_on_the_sphere(self) -> None:
        """
        Three integration choices are available: B-bar (default), PSRI, and
        full integration, and `shell_modal_analysis_spline` claims B-bar
        leaves the membrane-dominated sphere spectrum unchanged. The sphere's
        n = 2 branch is 99.9 % membrane energy, so it is exactly the case where
        no locking cure should matter, and all three must agree.

        Measured first-cluster means: B-bar 576.63, PSRI 576.51, full 577.30 Hz,
        a spread of 0.78 Hz. The tolerance is 0.5 % (2.9 Hz), about four times
        that spread, and far below the tens of percent by which a locking cure
        moves a bending-dominated problem.
        """
        elements = {
            "bbar": {"locking_cure": ShellElement.LockingCure.BBAR},
            "psri": {"locking_cure": ShellElement.LockingCure.PSRI},
            "full": {"locking_cure": ShellElement.LockingCure.FULL},
        }
        means = {}
        for label, element_kwargs in elements.items():
            solver = steel_sphere_solver(**element_kwargs)
            result = solver.solve_modes_slepc(num_modes=NUM_RIGID_BODY_MODES + 5)
            means[label] = float(
                result.frequencies[NUM_RIGID_BODY_MODES:].mean()
            )
        spread = max(means.values()) - min(means.values())
        self.assertLess(
            spread,
            0.005 * IGA_FIRST_ELASTIC_HZ,
            msg=f"Integration schemes disagree on the sphere: {means}",
        )
        for label, mean_frequency in means.items():
            with self.subTest(scheme=label):
                self.assertAlmostEqual(
                    mean_frequency,
                    IGA_FIRST_ELASTIC_HZ,
                    delta=0.01 * IGA_FIRST_ELASTIC_HZ,
                )

    def test_p3_reproduces_the_iga_reference(self) -> None:
        """
        P3 is what the recorded production runs used (`field_degree: 3` in every
        archived modal log), so it must be covered even though it is the
        expensive configuration. It still costs a fraction of a second at h = 0.6.

        Same 1 % tolerance as the P2 anchor, and for the same reason; the
        measured error is -0.10 %, so there is a factor of ten in hand. The point
        is that raising the order does not disturb the spectrum: B-bar projects
        onto DG_{p-1}, so the reduced space changes with the degree and a wrong
        projection space would show up here and not at P2.

        h = 0.6 rather than the h = 0.8 used elsewhere in this file: at h = 0.8
        the sphere has only ~50 cells and the order-2 geometry error dominates
        (-0.47 %), which would leave the bound barely a factor of two and make
        this a mesh-quality assay rather than a check on the cubic element.
        """
        solver = steel_sphere_solver(
            0.6, 2, degree=3, locking_cure=ShellElement.LockingCure.BBAR
        )
        result = solver.solve_modes_slepc(num_modes=NUM_RIGID_BODY_MODES + 5)
        elastic = result.frequencies[NUM_RIGID_BODY_MODES:]
        self.assertAlmostEqual(
            float(elastic.mean()),
            IGA_FIRST_ELASTIC_HZ,
            delta=0.01 * IGA_FIRST_ELASTIC_HZ,
            msg=f"P3 first cluster: {elastic}",
        )
        rigid = np.abs(result.eigenvalues[:NUM_RIGID_BODY_MODES]).max()
        self.assertGreater(
            abs(result.eigenvalues[NUM_RIGID_BODY_MODES]) / max(rigid, 1.0e-300),
            1.0e8,
            msg="P3 loses the rigid-body nullspace.",
        )

    def test_spurious_modes_appear_only_when_both_guards_are_off(self) -> None:
        """
        The drilling rotation `theta.n` is not a shell degree of freedom:
        nothing in the membrane/bending/shear energy constrains it, so it needs
        either a stabilisation stiffness (`drilling=True`) or zero inertia
        (`project_rotary_inertia=True`) or the discrete problem carries spurious
        zero-stiffness-but-finite-mass modes low in the spectrum.

        Measured first non-rigid frequency on the anchor mesh: 576.6 Hz for the
        three configurations with at least one guard active, and 69.6 Hz (a
        factor of 8.3 below the physical mode, and clearly not a shell mode)
        when both are off.

        Asserted as "within 1 % of the reference" versus "below half of it",
        which is unambiguous either way at this separation. The second bound is
        deliberately loose: with both guards off nothing constrains the drilling
        rotation, so the spurious frequency is strongly mesh-dependent (380 Hz at
        h = 0.8, 24 Hz at h = 0.3) and a bound near the measured 69.6 Hz would
        pin the anchor mesh rather than the property.
        """
        for drilling, project in ((True, True), (True, False), (False, True)):
            with self.subTest(drilling=drilling, project_rotary_inertia=project):
                solver = steel_sphere_solver(
                    locking_cure=ShellElement.LockingCure.BBAR,
                    drilling=drilling,
                    project_rotary_inertia=project,
                )
                result = solver.solve_modes_slepc(
                    num_modes=NUM_RIGID_BODY_MODES + 1
                )
                self.assertAlmostEqual(
                    float(result.frequencies[NUM_RIGID_BODY_MODES]),
                    IGA_FIRST_ELASTIC_HZ,
                    delta=0.01 * IGA_FIRST_ELASTIC_HZ,
                )

        unguarded = steel_sphere_solver(
            locking_cure=ShellElement.LockingCure.BBAR,
            drilling=False,
            project_rotary_inertia=False,
        )
        result = unguarded.solve_modes_slepc(num_modes=NUM_RIGID_BODY_MODES + 1)
        self.assertLess(
            float(result.frequencies[NUM_RIGID_BODY_MODES]),
            0.5 * IGA_FIRST_ELASTIC_HZ,
            msg=(
                "With both the drilling stiffness and the rotary-inertia "
                "projection disabled, spurious drilling modes are expected below "
                "the first elastic mode."
            ),
        )

    def test_spectrum_is_insensitive_to_the_drilling_stiffness(self) -> None:
        """
        The drilling term is a stabilisation, not physics, so the physical
        spectrum must not depend on how hard it is applied. Measured across four
        orders of magnitude in `drilling_stiffness_factor` (0.01 to 100), the
        first elastic cluster moves by 8.9e-5 relative (0.05 Hz out of 576.6).

        The bound is 1e-3 relative, an order above the measured drift over a
        range far wider than anyone would use, and two orders below the 0.5 %
        agreement the physical assertions elsewhere demand. A term that had
        become load-bearing (a units slip in `E h^3 / h_cell^2`, say) would show
        up here as a first-order dependence.
        """
        means = []
        for factor in (0.01, 100.0):
            solver = steel_sphere_solver(
                locking_cure=ShellElement.LockingCure.BBAR,
                drilling_stiffness_factor=factor,
            )
            result = solver.solve_modes_slepc(num_modes=NUM_RIGID_BODY_MODES + 5)
            means.append(float(result.frequencies[NUM_RIGID_BODY_MODES:].mean()))
        self.assertAlmostEqual(
            means[1] / means[0],
            1.0,
            delta=1.0e-3,
            msg=f"First cluster moved with the drilling stiffness: {means}",
        )

    def test_spectral_shift_scales_with_the_membrane_eigenvalue(self) -> None:
        """
        The shift is `1e-3 * E / (rho L^2)`, deliberately geometry-scale-aware
        rather than mesh-dependent. Two consequences are pinned: it does not move
        when the mesh is refined at fixed geometry, and it falls as `1/L^2` when
        the geometry is scaled, which is how it stays a fixed fraction of the
        elastic spectrum (frequencies also scale as 1/L).

        It must also stay well below the first elastic eigenvalue, or the
        shift-invert target would sit inside the physical spectrum: measured
        ratio is 6.0e3 / 1.3e7, i.e. 4.6e-4.
        """
        coarse = steel_sphere_solver(0.8, 2, locking_cure=ShellElement.LockingCure.BBAR)
        fine = steel_sphere_solver(locking_cure=ShellElement.LockingCure.BBAR)
        # `_spectral_shift` reads `length` from a `max`/`min` reduction over mesh
        # node coordinates (an exact selection, no rounding) and then applies one
        # squaring, one product and one division. The sphere's axis extremes are
        # CAD model vertices, which gmsh places at their exact model coordinate
        # regardless of the surrounding element size, so refining the mesh leaves
        # `length` (and hence the shift) bit-identical (the two calls below
        # differ by 0.0 in the current pipeline). The delta is a float64 margin
        # for the three arithmetic steps that follow, each correctly-rounded to
        # half a ULP: 16 ULP of the shift's own scale, `1e-3 * E / (rho * (2 *
        # RADIUS)**2)`, covers a change of rounding mode or evaluation order
        # while staying tighter than the previous, undocumented 1e-9 bound.
        shift_scale = 1.0e-3 * STEEL_SHELL["youngs_modulus"] / (
            STEEL_SHELL["density"] * (2.0 * RADIUS) ** 2
        )
        self.assertAlmostEqual(
            coarse._spectral_shift(),
            fine._spectral_shift(),
            delta=16 * sys.float_info.epsilon * shift_scale,
        )

        big_problem = GeometricallyExactShellProblem(
            make_curved_sphere_mesh(2.0 * RADIUS, 2.0 * ANCHOR_MESH_SIZE, order=2),
            element=ShellElement(locking_cure=ShellElement.LockingCure.BBAR),
            **STEEL_SHELL,
        )
        big = GeometricallyExactShellModalSolver(big_problem)
        self.assertAlmostEqual(
            big._spectral_shift() / fine._spectral_shift(),
            0.25,
            delta=1.0e-6,
            msg="Spectral shift must scale as 1/L^2 with the geometry.",
        )

        result = fine.solve_modes_slepc(num_modes=NUM_RIGID_BODY_MODES + 1)
        self.assertLess(
            fine._spectral_shift(),
            1.0e-2 * result.eigenvalues[NUM_RIGID_BODY_MODES],
            msg="Spectral shift is not safely below the first elastic eigenvalue.",
        )

    def test_the_factor_solver_reaches_the_guard_as_its_matsolvertype(self) -> None:
        """
        `PC.setFactorSolverType` is a pure setter: it accepts any value and the
        failure only appears inside the factorization, as an error that does not
        name the missing backend. So `FactorSolver.require` must fire before the
        assembly, and it must be handed the member's PETSc `MatSolverType` string
        rather than the member itself: a `FactorSolver` would satisfy neither
        the availability check nor PETSc. The check's own behaviour is pinned in
        `test_crank_nicolson_fem_scheme`.
        """
        crossed: list[str] = []

        def stop_at_the_guard(solver_type: str, purpose: str) -> None:
            crossed.append(solver_type)
            raise RuntimeError(f"stopped before assembling for {purpose}")

        with mock.patch(
            "fem.shells.geometrically_exact_shell_modal_solver.FactorSolver.require",
            stop_at_the_guard,
        ):
            with self.assertRaises(RuntimeError) as raised:
                self.solver.solve_modes_slepc(
                    num_modes=2, factor_solver=FactorSolver.SUPERLU_DIST
                )

        self.assertEqual(crossed, [FactorSolver.SUPERLU_DIST.value])
        self.assertIn("the shift-invert modal solve", str(raised.exception))

    def test_builtin_factor_solver_is_accepted(self) -> None:
        """
        The `FactorSolver.PETSC` special case is load-bearing, not
        decorative: `PETSc.Sys.hasExternalPackage("petsc")` returns False,
        since PETSc's own factorization is not an "external package", so without the
        special case the one backend that is always available would be the one
        rejected. Asserted here through a real solve, so it covers the whole
        shift-invert path and not just the guard, including the second crossing
        of the enum, into `PC.setFactorSolverType`.
        """
        result = self.solver.solve_modes_slepc(
            num_modes=8, factor_solver=FactorSolver.PETSC
        )
        self.assertAlmostEqual(
            float(result.frequencies[NUM_RIGID_BODY_MODES:].mean()),
            IGA_FIRST_ELASTIC_HZ,
            delta=0.01 * IGA_FIRST_ELASTIC_HZ,
        )

    def test_missing_slepc_raises_an_actionable_import_error(self) -> None:
        """The scalable path is optional in principle; if slepc4py is absent the
        message must name it and point at the scipy alternative, not surface as a
        bare ModuleNotFoundError from the middle of a solve."""
        with mock.patch.dict(sys.modules, {"slepc4py": None}):
            with self.assertRaises(ImportError) as raised:
                self.solver.solve_modes_slepc(num_modes=2)
        self.assertIn("slepc4py", str(raised.exception))

    def test_elastic_spectra_agree(self) -> None:
        """
        Two solve paths exist: SLEPc shift-invert (production, memory-scalable)
        and a scipy/ARPACK path that additionally reports the per-mode energy
        split. The entry point picks between them with a flag, so they must give
        the same physics.

        Compared as a cluster mean rather than mode by mode: within a five-fold
        degenerate cluster the individual eigenvalues are not separated (spread
        0.072 Hz), so which member each solver returns first is arbitrary, while
        the cluster mean is invariant to that choice: it agrees to 1e-11 here.
        The 1e-4 bound is set by ARPACK's own run-to-run noise on this problem
        (~1e-6 relative; it starts from an unseeded random vector) with two
        orders in hand, not by the agreement measured today.
        """
        slepc_mean = float(self.slepc.frequencies[NUM_RIGID_BODY_MODES:].mean())
        scipy_mean = float(self.scipy.frequencies[NUM_RIGID_BODY_MODES:].mean())
        self.assertAlmostEqual(
            scipy_mean / slepc_mean, 1.0, delta=1.0e-4, msg=
            f"scipy {scipy_mean:.6f} Hz vs SLEPc {slepc_mean:.6f} Hz."
        )

    def test_both_paths_find_the_same_rigid_body_count(self) -> None:
        for label, result in (("slepc", self.slepc), ("scipy", self.scipy)):
            with self.subTest(path=label):
                rigid = np.abs(result.eigenvalues[:NUM_RIGID_BODY_MODES]).max()
                elastic = abs(result.eigenvalues[NUM_RIGID_BODY_MODES])
                self.assertGreater(elastic / max(rigid, 1.0e-300), 1.0e8)

    def test_first_elastic_mode_is_membrane_dominated(self) -> None:
        """
        Physics, not numerics: the n = 2 spheroidal mode of a thin sphere stores
        its energy in stretching, not bending, which is why its frequency
        scales as 1/R and why the IGA reference could be rescaled from R = 0.1135
        to R = 1. Measured membrane fraction is 0.99981; the bound is 0.99.

        This is also what makes the sphere a poor locking test and a good
        formulation test, and it is only computable on the scipy path.
        """
        energies = self.scipy.mode_energies[NUM_RIGID_BODY_MODES]
        membrane_fraction = energies[0] / energies.sum()
        self.assertGreater(
            membrane_fraction,
            0.99,
            msg=f"membrane/bending/shear/drilling energies: {energies}",
        )

    def test_slepc_path_reports_no_energy_split(self) -> None:
        """Documented behaviour: the scalable path skips the diagnostic blocks,
        so its `mode_energies` are zeros rather than a stale or wrong split."""
        np.testing.assert_array_equal(
            self.slepc.mode_energies, np.zeros((self.num_modes, 4))
        )
