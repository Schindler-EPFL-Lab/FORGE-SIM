"""
Tests for the geometrically-exact Reissner-Mindlin (Naghdi) shell modal solver.

The anchor is the canonical unit steel sphere -- R = 1 m, E = 193.05 GPa,
nu = 0.28, rho = 8025.9 kg/m^3, h = 1.5875 mm -- which the IGA side pins
independently in `tests/test_steel_sphere_modal_analysis.py` at the repository
root (first elastic mode ~577 Hz, five-fold degenerate). Reproducing that
spectrum here is a genuine cross-validation: two unrelated discretisations (IGA
on a six-patch B-spline cubed sphere, FEM on an unstructured curved triangle
mesh) computing the same physics. It is the test most likely to catch a real
defect in the formulation, because the failure mode this module exists to fix --
loss of rigid-body invariance on curved geometry -- shows up as a wrong low
spectrum, not as a crash.

The sphere is also an analytically known problem: for a thin spherical shell the
membrane-dominated spheroidal branch gives 576.3 Hz (n=2, x5), 681.0 Hz (n=3,
x7) and 721.8 Hz (n=4, x9) at this material and radius, so the whole low
spectrum -- not just its first cluster -- is pinned against a reference the code
had no hand in producing.

Everything here is gmsh + DOLFINx only; no Mitsuba, no spline. The spline-side
pipeline is covered by `test_spline_to_shell_mesh.py` and
`test_shell_modal_analysis_spline.py`.

Run from the repository root::

    conda run -n dolfinx-rebel-shape python -m unittest discover dolfinx/tests
"""

import sys
import unittest
import warnings
from unittest import mock

import numpy as np
import scipy.sparse as sp

# Imported for its module-level side effect: `_fem_fixtures` puts `dolfinx/` and
# the repository root on `sys.path`.
import _fem_fixtures  # noqa: F401

import ufl
from dolfinx import fem

from geometrically_exact_shell_solver import (
    GeometricallyExactShellSolver,
    ShellElement,
    make_curved_sphere_mesh,
)

# Canonical unit steel sphere, identical to the IGA reference test.
STEEL_SHELL = {
    "youngs_modulus": 193.05e9,
    "poisson_ratio": 0.28,
    "density": 8025.9,
    "thickness": 1.5875e-3,
}
RADIUS = 1.0

# Analytical thin-spherical-shell frequencies (Hz) of the first three
# membrane-dominated spheroidal clusters at this material and radius, together
# with their degeneracies (2n+1). Quoted in the class docstring of
# `GeometricallyExactShellSolver`; the IGA solver reproduces them too.
ANALYTIC_CLUSTERS = ((576.3, 5), (681.0, 7), (721.8, 9))
IGA_FIRST_ELASTIC_HZ = 577.0
NUM_RIGID_BODY_MODES = 6

# Element size for the anchor mesh. An h-refinement sweep (h = 0.8 -> 0.3) puts
# the first elastic cluster at 579.05, 576.89, 576.63, 576.57, 576.38 Hz, so by
# h = 0.5 the discretisation error is already down to ~0.06 % while the mesh is
# still only ~150 cells and the whole solve takes a fraction of a second.
ANCHOR_MESH_SIZE = 0.5

# Cache meshes and solvers across test classes: building a solver JIT-compiles
# its variational forms, which costs far more than the eigensolve at these
# sizes, and several classes want the same configuration.
_MESH_CACHE: dict[tuple[float, int], object] = {}
_SOLVER_CACHE: dict[tuple, GeometricallyExactShellSolver] = {}


def sphere_mesh(mesh_size: float = ANCHOR_MESH_SIZE, order: int = 2):
    """A cached curved sphere mesh of the given element size and geometry order."""
    key = (mesh_size, order)
    if key not in _MESH_CACHE:
        _MESH_CACHE[key] = make_curved_sphere_mesh(RADIUS, mesh_size, order=order)
    return _MESH_CACHE[key]


def steel_sphere_solver(
    mesh_size: float = ANCHOR_MESH_SIZE, order: int = 2, **element_kwargs
) -> GeometricallyExactShellSolver:
    """A cached steel-sphere solver; `element_kwargs` go to `ShellElement`."""
    key = (mesh_size, order) + tuple(sorted(element_kwargs.items()))
    if key not in _SOLVER_CACHE:
        _SOLVER_CACHE[key] = GeometricallyExactShellSolver(
            sphere_mesh(mesh_size, order),
            element=ShellElement(**element_kwargs),
            **STEEL_SHELL,
        )
    return _SOLVER_CACHE[key]


def cluster_slices():
    """Index slices of the analytic clusters in the sorted spectrum."""
    start = NUM_RIGID_BODY_MODES
    for frequency, degeneracy in ANALYTIC_CLUSTERS:
        yield frequency, slice(start, start + degeneracy)
        start += degeneracy


NUM_ANCHOR_MODES = NUM_RIGID_BODY_MODES + sum(d for _, d in ANALYTIC_CLUSTERS)


class TestSteelSphereSpectrum(unittest.TestCase):
    """
    The cross-validation anchor: does the FEM shell reproduce the steel sphere's
    spectrum, as the IGA solver and the analytical thin-shell solution give it?

    B-bar and P2/P2 are the production defaults of `shell_modal_analysis_spline`.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.solver = steel_sphere_solver(bbar=True, psri=False)
        cls.result = cls.solver.solve_modes_slepc(num_modes=NUM_ANCHOR_MODES)

    def test_six_rigid_body_modes_carry_no_energy(self) -> None:
        """
        A closed shell is unsupported, so exactly six rigid-body motions must be
        zero-energy. The geometrically-exact strain measures vanish on them
        identically, so the computed eigenvalues are pure round-off about zero and
        the gap to the first elastic mode is enormous (~1e14 as measured).

        The threshold is 1e8: eight orders below what is observed, but still ten
        orders above what the defect this module exists to cure would produce --
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
        h-refinement sweep in the module header), so 1 % is generous against both
        while remaining far tighter than any formulation defect -- flat facets at
        the same mesh size give 437 Hz, a 24 % error.
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

        Tolerance 1.5 %: the measured errors are +0.057 %, +0.223 % and +0.680 %
        -- they grow with mode number because a fixed mesh resolves a shorter
        wavelength less well -- so 1.5 % covers the n = 4 cluster with a factor
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
        about three times the largest measured spread -- tight enough that a
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
        """`f = sqrt(|lambda|) / 2 pi` -- the contract every caller assumes."""
        expected = np.sqrt(np.abs(self.result.eigenvalues)) / (2.0 * np.pi)
        np.testing.assert_allclose(self.result.frequencies, expected, rtol=0.0)

    def test_result_shapes(self) -> None:
        self.assertEqual(self.result.ndofs, self.solver.ndofs)
        self.assertEqual(
            self.result.eigenvectors.shape, (self.solver.ndofs, NUM_ANCHOR_MODES)
        )
        self.assertEqual(self.result.eigenvalues.shape, (NUM_ANCHOR_MODES,))


class TestRigidBodyInvariance(unittest.TestCase):
    """
    The property this formulation was written for: the strain measures vanish on
    `u = c + omega x x, theta = omega` at ANY geometry order, because they are
    built coordinate-free from the surface normal rather than from a flat-facet
    frame. `rigid_body_energies` evaluates the Rayleigh quotient of each of the
    six analytic rigid-body fields directly, which isolates the formulation from
    the eigensolver.
    """

    def test_flat_geometry_gives_zero_rigid_body_energy(self) -> None:
        """
        On an order-1 (flat-facet) mesh the interpolated rigid-body fields are
        represented exactly by the P2 space, so the Rayleigh quotients come out
        as exact zeros -- the strongest possible statement of rigid-body
        invariance, and the claim in the class docstring.

        Asserted as a bound rather than as `== 0.0` even so. The exact zero
        depends on `rigid_body_energies` flooring the quotient with
        `max(tot, 0.0)`, so it holds only while the round-off residue lands on
        the non-positive side; a BLAS or basix change could move it to the
        positive side without disturbing the physical claim. 1e-9 Hz against a
        first elastic mode of 576 Hz is that claim, and nothing weaker.
        """
        solver = steel_sphere_solver(order=1, bbar=True, psri=False)
        for name, frequency in solver.rigid_body_energies().items():
            with self.subTest(field=name):
                self.assertLessEqual(
                    frequency,
                    1.0e-9,
                    msg=f"Rigid-body field {name} has energy: {frequency} Hz.",
                )

    def test_curved_geometry_rigid_body_energy_is_round_off(self) -> None:
        """
        On a curved (order-2) mesh the quotients are no longer exact zeros: the
        analytic rotation field is interpolated at the P2 nodes, and the geometric
        normal of the curved element is not exactly the sphere normal in between.
        The residual is nonetheless ~1.5e-5 Hz against a first elastic mode of
        576 Hz, i.e. ten orders of magnitude down in energy.

        The bound is 1e-3 Hz, two orders above what is measured and six orders
        below the first elastic mode.
        """
        solver = steel_sphere_solver(bbar=True, psri=False)
        for name, frequency in solver.rigid_body_energies().items():
            with self.subTest(field=name):
                self.assertLess(frequency, 1.0e-3, msg=f"field {name}")

    def test_flat_geometry_degrades_accuracy_not_rigid_body_modes(self) -> None:
        """
        Flat facets are an accuracy question, not a correctness one: the O(h^2)
        geometry error puts the first cluster at ~437 Hz with a ~120 Hz spread at
        this mesh size, while the rigid-body modes stay exactly zero-energy
        (previous test). This pins the class docstring's claim that curved
        geometry is "about ACCURACY, not RB correctness" -- and guards against
        someone silently defaulting the pipeline to order 1.
        """
        solver = steel_sphere_solver(order=1, bbar=True, psri=False)
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


class TestLockingCures(unittest.TestCase):
    """
    Three integration choices are available -- B-bar (default), PSRI, and full
    integration -- and `shell_modal_analysis_spline` claims B-bar "leaves the
    membrane-dominated sphere spectrum unchanged". The sphere's n = 2 branch is
    99.9 % membrane energy, so it is exactly the case where no locking cure
    should matter, and all three must agree.
    """

    ELEMENTS = {
        "bbar": {"bbar": True, "psri": False},
        "psri": {"bbar": False, "psri": True},
        "full": {"bbar": False, "psri": False},
    }

    def test_all_integration_schemes_agree_on_the_sphere(self) -> None:
        """
        Measured first-cluster means: B-bar 576.63, PSRI 576.51, full 577.30 Hz,
        a spread of 0.78 Hz. The tolerance is 0.5 % (2.9 Hz), about four times
        that spread, and far below the tens of percent by which a locking cure
        moves a bending-dominated problem.
        """
        means = {}
        for label, element_kwargs in self.ELEMENTS.items():
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

    def test_element_label_reports_the_active_scheme(self) -> None:
        """`label()` goes into the run log; it must not claim a cure that is off.
        B-bar takes precedence over PSRI, so `bbar=True, psri=True` reads Bbar."""
        self.assertEqual(
            ShellElement(degree=3, bbar=True, psri=True).label(),
            "P3/P3+Bbar+geometric",
        )
        self.assertEqual(
            ShellElement(degree=2, bbar=False, psri=True).label(),
            "P2/P2+PSRI+geometric",
        )
        self.assertEqual(
            ShellElement(degree=2, bbar=False, psri=False).label(),
            "P2/P2+full+geometric",
        )

    def test_reduced_quadrature_degree_is_order_aware(self) -> None:
        """PSRI's reduced degree defaults to 2*(p-1) -- degree 2 hourglasses at
        P3, per the class docstring -- and an explicit setting wins."""
        for degree in (2, 3):
            with self.subTest(degree=degree):
                solver = GeometricallyExactShellSolver(
                    sphere_mesh(0.8, 2),
                    element=ShellElement(degree=degree, bbar=False, psri=True),
                    **STEEL_SHELL,
                )
                self.assertEqual(solver.reduced_quad, 2 * (degree - 1))
                self.assertEqual(solver.quad_degree, 2 * degree + 2)
        solver = GeometricallyExactShellSolver(
            sphere_mesh(0.8, 2),
            element=ShellElement(degree=2, bbar=False, psri=True, reduced_quad=1),
            **STEEL_SHELL,
        )
        self.assertEqual(solver.reduced_quad, 1)


class TestHigherOrderElements(unittest.TestCase):
    """
    P3 is what the recorded production runs used (`field_degree: 3` in every
    archived modal log), so it must be covered even though it is the expensive
    configuration. It still costs a fraction of a second at h = 0.6.
    """

    def test_p3_reproduces_the_iga_reference(self) -> None:
        """
        Same 1 % tolerance as the P2 anchor, and for the same reason; the
        measured error is -0.10 %, so there is a factor of ten in hand. The point
        is that raising the order does not disturb the spectrum -- B-bar projects
        onto DG_{p-1}, so the reduced space changes with the degree and a wrong
        projection space would show up here and not at P2.

        h = 0.6 rather than the h = 0.8 used elsewhere in this file: at h = 0.8
        the sphere has only ~50 cells and the order-2 GEOMETRY error dominates
        (-0.47 %), which would leave the bound barely a factor of two and make
        this a mesh-quality assay rather than a check on the cubic element.
        """
        solver = steel_sphere_solver(0.6, 2, degree=3, bbar=True, psri=False)
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


class TestDrillingStabilisation(unittest.TestCase):
    """
    The drilling rotation `theta.n` is not a shell degree of freedom: nothing in
    the membrane/bending/shear energy constrains it, so it needs either a
    stabilisation stiffness (`drilling=True`) or zero inertia
    (`project_rotary_inertia=True`) or the discrete problem carries spurious
    zero-stiffness-but-finite-mass modes low in the spectrum.
    """

    def test_spurious_modes_appear_only_when_both_guards_are_off(self) -> None:
        """
        Measured first non-rigid frequency on the anchor mesh: 576.6 Hz for the
        three configurations with at least one guard active, and 69.6 Hz -- a
        factor of 8.3 below the physical mode, and clearly not a shell mode --
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
                    bbar=True,
                    psri=False,
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
            bbar=True, psri=False, drilling=False, project_rotary_inertia=False
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

    def test_drilling_penalty_is_objective_on_rigid_body_motions(self) -> None:
        """
        A stabilisation may not do work on a rigid motion, or it would stiffen
        the six modes it is supposed to leave alone. It does not, identically:
        `w_dr = 1/2 n.curl(u) - theta.n` gives `n.omega - omega.n = 0` for
        `u = c + omega x x, theta = omega`.

        Measured drilling energy on the six analytic fields is ~1e-11 against an
        elastic-scale reference of ~1e9, i.e. 1e-20 relative. The bound is 1e-15
        relative to that reference -- five orders above what is measured, and far
        below the 1e-5 energy fraction the penalty legitimately carries in the
        PHYSICAL modes (it is a penalty, not an exactly-null term, so its
        objectivity holds only on rigid motions).
        """
        solver = steel_sphere_solver(bbar=True, psri=False)
        solver._ensure_scipy_assembled()
        displacement_space, displacement_map = solver.V.sub(0).collapse()
        rotation_space, rotation_map = solver.V.sub(1).collapse()
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
                    phi = np.zeros(solver.ndofs)
                    displacement = fem.Function(displacement_space)
                    displacement.interpolate(field)
                    rotation_function = fem.Function(rotation_space)
                    rotation_function.interpolate(
                        lambda x, w=rotation: np.tile(w[:, None], (1, x.shape[1]))
                    )
                    phi[displacement_map] = displacement.x.array
                    phi[rotation_map] = rotation_function.x.array
                    drilling_energy = abs(float(phi @ (solver.K_drill @ phi)))
                    mass = float(phi @ (solver.M @ phi))
                    self.assertLess(drilling_energy / (mass * reference), 1.0e-15)

    def test_drilling_penalty_is_what_makes_the_stiffness_non_singular(self) -> None:
        """
        The two guards are NOT redundant, and this is the difference: the
        drilling penalty gives the unconstrained rotation STIFFNESS, while
        `project_rotary_inertia` only removes its MASS. Both clean up a modal
        spectrum, but only the penalty leaves `K` with the six-dimensional
        nullspace it should have.

        Measured with the inertia projection on and the penalty off: the nullity
        of `K` is 65 at h = 0.6 and 91 at h = 0.5 -- it grows with the number of
        rotation nodes -- against 6 with the penalty on. The extra directions are
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
                    bbar=True, psri=False, drilling=drilling,
                    project_rotary_inertia=True,
                )
                solver._ensure_scipy_assembled()
                stiffness = solver.K.toarray()
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

    def test_spectrum_is_insensitive_to_the_drilling_stiffness(self) -> None:
        """
        The drilling term is a stabilisation, not physics, so the physical
        spectrum must not depend on how hard it is applied. Measured across four
        orders of magnitude in `drilling_stiffness_factor` (0.01 to 100), the
        first elastic cluster moves by 8.9e-5 relative -- 0.05 Hz out of 576.6.

        The bound is 1e-3 relative, an order above the measured drift over a
        range far wider than anyone would use, and two orders below the 0.5 %
        agreement the physical assertions elsewhere in this file demand. A term
        that had become load-bearing (a units slip in `E h^3 / h_cell^2`, say)
        would show up here as a first-order dependence.
        """
        means = []
        for factor in (0.01, 100.0):
            solver = steel_sphere_solver(
                bbar=True, psri=False, drilling_stiffness_factor=factor
            )
            result = solver.solve_modes_slepc(num_modes=NUM_RIGID_BODY_MODES + 5)
            means.append(float(result.frequencies[NUM_RIGID_BODY_MODES:].mean()))
        self.assertAlmostEqual(
            means[1] / means[0],
            1.0,
            delta=1.0e-3,
            msg=f"First cluster moved with the drilling stiffness: {means}",
        )

    def test_drilling_stiffness_factor_scales_the_drilling_block(self) -> None:
        """`drilling_stiffness_factor` is a plain linear scale on the drilling
        stiffness (the parameter the entry point exposes), and nothing else."""
        base = steel_sphere_solver(0.8, 2, bbar=True, psri=False)
        scaled = steel_sphere_solver(
            0.8, 2, bbar=True, psri=False, drilling_stiffness_factor=3.0
        )
        base._ensure_scipy_assembled()
        scaled._ensure_scipy_assembled()
        # The factor multiplies the integrand rather than the assembled matrix, so
        # the quadrature sums are re-evaluated and agreement is to round-off of the
        # block norm, not of each entry: entries span fifteen orders of magnitude
        # and the tiny ones carry no relative accuracy. Hence an absolute tolerance
        # scaled by the block norm (measured residual ~1e-14 on a norm of 2e3).
        drill_scale = abs(base.K_drill).max()
        np.testing.assert_allclose(
            scaled.K_drill.toarray(),
            3.0 * base.K_drill.toarray(),
            rtol=0.0,
            atol=1.0e-12 * 3.0 * drill_scale,
        )
        np.testing.assert_allclose(
            scaled.K_bend.toarray(),
            base.K_bend.toarray(),
            rtol=0.0,
            atol=1.0e-12 * abs(base.K_bend).max(),
        )


class TestAssembledOperators(unittest.TestCase):
    """
    Structural properties the two eigensolvers rely on. Both `scipy.eigsh` with
    `sigma=0` and SLEPc's `GHEP` problem type assume a symmetric pair with an SPD
    mass matrix; if that silently stopped holding, the eigenvalues would still
    come back, just wrong.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.solver = steel_sphere_solver(0.8, 2, bbar=True, psri=False)
        cls.solver._ensure_scipy_assembled()

    @staticmethod
    def _relative_asymmetry(matrix: sp.spmatrix) -> float:
        largest = abs(matrix).max()
        if largest == 0.0:
            return 0.0
        return float(abs(matrix - matrix.T).max() / largest)

    def test_every_stiffness_block_and_the_mass_are_symmetric(self) -> None:
        """
        Both eigensolvers assume a symmetric pair, so every block must be
        symmetric to round-off. 1e-12 is far above the measured ~1e-16 and far
        below any real defect.

        `K_mem` is in this list on purpose. Under B-bar it is not assembled from
        a bilinear form but as the triple product `Pi^T G Pi`, whose symmetry
        rests on `G` being self-adjoint -- which holds only because the membrane
        assumed-strain space is the symmetric TANGENTIAL tensors. Carrying it as
        unconstrained 3x3 tensors, a strictly larger space, leaves this block
        asymmetric at ~1e-3 (see
        `test_membrane_material_operator_is_self_adjoint_on_the_reduced_space`).
        """
        for name in ("M", "K", "K_mem", "K_bend", "K_shear", "K_drill"):
            with self.subTest(block=name):
                self.assertLess(
                    self._relative_asymmetry(getattr(self.solver, name)), 1.0e-12
                )

    def test_membrane_material_operator_is_self_adjoint_on_the_reduced_space(
        self,
    ) -> None:
        """
        The precondition the paper's symmetry argument rests on, checked where it
        is actually needed: on the B-bar reduced space `W`.

        `G = <h C_ps(Q), R>` is the material-weighted reduced-space mass, and the
        projected stiffness `Pi^T G Pi` is symmetric if and only if `G` is.
        `C_ps(e) = lambda_ps tr(e) P + 2 mu e` is self-adjoint exactly on
        TANGENTIAL tensors: the antisymmetric part of the pairing reduces to
        `lambda_ps (tr(q) <P, r> - tr(r) <P, q>)`, and `<P, e> == tr(e)` holds iff
        `e` is tangential. `W` is therefore carried as the three components of a
        symmetric tangential tensor in the local frame, not as unconstrained 3x3
        tensors -- the latter is a strictly larger space containing directions on
        which `C_ps` is not self-adjoint, and it left `G` asymmetric by 0.21-0.27
        relative and the assembled block by ~1e-3.

        Measured now: 5.7e-17 (P2, h = 0.8), 8.1e-17 (P2, h = 0.5), 1.2e-16 (P3),
        i.e. round-off. The bound is 1e-12, four orders above that and nine below
        what the unconstrained space gave.
        """
        g = self.solver._scipy(self.solver._bbar_mem[2])
        self.assertLess(self._relative_asymmetry(g), 1.0e-12)

    def test_membrane_reduced_space_is_the_symmetric_tangential_tensors(
        self,
    ) -> None:
        """
        The space has three components per point, not nine: a symmetric
        tangential tensor has exactly three independent entries in a tangent
        frame. Checked as a dimension count against the shear space, which is a
        genuine three-vector on the same `DG_{p-1}` scalar basis, so the two must
        have identical dimension.

        This is what stops the space silently widening back to unconstrained 3x3
        tensors, which is the only way the symmetry above can be lost.
        """
        membrane = self.solver._scipy(self.solver._bbar_mem[1])
        shear = self.solver._scipy(self.solver._bbar_shear[1])
        self.assertEqual(membrane.shape, shear.shape)

        n_cells = self.solver.msh.topology.index_map(
            self.solver.msh.topology.dim
        ).size_local
        scalar_dofs_per_cell = self.solver.el.degree * (self.solver.el.degree + 1) // 2
        self.assertEqual(membrane.shape[0], 3 * scalar_dofs_per_cell * n_cells)

    def test_reduced_space_tensors_are_symmetric_and_tangential(self) -> None:
        """
        The reconstruction `_tangential_tensor` is what makes the space what it
        claims to be, so it is checked directly rather than only through its
        consequences: for arbitrary components the rebuilt tensor `E` must
        satisfy `E == E^T` and `E n == 0` pointwise.

        Evaluated by projecting `|E - E^T|^2` and `|E n|^2` onto DG0 and taking
        the largest cell value, normalised by `|E|^2`. Bound 1e-20 on a squared
        relative quantity, i.e. 1e-10 on the quantity itself; measured ~1e-32.
        """
        solver = self.solver
        components = ufl.as_vector([1.0, -2.0, 3.0])
        tensor = solver._tangential_tensor(components)
        norm = ufl.inner(tensor, tensor)
        measures = {
            "asymmetry": ufl.inner(tensor - ufl.transpose(tensor),
                                   tensor - ufl.transpose(tensor)) / norm,
            "normal_component": ufl.dot(ufl.dot(tensor, solver.n),
                                        ufl.dot(tensor, solver.n)) / norm,
        }
        dg0 = fem.functionspace(solver.msh, ("DG", 0))
        for name, measure in measures.items():
            with self.subTest(quantity=name):
                projected = fem.Function(dg0)
                projected.interpolate(
                    fem.Expression(measure, dg0.element.interpolation_points())
                )
                self.assertLess(float(np.abs(projected.x.array).max()), 1.0e-20)

    def test_both_solve_paths_use_the_same_stiffness(self) -> None:
        """
        The two paths build the stiffness differently: the scipy path sums four
        separately assembled blocks, while the SLEPc path assembles one combined
        UFL form (`_a_stiff`) and, under B-bar, adds the two element-local blocks
        to it as matrices. Nothing forces those to agree -- a term could be
        double-counted or dropped on one side only, and the symptom would be two
        slightly different spectra rather than an error.

        Reassembled here exactly as `solve_modes_slepc` does. Tolerance 1e-12
        relative to the matrix norm: the two orderings of the same sums differ
        only by floating-point association.
        """
        combined = self.solver._scipy(self.solver._a_stiff)
        if self.solver.el.bbar:
            membrane, shear = self.solver._bbar_blocks()
            combined = combined + membrane + shear
        self.assertLess(
            abs(combined - self.solver.K).max() / abs(self.solver.K).max(), 1.0e-12
        )

    def test_mass_matrix_is_positive_definite(self) -> None:
        """Required by both eigensolvers. Checked via the smallest eigenvalue of
        a dense symmetric eigendecomposition -- affordable at this mesh size."""
        smallest = float(np.linalg.eigvalsh(self.solver.M.toarray()).min())
        self.assertGreater(smallest, 0.0, msg=f"M has eigenvalue {smallest:.3e}.")

    def test_stiffness_nullspace_is_six_dimensional(self) -> None:
        """
        The formulation's central invariant, stated without the eigensolver: K
        must have exactly six near-null directions (the rigid-body motions) and
        the seventh must be far away.

        Taken on the symmetric part, because `eigvalsh` reads only one triangle
        and would otherwise silently analyse a matrix that is not K (see
        `test_bbar_membrane_block_is_symmetric_to_the_projection_error`); both
        eigensolvers likewise see only the symmetric part.

        The threshold separating "null" from "elastic" is placed at 1e-8 of the
        largest stiffness eigenvalue. Measured, the six null eigenvalues are
        ~1e-7 and the seventh is ~1e2 against a largest of 6.7e9, so the gap
        spans nine orders and the classification does not depend on where in it
        the cut is made.
        """
        stiffness = self.solver.K.toarray()
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
        `K phi` is zero for each of the six analytic rigid-body fields -- not
        merely `phi^T K phi`.

        Bound 1e-12 on `||K phi||_inf / (||K||_max ||phi||_inf)`; measured ~1e-15,
        i.e. accumulated round-off over the assembly, with three orders of slack.
        """
        displacement_space, displacement_map = self.solver.V.sub(0).collapse()
        rotation_space, rotation_map = self.solver.V.sub(1).collapse()
        stiffness_scale = abs(self.solver.K).max()

        for axis in range(3):
            direction = np.eye(3)[axis]
            for name, displacement_field, rotation in (
                ("translation", lambda x, d=direction: np.tile(
                    d[:, None], (1, x.shape[1])), np.zeros(3)),
                ("rotation", lambda x, d=direction: np.cross(d, x.T).T, direction),
            ):
                with self.subTest(field=f"{name}_{axis}"):
                    phi = np.zeros(self.solver.ndofs)
                    displacement = fem.Function(displacement_space)
                    displacement.interpolate(displacement_field)
                    rotation_function = fem.Function(rotation_space)
                    rotation_function.interpolate(
                        lambda x, w=rotation: np.tile(w[:, None], (1, x.shape[1]))
                    )
                    phi[displacement_map] = displacement.x.array
                    phi[rotation_map] = rotation_function.x.array
                    residual = np.abs(self.solver.K @ phi).max() / (
                        stiffness_scale * np.abs(phi).max()
                    )
                    self.assertLess(residual, 1.0e-12)

    def test_bbar_blocks_are_cached(self) -> None:
        """The blocks cost a sparse triple product each and are used by both
        solve paths, so they are computed once."""
        first = self.solver._bbar_blocks()
        self.assertIs(first, self.solver._bbar_blocks())

    def test_spectral_shift_scales_with_the_membrane_eigenvalue(self) -> None:
        """
        The shift is `1e-3 * E / (rho L^2)`, deliberately geometry-scale-aware
        rather than mesh-dependent. Two consequences are pinned: it does not move
        when the mesh is refined at fixed geometry, and it falls as `1/L^2` when
        the geometry is scaled -- which is how it stays a fixed fraction of the
        elastic spectrum (frequencies also scale as 1/L).

        It must also stay well below the first elastic eigenvalue, or the
        shift-invert target would sit inside the physical spectrum: measured
        ratio is 6.0e3 / 1.3e7, i.e. 4.6e-4.
        """
        coarse = steel_sphere_solver(0.8, 2, bbar=True, psri=False)
        fine = steel_sphere_solver(bbar=True, psri=False)
        self.assertAlmostEqual(
            coarse._spectral_shift(), fine._spectral_shift(), delta=1.0e-9
        )

        big = GeometricallyExactShellSolver(
            make_curved_sphere_mesh(2.0 * RADIUS, 2.0 * ANCHOR_MESH_SIZE, order=2),
            element=ShellElement(bbar=True, psri=False),
            **STEEL_SHELL,
        )
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


class TestBlockDiagonalInverse(unittest.TestCase):
    """
    `_block_diag_inverse` is the fast path that makes B-bar affordable: it
    exploits the fact that a DG mass matrix is block-diagonal per cell. Its
    return contract is "the inverse, or `None` if the structure is not what I
    assume" -- and the caller silently falls back to a global sparse solve, so a
    wrong answer here would be invisible.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.solver = steel_sphere_solver(0.8, 2, bbar=True, psri=False)

    def test_inverts_the_reduced_space_mass_matrix(self) -> None:
        """Round-off bound 1e-10: these are small (3x3-per-node) well-conditioned
        blocks scaled by cell area, so the residual is ~1e-16 in practice."""
        _, mass_form, _, _ = self.solver._bbar_shear
        mass = self.solver._scipy(mass_form)
        inverse = self.solver._block_diag_inverse(mass)
        self.assertIsNotNone(inverse)
        residual = (inverse @ mass) - sp.identity(mass.shape[0], format="csr")
        self.assertLess(abs(residual).max(), 1.0e-10)

    def test_returns_none_for_a_non_block_diagonal_matrix(self) -> None:
        """The guard that keeps the caller's fallback honest: a matrix with an
        entry outside the per-cell blocks must be rejected, not inverted."""
        _, mass_form, _, _ = self.solver._bbar_shear
        mass = self.solver._scipy(mass_form).tolil()
        mass[0, mass.shape[1] - 1] = 1.0  # couples the first and last cells
        self.assertIsNone(self.solver._block_diag_inverse(mass.tocsr()))

    def test_returns_none_when_the_size_is_not_a_cell_multiple(self) -> None:
        """The other structural guard: a matrix whose dimension is not divisible
        by the cell count cannot be the expected uniform-block-size layout."""
        _, mass_form, _, _ = self.solver._bbar_shear
        mass = self.solver._scipy(mass_form)
        trimmed = mass[:-1, :-1].tocsr()
        self.assertIsNone(self.solver._block_diag_inverse(trimmed))


class TestSolvePathsAgree(unittest.TestCase):
    """
    Two solve paths exist: SLEPc shift-invert (production, memory-scalable) and
    a scipy/ARPACK path that additionally reports the per-mode energy split. The
    entry point picks between them with a flag, so they must give the same
    physics.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.solver = steel_sphere_solver(bbar=True, psri=False)
        cls.num_modes = NUM_RIGID_BODY_MODES + 5
        cls.slepc = cls.solver.solve_modes_slepc(num_modes=cls.num_modes)
        cls.scipy = cls.solver.solve_modes(num_modes=cls.num_modes)

    def test_elastic_spectra_agree(self) -> None:
        """
        Compared as a cluster mean rather than mode by mode: within a five-fold
        degenerate cluster the individual eigenvalues are not separated (spread
        0.072 Hz), so which member each solver returns first is arbitrary, while
        the cluster mean is invariant to that choice -- it agrees to 1e-11 here.
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
        its energy in stretching, not bending -- which is why its frequency
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


class TestSolverPreconditions(unittest.TestCase):
    """
    The guards `solve_modes_slepc` runs before it spends anything. Each replaces
    a failure that used to surface late and misleadingly.
    """

    def test_unavailable_factor_solver_is_rejected_up_front(self) -> None:
        """
        `PC.setFactorSolverType` is a pure setter: it accepts any string and the
        failure only appears inside the factorization, as an error that does not
        name the missing backend. The check must fire before the assembly.
        """
        solver = steel_sphere_solver(0.8, 2, bbar=True, psri=False)
        with self.assertRaises(RuntimeError) as raised:
            solver.solve_modes_slepc(num_modes=2, factor_solver="not_a_solver")
        self.assertIn("not_a_solver", str(raised.exception))

    def test_builtin_factor_solver_is_accepted(self) -> None:
        """
        The `_BUILTIN_FACTOR_SOLVER` special case is load-bearing, not
        decorative: `PETSc.Sys.hasExternalPackage("petsc")` returns False --
        PETSc's own factorization is not an "external package" -- so without the
        special case the one backend that is always available would be the one
        rejected.
        """
        solver = steel_sphere_solver(0.8, 2, bbar=True, psri=False)
        result = solver.solve_modes_slepc(num_modes=8, factor_solver="petsc")
        self.assertAlmostEqual(
            float(result.frequencies[NUM_RIGID_BODY_MODES:].mean()),
            IGA_FIRST_ELASTIC_HZ,
            delta=0.01 * IGA_FIRST_ELASTIC_HZ,
        )

    def test_missing_slepc_raises_an_actionable_import_error(self) -> None:
        """The scalable path is optional in principle; if slepc4py is absent the
        message must name it and point at the scipy alternative, not surface as a
        bare ModuleNotFoundError from the middle of a solve."""
        solver = steel_sphere_solver(0.8, 2, bbar=True, psri=False)
        with mock.patch.dict(sys.modules, {"slepc4py": None}):
            with self.assertRaises(ImportError) as raised:
                solver.solve_modes_slepc(num_modes=2)
        self.assertIn("slepc4py", str(raised.exception))


class TestNormalChoice(unittest.TestCase):
    """`normal="analytic_sphere"` is a validation aid, valid only on geometry
    that really lies on the unit sphere."""

    def test_analytic_normal_on_a_flat_mesh_warns(self) -> None:
        """The constructor's guard: an analytic normal is inconsistent with a
        flat element's own geometry, which corrupts the rigid-body modes."""
        with self.assertWarns(RuntimeWarning):
            GeometricallyExactShellSolver(
                sphere_mesh(0.8, 1),
                element=ShellElement(normal="analytic_sphere"),
                **STEEL_SHELL,
            )

    def test_geometric_normal_on_a_flat_mesh_does_not_warn(self) -> None:
        """The default is safe at every geometry order, so it must stay quiet."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            GeometricallyExactShellSolver(
                sphere_mesh(0.8, 1), element=ShellElement(), **STEEL_SHELL
            )


class TestCurvedSphereMesh(unittest.TestCase):
    """`make_curved_sphere_mesh` is the shared gmsh sphere builder -- production
    code, and the geometry every test above stands on."""

    def test_order_two_nodes_lie_on_the_sphere(self) -> None:
        """
        The point of an isoparametric mesh: every geometry node, mid-edge nodes
        included, sits on the exact surface. gmsh places them by projecting onto
        the OCC sphere, so the residual is its own geometric tolerance -- 1e-9
        relative is generous against the measured ~1e-15 and would still catch
        a mesh whose curved nodes were interpolated rather than projected (those
        sit ~h^2/8 ~ 3e-2 off a radius-1 sphere).
        """
        for radius in (1.0, 2.5):
            with self.subTest(radius=radius):
                msh = make_curved_sphere_mesh(radius, 0.8, order=2)
                radii = np.linalg.norm(msh.geometry.x, axis=1)
                np.testing.assert_allclose(radii, radius, rtol=1.0e-9)

    def test_order_one_has_three_nodes_per_cell_and_order_two_has_six(self) -> None:
        """Order 1 gives P1 triangles, order 2 gives P2 (three vertices plus
        three mid-edge nodes) -- the layout `_cell_size` relies on when it takes
        the first three geometry nodes as the vertices."""
        for order, nodes_per_cell in ((1, 3), (2, 6)):
            with self.subTest(order=order):
                msh = make_curved_sphere_mesh(1.0, 0.8, order)
                self.assertEqual(msh.geometry.dofmap.shape[1], nodes_per_cell)

    def test_repeated_calls_succeed(self) -> None:
        """gmsh is a process-global singleton; the builder must leave it clean so
        a suite can build many meshes (and so a leaked session elsewhere in the
        process does not poison it)."""
        first = make_curved_sphere_mesh(1.0, 0.9, order=1)
        second = make_curved_sphere_mesh(1.0, 0.9, order=1)
        np.testing.assert_allclose(first.geometry.x, second.geometry.x)

    def test_cell_size_matches_the_longest_triangle_edge(self) -> None:
        """`_cell_size` feeds both the PSRI weight `(t/h)^2` and the drilling
        stiffness `E t^3 / h^2`, so an off-by-a-factor here would silently
        mis-weight two energy terms. Recomputed here from the vertices."""
        solver = steel_sphere_solver(0.8, 2, bbar=True, psri=False)
        vertices = solver.msh.geometry.x[np.asarray(solver.msh.geometry.dofmap)[:, :3]]
        expected = np.max(
            [
                np.linalg.norm(vertices[:, a] - vertices[:, b], axis=1)
                for a, b in ((0, 1), (1, 2), (0, 2))
            ],
            axis=0,
        )
        space = solver.h_cell.function_space
        np.testing.assert_allclose(
            solver.h_cell.x.array[space.dofmap.list[:, 0]], expected, rtol=1.0e-14
        )


if __name__ == "__main__":
    unittest.main()
