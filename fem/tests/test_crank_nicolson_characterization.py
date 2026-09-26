"""
Property tests for `FEMHeatProblem` / `CrankNicolsonFEMScheme` on two small,
fully deterministic reference scenarios.

Where `test_fem_heat_problem.py` and `test_crank_nicolson_fem_scheme.py` pin
properties that any correct heat solver has, this module pins the same kind
of property for two runs on one shared mesh (a uniform-material scenario
and a two-material one), but checks their diagnostics against quantities
derived from the physics and the discretization, never against a committed
set of numbers. The two-material scenario is what exercises
`FEMHeatProblem` with a genuinely spatially varying conductivity, density,
specific heat and thickness: the reason those are held as `fem.Function`
rather than scalars in the first place.

A spatially uniform source on a uniform material is what makes the
uniform scenario's derivation possible. The load vector then satisfies
`F = (Q / rho c d) * M @ 1` exactly (both the source and mass forms
integrate the same basis functions, one against a constant coefficient,
the other against the partition of unity), so `F` is exactly parallel to
the mass matrix's row-sum vector. Substituting into the Crank-Nicolson
update shows the source excites only the constant mode: the mean grows by
exactly the known source rate, and the field's non-constant part evolves
under precisely the homogeneous recursion
`test_crank_nicolson_fem_scheme.py` proves damps monotonically, with the
source contributing to it not at all. That decomposition is what licenses
`test_final_field_variance_decays_monotonically` and does not survive the
two-material scenario; see `TestCrankNicolsonCharacterization`'s class
docstring for why.

Caveat: both scenarios are pinned to the mesh gmsh produces for these
parameters. `num_dofs` / `num_cells` are asserted first and exactly, so a
gmsh version bump that remeshes the sphere fails with an obvious message
instead of an opaque numerical mismatch.
"""

import unittest
from dataclasses import dataclass, fields

import numpy as np
from _fem_fixtures import (
    CONCRETE,
    STEEL,
    function_from_dof_values,
    make_sphere_mesh,
    two_material_funcs,
    uniform_material_funcs,
)
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc

from fem.heat.crank_nicolson_fem_scheme import CrankNicolsonFEMScheme
from fem.heat.fem_heat_problem import FEMHeatProblem


@dataclass(frozen=True, kw_only=True)
class ReferenceScenarioDiagnostics:
    """
    The scalar diagnostics `FEMHeatProblem` / `CrankNicolsonFEMScheme` report
    for the reference scenario.

    This record is the schema `TestCrankNicolsonCharacterization` checks: a
    field added here without also extending one of its assertions carries no
    coverage at all, the suite would stay green while pinning nothing for
    it.
    """

    num_dofs: int
    """Global degrees of freedom of the P1 space on the gmsh sphere."""
    num_cells: int
    """Triangles of that sphere; guards the rest against a gmsh version bump."""
    relaxation_time: float
    """Natural relaxation time T_relax from the SLEPc eigensolve [s]."""
    dt: float
    """Time step the normalized `dt_over_t_relax` resolved to [s]."""
    heat_generation_rate: float
    """Assembled source power integral(Q) dA [W]."""
    total_heat: list[float]
    """Total thermal energy at each recorded snapshot [J], in time order."""
    final_temperature_variance: list[float]
    """
    Mass-weighted energy of the field's non-constant part at each recorded
    snapshot [K^2], in time order.
    """


@dataclass(frozen=True, kw_only=True)
class _ReferenceScenarioOracle:
    """
    Reference quantities computed independently of `FEMHeatProblem`'s own
    diagnostics, against which `ReferenceScenarioDiagnostics` is checked.

    Not part of the diagnostics schema: these are the test's own
    expectations, not solver output.
    """

    geometric_surface_area: float
    """
    Sphere area from the mesh's vertex coordinates and triangle connectivity
    alone, via the elementary cross-product formula: no UFL form and no
    quadrature rule, so it is a genuine oracle for the `dx`-integrated area
    DOLFINx assembles internally.
    """
    mass_row_sum_total: float
    """
    `1^T M 1`, the discrete thermal mass: the mass matrix's row-sum total,
    independent of whatever temperature field it is later multiplied
    against.
    """
    initial_mass_matrix_total_heat: float
    """
    `1^T M u^0` by direct matrix-vector product, an independent path to the
    same quantity `FEMHeatProblem.compute_total_heat()` reports from its own
    UFL form.
    """


@dataclass(frozen=True, kw_only=True)
class TwoMaterialScenarioDiagnostics:
    """
    The scalar diagnostics `FEMHeatProblem` / `CrankNicolsonFEMScheme` report
    for the two-material scenario: the properties from
    `ReferenceScenarioDiagnostics` that survive a spatially varying
    conductivity, density, specific heat and thickness. See
    `TestCrankNicolsonCharacterization`'s class docstring for what does not
    and why.

    `num_dofs` / `num_cells` are not repeated here: this scenario runs on the
    exact mesh and function space `test_mesh_matches_the_expected_discretization`
    already guards, not a second one.
    """

    dt: float
    """Time step the normalized `dt_over_t_relax` resolved to [s]."""
    heat_generation_rate: float
    """Assembled source power integral(Q) dA [W]."""
    total_heat: list[float]
    """Total thermal energy at each recorded snapshot [J], in time order."""


@dataclass(frozen=True, kw_only=True)
class _TwoMaterialScenarioOracle:
    """
    Reference quantities computed independently of `FEMHeatProblem`'s own
    diagnostics, against which `TwoMaterialScenarioDiagnostics` is checked.

    Not part of the diagnostics schema: these are the test's own
    expectations, not solver output.
    """

    mass_row_sum_actual: float
    """`1^T M 1` by direct matrix-vector product on the assembled mass matrix."""
    mass_row_sum_expected: float
    """
    The same quantity from the closed-form barycentric-moment quadrature
    (`_exact_mass_row_sum`), independent of DOLFINx's own quadrature choice
    and, unlike the uniform scenario's `geometric_surface_area * constant`,
    correct on the triangles straddling the material boundary too.
    """
    heat_generation_rate_expected: float
    """
    `integral(Q) dA` from the exact P1 vertex-average rule
    (`_exact_source_integral`), checked against
    `TwoMaterialScenarioDiagnostics.heat_generation_rate`.
    """
    initial_mass_matrix_total_heat: float
    """
    `1^T M u^0` by direct matrix-vector product, an independent path to the
    same quantity `FEMHeatProblem.compute_total_heat()` reports from its own
    UFL form.
    """


class TestCrankNicolsonCharacterization(unittest.TestCase):
    """
    Checks two reference scenarios' diagnostics against relations derived
    from the discretization, rather than against a committed set of numbers.

    A spatially uniform source on a uniform material makes the discrete
    conservation law `1^T M u^{n+1} = 1^T M u^n + dt * 1^T F` (which needs
    only `K 1 = 0` and holds for any material distribution) decouple the
    field's mean from its spread in the uniform scenario, because `F` is
    then exactly parallel to `M @ 1`: the source contributes to the mean
    alone, and the non-constant part evolves under the same homogeneous
    recursion `test_crank_nicolson_fem_scheme.py` proves damps
    monotonically. That decomposition is what licenses
    `test_final_field_variance_decays_monotonically`.

    It does not survive the two-material scenario, and no analogous
    assertion is made there. The parallelism relied on `(M @ 1)_i` and
    `F_i` reducing to the same integral of `phi_i` up to one shared scalar
    factor, true only because a uniform `Q / (rho c d)` is that shared
    factor. With `rho c d` varying spatially, `(M @ 1)_i = integral(rho c d
    phi_i) dA` while `F_i = integral(Q phi_i) dA` are integrals of two
    unrelated functions against the same `phi_i`, and nothing makes them
    proportional. What does survive is the conservation law itself (it
    never used the parallelism, only `K 1 = 0`, which holds for any
    conductivity field: each row of `K` integrates `grad(phi_i) .
    grad(constant) = 0` regardless of what multiplies it) and the mass
    matrix's row sum, generalized from `constant * area` to the exact
    per-material-region integral. Both are checked below on a genuinely
    varying conductivity, density, specific heat and thickness, the
    reason `FEMHeatProblem` holds them as `fem.Function` rather than
    scalars, sharing the mesh and function space the uniform scenario
    already built and `test_mesh_matches_the_expected_discretization`
    already guards, rather than building and guarding a second one.
    """

    MESH_SIZE = 0.35
    SPHERE_RADIUS = 1.0
    DT_OVER_T_RELAX = 0.02
    NUM_STEPS = 12
    SNAPSHOT_INTERVAL = 3
    SOURCE = 40.0
    TWO_MATERIAL_ACTIVE_SOURCE = 40.0
    """Source on the active (steel) hemisphere only; zero on the passive one."""

    EXPECTED_NUM_DOFS = 129
    """Global DOFs of the P1 space gmsh produces at `MESH_SIZE` / `SPHERE_RADIUS`."""
    EXPECTED_NUM_CELLS = 254
    """Triangles of that same mesh."""

    EXPECTED_DIAGNOSTIC_FIELDS = (
        "num_dofs",
        "num_cells",
        "relaxation_time",
        "dt",
        "heat_generation_rate",
        "total_heat",
        "final_temperature_variance",
    )

    EXPECTED_TWO_MATERIAL_DIAGNOSTIC_FIELDS = (
        "dt",
        "heat_generation_rate",
        "total_heat",
    )

    ROUND_OFF_RTOL = 1.0e-10
    """
    Relative tolerance for algebraic identities that are exact in exact
    arithmetic: `K 1 = 0`, a constant or exactly-quadrated source/mass
    integral, and the discrete conservation law. Measured disagreement is
    3e-16 to 8e-16 relative (a few float ops, twelve MUMPS solves for the
    multi-step identity, or the barycentric-moment quadrature); 1e-10
    leaves six orders of margin while still failing a dropped coefficient.
    """

    RELAXATION_TIME_BRACKET = (0.85, 1.15)
    """
    Multiplicative window around the continuum estimate `R^2 / (2 k / (rho
    c))` (reciprocal of the sphere's first Laplace-Beltrami eigenvalue)
    `relaxation_time` must fall within. Observed across two discretizations:
    ratio to continuum measured at 0.9716 on this 254-triangle mesh and
    0.9549 on a coarser 154-triangle one; +-15% margins both without
    losing sensitivity to a materially wrong eigensolve.
    """

    @classmethod
    def setUpClass(cls) -> None:
        msh = make_sphere_mesh(
            radius=cls.SPHERE_RADIUS, mesh_size=cls.MESH_SIZE, comm=MPI.COMM_SELF
        )
        V = fem.functionspace(msh, ("Lagrange", 1))
        cls.result, cls.oracle = cls._run_reference_scenario(msh, V)
        cls.two_material_result, cls.two_material_oracle = (
            cls._run_two_material_scenario(msh, V)
        )

    @classmethod
    def _run_reference_scenario(
        cls, msh: mesh.Mesh, V: fem.FunctionSpace
    ) -> tuple[ReferenceScenarioDiagnostics, _ReferenceScenarioOracle]:
        """
        Runs the reference heat scenario and returns its diagnostics
        alongside the quantities they are independently checked against.

        The initial field is non-uniform (a top-to-bottom gradient plus an
        azimuthal ripple) so conduction is doing real work; the material is
        uniform concrete and the source is spatially uniform, which is what
        makes the mean/spread decomposition in the class docstring exact.
        The time step comes from the relaxation eigensolve, so
        `relaxation_time` is characterized as well.

        `msh` / `V` are built once in `setUpClass` and shared with
        `_run_two_material_scenario`.
        """
        dof_coordinates = V.tabulate_dof_coordinates()
        x, y, z = dof_coordinates.T

        u0 = function_from_dof_values(V, 300.0 + 20.0 * z + 5.0 * x * y)
        material_funcs = uniform_material_funcs(V, CONCRETE, source=cls.SOURCE)

        problem = FEMHeatProblem(
            msh=msh, V=V, u0=u0, material_funcs=material_funcs, comm=MPI.COMM_SELF
        )
        scheme = CrankNicolsonFEMScheme.from_normalized_time_step(
            problem=problem,
            dt_over_t_relax=cls.DT_OVER_T_RELAX,
            comm=MPI.COMM_SELF,
        )

        mass_matrix = problem.mass_matrix
        initial_mass_matrix_total_heat = cls._mass_matrix_total_heat(
            mass_matrix, problem.state
        )
        mass_row_sum_total = cls._mass_row_sum_total(mass_matrix)

        # Snapshots are taken exactly the way `FEMHeatRun.march` takes them:
        # standalone state copies during the loop, diagnostics afterwards.
        snapshots = [problem.snapshot_state()]
        for step in range(1, cls.NUM_STEPS + 1):
            scheme.step()
            if step % cls.SNAPSHOT_INTERVAL == 0:
                snapshots.append(problem.snapshot_state())

        total_heat = []
        variance = []
        for snapshot in snapshots:
            problem.load_state(snapshot)
            total_heat.append(problem.compute_total_heat())
            variance.append(
                cls._mass_weighted_variance(
                    mass_matrix, problem.state, mass_row_sum_total
                )
            )
            snapshot.destroy()

        diagnostics = ReferenceScenarioDiagnostics(
            num_dofs=int(V.dofmap.index_map.size_global),
            num_cells=int(msh.topology.index_map(2).size_global),
            relaxation_time=float(scheme.relaxation_time),
            dt=float(scheme.dt),
            heat_generation_rate=float(problem.compute_heat_generation_rate()),
            total_heat=[float(value) for value in total_heat],
            final_temperature_variance=[float(value) for value in variance],
        )
        oracle = _ReferenceScenarioOracle(
            geometric_surface_area=cls._geometric_surface_area(msh),
            mass_row_sum_total=mass_row_sum_total,
            initial_mass_matrix_total_heat=initial_mass_matrix_total_heat,
        )
        return diagnostics, oracle

    @staticmethod
    def _triangle_areas(msh: mesh.Mesh) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns each triangle's vertex-index triple and its area, from the
        mesh's own vertex coordinates and triangle connectivity via the
        elementary cross-product formula: no UFL form and no quadrature
        rule, so this is a genuine oracle for whatever DOLFINx assembles
        internally, shared by every geometric check in this module.
        """
        vertices = msh.geometry.x
        triangles = msh.topology.connectivity(2, 0).array.reshape(-1, 3)
        edge1 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
        edge2 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
        areas = 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)
        return triangles, areas

    @classmethod
    def _geometric_surface_area(cls, msh: mesh.Mesh) -> float:
        """Returns the sphere's total surface area (see `_triangle_areas`)."""
        return float(cls._triangle_areas(msh)[1].sum())

    @staticmethod
    def _mass_row_sum_total(mass_matrix: PETSc.Mat) -> float:
        """
        Returns `1^T M 1`: the mass matrix's total row sum, i.e. the discrete
        surface integral of the mass coefficient alone.
        """
        ones = mass_matrix.createVecLeft()
        ones.set(1.0)
        weighted = mass_matrix.createVecLeft()
        mass_matrix.mult(ones, weighted)
        return float(ones.dot(weighted))

    @staticmethod
    def _mass_matrix_total_heat(mass_matrix: PETSc.Mat, state: PETSc.Vec) -> float:
        """
        Returns `1^T M u`, the total heat by direct matrix-vector product,
        the same quantity `FEMHeatProblem.compute_total_heat()` reports from
        its own, independently written UFL form.
        """
        ones = mass_matrix.createVecLeft()
        ones.set(1.0)
        weighted = mass_matrix.createVecLeft()
        mass_matrix.mult(state, weighted)
        return float(ones.dot(weighted))

    @staticmethod
    def _mass_weighted_variance(
        mass_matrix: PETSc.Mat, state: PETSc.Vec, mass_row_sum_total: float
    ) -> float:
        """
        Returns `u^T M u - (1^T M u)^2 / (1^T M 1)`: the mass-weighted energy
        of the field's non-constant part, the quantity Crank-Nicolson damps
        at every step (compare `test_crank_nicolson_fem_scheme.py`'s
        `_mass_weighted_variance`, which computes the same quantity through
        `compute_total_heat()` rather than a direct matrix-vector product).
        """
        ones = mass_matrix.createVecLeft()
        ones.set(1.0)
        weighted = mass_matrix.createVecLeft()
        mass_matrix.mult(state, weighted)
        second_moment = state.dot(weighted)
        total_heat = ones.dot(weighted)
        return second_moment - total_heat**2 / mass_row_sum_total

    @classmethod
    def _run_two_material_scenario(
        cls, msh: mesh.Mesh, V: fem.FunctionSpace
    ) -> tuple[TwoMaterialScenarioDiagnostics, _TwoMaterialScenarioOracle]:
        """
        Runs the two-material heat scenario on the shared `msh` / `V` and
        returns its diagnostics alongside the quantities they are
        independently checked against.

        The active (steel) hemisphere carries the whole source over a
        passive (concrete) base, mirroring the two-material heat scripts'
        facade split; the initial field is the same non-uniform one the
        uniform scenario uses. See the class docstring for which of the
        uniform scenario's properties transfer here and which do not.
        """
        dof_coordinates = V.tabulate_dof_coordinates()
        x, y, z = dof_coordinates.T
        active_mask = z > 0.0

        u0 = function_from_dof_values(V, 300.0 + 20.0 * z + 5.0 * x * y)
        material_funcs = two_material_funcs(
            V,
            active_mask=active_mask,
            active_material=STEEL,
            passive_material=CONCRETE,
            active_source=cls.TWO_MATERIAL_ACTIVE_SOURCE,
        )

        problem = FEMHeatProblem(
            msh=msh, V=V, u0=u0, material_funcs=material_funcs, comm=MPI.COMM_SELF
        )
        scheme = CrankNicolsonFEMScheme.from_normalized_time_step(
            problem=problem,
            dt_over_t_relax=cls.DT_OVER_T_RELAX,
            comm=MPI.COMM_SELF,
        )

        mass_matrix = problem.mass_matrix
        initial_mass_matrix_total_heat = cls._mass_matrix_total_heat(
            mass_matrix, problem.state
        )

        snapshots = [problem.snapshot_state()]
        for step in range(1, cls.NUM_STEPS + 1):
            scheme.step()
            if step % cls.SNAPSHOT_INTERVAL == 0:
                snapshots.append(problem.snapshot_state())

        total_heat = []
        for snapshot in snapshots:
            problem.load_state(snapshot)
            total_heat.append(problem.compute_total_heat())
            snapshot.destroy()

        diagnostics = TwoMaterialScenarioDiagnostics(
            dt=float(scheme.dt),
            heat_generation_rate=float(problem.compute_heat_generation_rate()),
            total_heat=[float(value) for value in total_heat],
        )
        oracle = _TwoMaterialScenarioOracle(
            mass_row_sum_actual=cls._mass_row_sum_total(mass_matrix),
            mass_row_sum_expected=cls._exact_mass_row_sum(msh, active_mask),
            heat_generation_rate_expected=cls._exact_source_integral(
                msh, active_mask, cls.TWO_MATERIAL_ACTIVE_SOURCE
            ),
            initial_mass_matrix_total_heat=initial_mass_matrix_total_heat,
        )
        return diagnostics, oracle

    @classmethod
    def _exact_mass_row_sum(cls, msh: mesh.Mesh, active_mask: np.ndarray) -> float:
        """
        Returns `1^T M 1` for the two-material scenario in closed form.

        `rho c d` is a product of three per-vertex-linear (P1) fields, so on
        each triangle it is an exact cubic polynomial in the barycentric
        coordinates `lambda_0, lambda_1, lambda_2`. Every vertex takes only
        the active or the passive material's triple, never a blend, so a
        triangle's three corners pick out 27 `(i, j, k)` terms of the
        standard monomial formula

            integral(lambda_i^a lambda_j^b lambda_k^c) dA
                = (a! b! c!) / (a + b + c + 2)! * 2 * Area,

        whose weight depends only on how many of `i, j, k` coincide:
        `Area / 10` if all three, `Area / 30` if exactly two, `Area / 60` if
        all distinct. On the reference mesh this reproduces the assembled
        `1^T M 1` to 3e-16 relative, confirming DOLFINx's own quadrature
        degree is exact for it, so `ROUND_OFF_RTOL` applies here too.

        `active_mask` is a boolean array over the DOFs of `V`. For a P1
        space on this order-1 mesh, run on a single rank, those DOFs
        coincide 1:1 and in the same order with `msh`'s own geometry
        vertices (confirmed by the 3e-16 agreement above, which an
        ordering mismatch could not produce), so the mask indexes the
        mesh's triangle connectivity directly.
        """
        triangles, areas = cls._triangle_areas(msh)
        active_triple = np.array([STEEL.density, STEEL.specific_heat, STEEL.thickness])
        passive_triple = np.array(
            [CONCRETE.density, CONCRETE.specific_heat, CONCRETE.thickness]
        )
        per_vertex_triple = np.where(
            active_mask[:, None], active_triple, passive_triple
        )
        corner = per_vertex_triple[triangles]
        rho, c, d = corner[:, :, 0], corner[:, :, 1], corner[:, :, 2]

        index = np.arange(3)
        i, j, k = np.meshgrid(index, index, index, indexing="ij")
        all_same = (i == j) & (j == k)
        two_same = ~all_same & ((i == j) | (j == k) | (i == k))
        weight = np.select(
            [all_same, two_same], [1.0 / 10.0, 1.0 / 30.0], default=1.0 / 60.0
        )

        return float(areas @ np.einsum("ti,tj,tk,ijk->t", rho, c, d, weight))

    @classmethod
    def _exact_source_integral(
        cls, msh: mesh.Mesh, active_mask: np.ndarray, active_source: float
    ) -> float:
        """
        Returns `integral(Q) dA` for the two-material scenario in closed
        form.

        Unlike the mass coefficient, `Q` is itself a single per-vertex-linear
        (P1) field, `active_source` on active DOFs, zero elsewhere, not a
        product of several, so the standard vertex-average rule for a linear
        function, `integral over a triangle = Area / 3 * (sum of its three
        corner values)`, is already exact; no boundary-triangle correction
        is needed the way it is for `_exact_mass_row_sum`.
        """
        triangles, areas = cls._triangle_areas(msh)
        corner_source = np.where(active_mask, active_source, 0.0)[triangles]
        return float(areas @ corner_source.mean(axis=1))

    def test_mesh_matches_the_expected_discretization(self):
        """
        Guards the rest of the module: every other assertion, for both the
        uniform and the two-material scenario, only means anything on the
        mesh it was derived for; the two scenarios share the exact `msh` /
        `V` objects built in `setUpClass`, so this is the only guard either
        needs. A different gmsh version (or a change to `make_sphere_mesh`)
        shows up here rather than as an unexplained numerical mismatch
        downstream.
        """
        self.assertEqual(self.result.num_dofs, self.EXPECTED_NUM_DOFS)
        self.assertEqual(self.result.num_cells, self.EXPECTED_NUM_CELLS)

    def test_diagnostics_schema_matches_the_expected_fields(self):
        """
        `ReferenceScenarioDiagnostics` has exactly the fields this module's
        other tests know how to check, no more and no less: a field added
        to the record without extending one of them would otherwise carry no
        coverage at all.
        """
        record_fields = {field.name for field in fields(ReferenceScenarioDiagnostics)}
        self.assertEqual(record_fields, set(self.EXPECTED_DIAGNOSTIC_FIELDS))

    def test_two_material_diagnostics_schema_matches_the_expected_fields(self):
        """
        `TwoMaterialScenarioDiagnostics` has exactly the fields this
        module's two-material tests know how to check: the same discipline
        as `test_diagnostics_schema_matches_the_expected_fields`, over the
        smaller record the varying-coefficient scenario actually supports.
        """
        record_fields = {
            field.name for field in fields(TwoMaterialScenarioDiagnostics)
        }
        self.assertEqual(
            record_fields, set(self.EXPECTED_TWO_MATERIAL_DIAGNOSTIC_FIELDS)
        )

    def test_time_step_characterization(self):
        """
        `dt` must be exactly `dt_over_t_relax * relaxation_time` by
        construction, and `relaxation_time` itself must fall within the
        observed discretization-error window of the continuum estimate (see
        `RELAXATION_TIME_BRACKET`).
        """
        self.assertAlmostEqual(
            self.result.dt / self.result.relaxation_time,
            self.DT_OVER_T_RELAX,
            delta=self.ROUND_OFF_RTOL * self.DT_OVER_T_RELAX,
        )

        alpha = CONCRETE.conductivity / (CONCRETE.density * CONCRETE.specific_heat)
        continuum_relaxation_time = self.SPHERE_RADIUS**2 / (2.0 * alpha)
        lower, upper = self.RELAXATION_TIME_BRACKET
        self.assertGreater(
            self.result.relaxation_time, lower * continuum_relaxation_time
        )
        self.assertLess(self.result.relaxation_time, upper * continuum_relaxation_time)

    def test_mass_matrix_row_sum_matches_the_geometric_area(self):
        """
        `1^T M 1` must equal `rho c d * area` for a uniform material: the
        mass matrix's total row sum is the discrete surface integral of a
        constant coefficient, which P1 quadrature integrates exactly.
        """
        thermal_mass_coefficient = (
            CONCRETE.density * CONCRETE.specific_heat * CONCRETE.thickness
        )
        expected = thermal_mass_coefficient * self.oracle.geometric_surface_area
        self.assertAlmostEqual(
            self.oracle.mass_row_sum_total / expected, 1.0, delta=self.ROUND_OFF_RTOL
        )

    def test_heat_generation_rate_matches_the_geometric_area(self):
        """
        `integral(Q) dA` for a spatially uniform `Q` is exactly `Q * area`:
        P1 quadrature integrates a constant exactly, so this is an identity,
        not an approximation.
        """
        expected = self.SOURCE * self.oracle.geometric_surface_area
        self.assertAlmostEqual(
            self.result.heat_generation_rate / expected, 1.0, delta=self.ROUND_OFF_RTOL
        )

    def test_total_heat_is_conserved_up_to_the_known_source_growth(self):
        """
        `compute_total_heat()` at t=0 must agree with the matrix-vector
        identity `1^T M u^0`, and every later snapshot must equal the
        initial value plus the exact discrete growth
        `step * dt * heat_generation_rate`: the conservation law
        `1^T M u^{n+1} = 1^T M u^n + dt * 1^T F` applied `step` times, using
        only `K 1 = 0`.
        """
        self.assertAlmostEqual(
            self.result.total_heat[0] / self.oracle.initial_mass_matrix_total_heat,
            1.0,
            delta=self.ROUND_OFF_RTOL,
        )

        step_indices = range(
            0, self.NUM_STEPS + self.SNAPSHOT_INTERVAL, self.SNAPSHOT_INTERVAL
        )
        predicted = [
            self.result.total_heat[0]
            + step * self.result.dt * self.result.heat_generation_rate
            for step in step_indices
        ]
        np.testing.assert_allclose(
            self.result.total_heat, predicted, rtol=self.ROUND_OFF_RTOL
        )

    def test_final_field_variance_decays_monotonically(self):
        """
        The mass-weighted variance, the energy of the field's non-constant
        part, must decrease at every snapshot: Crank-Nicolson's per-mode
        amplification factor `(1 - lambda dt/2) / (1 + lambda dt/2)` has
        magnitude strictly below one for every `lambda > 0`, and the uniform
        source here contributes nothing to this quantity (see the class
        docstring). A sign error in the stiffness term would still conserve
        `total_heat` but would grow this instead of shrinking it.
        """
        variance = self.result.final_temperature_variance
        for previous, current in zip(variance, variance[1:]):
            self.assertLess(current, previous)

        # Not just formally decreasing: over 12 steps at dt = 0.02 T_relax the
        # observed decay is to ~61% of the initial value. A threshold at 90%
        # still fails a scheme whose damping has become numerically negligible
        # (e.g. a stiffness matrix scaled down by orders of magnitude) while
        # giving comfortable margin over the real decay.
        self.assertLess(variance[-1] / variance[0], 0.9)

    def test_two_material_mass_row_sum_matches_the_exact_quadrature(self):
        """
        `1^T M 1` must equal the closed-form barycentric-moment integral of
        `rho c d` over the mesh (`_exact_mass_row_sum`). Unlike the uniform
        scenario, this is not `constant * area`: the coefficient is
        genuinely non-constant, and non-linear (a product of three P1
        fields), on the triangles straddling the material boundary.

        This is the assertion that actually needs the varying coefficient
        to be right, as opposed to merely self-consistent. Collapsing the
        two materials to one everywhere, in both the mass form and
        `compute_total_heat()`'s own form, so the two stay in agreement
        with each other, leaves the growth-law identity below exactly
        satisfied (it only ever compares the solver's diagnostics against
        themselves), and moves this one by 72%: confirmed by construction,
        not merely argued.
        """
        self.assertAlmostEqual(
            self.two_material_oracle.mass_row_sum_actual
            / self.two_material_oracle.mass_row_sum_expected,
            1.0,
            delta=self.ROUND_OFF_RTOL,
        )

    def test_two_material_heat_generation_rate_matches_the_geometric_area(self):
        """
        `integral(Q) dA` must equal the exact P1 vertex-average quadrature
        of the per-DOF source field (`_exact_source_integral`), the
        two-material analogue of
        `test_heat_generation_rate_matches_the_geometric_area`, generalized
        from a spatially uniform `Q` to one that is zero on the passive
        hemisphere.
        """
        self.assertAlmostEqual(
            self.two_material_result.heat_generation_rate
            / self.two_material_oracle.heat_generation_rate_expected,
            1.0,
            delta=self.ROUND_OFF_RTOL,
        )

    def test_two_material_total_heat_is_conserved_up_to_the_known_source_growth(self):
        """
        The discrete conservation law transfers unchanged from the uniform
        scenario: `1^T M u^{n+1} = 1^T M u^n + dt * 1^T F` needs only
        `K 1 = 0`, which holds for any conductivity field because every row
        of `K` integrates `grad(phi_i) . grad(constant) = 0` regardless of
        what multiplies it. So `compute_total_heat()` at t=0 must still
        agree with `1^T M u^0`, and every later snapshot must still equal
        the initial value plus `step * dt * heat_generation_rate`.

        This is a self-consistency check, not a correctness one: it
        compares the solver's own diagnostics against each other, and would
        pass unchanged even if every material field were silently collapsed
        to one; `test_two_material_mass_row_sum_matches_the_exact_quadrature`
        is the assertion that catches that.
        """
        self.assertAlmostEqual(
            self.two_material_result.total_heat[0]
            / self.two_material_oracle.initial_mass_matrix_total_heat,
            1.0,
            delta=self.ROUND_OFF_RTOL,
        )

        step_indices = range(
            0, self.NUM_STEPS + self.SNAPSHOT_INTERVAL, self.SNAPSHOT_INTERVAL
        )
        predicted = [
            self.two_material_result.total_heat[0]
            + step
            * self.two_material_result.dt
            * self.two_material_result.heat_generation_rate
            for step in step_indices
        ]
        np.testing.assert_allclose(
            self.two_material_result.total_heat, predicted, rtol=self.ROUND_OFF_RTOL
        )
