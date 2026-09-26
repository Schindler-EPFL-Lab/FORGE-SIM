import unittest
from dataclasses import replace
from unittest import mock

import numpy as np
import ufl
from _fem_fixtures import (
    CONCRETE,
    constant_function,
    function_from_dof_values,
    make_sphere_mesh,
    state_array,
    uniform_material_funcs,
)
from dolfinx import fem
from mpi4py import MPI

from fem.heat.crank_nicolson_fem_scheme import CrankNicolsonFEMScheme
from fem.heat.fem_heat_problem import FEMHeatProblem
from fem.heat.material_functions import MaterialFunctions
from fem.petsc_factor_solver import FactorSolver


class TestCrankNicolsonFEMScheme(unittest.TestCase):
    """
    Property tests on a coarse closed sphere with homogeneous material.

    These tests pin down invariants of the discretisation rather than any
    particular number: conservation of thermal energy, the constant
    nullspace of the stiffness operator on a closed manifold, linearity in
    the source, monotone decay of the non-constant part of the field, and
    the second-order accuracy of the Crank-Nicolson update. Together they
    are what a refactor of the scheme has to keep true; a sign error, a
    lost thickness factor, or a mishandled Crank-Nicolson half-step breaks
    at least one of them.

    Everything runs on a coarse closed sphere built by `_fem_fixtures`; no
    external data is needed.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Mesh generation is the slow part of the suite, so it happens once.
        cls.comm = MPI.COMM_SELF
        cls.msh = make_sphere_mesh(comm=cls.comm)
        cls.V = fem.functionspace(cls.msh, ("Lagrange", 1))
        cls.dof_coordinates = cls.V.tabulate_dof_coordinates()

    def _make_problem(
        self,
        u0: fem.Function,
        source: float = 0.0,
        material_funcs: MaterialFunctions | None = None,
    ) -> FEMHeatProblem:
        if material_funcs is None:
            material_funcs = uniform_material_funcs(self.V, CONCRETE, source=source)
        return FEMHeatProblem(
            msh=self.msh, V=self.V, u0=u0, material_funcs=material_funcs, comm=self.comm
        )

    def _make_scheme(
        self,
        u0: fem.Function,
        dt: float = 1.0e4,
        source: float = 0.0,
        material_funcs: MaterialFunctions | None = None,
    ) -> tuple[FEMHeatProblem, CrankNicolsonFEMScheme]:
        """
        Builds a problem and a scheme with an explicit physical time step. The
        default `dt` is ~2% of the relaxation time of this mesh/material pair,
        the regime the runners use.
        """
        problem = self._make_problem(u0, source=source, material_funcs=material_funcs)
        scheme = CrankNicolsonFEMScheme(
            problem=problem, dt=dt, comm=self.comm
        )
        return problem, scheme

    def _linear_temperature_field(self) -> fem.Function:
        """
        A smooth, strongly non-uniform initial field: 300 K plus a 20 K
        top-to-bottom gradient and a small azimuthal ripple, so that many
        eigenmodes of (K, M) are excited rather than just the slowest one.
        """
        x, y, z = self.dof_coordinates.T
        return function_from_dof_values(self.V, 300.0 + 20.0 * z + 5.0 * x * y)

    def _mass_weighted_variance(self, problem: FEMHeatProblem) -> float:
        """
        Returns the squared mass-matrix norm of the non-constant part of the
        current field,

            var = u^T M u - (1^T M u)^2 / (1^T M 1),

        assembled as `int(rho c d u^2) - H^2 / int(rho c d)`. Because the
        constant vector spans the kernel of K, this is exactly the energy of the
        components that Crank-Nicolson damps.
        """
        u = problem.solution
        material = problem.material_funcs
        mass_coeff = material.density * material.specific_heat * material.thickness
        second_moment = fem.assemble_scalar(
            fem.form(mass_coeff * ufl.inner(u, u) * ufl.dx)
        )
        thermal_mass = fem.assemble_scalar(fem.form(mass_coeff * ufl.dx))
        total_heat = problem.compute_total_heat()
        return second_moment - total_heat**2 / thermal_mass

    # --------------------------------------------------------------- energy
    def test_total_heat_is_conserved_without_a_source(self):
        """
        With no source and no boundary (a closed manifold), the heat equation
        conserves total thermal energy `int(rho c d u) dA` exactly: the discrete
        statement is `1^T M u^{n+1} = 1^T M u^n`, which follows from `K 1 = 0`
        applied to both Crank-Nicolson half-steps. Any loss of that property
        (a wrong mass/stiffness pairing, an inconsistent half-step, or an
        under-converged linear solve) shows up here as drift.
        """
        problem, scheme = self._make_scheme(self._linear_temperature_field())
        initial_heat = problem.compute_total_heat()

        drifts = []
        for _ in range(40):
            scheme.step()
            drifts.append(abs(problem.compute_total_heat() - initial_heat))

        # The direct Cholesky solve conserves energy to round-off: the observed
        # drift here is ~3e-15 relative, and ~1e-15 per snapshot on production
        # runs with 110k vertices. 1e-10 leaves five orders of margin for mesh
        # size, MUMPS version and BLAS threading differences while still failing
        # instantly on any real conservation defect (which manifests at 1e-3 or
        # worse).
        self.assertLess(max(drifts) / abs(initial_heat), 1e-10)

    def test_uniform_source_heats_at_the_analytic_rate(self):
        """
        A spatially uniform source on a uniform material has the exact solution
        `u(t) = u0 + Q t / (rho c d)`: it stays uniform, so the conduction term
        vanishes identically and Crank-Nicolson integrates the remaining ODE
        exactly. This is a known-answer test of how the three forms are coupled:
        it fails if the source is integrated with a spurious thickness
        factor, or if the mass coefficient loses one of rho, c, d.
        """
        source = 12.0
        initial_temperature = 300.0
        num_steps = 10
        dt = 1.0e4

        problem, scheme = self._make_scheme(
            constant_function(self.V, initial_temperature), dt=dt, source=source
        )
        for _ in range(num_steps):
            scheme.step()

        thermal_mass = CONCRETE.density * CONCRETE.specific_heat * CONCRETE.thickness
        expected = initial_temperature + source * (num_steps * dt) / thermal_mass
        # Exact in exact arithmetic; the error is pure round-off amplified by the
        # linear solve, measured at ~1e-13 K absolute on ~3e2 K values. 1e-9
        # relative keeps three orders of margin and still catches a missing
        # thickness factor (which would be a 5x error here).
        temperatures = state_array(problem)
        self.assertLess(np.abs(temperatures - expected).max() / expected, 1e-9)

    # ------------------------------------------------------- nullspace of K
    def test_constant_field_is_a_steady_state(self):
        """
        On a closed manifold with pure Neumann conditions the stiffness matrix is
        only positive semi-definite, not definite: constants are in its kernel
        (`lambda_0 = 0`), which is why the relaxation eigensolve has to skip the
        first mode and why its shift-invert needs a negative shift. The physical face of
        that algebraic statement is that a uniform temperature with no source is
        a steady state, and nothing else in the pipeline checks it.
        """
        initial_temperature = 300.0
        problem, scheme = self._make_scheme(
            constant_function(self.V, initial_temperature)
        )

        for _ in range(20):
            scheme.step()

        # In exact arithmetic the state is unchanged, so the only admissible
        # error is round-off in the Cholesky solve; measured max deviation is
        # ~1e-13 K. 1e-9 relative is a round-off gate with margin, and is
        # orders below the O(1) drift a broken nullspace would cause.
        temperatures = state_array(problem)
        deviation = np.abs(temperatures - initial_temperature).max()
        self.assertLess(deviation / initial_temperature, 1e-9)

    def test_diffusion_damps_the_non_constant_field_monotonically(self):
        """
        Crank-Nicolson amplifies the eigenmode with eigenvalue `lambda` by
        `(1 - lambda dt/2) / (1 + lambda dt/2)`, whose magnitude is strictly below
        one for every `lambda > 0`. So the mass-weighted energy of the
        non-constant part of the field must decrease at each step without
        exception, while the constant part (the total heat) is untouched. This
        is the test that separates diffusion from anti-diffusion: a sign error
        in the stiffness term conserves energy just as well but grows the field
        instead of smoothing it.
        """
        problem = self._make_problem(self._linear_temperature_field())
        scheme = CrankNicolsonFEMScheme.from_normalized_time_step(
            problem=problem,
            dt_over_t_relax=0.05,
            comm=self.comm,
        )

        initial_heat = problem.compute_total_heat()
        variances = [self._mass_weighted_variance(problem)]
        heats = [initial_heat]
        for _ in range(20):
            scheme.step()
            variances.append(self._mass_weighted_variance(problem))
            heats.append(problem.compute_total_heat())

        # Strict monotonicity is an exact property of the scheme; the per-step
        # decrease here is >=1% of the current value, i.e. ~13 orders above the
        # round-off floor, so a bare `<` comparison is not fragile.
        for previous, current in zip(variances, variances[1:]):
            self.assertLess(current, previous)

        # Over 20 steps at dt = 0.05 T_relax the slowest mode alone decays by
        # ~0.95^20 = 0.36, so anything close to 1.0 means the field is not
        # actually diffusing.
        self.assertLess(variances[-1] / variances[0], 0.5)

        # Smoothing must not move energy around: same gate as the conservation
        # test above.
        self.assertLess(
            max(abs(h - initial_heat) for h in heats) / abs(initial_heat), 1e-10
        )

    # ------------------------------------------------------ source linearity
    def test_scale_source_scales_the_generation_rate(self):
        """
        `scale_source(f)` must scale the power reported as `int(Q) dA` by exactly
        `f`, because the runners use it to hit a target temperature rise and then
        report the resulting wattage. It scales the assembled load vector and the
        source Function together; if the two ever diverge, the reported power
        stops describing the simulation that was run.
        """
        problem, scheme = self._make_scheme(constant_function(self.V, 0.0), source=7.0)
        base_rate = problem.compute_heat_generation_rate()
        self.assertGreater(base_rate, 0.0)

        factor = 3.0
        scheme.scale_source(factor)
        scaled_rate = problem.compute_heat_generation_rate()

        # Scaling then integrating differs from integrating then scaling only by
        # the rounding of one multiplication per DOF; 1e-14 relative is a
        # round-off gate (measured: exact to the last bit).
        self.assertAlmostEqual(scaled_rate / base_rate, factor, delta=1e-14 * factor)

    def test_scale_source_scales_the_solution_linearly(self):
        """
        The heat equation is linear in the source, so with a zero initial field
        the whole trajectory is proportional to the source amplitude. This pins
        the half of `scale_source` that acts during stepping: that it rescales
        the assembled load vector `F` used inside `step()`, not just the
        diagnostic Function.
        """
        num_steps = 8
        factor = 3.0

        baseline_problem, baseline_scheme = self._make_scheme(
            constant_function(self.V, 0.0), source=5.0
        )
        scaled_problem, scaled_scheme = self._make_scheme(
            constant_function(self.V, 0.0), source=5.0
        )
        scaled_scheme.scale_source(factor)

        for _ in range(num_steps):
            baseline_scheme.step()
            scaled_scheme.step()

        baseline_temperatures = state_array(baseline_problem)
        scaled_temperatures = state_array(scaled_problem)
        self.assertGreater(np.abs(baseline_temperatures).max(), 0.0)

        # Exact linearity in exact arithmetic; the two trajectories differ only
        # by rounding of the scaled right-hand side inside the triangular solves.
        # 1e-12 relative is ~4 orders above the observed 1e-16 difference.
        np.testing.assert_allclose(
            scaled_temperatures, factor * baseline_temperatures, rtol=1e-12
        )

    # ----------------------------------------------------- time-step contract
    def test_normalized_time_step_round_trip(self):
        """
        `from_normalized_time_step(x)` is the knob that makes an IGA run and an
        FEM run comparable: both resolve `dt = x * T_relax` from their own
        assembled (K, M). This checks the arithmetic identity and that the cached
        `relaxation_time` the scheme reports is the one it actually used: a
        stale or recomputed cache here would silently desynchronise the two
        discretisations.
        """
        dt_over_t_relax = 0.02
        problem = self._make_problem(self._linear_temperature_field())
        scheme = CrankNicolsonFEMScheme.from_normalized_time_step(
            problem=problem,
            dt_over_t_relax=dt_over_t_relax,
            comm=self.comm,
        )

        relaxation_time = scheme.relaxation_time
        self.assertTrue(np.isfinite(relaxation_time))
        self.assertGreater(relaxation_time, 0.0)

        # A single floating-point product; equality is exact up to the last bit.
        self.assertAlmostEqual(
            scheme.dt / relaxation_time, dt_over_t_relax, delta=1e-15 * dt_over_t_relax
        )

    def test_relaxation_time_matches_the_slowest_decay_rate(self):
        """
        `T_relax = 1 / lambda_1` is meant to be the actual physical slowest decay
        time of the semi-discrete system, not an arbitrary scale. Stepping for one
        relaxation time must therefore damp the non-constant field by roughly
        `1/e`. This is an end-to-end check of the eigensolve (target, shift sign,
        and the filtering of the zero constant mode) against the time
        integration, in a way that a pure eigenvalue assertion cannot be.
        """
        dt_over_t_relax = 0.01
        num_steps = 100
        problem = self._make_problem(
            # A pure z-gradient projects strongly onto the slowest (degree-1
            # spherical harmonic) modes, so the late-time decay is governed by
            # lambda_1 with little contamination from faster modes.
            function_from_dof_values(
                self.V, 300.0 + 20.0 * self.dof_coordinates[:, 2]
            )
        )
        scheme = CrankNicolsonFEMScheme.from_normalized_time_step(
            problem=problem,
            dt_over_t_relax=dt_over_t_relax,
            comm=self.comm,
        )

        initial_variance = self._mass_weighted_variance(problem)
        for _ in range(num_steps):
            scheme.step()
        final_variance = self._mass_weighted_variance(problem)

        # The variance is quadratic in the field, so after t = T_relax it decays
        # by ~exp(-2) = 0.135. Crank-Nicolson's amplification factor is not
        # exp(-lambda dt) but its (2,2)-Pade approximant, and faster modes decay
        # first, so the realised ratio is only close to that value, hence a
        # generous factor-of-two window rather than a tight tolerance. It still
        # fails hard if the eigensolve returns the constant mode (ratio -> 1) or
        # a fast mode (ratio -> 0).
        decay = final_variance / initial_variance
        self.assertGreater(decay, 0.5 * np.exp(-2.0))
        self.assertLess(decay, 2.0 * np.exp(-2.0))

    def test_relaxation_time_scales_with_the_material_coefficients(self):
        """
        The slowest decay rate of the semi-discrete system is set by the ratio of
        the stiffness and mass coefficients, `k d / (rho c d)`, so

            T_relax  is independent of the thickness d,
            T_relax  is inversely proportional to the conductivity k,
            T_relax  is proportional to rho and to c.

        The thickness invariance is the sharp one: it holds only because `d`
        multiplies the stiffness form and the mass form alike, so it fails the
        moment the factor is dropped from (or duplicated in) one of them: a
        mistake that a single-material run with d = 1 would never reveal.
        """

        def relaxation_time_for(**overrides: float) -> float:
            material = replace(CONCRETE, **overrides)
            _, scheme = self._make_scheme(
                constant_function(self.V, 300.0),
                material_funcs=uniform_material_funcs(self.V, material),
            )
            return scheme.relaxation_time

        base = relaxation_time_for()
        cases = {
            "thickness": (
                relaxation_time_for(thickness=2.0 * CONCRETE.thickness),
                1.0,
            ),
            "conductivity": (
                relaxation_time_for(conductivity=2.0 * CONCRETE.conductivity),
                0.5,
            ),
            "density": (relaxation_time_for(density=2.0 * CONCRETE.density), 2.0),
            "specific_heat": (
                relaxation_time_for(specific_heat=2.0 * CONCRETE.specific_heat),
                2.0,
            ),
        }

        # Exact identities of the generalized eigenproblem; the achievable
        # accuracy is set by the SLEPc solve, whose relative tolerance is 1e-8
        # (`_compute_relaxation_time(tol=...)`). 1e-6 is a safe gate above that
        # convergence noise and four orders below the factor-of-two errors this
        # test is designed to catch.
        for name, (relaxation_time, expected_ratio) in cases.items():
            with self.subTest(scaled=name):
                self.assertAlmostEqual(
                    relaxation_time / base, expected_ratio, delta=1e-6 * expected_ratio
                )

    def test_rejects_non_positive_time_steps(self):
        """
        Both construction paths validate their time-step argument up front; a
        non-positive `dt` would make `A = M + dt/2 K` indefinite and the Cholesky
        factorization would fail much later, with a far less legible error.
        """
        u0 = constant_function(self.V, 300.0)
        with self.assertRaises(ValueError):
            self._make_scheme(u0, dt=0.0)
        with self.assertRaises(ValueError):
            CrankNicolsonFEMScheme.from_normalized_time_step(
                problem=self._make_problem(u0),
                dt_over_t_relax=-0.1,
                comm=self.comm,
            )

    def test_a_diverged_step_raises_instead_of_integrating_onward(self):
        """
        A per-step solve PETSc reports as diverged stops the run, naming which
        step it was and what PETSc said.

        The factorization is validated once at construction and then reused for
        every one of the several hundred steps, so nothing downstream would
        notice one that failed later (MUMPS exhausting its workspace, or a
        non-finite right-hand side). The stepping state must be left as it was:
        a run that carried a garbage solve onward would still produce a
        plausible-looking temperature field and a plausible-looking energy log.

        PETSc cannot be made to diverge on demand on this SPD system, so the
        `KSP` is swapped for one reporting `KSP_DIVERGED_PC_FAILED` (-11) after
        two real steps, which is also what pins the reported index to the step
        that actually failed rather than to the first.
        """
        problem, scheme = self._make_scheme(self._linear_temperature_field())
        scheme.step()
        scheme.step()
        before = state_array(problem)

        diverged = mock.Mock()
        diverged.getConvergedReason.return_value = -11
        with mock.patch.object(scheme, "_ksp", diverged):
            with self.assertRaises(RuntimeError) as raised:
                scheme.step()

        self.assertIn("-11", str(raised.exception))
        self.assertIn("step 2", str(raised.exception))
        np.testing.assert_array_equal(state_array(problem), before)

    # ------------------------------------------------------- factor solver
    def test_factor_solver_require_accepts_the_builtin_backend(self):
        """
        `PETSc.Sys.hasExternalPackage("petsc")` reports False, since PETSc's own
        factorization is not registered as an "external package", so without
        the `FactorSolver.PETSC` special case, the one backend guaranteed to be
        available under every configuration would be the one `FactorSolver.require`
        rejects.

        Both of this scheme's factor-solver checks (`_build_ksp`'s
        `LINEAR_FACTOR_SOLVER` and `_compute_relaxation_time`'s
        `RELAXATION_FACTOR_SOLVER`) route through the shared
        `fem.petsc_factor_solver.FactorSolver.require`; this pins the special
        case at its shared source rather than only observing it through a full
        scheme build. Confirmed load-bearing: with the early return deleted,
        this very call raises `RuntimeError` on the one backend that must never
        be refused.
        """
        FactorSolver.require(FactorSolver.PETSC.value, "a test purpose")

    def test_factor_solver_require_rejects_an_unconfigured_backend(self):
        """
        A misspelt or unconfigured PETSc `MatSolverType` is refused immediately
        and by name, rather than surfacing later as an opaque error from inside
        a Cholesky factorization, the failure mode `FactorSolver.require`
        exists to replace (see its module docstring).
        """
        with self.assertRaises(RuntimeError) as raised:
            FactorSolver.require("not_a_solver", "a test purpose")
        self.assertIn("not_a_solver", str(raised.exception))
        self.assertIn("a test purpose", str(raised.exception))

    # ------------------------------------------------------ temporal accuracy
    def test_time_integration_is_second_order_accurate(self):
        """
        Crank-Nicolson is the centred (theta = 1/2) rule, and its defining
        property is second-order accuracy in dt: halving the step must quarter
        the error. Refining a fixed physical interval and measuring the ratio of
        successive errors against a much finer reference pins the fact that
        the two half-steps are symmetric: the `+0.5*dt*K` / `-0.5*dt*K` pair.
        Any other weighting still conserves energy and still smooths, but
        drops the order to one (ratio ~2), or, if the weights do not sum to
        one, converges to the solution of a different conductivity altogether.
        """
        final_time = 4.0e5
        source = 6.0

        def solve(num_steps: int) -> np.ndarray:
            problem, scheme = self._make_scheme(
                self._linear_temperature_field(),
                dt=final_time / num_steps,
                source=source,
            )
            for _ in range(num_steps):
                scheme.step()
            return state_array(problem)

        reference = solve(320)
        errors = [
            np.abs(solve(num_steps) - reference).max() for num_steps in (10, 20, 40)
        ]

        # Measured ratios are 4.29 and 4.05 (the coarsest levels are not yet
        # asymptotic, hence the upper end of the window). [3.0, 5.0] brackets
        # second order with room for the pre-asymptotic offset while excluding
        # first order (ratio 2) and a non-convergent scheme (ratio -> 1).
        for coarse, fine in zip(errors, errors[1:]):
            self.assertGreater(coarse / fine, 3.0)
            self.assertLess(coarse / fine, 5.0)
