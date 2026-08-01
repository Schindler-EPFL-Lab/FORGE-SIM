"""
Physics property tests for `CrankNicolsonFEMSolver`.

These tests pin down the *invariants of the discretisation* rather than any
particular number: conservation of thermal energy, the constant nullspace of the
stiffness operator on a closed manifold, linearity in the source, monotone decay
of the non-constant part of the field, and the decoupling of the stepping state
from the diagnostic `fem.Function`. Together they are what a refactor of the
solver has to keep true; a sign error, a lost thickness factor, a mishandled
Crank-Nicolson half-step or a re-introduction of the "diagnostics corrupt the
time loop" bug all break at least one of them.

Everything runs on a coarse closed sphere built by `_fem_fixtures`; no external
data is needed. Run from the repository root::

    conda run -n dolfinx-rebel-shape python -m unittest discover dolfinx/tests
"""

import unittest

import numpy as np
import ufl

from dolfinx import fem
from mpi4py import MPI

from _fem_fixtures import (
    CONCRETE,
    constant_function,
    function_from_dof_values,
    make_sphere_mesh,
    state_array,
    uniform_material_funcs,
)
from crank_nicolson_fem_solver import CrankNicolsonFEMSolver


class TestCrankNicolsonFEMSolver(unittest.TestCase):
    """
    Property tests on a coarse closed sphere with homogeneous material.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Mesh generation is the slow part of the suite, so it happens once.
        cls.comm = MPI.COMM_SELF
        cls.msh = make_sphere_mesh(comm=cls.comm)
        cls.V = fem.functionspace(cls.msh, ("Lagrange", 1))
        cls.dof_coordinates = cls.V.tabulate_dof_coordinates()

    def _make_solver(
        self,
        u0: fem.Function,
        dt: float = 1.0e4,
        source: float = 0.0,
        material_funcs: dict[str, fem.Function] | None = None,
    ) -> CrankNicolsonFEMSolver:
        """
        Builds a solver with an explicit physical time step. The default `dt` is
        ~2% of the relaxation time of this mesh/material pair, the regime the
        runners use.
        """
        if material_funcs is None:
            material_funcs = uniform_material_funcs(self.V, CONCRETE, source=source)
        return CrankNicolsonFEMSolver(
            msh=self.msh,
            V=self.V,
            u0=u0,
            material_funcs=material_funcs,
            dt=dt,
            comm=self.comm,
        )

    def _linear_temperature_field(self) -> fem.Function:
        """
        A smooth, strongly non-uniform initial field: 300 K plus a 20 K
        top-to-bottom gradient and a small azimuthal ripple, so that many
        eigenmodes of (K, M) are excited rather than just the slowest one.
        """
        x, y, z = self.dof_coordinates.T
        return function_from_dof_values(self.V, 300.0 + 20.0 * z + 5.0 * x * y)

    def _mass_weighted_variance(self, solver: CrankNicolsonFEMSolver) -> float:
        """
        Returns the squared mass-matrix norm of the *non-constant* part of the
        current field,

            var = u^T M u - (1^T M u)^2 / (1^T M 1),

        assembled as `int(rho c d u^2) - H^2 / int(rho c d)`. Because the
        constant vector spans the kernel of K, this is exactly the energy of the
        components that Crank-Nicolson damps.
        """
        u = solver.solution
        mass_coeff = solver.rho * solver.c * solver.d
        second_moment = fem.assemble_scalar(
            fem.form(mass_coeff * ufl.inner(u, u) * ufl.dx)
        )
        thermal_mass = fem.assemble_scalar(fem.form(mass_coeff * ufl.dx))
        total_heat = solver.compute_total_heat()
        return second_moment - total_heat**2 / thermal_mass

    # --------------------------------------------------------------- energy
    def test_total_heat_is_conserved_without_a_source(self):
        """
        With no source and no boundary (a closed manifold), the heat equation
        conserves total thermal energy `int(rho c d u) dA` exactly: the discrete
        statement is `1^T M u^{n+1} = 1^T M u^n`, which follows from `K 1 = 0`
        applied to both Crank-Nicolson half-steps. Any loss of that property --
        a wrong mass/stiffness pairing, an inconsistent half-step, or an
        under-converged linear solve -- shows up here as drift.
        """
        solver = self._make_solver(self._linear_temperature_field())
        initial_heat = solver.compute_total_heat()

        drifts = []
        for _ in range(40):
            solver.step()
            drifts.append(abs(solver.compute_total_heat() - initial_heat))

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
        exactly. This is a known-answer test of the *coupling* between the three
        forms -- it fails if the source is integrated with a spurious thickness
        factor, or if the mass coefficient loses one of rho, c, d.
        """
        source = 12.0
        initial_temperature = 300.0
        num_steps = 10
        dt = 1.0e4

        solver = self._make_solver(
            constant_function(self.V, initial_temperature), dt=dt, source=source
        )
        for _ in range(num_steps):
            solver.step()

        thermal_mass = (
            CONCRETE["density"] * CONCRETE["specific_heat"] * CONCRETE["thickness"]
        )
        expected = initial_temperature + source * (num_steps * dt) / thermal_mass
        # Exact in exact arithmetic; the error is pure round-off amplified by the
        # linear solve, measured at ~1e-13 K absolute on ~3e2 K values. 1e-9
        # relative keeps three orders of margin and still catches a missing
        # thickness factor (which would be a 5x error here).
        temperatures = state_array(solver)
        self.assertLess(np.abs(temperatures - expected).max() / expected, 1e-9)

    # ------------------------------------------------------- nullspace of K
    def test_constant_field_is_a_steady_state(self):
        """
        On a closed manifold with pure Neumann conditions the stiffness matrix is
        only positive *semi*-definite: constants are in its kernel (`lambda_0 =
        0`), which is why the relaxation eigensolve has to skip the first mode
        and why its shift-invert needs a negative shift. The physical face of
        that algebraic statement is that a uniform temperature with no source is
        a steady state -- and nothing else in the pipeline checks it.
        """
        initial_temperature = 300.0
        solver = self._make_solver(constant_function(self.V, initial_temperature))

        for _ in range(20):
            solver.step()

        # In exact arithmetic the state is unchanged, so the only admissible
        # error is round-off in the Cholesky solve; measured max deviation is
        # ~1e-13 K. 1e-9 relative is a round-off gate with margin, and is
        # orders below the O(1) drift a broken nullspace would cause.
        temperatures = state_array(solver)
        deviation = np.abs(temperatures - initial_temperature).max()
        self.assertLess(deviation / initial_temperature, 1e-9)

    def test_diffusion_damps_the_non_constant_field_monotonically(self):
        """
        Crank-Nicolson amplifies the eigenmode with eigenvalue `lambda` by
        `(1 - lambda dt/2) / (1 + lambda dt/2)`, whose magnitude is strictly below
        one for every `lambda > 0`. So the mass-weighted energy of the
        non-constant part of the field must decrease at *every* step, while the
        constant part (the total heat) is untouched. This is the test that
        separates diffusion from anti-diffusion: a sign error in the stiffness
        term conserves energy just as well but grows the field instead of
        smoothing it.
        """
        solver = CrankNicolsonFEMSolver.from_normalized_time_step(
            msh=self.msh,
            V=self.V,
            u0=self._linear_temperature_field(),
            material_funcs=uniform_material_funcs(self.V, CONCRETE),
            dt_over_t_relax=0.05,
            comm=self.comm,
        )

        initial_heat = solver.compute_total_heat()
        variances = [self._mass_weighted_variance(solver)]
        heats = [initial_heat]
        for _ in range(20):
            solver.step()
            variances.append(self._mass_weighted_variance(solver))
            heats.append(solver.compute_total_heat())

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
        `scale_source(f)` must scale the *reported* power `int(Q) dA` by exactly
        `f`, because the runners use it to hit a target temperature rise and then
        report the resulting wattage. It scales the assembled load vector and the
        source Function together; if the two ever diverge, the reported power
        stops describing the simulation that was run.
        """
        solver = self._make_solver(constant_function(self.V, 0.0), source=7.0)
        base_rate = solver.compute_heat_generation_rate()
        self.assertGreater(base_rate, 0.0)

        factor = 3.0
        solver.scale_source(factor)
        scaled_rate = solver.compute_heat_generation_rate()

        # Scaling then integrating differs from integrating then scaling only by
        # the rounding of one multiplication per DOF; 1e-14 relative is a
        # round-off gate (measured: exact to the last bit).
        self.assertAlmostEqual(scaled_rate / base_rate, factor, delta=1e-14 * factor)

    def test_scale_source_scales_the_solution_linearly(self):
        """
        The heat equation is linear in the source, so with a zero initial field
        the whole trajectory is proportional to the source amplitude. This pins
        the *dynamic* half of `scale_source`: that it rescales the assembled load
        vector `F` used inside `step()`, not just the diagnostic Function.
        """
        num_steps = 8
        factor = 3.0

        baseline = self._make_solver(constant_function(self.V, 0.0), source=5.0)
        scaled = self._make_solver(constant_function(self.V, 0.0), source=5.0)
        scaled.scale_source(factor)

        for _ in range(num_steps):
            baseline.step()
            scaled.step()

        baseline_temperatures = state_array(baseline)
        scaled_temperatures = state_array(scaled)
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
        `relaxation_time` the solver reports is the one it actually used -- a
        stale or recomputed cache here would silently desynchronise the two
        discretisations.
        """
        dt_over_t_relax = 0.02
        solver = CrankNicolsonFEMSolver.from_normalized_time_step(
            msh=self.msh,
            V=self.V,
            u0=self._linear_temperature_field(),
            material_funcs=uniform_material_funcs(self.V, CONCRETE),
            dt_over_t_relax=dt_over_t_relax,
            comm=self.comm,
        )

        relaxation_time = solver.relaxation_time
        self.assertTrue(np.isfinite(relaxation_time))
        self.assertGreater(relaxation_time, 0.0)

        # A single floating-point product; equality is exact up to the last bit.
        self.assertAlmostEqual(
            solver.dt / relaxation_time, dt_over_t_relax, delta=1e-15 * dt_over_t_relax
        )

    def test_relaxation_time_matches_the_slowest_decay_rate(self):
        """
        `T_relax = 1 / lambda_1` is meant to be the *physical* slowest decay time
        of the semi-discrete system, not an arbitrary scale. Stepping for one
        relaxation time must therefore damp the non-constant field by roughly
        `1/e`. This is an end-to-end check of the eigensolve (target, shift sign,
        and the filtering of the zero constant mode) against the time
        integration, in a way that a pure eigenvalue assertion cannot be.
        """
        dt_over_t_relax = 0.01
        num_steps = 100
        solver = CrankNicolsonFEMSolver.from_normalized_time_step(
            msh=self.msh,
            V=self.V,
            # A pure z-gradient projects strongly onto the slowest (degree-1
            # spherical harmonic) modes, so the late-time decay is governed by
            # lambda_1 with little contamination from faster modes.
            u0=function_from_dof_values(
                self.V, 300.0 + 20.0 * self.dof_coordinates[:, 2]
            ),
            material_funcs=uniform_material_funcs(self.V, CONCRETE),
            dt_over_t_relax=dt_over_t_relax,
            comm=self.comm,
        )

        initial_variance = self._mass_weighted_variance(solver)
        for _ in range(num_steps):
            solver.step()
        final_variance = self._mass_weighted_variance(solver)

        # The variance is quadratic in the field, so after t = T_relax it decays
        # by ~exp(-2) = 0.135. Crank-Nicolson's amplification factor is not
        # exp(-lambda dt) but its (2,2)-Pade approximant, and faster modes decay
        # first, so the realised ratio is only close to that value -- hence a
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
        multiplies *both* forms, so it fails the moment the factor is dropped
        from (or duplicated in) one of them -- a mistake that a single-material
        run with d = 1 would never reveal.
        """

        def relaxation_time_for(**overrides: float) -> float:
            material = {**CONCRETE, **overrides}
            solver = CrankNicolsonFEMSolver(
                msh=self.msh,
                V=self.V,
                u0=constant_function(self.V, 300.0),
                material_funcs=uniform_material_funcs(self.V, material),
                dt=1.0e4,
                comm=self.comm,
            )
            return solver.relaxation_time

        base = relaxation_time_for()
        cases = {
            "thickness": (
                relaxation_time_for(thickness=2.0 * CONCRETE["thickness"]),
                1.0,
            ),
            "conductivity": (
                relaxation_time_for(conductivity=2.0 * CONCRETE["conductivity"]),
                0.5,
            ),
            "density": (relaxation_time_for(density=2.0 * CONCRETE["density"]), 2.0),
            "specific_heat": (
                relaxation_time_for(specific_heat=2.0 * CONCRETE["specific_heat"]),
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
            self._make_solver(u0, dt=0.0)
        with self.assertRaises(ValueError):
            CrankNicolsonFEMSolver.from_normalized_time_step(
                msh=self.msh,
                V=self.V,
                u0=u0,
                material_funcs=uniform_material_funcs(self.V, CONCRETE),
                dt_over_t_relax=-0.1,
                comm=self.comm,
            )

    # ------------------------------------------------------ temporal accuracy
    def test_time_integration_is_second_order_accurate(self):
        """
        Crank-Nicolson is the centred (theta = 1/2) rule, and its defining
        property is second-order accuracy in dt: halving the step must quarter
        the error. Refining a fixed physical interval and measuring the ratio of
        successive errors against a much finer reference pins the *symmetry* of
        the two half-steps -- the `+0.5*dt*K` / `-0.5*dt*K` pair. Any other
        weighting still conserves energy and still smooths, but drops the order
        to one (ratio ~2), or, if the weights no longer sum to one, converges to
        the solution of a different conductivity altogether.
        """
        final_time = 4.0e5
        source = 6.0

        def solve(num_steps: int) -> np.ndarray:
            solver = self._make_solver(
                self._linear_temperature_field(),
                dt=final_time / num_steps,
                source=source,
            )
            for _ in range(num_steps):
                solver.step()
            return state_array(solver)

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

    # ------------------------------------------------------- state handling
    def test_snapshot_and_load_state_round_trip(self):
        """
        The runners collect `snapshot_state()` vectors during the time loop and
        replay them with `load_state()` afterwards to compute diagnostics and
        write output. That replay has to be lossless -- the snapshot is a
        standalone copy, so stepping past it and coming back must restore the
        field and its total heat bit-for-bit.
        """
        solver = self._make_solver(self._linear_temperature_field())
        for _ in range(3):
            solver.step()

        snapshot = solver.snapshot_state()
        snapshot_temperatures = state_array(solver)
        snapshot_heat = solver.compute_total_heat()

        for _ in range(5):
            solver.step()
        self.assertGreater(
            np.abs(state_array(solver) - snapshot_temperatures).max(), 0.0
        )

        solver.load_state(snapshot)
        # A vector copy: restoration is exact, so this is asserted bitwise. Any
        # tolerance here would hide an aliasing bug (e.g. a snapshot that is a
        # view of the live state rather than a copy).
        np.testing.assert_array_equal(state_array(solver), snapshot_temperatures)
        self.assertEqual(solver.compute_total_heat(), snapshot_heat)
        snapshot.destroy()

    def test_diagnostics_between_steps_do_not_perturb_the_trajectory(self):
        """
        The stepping state deliberately lives in plain PETSc vectors, decoupled
        from `u_n`: reading `solution` or `compute_total_heat()` between steps
        syncs *into* the Function and must never feed back. An earlier design did
        feed back -- taking snapshots mid-run silently corrupted the results,
        which is why `run_fem_heat_simulation` defers all form assembly to after
        the loop. This test locks the decoupling in, so that a future refactor
        cannot quietly reintroduce it.
        """
        num_steps = 10

        undisturbed = self._make_solver(self._linear_temperature_field(), source=4.0)
        for _ in range(num_steps):
            undisturbed.step()

        observed = self._make_solver(self._linear_temperature_field(), source=4.0)
        for _ in range(num_steps):
            observed.step()
            observed.compute_total_heat()
            observed.solution.x.array[0]
            observed.snapshot_state().destroy()

        # The two solvers execute the same sequence of operations, but MUMPS is
        # only bitwise reproducible at OMP_NUM_THREADS=1: with threading on, two
        # identical runs in one process already differ by ~7e-16 relative, so a
        # bitwise assertion would be flaky. 1e-13 sits three orders above that
        # floor and still an order of magnitude *below* anything a feedback path
        # could cause -- the historical version of this bug produced visibly
        # wrong peak temperatures, not last-bit noise.
        np.testing.assert_allclose(
            state_array(observed), state_array(undisturbed), rtol=1e-13
        )


if __name__ == "__main__":
    unittest.main()
