import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc

from fem.dolfinx_mesh import create_manifold_mesh
from fem.fem_field_data import FEMFieldData
from fem.fem_mesh_data import FEMMeshData
from fem.heat.crank_nicolson_fem_scheme import CrankNicolsonFEMScheme
from fem.heat.fem_heat_problem import FEMHeatProblem
from fem.heat.fem_heat_solution import FEMHeatSolution
from fem.heat.fem_simulation_config import FEMSimulationConfig
from fem.heat.fem_simulation_diagnostics import FEMSimulationDiagnostics
from fem.heat.material_functions import MaterialFunctions
from fem.heat.snapshot_diagnostics import SnapshotDiagnostics
from fem.heat.vertex_interpolant import VertexInterpolant
from fem.xdmf_time_series import XDMFTimeSeries
from forge_sim import logger

DiagnosticsCallback = Callable[[SnapshotDiagnostics], None]
"""
Invoked once per saved snapshot, in time order, from `FEMHeatRun.march`'s
post-loop diagnostics pass (never from the stepping loop).
"""


class FEMHeatRun:
    """
    One DOLFINx Crank-Nicolson heat run: the mesh, the space, the fields, the
    problem and its scheme.

    Construction performs the whole setup: the DOLFINx mesh, the Lagrange space the
    temperature lives in, the nearest-vertex interpolant, the initial condition, the
    five material functions, the `FEMHeatProblem` and its `CrankNicolsonFEMScheme`
    with its resolved time step, and the `target_delta_temp` source scaling, in
    that order, and holds nothing a caller may reassign afterwards. `march` then
    runs the time loop and returns the series as a value; nothing is left on the
    instance for a second call to read.

    Two callers of `march` exist and they differ in exactly two things, which is
    why this is a class rather than two copies of the sequence: which vertex
    ordering the snapshots are gathered onto (`FEMHeatSolution.gather_at`), and
    what is done with them afterwards. `run_with_xdmf` writes an XDMF series in
    the mesh's own order; `fem/end_to_end/heat_comparison/scene_heat_comparison.py`
    gathers onto the IGA export mesh's order and diffs.

    The problem's stepping state advances during `march`, which is the one piece of
    mutability here and is the point of a time integrator: what is fixed at
    construction is the assembled operators, the factorization and the resolved dt,
    and none of those is re-derived. This mirrors the IGA-side split between
    `HeatSimulation` and `CrankNicolsonScheme`.
    """

    def __init__(
        self,
        mesh_data: FEMMeshData,
        field_data: FEMFieldData,
        config: FEMSimulationConfig,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        """
        Builds the DOLFINx mesh, the material functions and the Crank-Nicolson
        problem and scheme for one heat run, in the order this class's docstring
        pins.

        `mesh_data` is the triangle-mesh geometry and `field_data` the vertex-based
        initial temperature and material fields sampled on it, typically both
        from `SplineToFEMProjector`, though nothing here reads a spline directly.
        `config` resolves the time step and the output/discretization options, and
        `comm` is the MPI communicator.

        :raises ValueError: If `config` resolves to a non-positive time step.
        :raises RuntimeError: If PETSc lacks the required factorization backend, or
            if a Cholesky factorization cannot be set up.
        :raises ImportError: If the resolved step comes from `dt_over_t_relax` and
            slepc4py is not available.
        """
        self._mesh_data = mesh_data
        self._config = config
        self._comm = comm

        self._mesh = self.create_dolfinx_mesh(mesh_data, comm)
        self._function_space = fem.functionspace(
            self._mesh, ("Lagrange", config.fem_degree)
        )
        if comm.rank == 0:
            logger.info(
                f"Mesh: {mesh_data.vertices.shape[0]} vertices, "
                f"{mesh_data.faces.shape[0]} faces. "
                f"FEM degree: P{config.fem_degree}, "
                f"DOFs: {self._function_space.dofmap.index_map.size_global}."
            )

        interpolate = self.vertex_interpolant(mesh_data.vertices)
        initial_temperature = interpolate(self._function_space, field_data.temperature)
        material_funcs = self.material_functions(
            field_data, self._function_space, interpolate
        )

        self._problem, self._scheme, self._relaxation_time = self._build_scheme(
            initial_temperature, material_funcs
        )
        self._applied_heat_source = self._apply_target_delta_temp_scaling()

    @staticmethod
    def create_dolfinx_mesh(
        mesh_data: FEMMeshData,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> mesh.Mesh:
        """
        Builds a DOLFINx mesh directly from triangle-mesh geometry.

        The coordinate element is a P1 Lagrange triangle with a 3-vector value
        shape, which is what makes this a 2D manifold embedded in 3-space.

        DOLFINx permutes the vertex ordering relative to the input, so field
        interpolation matches by nearest vertex; see `vertex_interpolant`.

        The direct route agrees with a meshio -> `.msh` -> `gmshio.model_to_mesh`
        round-trip on identical geometry, identical P1 dof coordinates and a
        bitwise-identical assembled stiffness operator, on a single rank.

        `mesh_data` supplies the vertices and faces of the tessellated surface;
        it must be the same on every rank, since only rank 0's copy is read;
        `create_manifold_mesh` explains why. `comm` is the MPI communicator; the
        FEM heat pipeline runs serially, so the single-rank path is the one
        exercised in anger, and the rank-0 gating in `create_manifold_mesh` is
        what keeps the multi-rank path from silently duplicating.

        :returns: DOLFINx mesh (topological dim 2, geometric dim 3).
        """
        return create_manifold_mesh(
            comm=comm,
            cells=mesh_data.faces,
            points=mesh_data.vertices,
            degree=1,
        )

    @staticmethod
    def vertex_interpolant(
        original_vertices: npt.NDArray[np.float64],
    ) -> VertexInterpolant:
        """
        Builds the interpolant that maps vertex-based field values onto a
        DOLFINx function space by nearest-vertex matching, as used by
        `material_functions` and the initial temperature.

        :returns: A `VertexInterpolant` bound to `original_vertices`; see that
            class for the matching rule.
        """
        return VertexInterpolant(original_vertices)

    @staticmethod
    def material_functions(
        field_data: FEMFieldData,
        function_space: fem.FunctionSpace,
        interpolate: VertexInterpolant,
    ) -> MaterialFunctions:
        """
        Interpolates the five material fields of `field_data` onto `function_space`.

        The fields are named one by one rather than reflected over `FEMFieldData`,
        which also carries the initial temperature, not a material property, and
        interpolated separately by the caller.
        """
        return MaterialFunctions(
            conductivity=interpolate(function_space, field_data.conductivity),
            density=interpolate(function_space, field_data.density),
            specific_heat=interpolate(function_space, field_data.specific_heat),
            thickness=interpolate(function_space, field_data.thickness),
            source=interpolate(function_space, field_data.source),
        )

    def march(
        self,
        *,
        retain_snapshots: bool = True,
        diagnostics_callback: DiagnosticsCallback | None = None,
    ) -> FEMHeatSolution:
        """
        Runs the time loop and returns the snapshot series as a value.

        Snapshots are taken at t = 0, at every `config.save_interval`-th step, and
        always at the final step even when it is not a multiple of the interval.
        The hot loop does nothing but advance the scheme and copy out snapshots as
        plain numpy arrays; all DOLFINx form assembly (the energy-drift
        diagnostics below) and XDMF I/O are deferred to after it, since performing
        those on the problem's function between Crank-Nicolson steps corrupted the
        time-stepping whenever a snapshot fell on an intermediate step.

        `retain_snapshots` selects whether the returned solution keeps the
        snapshot arrays, combined here with the rank check since only rank 0
        gathers or writes; discarding them on a diagnostics-only run saves
        `n_snapshots * n_local_dofs * 8` bytes for nothing. `diagnostics_callback`,
        when given, is invoked once per saved snapshot, in time order, from the
        post-loop pass below (never from the stepping loop), and only on rank
        0, mirroring the rank-guarded logging, so it must not perform collective
        MPI operations.

        :returns: The run's diagnostics, and the snapshot series (empty unless
            `retain_snapshots` and rank 0) with the dof coordinates it can be
            resampled by.
        :raises RuntimeError: If `retain_snapshots` and run on more than one
            rank: `dof_coordinates` includes ghost dofs while a snapshot
            holds owned dofs only, and there is no MPI gather in `fem/heat`
            to reconcile the two.
        """
        if retain_snapshots and self._comm.size > 1:
            raise RuntimeError(
                "FEMHeatRun.march with retain_snapshots=True is implemented "
                "for serial runs only (MPI.COMM_SELF); dof_coordinates "
                "includes ghost dofs while each snapshot holds owned dofs "
                "only, and fem/heat has no MPI gather to reconcile them."
            )

        if self._comm.rank == 0:
            logger.info(
                f"Simulation: dt={self._scheme.dt:.6g} s, "
                f"steps={self._config.num_time_steps}, "
                f"T_final={self._config.num_time_steps * self._scheme.dt:.6g} s."
            )

        save_steps = {
            step
            for step in range(1, self._config.num_time_steps + 1)
            if step % self._config.save_interval == 0
            or step == self._config.num_time_steps
        }
        # Snapshots are standalone PETSc-vector copies of the problem's internal
        # state (problem.snapshot_state()), taken during stepping. They never touch
        # a fem.Function, so collecting them mid-run cannot disturb the
        # integration.
        snapshot_times = [0.0]
        snapshot_vectors = [self._problem.snapshot_state()]

        start_time = time.time()
        for step in range(1, self._config.num_time_steps + 1):
            self._scheme.step()
            if step in save_steps:
                snapshot_times.append(step * self._scheme.dt)
                snapshot_vectors.append(self._problem.snapshot_state())
        elapsed = time.time() - start_time

        if self._comm.rank == 0:
            logger.info(
                f"Completed {self._config.num_time_steps} steps in {elapsed:.2f} s "
                f"({elapsed / self._config.num_time_steps:.4f} s/step)."
            )

        heat_generation_rate = self._problem.compute_heat_generation_rate()
        if self._comm.rank == 0:
            logger.info(f"Heat generation rate: {heat_generation_rate:.6e} W")

        snapshot_diagnostics, snapshot_arrays = self._replay_snapshots(
            snapshot_times,
            snapshot_vectors,
            heat_generation_rate,
            diagnostics_callback,
            retain_snapshots and self._comm.rank == 0,
        )

        diagnostics = FEMSimulationDiagnostics(
            dt=self._scheme.dt,
            relaxation_time=self._relaxation_time,
            heat_generation_rate=heat_generation_rate,
            applied_heat_source=self._applied_heat_source,
            snapshots=tuple(snapshot_diagnostics),
        )
        return FEMHeatSolution(
            diagnostics=diagnostics,
            times=tuple(snapshot_times),
            snapshots=tuple(snapshot_arrays),
            dof_coordinates=self._function_space.tabulate_dof_coordinates(),
        )

    def run_with_xdmf(
        self,
        output_folder: Path,
        xdmf_name: str,
        diagnostics_callback: DiagnosticsCallback | None = None,
    ) -> FEMSimulationDiagnostics:
        """
        Marches this run and, when `config.write_xdmf`, writes the result to XDMF.

        The FEM counterpart to the IGA `HeatSimulationRunner.run_heat_simulation_
        with_xdmf`: marches this run and, when `config.write_xdmf`, writes the
        resulting series to XDMF in the mesh's own vertex order, gathered from
        PETSc dof order by nearest-dof matching. Output goes through our own
        writer rather than DOLFINx `XDMFFile.write_function`, which corrupts
        repeated snapshot writes.

        The single- and two-material spline entry scripts differ only in how they
        project the material fields onto the mesh's vertices; everything
        downstream is this one method. `output_folder` is the directory for
        outputs (created if needed), and `xdmf_name` the file name for the XDMF
        output within it. `diagnostics_callback` is an optional hook receiving the
        energy diagnostics of each saved snapshot, in time order, as they are
        reported (see `march`).

        :returns: The run's diagnostics (step size, relaxation time, source power,
            and the per-snapshot energies), so a caller can report them without
            reaching into the solver.
        """
        output_folder.mkdir(parents=True, exist_ok=True)

        solution = self.march(
            retain_snapshots=self._config.write_xdmf,
            diagnostics_callback=diagnostics_callback,
        )

        if self._config.write_xdmf and self._comm.rank == 0:
            output_path = output_folder / xdmf_name
            XDMFTimeSeries.write(
                output_path,
                self._mesh_data.vertices,
                self._mesh_data.faces,
                solution.times,
                solution.gather_at(self._mesh_data.vertices),
                attribute_name="temperature",
                layout=XDMFTimeSeries.Layout.COMPACT,
                declare_float_precision=True,
            )
            logger.info(f"Solution written to {output_path}.")

        return solution.diagnostics

    def _build_scheme(
        self, initial_temperature: fem.Function, material_funcs: MaterialFunctions
    ) -> tuple[FEMHeatProblem, CrankNicolsonFEMScheme, float | None]:
        """
        Builds the problem and its Crank-Nicolson scheme, resolving the time
        step by precedence: `final_time` (dt = final_time / num_time_steps) >
        explicit `time_step_size` > normalized `dt_over_t_relax`.
        `FEMSimulationConfig.__post_init__` has already checked that exactly
        one of `time_step_size` and `dt_over_t_relax` is set when `final_time`
        is None.

        The problem is assembled once, up front, and shared by all three
        branches; `from_normalized_time_step` reads its (K, M) for the
        eigensolve directly rather than reassembling them.

        :returns: The problem, the scheme, and the relaxation time T_relax [s]
            the scheme was built from, which is None for the two branches that
            never compute one. Reporting it from here keeps `march` from
            touching `scheme.relaxation_time`, whose lazy getter would run the
            eigensolve those branches deliberately skip.
        """
        problem = FEMHeatProblem(
            msh=self._mesh,
            V=self._function_space,
            u0=initial_temperature,
            material_funcs=material_funcs,
            comm=self._comm,
        )

        if self._config.final_time is not None:
            dt = self._config.final_time / self._config.num_time_steps
            logger.info(
                f"final_time override: dt = {self._config.final_time:.6g} / "
                f"{self._config.num_time_steps} = {dt:.6g} s (eigensolve skipped)."
            )
            return (
                problem,
                CrankNicolsonFEMScheme(problem=problem, dt=dt, comm=self._comm),
                None,
            )

        if self._config.time_step_size is not None:
            return (
                problem,
                CrankNicolsonFEMScheme(
                    problem=problem,
                    dt=self._config.time_step_size,
                    comm=self._comm,
                ),
                None,
            )

        scheme = CrankNicolsonFEMScheme.from_normalized_time_step(
            problem=problem,
            dt_over_t_relax=self._config.dt_over_t_relax,
            comm=self._comm,
        )
        # Already cached by `from_normalized_time_step`, so this triggers no solve.
        return problem, scheme, scheme.relaxation_time

    def _apply_target_delta_temp_scaling(self) -> float | None:
        """
        Scales the source so an insulated source element rises by
        `config.target_delta_temp` over the full simulation, matching the IGA
        runner.

        The required heat source (insulated thin-shell element, no lateral
        conduction) is `Q = dT * rho * c * h / t_sim`, with `t_sim = num_steps *
        dt` and (rho, c, h) taken from `config.heat_source_material`.
        `FEMSimulationConfig.__post_init__` has already checked that a material
        is given whenever `target_delta_temp` is set.

        :returns: The applied heat source Q in W/m^2, or None when
            `config.target_delta_temp` is unset and the source is therefore left
            at whatever the projected source field holds (there is no single
            scalar Q in that case, since the field may vary per vertex).
            Returning it lets callers report Q without re-deriving the formula.
        """
        if self._config.target_delta_temp is None:
            return None
        assert self._config.heat_source_material is not None

        t_sim = self._config.num_time_steps * self._scheme.dt
        material = self._config.heat_source_material
        thermal_mass = material.density * material.specific_heat * material.thickness
        heat_source = self._config.target_delta_temp * thermal_mass / t_sim

        self._scheme.scale_source(heat_source)
        logger.info(
            f"target_delta_temp={self._config.target_delta_temp:.4g} K -> "
            f"Q={heat_source:.6g} W/m^2 (t_sim={t_sim:.6g} s, "
            f"thermal_mass=rho*c*h={thermal_mass:.6g} J/(m^2*K))."
        )
        return heat_source

    def _replay_snapshots(
        self,
        snapshot_times: list[float],
        snapshot_vectors: list[PETSc.Vec],
        heat_generation_rate: float,
        diagnostics_callback: DiagnosticsCallback | None,
        retain_snapshots: bool,
    ) -> tuple[list[SnapshotDiagnostics], list[npt.NDArray[np.float64]]]:
        """
        Loads each stored snapshot back into the problem and reports its energy.

        Stepping is finished by the time this runs, so the problem's function may
        be reused as the carrier for each snapshot; assembling the heat form on
        it is harmless now, and was not during the loop. The energy drift is
        measured against what the source alone accounts for, `H(0) + t * Q_total`.

        Each snapshot vector is destroyed once its values have been copied out, so
        the series never exists twice. `retain_snapshots` is already the combined
        "keep the arrays and this is rank 0" decision `march` made; logging and
        the callback are gated on rank 0 independently, since a non-zero rank
        still wants the drift computed even when it discards the array.

        :returns: The per-snapshot diagnostics and the snapshot values as numpy
            arrays, both in time order. The arrays are in PETSc dof order, and the
            list is empty when `retain_snapshots` is False.
        """
        initial_heat = None
        snapshot_diagnostics: list[SnapshotDiagnostics] = []
        snapshot_arrays: list[npt.NDArray[np.float64]] = []

        for snapshot_time, snapshot_vector in zip(snapshot_times, snapshot_vectors):
            self._problem.load_state(snapshot_vector)
            # Also syncs the state into u_n.
            total_heat = self._problem.compute_total_heat()
            if initial_heat is None:
                initial_heat = total_heat
            expected_heat = initial_heat + snapshot_time * heat_generation_rate
            relative_drift = (
                (total_heat - expected_heat) / expected_heat
                if expected_heat != 0
                else 0.0
            )

            diagnostics = SnapshotDiagnostics(
                time=snapshot_time,
                total_heat=total_heat,
                relative_drift=relative_drift,
            )
            snapshot_diagnostics.append(diagnostics)
            if self._comm.rank == 0:
                logger.info(
                    f"t={snapshot_time:.6g} s: H={total_heat:.6e} J, "
                    f"drift={relative_drift:+.2e}."
                )
                if diagnostics_callback is not None:
                    diagnostics_callback(diagnostics)

            if retain_snapshots:
                snapshot_arrays.append(snapshot_vector.getArray().copy())
            snapshot_vector.destroy()

        return snapshot_diagnostics, snapshot_arrays
