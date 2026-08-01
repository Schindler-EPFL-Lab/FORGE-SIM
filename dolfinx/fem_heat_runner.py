"""
Shared orchestration for DOLFINx FEM heat simulations on spline surfaces.

This module is the FEM counterpart to the IGA `HeatSimulationRunner`. It takes a
`SplineToFEMProjector` and the projected field data, builds the DOLFINx mesh and
material functions, sets up the Crank-Nicolson solver (with an explicit or a
relaxation-time-normalized time step), runs the time loop with energy-drift
reporting and XDMF export, and optionally visualizes the result.

The single- and two-material entry-point scripts differ only in how they project
the material fields; everything downstream is shared here.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import basix.ufl
import numpy as np
import ufl
from scipy.spatial import cKDTree

from dolfinx import fem, mesh
from mpi4py import MPI

from rebel_shape import logger
from rebel_shape.scripts.configs.thermal_material_config import ThermalMaterialConfig

from crank_nicolson_fem_solver import CrankNicolsonFEMSolver
from spline_to_fem_projector import FEMFieldData, FEMMeshData, SplineToFEMProjector
from visualization import visualize_temperature_field
from xdmf_time_series import XdmfLayout, write_time_series_xdmf

# Material-property keys expected by `CrankNicolsonFEMSolver`, in `FEMFieldData`.
_MATERIAL_FIELDS = ("conductivity", "density", "specific_heat", "thickness", "source")


@dataclass
class FEMSimulationConfig:
    """
    Time-stepping, output, and discretization options for an FEM heat run.

    The time step is resolved in this precedence order: `final_time` (dt =
    final_time / num_time_steps) > `time_step_size` (explicit dt) > the
    `dt_over_t_relax` normalized step. When `final_time` is None, exactly one of
    `time_step_size` and `dt_over_t_relax` must be set. `final_time` is the knob
    for matching a reference run's end time exactly (e.g. an IGA XDMF), which
    also skips the relaxation eigensolve.

    :ivar num_time_steps: Number of Crank-Nicolson steps.
    :ivar save_interval: Steps between XDMF snapshots / drift reports.
    :ivar final_time: Total simulated time [s] to hit exactly, or None.
    :ivar time_step_size: Explicit physical time step [s], or None.
    :ivar dt_over_t_relax: Time step normalized by `T_relax`, or None.
    :ivar target_delta_temp: Target temperature rise [K]; when set the source is
        scaled so an insulated source element rises by this much over the run.
    :ivar heat_source_material: Material whose (rho, c, h) define the source
        scaling; required iff `target_delta_temp` is set.
    :ivar fem_degree: Lagrange polynomial degree (1 = P1).
    :ivar write_xdmf: Whether to write the solution to XDMF.
    :ivar show_visualization: Whether to show the interactive PyVista viewer.
    """

    num_time_steps: int
    save_interval: int
    time_step_size: float | None
    dt_over_t_relax: float | None
    final_time: float | None = None
    target_delta_temp: float | None = None
    heat_source_material: ThermalMaterialConfig | None = None
    fem_degree: int = 1
    write_xdmf: bool = True
    show_visualization: bool = False


@dataclass(frozen=True)
class SnapshotDiagnostics:
    """
    Energy bookkeeping for one saved snapshot of the run.

    `total_heat` is the assembled thermal energy H = integral(rho c d u) dA and
    `relative_drift` is its departure from the energy the source alone accounts
    for, (H - H_expected) / H_expected with H_expected = H(0) + t * Q_total. Both
    are global (MPI-reduced) values.

    :ivar time: Snapshot time [s], measured from the start of the run.
    :ivar total_heat: Total thermal energy at that time [J].
    :ivar relative_drift: Relative energy drift, dimensionless (0.0 when the
        expected energy is exactly zero).
    """

    time: float
    total_heat: float
    relative_drift: float


@dataclass(frozen=True)
class FEMSimulationDiagnostics:
    """
    What a completed FEM heat run reports about itself.

    Returned by `run_fem_heat_simulation` so a caller can log or record the run's
    characterization without reaching into the solver (or re-deriving the source
    scaling). Values are global, so every MPI rank receives the same record.

    :ivar dt: Physical time step actually used [s], however it was resolved.
    :ivar relaxation_time: Natural relaxation time T_relax [s] when the step was
        resolved from `dt_over_t_relax`, else None -- the other paths never
        compute it, and it is deliberately not triggered here just to report it
        (the eigensolve is expensive).
    :ivar heat_generation_rate: Total source power integral(Q) dA [W].
    :ivar applied_heat_source: The uniform source Q [W/m^2] imposed by
        `target_delta_temp` scaling, or None when no such scaling was applied.
    :ivar snapshots: Per-snapshot energy diagnostics, in time order, starting at
        t = 0.
    """

    dt: float
    relaxation_time: float | None
    heat_generation_rate: float
    applied_heat_source: float | None
    snapshots: tuple[SnapshotDiagnostics, ...]


# Invoked once per saved snapshot, in time order, from the post-loop diagnostics
# pass (never from the stepping loop -- see `run_fem_heat_simulation`).
DiagnosticsCallback = Callable[[SnapshotDiagnostics], None]


def create_dolfinx_mesh(
    mesh_data: FEMMeshData,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> mesh.Mesh:
    """
    Builds a DOLFINx mesh directly from triangle-mesh geometry.

    The coordinate element is a P1 Lagrange triangle with a 3-vector value
    shape, which is what makes this a 2D manifold embedded in 3-space. This
    replaces an earlier meshio -> temporary `.msh` -> `gmsh.merge` ->
    `gmshio.model_to_mesh` round-trip; the two were verified to produce
    identical geometry, identical P1 dof coordinates and a bitwise-identical
    assembled stiffness operator, on a single rank.

    DOLFINx permutes the vertex ordering relative to the input (as the gmsh
    route did too), so field interpolation matches by nearest vertex -- see
    `_build_vertex_interpolant`.

    `mesh_data` is not MPI-aware: every rank holds the same complete mesh. Handing
    that whole copy to `create_mesh` on every rank would make each one contribute
    the full topology, so an N-rank run would build N duplicates of the mesh with
    no error raised. Only rank 0 therefore supplies the cells and points; the rest
    supply empty arrays and receive their share from the partitioner. This mirrors
    what the previous `gmshio.model_to_mesh` route did by passing the gmsh model
    on rank 0 alone.

    :param mesh_data: Vertices and faces of the tessellated surface. Must be the
        same on every rank; only rank 0's copy is read.
    :param comm: MPI communicator. The FEM heat pipeline runs serially, so the
        single-rank path is the one exercised in anger; the rank-0 gating above
        is what keeps the multi-rank path from silently duplicating.
    :returns: DOLFINx mesh (topological dim 2, geometric dim 3).
    """
    coordinate_element = basix.ufl.element("Lagrange", "triangle", 1, shape=(3,))
    if comm.rank == 0:
        cells = mesh_data.faces.astype(np.int64)
        points = mesh_data.vertices
    else:
        cells = np.empty((0, 3), dtype=np.int64)
        points = np.empty((0, 3), dtype=mesh_data.vertices.dtype)
    return mesh.create_mesh(comm, cells, points, ufl.Mesh(coordinate_element))


def _build_vertex_interpolant(original_vertices: np.ndarray):
    """
    Builds an interpolant that maps vertex-based field values onto a DOLFINx
    function by nearest-vertex matching.

    A single KD-tree of the original tessellation vertices is built once and
    reused for every field. For each function DOF coordinate, the value of the
    nearest original vertex is assigned. For P1 this is exact (DOFs are
    vertices); for higher degrees the extra DOFs take their nearest vertex value.

    :param original_vertices: Tessellation vertices, shape (n_vertices, 3).
    :returns: A callable `(V, values) -> fem.Function`.
    """
    tree = cKDTree(original_vertices)

    def interpolate(V: fem.FunctionSpace, values: np.ndarray) -> fem.Function:
        function = fem.Function(V)

        def evaluate(coords: np.ndarray) -> np.ndarray:
            _, indices = tree.query(coords.T, k=1)
            return values[indices]

        function.interpolate(evaluate)
        return function

    return interpolate


def _build_solver(
    dolfinx_mesh: mesh.Mesh,
    V: fem.FunctionSpace,
    u0: fem.Function,
    material_funcs: dict[str, fem.Function],
    config: FEMSimulationConfig,
    comm: MPI.Comm,
) -> tuple[CrankNicolsonFEMSolver, float | None]:
    """
    Builds the Crank-Nicolson solver, resolving the time step by precedence:
    `final_time` (dt = final_time / num_time_steps) > explicit `time_step_size` >
    normalized `dt_over_t_relax`. When `final_time` is None, exactly one of
    `time_step_size` and `dt_over_t_relax` must be set.

    :returns: The solver and the relaxation time T_relax [s] it was built from,
        which is None for the two branches that never compute one. Reporting it
        from here keeps callers from touching `solver.relaxation_time`, whose
        lazy getter would run the eigensolve those branches deliberately skip.
    """
    if config.final_time is not None:
        dt = config.final_time / config.num_time_steps
        logger.info(
            f"final_time override: dt = {config.final_time:.6g} / "
            f"{config.num_time_steps} = {dt:.6g} s (eigensolve skipped)."
        )
        return (
            CrankNicolsonFEMSolver(
                msh=dolfinx_mesh,
                V=V,
                u0=u0,
                material_funcs=material_funcs,
                dt=dt,
                comm=comm,
            ),
            None,
        )

    if (config.time_step_size is None) == (config.dt_over_t_relax is None):
        raise ValueError(
            "Exactly one of `time_step_size` and `dt_over_t_relax` must be set "
            "when `final_time` is not given."
        )

    if config.time_step_size is not None:
        return (
            CrankNicolsonFEMSolver(
                msh=dolfinx_mesh,
                V=V,
                u0=u0,
                material_funcs=material_funcs,
                dt=config.time_step_size,
                comm=comm,
            ),
            None,
        )

    solver = CrankNicolsonFEMSolver.from_normalized_time_step(
        msh=dolfinx_mesh,
        V=V,
        u0=u0,
        material_funcs=material_funcs,
        dt_over_t_relax=config.dt_over_t_relax,
        comm=comm,
    )
    # Already cached by `from_normalized_time_step`, so this triggers no solve.
    return solver, solver.relaxation_time


def _apply_target_delta_temp_scaling(
    solver: CrankNicolsonFEMSolver,
    config: FEMSimulationConfig,
) -> float | None:
    """
    Scales the source so an insulated source element rises by
    `config.target_delta_temp` over the full simulation, matching the IGA runner.

    The required heat source (insulated thin-shell element, no lateral
    conduction) is `Q = dT * rho * c * h / t_sim`, with `t_sim = num_steps * dt`
    and (rho, c, h) taken from `config.heat_source_material`.

    :returns: The applied heat source Q in W/m^2, or None when
        `config.target_delta_temp` is unset and the source is therefore left at
        whatever the projected source field holds (there is no single scalar Q in
        that case, since the field may vary per vertex). Returning it lets
        callers report Q without re-deriving the formula.
    """
    if config.target_delta_temp is None:
        return None
    if config.heat_source_material is None:
        raise ValueError(
            "`heat_source_material` must be provided when `target_delta_temp` "
            "is set."
        )

    t_sim = config.num_time_steps * solver.dt
    material = config.heat_source_material
    thermal_mass = material.density * material.specific_heat * material.thickness
    heat_source = config.target_delta_temp * thermal_mass / t_sim

    solver.scale_source(heat_source)
    logger.info(
        f"target_delta_temp={config.target_delta_temp:.4g} K -> "
        f"Q={heat_source:.6g} W/m^2 (t_sim={t_sim:.6g} s, "
        f"thermal_mass=rho*c*h={thermal_mass:.6g} J/(m^2*K))."
    )
    return heat_source


def run_fem_heat_simulation(
    projector: SplineToFEMProjector,
    field_data: FEMFieldData,
    output_folder: Path,
    xdmf_name: str,
    config: FEMSimulationConfig,
    comm: MPI.Comm = MPI.COMM_WORLD,
    diagnostics_callback: DiagnosticsCallback | None = None,
) -> FEMSimulationDiagnostics:
    """
    Runs the FEM heat simulation on a projected spline surface.

    Builds the DOLFINx mesh and material functions from `projector`/`field_data`,
    sets up the Crank-Nicolson solver per `config`, runs the time loop with
    energy-drift reporting and optional XDMF export, and optionally visualizes
    the temperature evolution.

    :param projector: The spline-to-FEM projector (provides the mesh geometry).
    :param field_data: Field values projected onto the mesh vertices.
    :param output_folder: Directory for outputs (created if needed).
    :param xdmf_name: File name for the XDMF output within `output_folder`.
    :param config: Time-stepping, output, and discretization options.
    :param comm: MPI communicator.
    :param diagnostics_callback: Optional hook receiving the energy diagnostics
        of each saved snapshot, in time order, as they are reported. It is
        invoked from the post-loop diagnostics pass -- never from the stepping
        loop, which must stay free of DOLFINx form assembly (see below) -- and
        only on rank 0, mirroring the rank-guarded logging, so it must not
        perform collective MPI operations.
    :returns: The run's diagnostics (step size, relaxation time, source power,
        and the per-snapshot energies), so a caller can report them without
        reaching into the solver.
    """
    rank = comm.rank
    output_folder.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ mesh
    mesh_data = projector.get_mesh_data()
    dolfinx_mesh = create_dolfinx_mesh(mesh_data, comm)
    V = fem.functionspace(dolfinx_mesh, ("Lagrange", config.fem_degree))

    if rank == 0:
        logger.info(
            f"Mesh: {mesh_data.vertices.shape[0]} vertices, "
            f"{mesh_data.faces.shape[0]} faces. "
            f"FEM degree: P{config.fem_degree}, "
            f"DOFs: {V.dofmap.index_map.size_global}."
        )

    # ----------------------------------------------------- functions & solver
    interpolate = _build_vertex_interpolant(mesh_data.vertices)
    u0 = interpolate(V, field_data.temperature)
    material_funcs = {
        name: interpolate(V, getattr(field_data, name)) for name in _MATERIAL_FIELDS
    }

    solver, relaxation_time = _build_solver(
        dolfinx_mesh, V, u0, material_funcs, config, comm
    )
    applied_heat_source = _apply_target_delta_temp_scaling(solver, config)

    if rank == 0:
        logger.info(
            f"Simulation: dt={solver.dt:.6g} s, steps={config.num_time_steps}, "
            f"T_final={config.num_time_steps * solver.dt:.6g} s."
        )

    # ----------------------------------------------------------- time loop
    # The hot loop does nothing but advance the solver and copy out snapshots as
    # plain numpy arrays. ALL DOLFINx form assembly (heat diagnostics) and XDMF
    # I/O is deferred to after the loop: performing those on the live solver
    # function between Crank-Nicolson steps corrupted the time-stepping (the
    # solution came out wrong whenever snapshots were taken at intermediate
    # steps, i.e. save_interval < num_time_steps). Keeping the stepping loop free
    # of DOLFINx form/IO calls avoids that, and is cleaner regardless.
    save_steps = {
        step
        for step in range(1, config.num_time_steps + 1)
        if step % config.save_interval == 0 or step == config.num_time_steps
    }
    # Snapshots are standalone PETSc-vector copies of the solver's internal
    # state (solver.snapshot_state()), taken during stepping. They never touch a
    # fem.Function, so collecting them mid-run cannot disturb the integration.
    snapshot_times = [0.0]
    snapshot_vecs = [solver.snapshot_state()]

    start_time = time.time()
    for step in range(1, config.num_time_steps + 1):
        solver.step()
        if step in save_steps:
            snapshot_times.append(step * solver.dt)
            snapshot_vecs.append(solver.snapshot_state())
    elapsed = time.time() - start_time

    if rank == 0:
        logger.info(
            f"Completed {config.num_time_steps} steps in {elapsed:.2f} s "
            f"({elapsed / config.num_time_steps:.4f} s/step)."
        )

    # ------------------------------------------ post-loop: diagnostics + output
    # Stepping is finished, so the solver function may be reused as the carrier
    # for each stored snapshot (writing/assembling on it is now harmless).
    heat_generation_rate = solver.compute_heat_generation_rate()
    if rank == 0:
        logger.info(f"Heat generation rate: {heat_generation_rate:.6e} W")

    # Output goes through our own writer (write_time_series_xdmf), NOT DOLFINx
    # XDMFFile.write_function (which corrupts repeated snapshot writes). Snapshot
    # values are in PETSc dof order; gather them into tessellation-vertex order
    # via nearest-dof matching for output.
    vertex_to_dof = None
    output_fields = []
    if config.write_xdmf and rank == 0:
        vertex_to_dof = cKDTree(V.tabulate_dof_coordinates()).query(
            mesh_data.vertices, k=1
        )[1]

    frames = [] if (rank == 0 and config.show_visualization) else None
    times = [] if (rank == 0 and config.show_visualization) else None

    initial_heat = None
    snapshot_diagnostics: list[SnapshotDiagnostics] = []
    for snapshot_time, snapshot_vec in zip(snapshot_times, snapshot_vecs):
        solver.load_state(snapshot_vec)
        total_heat = solver.compute_total_heat()  # also syncs the state into u_n
        if initial_heat is None:
            initial_heat = total_heat
        expected_heat = initial_heat + snapshot_time * heat_generation_rate
        relative_drift = (
            (total_heat - expected_heat) / expected_heat if expected_heat != 0 else 0.0
        )
        diagnostics = SnapshotDiagnostics(
            time=snapshot_time,
            total_heat=total_heat,
            relative_drift=relative_drift,
        )
        snapshot_diagnostics.append(diagnostics)
        if rank == 0:
            logger.info(
                f"t={snapshot_time:.6g} s: H={total_heat:.6e} J, "
                f"drift={relative_drift:+.2e}."
            )
            if diagnostics_callback is not None:
                diagnostics_callback(diagnostics)
        snapshot_array = snapshot_vec.getArray().copy()
        if config.write_xdmf and rank == 0:
            output_fields.append(snapshot_array[vertex_to_dof])
        if config.show_visualization and rank == 0:
            frames.append(snapshot_array)
            times.append(snapshot_time)
        snapshot_vec.destroy()

    if config.write_xdmf and rank == 0:
        # `output_folder` was created above, hence no parent-directory creation.
        write_time_series_xdmf(
            output_folder / xdmf_name,
            mesh_data.vertices,
            mesh_data.faces,
            snapshot_times,
            output_fields,
            attribute_name="temperature",
            layout=XdmfLayout.COMPACT,
            declare_float_precision=True,
        )
        logger.info(f"Solution written to {output_folder / xdmf_name}.")

    # --------------------------------------------------------- visualization
    if rank == 0 and config.show_visualization:
        visualize_temperature_field(
            V=V,
            u=solver.u_n,
            frames=frames,
            times=times,
            show=True,
        )

    return FEMSimulationDiagnostics(
        dt=solver.dt,
        relaxation_time=relaxation_time,
        heat_generation_rate=heat_generation_rate,
        applied_heat_source=applied_heat_source,
        snapshots=tuple(snapshot_diagnostics),
    )
