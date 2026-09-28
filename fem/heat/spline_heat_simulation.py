from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from dataclasses_reverse_cli.reverse_cli import ReverseCli
from mpi4py import MPI

from fem.fem_field_data import FEMFieldData
from fem.heat.fem_heat_runner import FEMHeatRun
from fem.heat.fem_simulation_config import FEMSimulationConfig
from fem.heat.fem_simulation_diagnostics import FEMSimulationDiagnostics
from fem.spline_to_fem_projector import SplineToFEMProjector
from forge_sim.scripts.configs.thermal_material_config import ThermalMaterialConfig


@dataclass(kw_only=True)
class SplineHeatSimulationParameters(ReverseCli, ABC):
    """
    Configuration shared by every FEM heat simulation on a spline surface.

    A subclass adds the material knobs its scenario needs and implements the
    three abstract members: which fields the projector is asked for, which
    material's (rho, c, h) a `target_delta_temp` source scaling uses, and what
    the output series is called. `run` does the rest.

    `heat_simulation_spline.py` and `heat_simulation_spline_two_materials.py`
    subclass this and differ in exactly one thing: how the material fields
    are projected off the spline. They share the same thirteen CLI knobs, the
    same validation rules, and the same load-project-solve sequence.

    The knobs deliberately mirror the IGA scripts in
    `forge_sim.scripts.heat_simulation`, defaults included, so that a given
    configuration resolves to a near-identical physical time step under either
    discretisation; `fem/tests/test_entry_points.py` pins that parity.
    """

    spline_path: Path
    """Path to the IGA spline surface with inpainted thermal data (JSON file)."""
    output_folder: Path = Path("./outputs/")
    """Directory for simulation outputs (created if it does not exist)."""

    num_time_steps: int = 500
    """Number of time steps for the simulation."""
    time_step_size: float | None = None
    """
    Explicit physical time step in seconds. Set exactly one of this and
    `dt_over_t_relax`.
    """
    dt_over_t_relax: float | None = 1e-5
    """
    Time step normalized by the natural relaxation time T_relax of the discretized
    system (physical dt = `dt_over_t_relax` * T_relax). T_relax is derived from the
    smallest non-zero generalized eigenvalue of the FEM-assembled (K, M), the same
    characterization the IGA solver uses, so the same value yields a near-identical
    physical dt on both. Set exactly one of this and `time_step_size`.
    """
    final_time: float | None = None
    """
    Total simulated time in seconds to reach exactly (dt = final_time /
    num_time_steps). Overrides `time_step_size` / `dt_over_t_relax` and skips the
    relaxation eigensolve. Use it to match a reference run's end time read from
    its XDMF, for a snapshot-aligned comparison.
    """
    save_interval: int = 50
    """Steps between XDMF snapshots and energy-drift reports."""

    tessellation_subdivisions: int = 512
    """Subdivisions per patch per direction (2 * value^2 triangles per patch)."""
    fem_degree: int = 1
    """Lagrange polynomial degree for the FEM basis (1 = P1)."""

    target_delta_temp: float | None = None
    """
    Target temperature rise (K) over the simulation duration, for the material
    that carries the source. When set, it overrides the script's own heat-source
    knob (`heat_source` / `facade_heat_source`): the source field is projected
    unit-valued and its strength is computed after assembly as
    Q = target_delta_temp * rho * c * h / t_sim (the rise an insulated element of
    that material would achieve).
    """

    target_length_scale: float | None = None
    """
    If set, rescales geometry so the longest axis of the control-point
    bounding box equals this value. The surface lies inside the control hull,
    so its own bounding box comes out slightly smaller.
    """

    write_xdmf: bool = True
    """Whether to write the solution to XDMF for ParaView / comparison."""

    def __post_init__(self) -> None:
        if not self.spline_path.exists():
            raise ValueError(f"Spline path {self.spline_path} does not exist.")
        if self.final_time is not None:
            if self.final_time <= 0.0:
                raise ValueError(
                    f"`final_time` must be a positive real number, got "
                    f"{self.final_time}."
                )
        elif (self.time_step_size is None) == (self.dt_over_t_relax is None):
            raise ValueError(
                "Exactly one of `time_step_size` and `dt_over_t_relax` must be set "
                "(or set `final_time`)."
            )
        if self.time_step_size is not None and self.time_step_size <= 0.0:
            raise ValueError(
                f"`time_step_size` must be a positive real number, got "
                f"{self.time_step_size}."
            )
        if self.dt_over_t_relax is not None and self.dt_over_t_relax <= 0.0:
            raise ValueError(
                f"`dt_over_t_relax` must be a positive real number, got "
                f"{self.dt_over_t_relax}."
            )
        if self.num_time_steps < 1:
            raise ValueError(
                f"`num_time_steps` must be a positive integer, got "
                f"{self.num_time_steps}."
            )
        if self.save_interval < 1:
            raise ValueError(
                f"`save_interval` must be a positive integer, got {self.save_interval}."
            )
        if self.tessellation_subdivisions < 1:
            raise ValueError(
                f"`tessellation_subdivisions` must be a positive integer, got "
                f"{self.tessellation_subdivisions}."
            )
        if self.fem_degree < 1:
            raise ValueError(f"`fem_degree` must be at least 1, got {self.fem_degree}.")
        if self.target_length_scale is not None and self.target_length_scale <= 0.0:
            raise ValueError(
                f"`target_length_scale` must be positive if provided, got "
                f"{self.target_length_scale}."
            )
        if self.target_delta_temp is not None and self.target_delta_temp <= 0.0:
            raise ValueError(
                f"`target_delta_temp` must be positive if provided, got "
                f"{self.target_delta_temp}."
            )

    @property
    @abstractmethod
    def xdmf_name(self) -> str:
        """
        File name of the XDMF time series written into `output_folder`.

        :raises NotImplementedError: If a subclass does not provide it.
        """
        raise NotImplementedError("A subclass must name its XDMF output.")

    @property
    @abstractmethod
    def heated_material(self) -> ThermalMaterialConfig:
        """
        The material whose (rho, c, h) set the `target_delta_temp` source scaling.

        It is the material actually being heated (the windows in a two-material
        facade run), and routing the wrong one mis-scales every source by the
        ratio of the two thermal masses, in a run that stays perfectly well-posed.

        :raises NotImplementedError: If a subclass does not provide it.
        """
        raise NotImplementedError("A subclass must name the heated material.")

    @abstractmethod
    def project_fields(self, projector: SplineToFEMProjector) -> FEMFieldData:
        """
        Projects the initial temperature and the material fields onto the mesh.

        This is the one step the two scenarios genuinely differ in. When
        `target_delta_temp` is set the source field is projected unit-valued and
        scaled after assembly, mirroring the IGA runner. `projector` is the
        projector built from `spline_path`, already tessellated.

        :returns: Per-vertex temperature and material fields.

        :raises NotImplementedError: If a subclass does not provide it.
        """
        raise NotImplementedError("A subclass must project its own fields.")

    @property
    def simulation_config(self) -> FEMSimulationConfig:
        """
        The runner configuration these parameters resolve to.

        `heat_source_material` is populated only alongside `target_delta_temp`,
        since it exists solely to supply that scaling's (rho, c, h); without a
        target rise the projected source field is used as it stands.
        """
        return FEMSimulationConfig(
            num_time_steps=self.num_time_steps,
            save_interval=self.save_interval,
            time_step_size=self.time_step_size,
            dt_over_t_relax=self.dt_over_t_relax,
            final_time=self.final_time,
            target_delta_temp=self.target_delta_temp,
            heat_source_material=(
                self.heated_material if self.target_delta_temp is not None else None
            ),
            fem_degree=self.fem_degree,
            write_xdmf=self.write_xdmf,
        )

    def run(self) -> FEMSimulationDiagnostics:
        """
        Loads the spline, projects the fields onto its tessellation, and runs the
        FEM heat simulation.

        This is the whole body of both entry scripts; what they contribute is the
        parameter subclass this is called on, already validated by its
        `__post_init__`.

        A Mitsuba variant must already be selected: loading the spline is a Dr.Jit
        operation, and `mi.set_variant` is a process-global side effect that belongs
        to the launching script, which sets it as its first act.

        :returns: The run's diagnostics: the resolved time step, the relaxation
            time it came from, the source power, and the per-snapshot energies. The
            same record is written to `FEMSimulationDiagnostics.JSON_NAME` in
            `output_folder`, on rank 0 only, since the XDMF series records none of it.
        """
        projector = SplineToFEMProjector(
            thermal_spline_path=self.spline_path,
            tessellation_subdivisions=self.tessellation_subdivisions,
            target_length_scale=self.target_length_scale,
        )
        field_data = self.project_fields(projector)

        fem_run = FEMHeatRun(
            mesh_data=projector.get_mesh_data(),
            field_data=field_data,
            config=self.simulation_config,
        )
        diagnostics = fem_run.run_with_xdmf(
            output_folder=self.output_folder, xdmf_name=self.xdmf_name
        )

        if MPI.COMM_WORLD.rank == 0:
            diagnostics.write_json(self.output_folder)
        return diagnostics
