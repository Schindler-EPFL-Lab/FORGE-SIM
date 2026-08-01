"""
DOLFINx FEM heat simulation on a tessellated spline surface (two materials).

This is the FEM counterpart to
`rebel_shape.scripts.heat_simulation.heat_simulation_two_materials` (the IGA
two-material runner). A facade mask spline selects between a base material
(concrete) and windows (glass): the mask is thresholded at the control points and
the per-material properties are interpolated by the spline basis onto the mesh,
reproducing the smooth transition zones the IGA solver integrates.

The parameters mirror the IGA script (normalized or explicit time step,
`target_delta_temp` source scaling, `target_length_scale` geometry rescaling, the
shared `ThermalMaterialConfig` for each material) so the two solvers can be
compared one-to-one.

Example usage::

    python heat_simulation_spline_two_materials.py \\
        --spline-path /path/to/thermal_spline.json \\
        --facade-mask-path /path/to/window_mask_spline.json
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import mitsuba as mi
import tyro

# Make the rebel_shape package importable when running this script directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape.scripts.configs.thermal_material_config import ThermalMaterialConfig

from fem_heat_runner import FEMSimulationConfig, run_fem_heat_simulation
from spline_to_fem_projector import SplineToFEMProjector


@dataclass(kw_only=True)
class HeatSimulationSplineTwoMaterialsParameters(ReverseCli):
    """
    Configuration for two-material FEM heat simulation on a spline surface.

    Material properties are spatially varied from a facade mask: control points
    whose mask value exceeds `facade_mask_threshold` are treated as glass
    (windows), the rest as concrete.
    """

    spline_path: Path
    """Path to the IGA spline surface with inpainted thermal data (JSON file)."""
    facade_mask_path: Path
    """Path to the window mask spline used to set facade material properties."""
    output_folder: Path = Path("./outputs/")
    """Directory for simulation outputs (created if it does not exist)."""

    num_time_steps: int = 500
    """Number of time steps for the simulation."""
    time_step_size: float | None = None
    """Explicit physical time step in seconds. Set exactly one of this and
    `dt_over_t_relax`."""
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
    facade_mask_threshold: float = 0.5
    """Threshold distinguishing concrete (below) from glass (above) in the mask."""

    tessellation_subdivisions: int = 512
    """Subdivisions per patch per direction (2 * value^2 triangles per patch)."""
    fem_degree: int = 1
    """Lagrange polynomial degree for the FEM basis (1 = P1)."""

    concrete_config: ThermalMaterialConfig = field(
        default_factory=lambda: ThermalMaterialConfig(
            conductivity=0.15, density=600.0, specific_heat=2400.0, thickness=0.02
        )
    )
    """Thermal material configuration for the base (non-window) region."""
    glass_config: ThermalMaterialConfig = field(
        default_factory=lambda: ThermalMaterialConfig(
            conductivity=0.9, density=2500.0, specific_heat=840.0, thickness=0.01
        )
    )
    """Thermal material configuration for the window region."""

    facade_heat_source: float = 0.1
    """Heat source applied to glass regions in W/m^2. Ignored when
    `target_delta_temp` is set."""
    target_delta_temp: float | None = 0.5
    """
    Target temperature rise (K) for glass regions over the simulation duration.
    When set, overrides `facade_heat_source`; the required source is computed
    after assembly as Q = target_delta_temp * rho_glass * c_glass * h_glass /
    t_sim (the rise an insulated glass element would achieve).
    """

    target_length_scale: float | None = 2.5
    """If set, rescales geometry so the longest axis of the CONTROL-POINT
    bounding box equals this value. The surface lies inside the control hull,
    so its own bounding box comes out slightly smaller."""

    write_xdmf: bool = True
    """Whether to write the solution to XDMF for ParaView / comparison."""
    show_visualization: bool = False
    """Whether to show the interactive PyVista viewer after the simulation."""

    def __post_init__(self) -> None:
        if not self.spline_path.exists():
            raise ValueError(f"Spline path {self.spline_path} does not exist.")
        if not self.facade_mask_path.exists():
            raise ValueError(
                f"Facade mask path {self.facade_mask_path} does not exist."
            )
        if self.final_time is not None:
            if self.final_time <= 0.0:
                raise ValueError("`final_time` must be a positive real number.")
        elif (self.time_step_size is None) == (self.dt_over_t_relax is None):
            raise ValueError(
                "Exactly one of `time_step_size` and `dt_over_t_relax` must be set "
                "(or set `final_time`)."
            )
        if self.time_step_size is not None and self.time_step_size <= 0.0:
            raise ValueError("`time_step_size` must be a positive real number.")
        if self.dt_over_t_relax is not None and self.dt_over_t_relax <= 0.0:
            raise ValueError("`dt_over_t_relax` must be a positive real number.")
        if self.num_time_steps < 1:
            raise ValueError("`num_time_steps` must be a positive integer.")
        if self.save_interval < 1:
            raise ValueError("`save_interval` must be a positive integer.")
        if self.tessellation_subdivisions < 1:
            raise ValueError("`tessellation_subdivisions` must be a positive integer.")
        if self.fem_degree < 1:
            raise ValueError("`fem_degree` must be at least 1.")
        if self.target_length_scale is not None and self.target_length_scale <= 0.0:
            raise ValueError("`target_length_scale` must be positive if provided.")
        if self.target_delta_temp is not None and self.target_delta_temp <= 0.0:
            raise ValueError("`target_delta_temp` must be positive if provided.")


def run(params: HeatSimulationSplineTwoMaterialsParameters) -> None:
    """Loads the splines, projects the two-material fields, and runs the FEM run."""
    # Mitsuba variant must be set before any spline loading / Dr.Jit operation.
    mi.set_variant("llvm_ad_mono")

    projector = SplineToFEMProjector(
        thermal_spline_path=params.spline_path,
        tessellation_subdivisions=params.tessellation_subdivisions,
        target_length_scale=params.target_length_scale,
    )

    # When targeting a temperature rise, the glass source is unit-valued and
    # scaled to the required Q after assembly (mirrors the IGA runner).
    glass_source = (
        1.0 if params.target_delta_temp is not None else params.facade_heat_source
    )
    field_data = projector.project_two_material_fields(
        material_mask_spline_path=params.facade_mask_path,
        material_threshold=params.facade_mask_threshold,
        concrete_conductivity=params.concrete_config.conductivity,
        concrete_density=params.concrete_config.density,
        concrete_specific_heat=params.concrete_config.specific_heat,
        concrete_thickness=params.concrete_config.thickness,
        glass_conductivity=params.glass_config.conductivity,
        glass_density=params.glass_config.density,
        glass_specific_heat=params.glass_config.specific_heat,
        glass_thickness=params.glass_config.thickness,
        glass_source=glass_source,
    )

    config = FEMSimulationConfig(
        num_time_steps=params.num_time_steps,
        save_interval=params.save_interval,
        time_step_size=params.time_step_size,
        dt_over_t_relax=params.dt_over_t_relax,
        final_time=params.final_time,
        target_delta_temp=params.target_delta_temp,
        heat_source_material=(
            params.glass_config if params.target_delta_temp is not None else None
        ),
        fem_degree=params.fem_degree,
        write_xdmf=params.write_xdmf,
        show_visualization=params.show_visualization,
    )

    run_fem_heat_simulation(
        projector=projector,
        field_data=field_data,
        output_folder=params.output_folder,
        xdmf_name="heat_simulation_two_materials.xdmf",
        config=config,
    )


if __name__ == "__main__":
    run(params=tyro.cli(HeatSimulationSplineTwoMaterialsParameters))
