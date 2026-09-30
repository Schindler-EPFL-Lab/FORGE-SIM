"""
Run a heat simulation on a tessellated IGA spline surface with two
different materials (concrete and glass) based on a facade mask.

The simulation uses the Crank-Nicolson time integration scheme to solve
the transient heat equation on the surface. Material properties are
spatially varied based on the facade mask, where regions above a
threshold are treated as glass (windows) and regions below as concrete.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import drjit as dr
import mitsuba as mi
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

from forge_sim import logger, tracking
from forge_sim.isogeometric_analysis.heat_simulation import HeatSimulation
from forge_sim.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)
from forge_sim.scripts.configs.thermal_material_config import ThermalMaterialConfig
from forge_sim.scripts.heat_simulation.heat_simulation_runner import (
    HeatSimulationRunner,
)


@dataclass
class HeatSimulationParametersTwoMaterials(ReverseCli):
    """
    Configuration parameters for heat simulation with two materials.

    This dataclass contains all parameters needed to run a heat simulation
    with spatially varying material properties determined by a facade mask.
    The mask is used to distinguish between concrete and glass regions,
    each with different thermal properties.
    """

    spline_path: Path
    """Path to the reconstructed IGA spline surface mesh with inpainted thermal data."""
    facade_mask_path: Path
    """Path to the window mask used to set the material properties of the facade."""
    output_folder: Path = Path("./outputs/")
    """Where to save the model and outputs (created if it does not exist)"""

    num_time_steps: int = 500
    """Number of time steps for the simulation."""
    dt_over_t_relax: float = 1e-5
    """
    Time step normalized by the natural relaxation time T_relax of the
    discretized system. The physical dt is `dt_over_t_relax * T_relax`,
    where T_relax = 1 / lambda_1 and lambda_1 is the smallest non-zero
    generalized eigenvalue of the assembled (K, M). Crank-Nicolson temporal
    error scales as O((dt_over_t_relax)^2); 0.01 is the canonical choice
    for high-accuracy runs, 0.05 trades modest accuracy for speed.
    """
    time_step_size: float | None = None
    """
    Explicit physical dt in seconds, overriding `dt_over_t_relax` when set,
    for anchoring this run's clock to a dt resolved elsewhere (e.g. an
    independent FEM run), rather than deriving one from this model's own
    relaxation time.
    """
    save_interval: int = 50
    """Interval at which to save the simulation state."""
    facade_mask_threshold: float = 0.5
    """Threshold to distinguish between concrete and glass in the facade mask."""

    xdmf_path: Path | None = None
    """
    Full path of the output XDMF file. Defaults to
    `output_folder / "heat_simulation_two_materials.xdmf"` when not given.
    """
    xdmf_subdivisions: int = 512
    """Per-patch tessellation for the XDMF export (2 * xdmf_subdivisions^2 tris)."""

    concrete_config: ThermalMaterialConfig = field(
        default_factory=lambda: ThermalMaterialConfig(
            conductivity=0.15, density=600.0, specific_heat=2400.0, thickness=0.02
        )
    )
    """Thermal material configuration for concrete."""

    glass_config: ThermalMaterialConfig = field(
        default_factory=lambda: ThermalMaterialConfig(
            conductivity=0.9,
            density=2500.0,
            specific_heat=840.0,
            thickness=0.01,
        )
    )
    """Thermal material configuration for glass."""
    facade_heat_source: float = 0.1
    """Heat source applied to the glass regions of the facade in W/m^2.
    Ignored when `target_delta_temp` is set."""

    target_delta_temp: float | None = 0.5
    """
    Target temperature rise (K) for glass regions over the simulation
    duration.  When set, overrides `facade_heat_source`.  The required
    heat source Q (W/m²) is computed after matrix assembly:
        Q = target_delta_temp * rho_glass * c_glass * h_glass / t_sim
    This is the temperature rise an *insulated* glass element would
    achieve; actual rise will be lower due to conduction losses.
    """

    target_length_scale: float | None = 2.5
    """
    If provided, uniformly rescales geometry so that the longest bbox axis equals
    this value.
    """

    use_gpu: bool = True
    """
    Whether to use the CUDA Mitsuba/Dr.Jit variant for tensor assembly. Note
    that the compute instance must have GPU support for this to work. For
    local testing, set to False.
    """

    def __post_init__(self):
        if not self.spline_path.exists():
            raise ValueError(f"Spline path {self.spline_path} does not exist.")
        if not self.facade_mask_path.exists():
            raise ValueError(
                f"Facade mask path {self.facade_mask_path} does not exist."
            )

        if self.num_time_steps < 1:
            raise ValueError("`num_time_steps` must be a positive integer.")
        if self.dt_over_t_relax <= 0:
            raise ValueError("`dt_over_t_relax` must be a positive real number.")
        if self.time_step_size is not None and self.time_step_size <= 0.0:
            raise ValueError("`time_step_size` must be a positive real number.")
        if self.save_interval < 1:
            raise ValueError("`save_interval` must be a positive integer.")
        if self.xdmf_subdivisions < 1:
            raise ValueError("`xdmf_subdivisions` must be a positive integer.")
        if self.target_length_scale is not None and self.target_length_scale <= 0.0:
            raise ValueError("`target_length_scale` must be positive if provided.")
        if self.target_delta_temp is not None and self.target_delta_temp <= 0.0:
            raise ValueError("`target_delta_temp` must be positive if provided.")


def simulate_heat_cn_two_materials(
    params: HeatSimulationParametersTwoMaterials,
    on_snapshot: Callable[[int, float, HeatSimulation], None] | None = None,
) -> HeatSimulationRunner:
    """
    Runs a heat simulation with spatially varying material properties.

    This function performs a transient heat simulation on a spline surface using the
    Crank-Nicolson method with two distinct materials: concrete and glass.
    Material properties are assigned based on a facade mask, where values above a
    threshold (0.5) indicate glass (windows) and values below indicate concrete.
    Glass regions also carry an internal heat source, set by `facade_heat_source`;
    concrete regions have none.

    The simulation loads both the geometry spline and the facade mask spline,
    thresholds the mask to create a binary material map. The two splines are assumed
    to originate from the same geometrical model, but with different inpainted
    modalities (thermal data vs. facade mask).

    `on_snapshot`, when given, is called at the XDMF save cadence with the
    step index, its physical time, and the live `HeatSimulation` the run
    solved with; see `HeatSimulationRunner.run_heat_simulation_with_xdmf`.

    :returns: The `HeatSimulationRunner` the simulation ran on, so a caller
        can read `heat_simulation` and `cn_scheme` diagnostics off it.
    """
    if params.use_gpu:
        mi.set_variant("cuda_ad_rgb")
        logger.info("Using the GPU for tensor assembly.")
    else:
        mi.set_variant("llvm_ad_rgb")
        logger.info("Using the CPU for tensor assembly.")

    tessellated_iga_spline = TessellatedIGASplineSurface.from_file(
        composite_spline_surface_path=params.spline_path
    )

    # Material properties are set based on the facade mask
    facade_spline = TessellatedIGASplineSurface.from_file(
        composite_spline_surface_path=params.facade_mask_path
    )

    if params.target_length_scale is not None:
        bbox_min = dr.min(tessellated_iga_spline.global_control_points, axis=0)
        bbox_max = dr.max(tessellated_iga_spline.global_control_points, axis=0)
        bbox_extent = bbox_max - bbox_min
        longest_extent = dr.max(bbox_extent)
        longest_extent_value = float(longest_extent.numpy())

        if longest_extent_value <= 0.0:
            raise ValueError(
                "Cannot apply `target_length_scale`: longest bbox extent is "
                "non-positive."
            )

        length_scale_factor = params.target_length_scale / longest_extent_value
        tessellated_iga_spline.global_control_points *= length_scale_factor
        facade_spline.global_control_points *= length_scale_factor

        logger.info(
            "Applied target length scaling to longest bbox axis: "
            f"{params.target_length_scale} (factor={length_scale_factor:.6g})."
        )

    heat_simulation_runner = HeatSimulationRunner(
        tessellated_iga_spline=tessellated_iga_spline,
        material_config=params.concrete_config,
    )

    thresholded_facade = (
        facade_spline.global_field_values > params.facade_mask_threshold
    )

    ones = dr.ones(
        dtype=mi.TensorXf,
        shape=(tessellated_iga_spline.global_field_values.shape[0], 1),
    )

    # Set material properties based on the facade mask
    heat_simulation_runner.conductivity = dr.select(
        thresholded_facade,
        params.glass_config.conductivity * ones,
        params.concrete_config.conductivity * ones,
    )
    heat_simulation_runner.density = dr.select(
        thresholded_facade,
        params.glass_config.density * ones,
        params.concrete_config.density * ones,
    )
    heat_simulation_runner.specific_heat = dr.select(
        thresholded_facade,
        params.glass_config.specific_heat * ones,
        params.concrete_config.specific_heat * ones,
    )
    heat_simulation_runner.thickness = dr.select(
        thresholded_facade,
        params.glass_config.thickness * ones,
        params.concrete_config.thickness * ones,
    )
    source_value = (
        1.0 if params.target_delta_temp is not None else params.facade_heat_source
    )
    heat_simulation_runner.source = dr.select(
        thresholded_facade, source_value * ones, 0.0 * ones
    )

    dt_kwargs = (
        {"time_step_size": params.time_step_size}
        if params.time_step_size is not None
        else {"dt_over_t_relax": params.dt_over_t_relax}
    )
    xdmf_path = (
        params.xdmf_path
        if params.xdmf_path is not None
        else params.output_folder / "heat_simulation_two_materials.xdmf"
    )

    def forward_snapshot(step_index: int, time: float) -> None:
        if on_snapshot is not None:
            assert heat_simulation_runner.heat_simulation is not None
            on_snapshot(step_index, time, heat_simulation_runner.heat_simulation)

    heat_simulation_runner.run_heat_simulation_with_xdmf(
        num_time_steps=params.num_time_steps,
        xdmf_path=xdmf_path,
        target_delta_temp=params.target_delta_temp,
        heat_source_material=params.glass_config
        if params.target_delta_temp is not None
        else None,
        xdmf_save_interval=params.save_interval,
        tessellation_subdivisions=params.xdmf_subdivisions,
        on_snapshot=forward_snapshot,
        **dt_kwargs,
    )

    return heat_simulation_runner


if __name__ == "__main__":
    parameters = tyro.cli(HeatSimulationParametersTwoMaterials)
    with tracking.run(artifact_directory=parameters.output_folder):
        tracking.log_params(parameters.to_loggable_dict())

        simulate_heat_cn_two_materials(params=parameters)
