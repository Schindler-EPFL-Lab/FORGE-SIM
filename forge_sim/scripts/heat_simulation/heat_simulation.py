"""
Run a heat simulation on a tessellated IGA spline surface loaded from a file.
The simulation uses the Crank-Nicolson time integration scheme to solve the
transient heat equation on the surface.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import drjit as dr
import mitsuba as mi
import tyro

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append("../")

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from forge_sim import logger, tracking
from forge_sim.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)
from forge_sim.scripts.configs.thermal_material_config import ThermalMaterialConfig
from forge_sim.scripts.heat_simulation.heat_simulation_runner import (
    HeatSimulationRunner,
)


@dataclass
class HeatSimulationParameters(ReverseCli):
    """
    Configuration parameters for single-material heat simulation.

    This dataclass contains all parameters needed to run a heat simulation
    with uniform material properties (concrete) across the entire surface.
    """

    spline_path: Path
    """Path to the reconstructed IGA spline surface mesh with inpainted thermal data."""
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
    save_interval: int = 50
    """Interval at which to save the simulation state."""

    concrete_config: ThermalMaterialConfig = field(
        default_factory=lambda: ThermalMaterialConfig(
            conductivity=2.0,
            density=2250.0,
            specific_heat=900.0,
            thickness=0.2,
        )
    )
    """Thermal material configuration for concrete."""

    target_delta_temp: float | None = None
    """
    Target temperature rise (K) over the simulation duration. When set,
    a uniform heat source is applied and its strength is computed after
    matrix assembly:  Q = target_delta_temp * rho * c * h / t_sim.
    This is the rise an *insulated* element would achieve.
    """

    heat_source: float = 0.0
    """Uniform heat source in W/m^2. Ignored when target_delta_temp is set."""

    target_length_scale: float | None = 4.0
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

        if self.num_time_steps < 1:
            raise ValueError("`num_time_steps` must be a positive integer!")
        if self.dt_over_t_relax <= 0:
            raise ValueError("`dt_over_t_relax` must be a positive real number!")
        if self.save_interval < 1:
            raise ValueError("`save_interval` must be a positive integer!")
        if self.target_length_scale is not None and self.target_length_scale <= 0.0:
            raise ValueError("`target_length_scale` must be positive if provided.")
        if self.target_delta_temp is not None and self.target_delta_temp <= 0.0:
            raise ValueError("`target_delta_temp` must be positive if provided.")


def simulate_heat_cn(params: HeatSimulationParameters) -> None:
    """
    Runs a complete heat simulation using the Crank-Nicolson time integration scheme.

    This function performs a full transient heat simulation on a spline surface using
    the Crank-Nicolson implicit method. It assembles the stiffness and mass matrices,
    as well as the source term vector, and iteratively updates the temperature field
    over a specified number of time steps. The simulation is based on the provided
    `params`, including the spline surface, material properties, and simulation
    parameters.
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

        logger.info(
            "Applied target length scaling to longest bbox axis: "
            f"{params.target_length_scale} (factor={length_scale_factor:.6g})."
        )

    heat_simulation_runner = HeatSimulationRunner(
        tessellated_iga_spline=tessellated_iga_spline,
        material_config=params.concrete_config,
    )

    if params.target_delta_temp is not None or params.heat_source != 0.0:
        source_value = (
            1.0 if params.target_delta_temp is not None else params.heat_source
        )
        ones = dr.ones(
            dtype=mi.TensorXf,
            shape=(tessellated_iga_spline.global_field_values.shape[0], 1),
        )
        heat_simulation_runner.source = source_value * ones

    heat_simulation_runner.run_heat_simulation_with_xdmf(
        num_time_steps=params.num_time_steps,
        dt_over_t_relax=params.dt_over_t_relax,
        xdmf_path=params.output_folder / "heat_simulation.xdmf",
        target_delta_temp=params.target_delta_temp,
        heat_source_material=params.concrete_config
        if params.target_delta_temp is not None
        else None,
        xdmf_save_interval=params.save_interval,
    )


if __name__ == "__main__":
    parameters = tyro.cli(HeatSimulationParameters)
    with tracking.run(artifact_directory=parameters.output_folder):
        tracking.log_params(parameters.to_loggable_dict())

        simulate_heat_cn(params=parameters)
