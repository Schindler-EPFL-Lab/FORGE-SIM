"""
Run a heat simulation on a tessellated IGA spline surface loaded from a file.
The simulation uses the Crank-Nicolson time integration scheme to solve the
transient heat equation on the surface.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import mitsuba as mi
import mlflow
import tyro

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append("../")

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape import logger
from rebel_shape.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)
from rebel_shape.scripts.configs.camera_ring_config import CameraRingConfig
from rebel_shape.scripts.configs.thermal_material_config import ThermalMaterialConfig
from rebel_shape.scripts.heat_simulation.heat_simulation_runner import (
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
    num_time_steps: int = 360
    """Number of time steps for the simulation."""
    time_step_size: float = 240.0
    """Size of each time step in seconds."""
    save_interval: int = 30
    """Interval at which to save the simulation state."""

    camera_config: CameraRingConfig = field(
        default_factory=lambda: CameraRingConfig(
            samples_per_pixel=4,
            sensor_count=6,
            sensor_resolution_height=480,
            sensor_resolution_width=480,
            sensor_fov=45.0,
        )
    )
    """Camera configuration for rendering the scene during the simulation."""
    texture_resolution: int = 512
    """Resolution of the texture used for the thermal data"""

    concrete_config: ThermalMaterialConfig = field(
        default_factory=lambda: ThermalMaterialConfig(
            conductivity=2.0,
            density=2250.0,
            specific_heat=900.0,
            thickness=0.2,
        )
    )
    """Thermal material configuration for concrete."""

    use_gpu: bool = True
    """
    Whether to use GPU for rendering or not. Note that the compute instance
    must have GPU support for this to work. For local testing, set to False.
    """

    def __post_init__(self):
        if not self.spline_path.exists():
            raise ValueError(f"Spline path {self.spline_path} does not exist.")

        if self.num_time_steps < 1:
            raise ValueError("`num_time_steps` must be a positive integer!")
        if self.time_step_size <= 0:
            raise ValueError("`time_step_size` must be a positive real number!")
        if self.save_interval < 1:
            raise ValueError("`save_interval` must be a positive integer!")
        if self.texture_resolution < 1:
            raise ValueError("`texture_resolution` must be a positive integer!")


def simulate_heat_cn(params: HeatSimulationParameters) -> None:
    """
    Runs a complete heat simulation using the Crank-Nicolson time integration scheme.

    This function performs a full transient heat simulation on a spline surface using
    the Crank-Nicolson implicit method. It assembles the stiffness and mass matrices,
    as well as the source term vector, and iteratively updates the temperature field
    over a specified number of time steps. The simulation is based on the provided
    `params`, including the spline surface, material properties, and simulation
    parameters.

    Every `params.save_interval` time steps, the current temperature field is rendered
    using Mitsuba in monochromatic mode (having rgb visualization would be very nice)
    and saved to the output folder.
    """
    if params.use_gpu:
        mi.set_variant("cuda_ad_rgb")
        logger.info("Using the GPU for rendering and tensor assembly.")
    else:
        mi.set_variant("llvm_ad_rgb")
        logger.info("Using the CPU for rendering and tensor assembly.")

    tessellated_iga_spline = TessellatedIGASplineSurface.from_file(
        composite_spline_surface_path=params.spline_path
    )

    # A single material is used - the properties are directly from the concrete config
    heat_simulation_runner = HeatSimulationRunner(
        tessellated_iga_spline=tessellated_iga_spline,
        material_config=params.concrete_config,
        camera_config=params.camera_config,
        texture_resolution=params.texture_resolution,
    )

    # Run the heat simulation
    heat_simulation_runner.run_heat_simulation(
        num_time_steps=params.num_time_steps,
        time_step_size=params.time_step_size,
        output_folder=params.output_folder,
    )


if __name__ == "__main__":
    parameters = tyro.cli(HeatSimulationParameters)
    mlflow.log_params(parameters.to_loggable_dict())

    simulate_heat_cn(params=parameters)
