"""
Run a heat simulation on a tessellated IGA spline surface with two
different materials (concrete and glass) based on a facade mask.

The simulation uses the Crank-Nicolson time integration scheme to solve
the transient heat equation on the surface. Material properties are
spatially varied based on the facade mask, where regions above a
threshold are treated as glass (windows) and regions below as concrete.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import drjit as dr
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

    num_time_steps: int = 360
    """Number of time steps for the simulation."""
    time_step_size: float = 240.0
    """Size of each time step in seconds."""
    save_interval: int = 30
    """Interval at which to save the simulation state."""
    facade_mask_threshold: float = 0.5
    """Threshold to distinguish between concrete and glass in the facade mask."""

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

    glass_config: ThermalMaterialConfig = field(
        default_factory=lambda: ThermalMaterialConfig(
            conductivity=0.9,
            density=2500.0,
            specific_heat=840.0,
            thickness=0.01,
        )
    )
    """Thermal material configuration for glass."""

    use_gpu: bool = True
    """
    Whether to use GPU for rendering or not. Note that the compute instance
    must have GPU support for this to work. For local testing, set to False.
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
        if self.time_step_size <= 0:
            raise ValueError("`time_step_size` must be a positive real number.")
        if self.save_interval < 1:
            raise ValueError("`save_interval` must be a positive integer.")
        if self.texture_resolution < 1:
            raise ValueError("`texture_resolution` must be a positive integer.")


def simulate_heat_cn_two_materials(
    params: HeatSimulationParametersTwoMaterials,
) -> None:
    """
    Runs a heat simulation with spatially varying material properties.

    This function performs a transient heat simulation on a spline surface using the
    Crank-Nicolson method with two distinct materials: concrete and glass.
    Material properties are assigned based on a facade mask, where values above a
    threshold (0.5) indicate glass (windows) and values below indicate concrete.
    Glass regions also have an internal heat source of 10 W/m^2, while concrete
    regions have no heat source.

    The simulation loads both the geometry spline and the facade mask spline,
    thresholds the mask to create a binary material map. The two splines are assumed
    to originate from the same geometrical model, but with different inpainted
    modalities (thermal data vs. facade mask).

    Every `params.save_interval` time steps, the current temperature
    field is rendered and saved to the output folder.
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

    heat_simulation_runner = HeatSimulationRunner(
        tessellated_iga_spline=tessellated_iga_spline,
        material_config=params.concrete_config,
        camera_config=params.camera_config,
        texture_resolution=params.texture_resolution,
    )

    # Material properties are set based on the facade mask
    facade_spline = TessellatedIGASplineSurface.from_file(
        composite_spline_surface_path=params.facade_mask_path
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
    heat_simulation_runner.source = dr.select(
        thresholded_facade, 10.0 * ones, 0.0 * ones
    )

    heat_simulation_runner.run_heat_simulation(
        num_time_steps=params.num_time_steps,
        time_step_size=params.time_step_size,
        output_folder=params.output_folder,
    )


if __name__ == "__main__":
    parameters = tyro.cli(HeatSimulationParametersTwoMaterials)
    mlflow.log_params(parameters.to_loggable_dict())

    simulate_heat_cn_two_materials(params=parameters)
