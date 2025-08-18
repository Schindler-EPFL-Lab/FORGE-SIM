"""
Run a heat simulation on a tessellated IGA spline surface loaded from a file.
The simulation uses the Crank-Nicolson time integration scheme to solve the
transient heat equation on the surface.
"""

import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import drjit as dr
import jsonpickle
import mitsuba as mi
import mlflow
import numpy as np
import scipy.sparse as sps
import tyro
from scipy.sparse import linalg

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append("./")

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape import logger
from rebel_shape.isogeometric_analysis.heat_simulation import HeatSimulation
from rebel_shape.meshing.composite_iga_spline_surface import CompositeIGASplineSurface
from rebel_shape.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from rebel_shape.plugin_wrappers.sampler import Sampler
from rebel_shape.property_optimizers.field_texture_optimizer import (
    FieldTextureOptimizer,
)
from rebel_shape.utils.image_utils import show_rendered_views
from rebel_shape.utils.sensor_utils import create_sensors_radial


@dataclass
class HeatSimulationParameters(ReverseCli):
    """Parameters for the heat simulation."""

    spline_path: Path
    """Path to the reconstructed IGA spline surface mesh with inpainted thermal data."""
    output_folder: Path = Path("./outputs/")
    """Where to save the model and outputs (created if it does not exist)"""
    num_time_steps: int = 360
    """Number of time steps for the simulation."""
    time_step_size: float = 60.0
    """Size of each time step in seconds."""
    save_interval: int = 30
    """Interval at which to save the simulation state."""

    texture_resolution: int = 512
    """Resolution of the texture used for the thermal data"""
    rendering_samples_per_pixel: int = 1
    """Number of samples per pixel used for rendering."""
    sensor_count: int = 6
    """Number of sensors placed in radial pattern around the origin."""
    radial_distance: float = 8.0
    """Distance of cameras from the origin."""
    sensor_resolution_width: int = 480
    """Resolution of the camera sensor along width"""
    sensor_resolution_height: int = 480
    """Resolution of the camera sensor along height"""
    sensor_fov: float = 45.0
    """Field of view of the camera sensors in degrees"""
    viewing_angle: float = -10.0
    """Camera viewing angle in degrees (negative looks down, positive looks up)"""

    concrete_conductivity: float = 2.0
    """Thermal conductivity of concrete in W/(m*K)"""
    concrete_density: float = 2250.0
    """Density of concrete in kg/m^3"""
    concrete_specific_heat: float = 900.0
    """Specific heat capacity of concrete in J/(kg*K)"""
    concrete_thickness: float = 0.2
    """Thickness of the concrete layer in meters"""

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
        if self.rendering_samples_per_pixel < 1:
            raise ValueError(
                "`rendering_samples_per_pixel` must be a positive integer!"
            )
        if self.sensor_count < 1:
            raise ValueError("`sensor_count` must be a positive integer!")
        if self.radial_distance <= 0:
            raise ValueError("`radial_distance` must be a positive real number!")
        if self.sensor_resolution_width < 1 or self.sensor_resolution_height < 1:
            raise ValueError(
                "`sensor_resolution_width` and `sensor_resolution_height` must be "
                "positive integers!"
            )
        if self.sensor_fov <= 0 or self.sensor_fov >= 180:
            raise ValueError(
                "`sensor_fov` must be a positive real number in the range (0, 180)!"
            )
        if self.viewing_angle <= -90 or self.viewing_angle >= 90:
            raise ValueError(
                "`viewing_angle` must be a real number in the range (-90, 90)!"
            )

        if self.concrete_conductivity <= 0:
            raise ValueError("`concrete_conductivity` must be a positive real number!")
        if self.concrete_density <= 0:
            raise ValueError("`concrete_density` must be a positive real number!")
        if self.concrete_specific_heat <= 0:
            raise ValueError("`concrete_specific_heat` must be a positive real number!")
        if self.concrete_thickness <= 0:
            raise ValueError("`concrete_thickness` must be a positive real number!")


def initialize_scene(
    spline_path: Path,
    sensor_count: int,
    radial_distance: float,
    viewing_angle: float,
    sensor_resolution: tuple[int, int],
    sensor_fov: float,
    samples_per_pixel: int,
    texture_resolution: int = 512,
) -> tuple["mi.Scene", list["mi.Sensor"], TessellatedIGASplineSurface]:
    """
    Creates and loads Mitsuba dictionary that represents the scene, as well as
    providing the sensor views and the tessellated IGA spline surface.

    `spline_path` is the path to the JSON file containing the serialized
    `TessellatedIGASplineSurface` object. Its texture is replaced with a monochromatic
    one of resolution `texture_resolution` x `texture_resolution`. This texture will
    carry the spline surface temperature data evaluated at specific parametric
    coordinates.

    Sensors are created in a radial pattern around the origin, with a specified
    `radial_distance` from the origin, where the origin is the mean of the global
    control points of the spline surface. The sensors are oriented to look at the
    origin with a specified `viewing_angle` in degrees (0 is looking in the plane,
    negative values look from downwards, positive values look from upwards).
    Cameras have `sensor_resolution` and `sensor_fov` field of view. `samples_per_pixel`
    specifies how many samples per pixel will be used for rendering.

    :returns: A tuple containing:

        - `scene_source`: The Mitsuba scene object loaded from the dictionary containing
            the spline surface and the temperature distribution texture.
        - `sensors`: A list of Mitsuba sensor objects created in a radial pattern
            around the origin, each with the specified resolution and field of view.
        - `tessellated_spline_surface`: The loaded tessellated IGA spline surface
            containing the composite spline and its tessellated mesh representation.

    :raises FileNotFoundError: If the `spline_path` does not exist.
    """

    with open(spline_path, "r") as tessellated_spline_file:
        tessellated_spline_surface: TessellatedIGASplineSurface = jsonpickle.decode(
            tessellated_spline_file.read()
        )

    scene_dict = {"type": "scene", "integrator": {"type": "path"}}

    with tempfile.TemporaryDirectory() as temp_dir:
        initial_texture_path = Path(temp_dir) / "initial_texture.exr"

        FieldTextureOptimizer.generate_empty_texture(
            base_color=mi.Color3f(0.5, 0.5, 0.5),
            texture_output_path=initial_texture_path,
            width=texture_resolution,
            height=texture_resolution,
        )

        for mesh in tessellated_spline_surface:
            mesh.save_to_ply(output_dir=Path(temp_dir), save_name=mesh.name)

            scene_dict[mesh.name] = {
                "type": "ply",
                "filename": temp_dir + f"/{mesh.name}.ply",
                "bsdf": DiffuseBSDF(
                    reflectance=DiffuseBSDF.Albedo(albedo=0.05),
                ).mitsuba_dict,
                "emitter": {
                    "type": "area",
                    "radiance": {
                        "type": "bitmap",
                        "filename": str(initial_texture_path),
                    },
                },
            }

        scene_source: "mi.Scene" = mi.load_dict(scene_dict)

    target_coords = tessellated_spline_surface.global_control_points.numpy().mean(
        axis=0
    )

    sensors = create_sensors_radial(
        sensor_count=sensor_count,
        radial_distance=radial_distance,
        target_coords=(target_coords[0], target_coords[1], target_coords[2]),
        viewing_angle=viewing_angle,
        sensor_resolution=sensor_resolution,
        sensor_fov=sensor_fov,
        sampler=Sampler(sample_count=samples_per_pixel, type=Sampler.Type.MULTIJITTER),
    )

    return scene_source, sensors, tessellated_spline_surface


def update_temperature(
    scene_params: "mi.SceneParameters",
    iga_spline: CompositeIGASplineSurface,
    texture_keys: list[str],
) -> None:
    """
    Updates the temperature textures in the scene parameters based on the current
    temperature field of the `iga_spline` surface.

    The user has to provide the `texture_keys` that correspond to the textures in the
    `scene_params` that will be updated with the new texture data.
    """
    max_value = mi.TensorXf([0.0])

    for i, (texture_key, patch) in enumerate(zip(texture_keys, iga_spline)):
        width, height = scene_params[texture_key].shape[:2]
        new_field_texture = patch.evaluate_field_texture(width=width, height=height)
        scene_params[texture_key] = new_field_texture

        max_value = dr.maximum(max_value, dr.max(new_field_texture))

    scene_params.update()
    logger.info(f"Maximum temperature value in textures: {max_value.numpy()[0]:.7f}")


def simulate_heat_cn(params: HeatSimulationParameters) -> None:
    """
    Runs a complete heat simulation using the Crank-Nicolson time integration scheme.

    This function performs a full transient heat simulation on a spline surface using
    the Crank-Nicolson implicit method. It assembles the stiffness and mass matrices,
    as well as the source term vector, and iteratively updates the temperature field
    over a specified number of time steps. The simulation is based on the provided
    `params`, including the spline surface, material properties, and simulation
    parameters.

    The simulation solves the heat equation:
        partial_t rho(x) c(x) d(x) u(x, t) - d(x) div_M (k(x) grad_M u(x, t)) = f(x, t),

    where rho(x) is density, c(x) is specific heat, k(x) is thermal conductivity,
    d(x) is thickness, and f(x, t) is the source term. The Crank-Nicolson method
    is used to discretize the time derivative, leading to an implicit system of
    equations at each time step. The scheme is second-order accurate in time and
    unconditionally stable for linear problems.

    The Crank-Nicolson update is given by:

        (M + 0.5*dt*K) T^(n+1) = (M - 0.5*dt*K) T^n + dt*F

    where M is the mass matrix, K is the stiffness matrix, F is the source term vector,
    dt is the time step, and T^n represents the temperature at time step n.

    The resulting system of equations can be solved in one of two ways:
        1. Using a direct solver (e.g., LU decomposition) for small systems.
        2. Using an iterative solver (e.g., conjugate gradient) for larger systems.
    The choice of solver depends on the size of the system and the available memory.
    Right now, a preconditioned conjugate gradient solver is used as a default,
    but due to the small problem size, it is not strictly necessary.

    Every `params.save_interval` time steps, the current temperature field is rendered
    using Mitsuba in monochromatic mode (having rgb visualization would be very nice)
    and saved to the output folder.
    """
    mi.set_variant("llvm_ad_mono")

    start_time = time.time()

    scene, sensors, tessellated_iga_spline = initialize_scene(
        spline_path=params.spline_path,
        sensor_count=params.sensor_count,
        radial_distance=params.radial_distance,
        viewing_angle=params.viewing_angle,
        sensor_resolution=(
            params.sensor_resolution_width,
            params.sensor_resolution_height,
        ),
        sensor_fov=params.sensor_fov,
        samples_per_pixel=params.rendering_samples_per_pixel,
        texture_resolution=params.texture_resolution,
    )
    scene_params = mi.traverse(scene)

    iga_spline = tessellated_iga_spline.composite_spline

    # Material properties
    conductivity = params.concrete_conductivity * dr.ones(
        dtype=mi.TensorXf, shape=(iga_spline.global_field_values.shape[0], 1)
    )
    density = params.concrete_density * dr.ones(
        dtype=mi.TensorXf, shape=(iga_spline.global_field_values.shape[0], 1)
    )
    specific_heat = params.concrete_specific_heat * dr.ones(
        dtype=mi.TensorXf, shape=(iga_spline.global_field_values.shape[0], 1)
    )
    source = dr.zeros(
        dtype=mi.TensorXf, shape=(iga_spline.global_field_values.shape[0], 1)
    )
    thickness = params.concrete_thickness * dr.ones(
        dtype=mi.TensorXf, shape=(iga_spline.global_field_values.shape[0], 1)
    )

    heat_simulation = HeatSimulation(
        composite_iga_spline=iga_spline,
        conductivity=conductivity,
        density=density,
        specific_heat=specific_heat,
        thickness=thickness,
        source_term=source,
    )

    setup_time = time.time() - start_time
    logger.info(f"Setup time: {setup_time:.3f} seconds")
    logger.info(f"Number of control points: {iga_spline.global_field_values.shape[0]}")

    start_time = time.time()
    local_indices, local_matrices = heat_simulation.assemble_local_stiffness_matrices()
    stiffness_matrix = heat_simulation.assemble_global_matrix(
        local_indices=local_indices, local_matrices=local_matrices
    )
    stiffness_matrix_time = time.time() - start_time
    logger.info(f"Stiffness matrix assembly time: {stiffness_matrix_time:.2f} seconds")
    logger.info(f"Stiffness matrix shape: {stiffness_matrix.shape}")
    logger.info(
        f"Number of non-zero entries in stiffness matrix: {stiffness_matrix.nnz}"
    )

    start_time = time.time()
    local_indices, local_matrices = heat_simulation.assemble_local_mass_matrices()
    mass_matrix = heat_simulation.assemble_global_matrix(
        local_indices=local_indices, local_matrices=local_matrices
    )
    mass_matrix_time = time.time() - start_time
    logger.info(f"Mass matrix assembly time: {mass_matrix_time:.3f} seconds")

    start_time = time.time()
    local_indices, local_vectors = heat_simulation.assemble_local_vectors(
        rhs_term=source
    )
    global_vector = heat_simulation.assemble_global_vector(
        local_indices=local_indices, local_vectors=local_vectors
    )
    global_vector_time = time.time() - start_time
    logger.info(f"Global vector assembly time: {global_vector_time:.3f} seconds")

    temp = heat_simulation.temperature.numpy().reshape(-1)

    system_matrix_low = mass_matrix - 0.5 * params.time_step_size * stiffness_matrix
    system_matrix_high = mass_matrix + 0.5 * params.time_step_size * stiffness_matrix

    property_key = "emitter.radiance.data"
    texture_keys = [f"{mesh.name}.{property_key}" for mesh in tessellated_iga_spline]

    # Compute lumped mass matrix
    lumped_mass_matrix_diag = mass_matrix.sum(axis=1).A1

    # Pre-conditioner for the conjugate gradient solver
    inv_lumped_mass_matrix = sps.diags(
        1.0 / lumped_mass_matrix_diag, format="csr", dtype=np.float64
    )

    start_time = time.time()
    for it in range(params.num_time_steps):
        if it % params.save_interval == 0:
            heat_simulation.temperature = mi.TensorXf(temp.reshape(-1, 1))
            logger.info(
                f"Total heat at time step {it}: {heat_simulation.total_heat()} J"
            )

            update_temperature(
                scene_params=scene_params,
                iga_spline=iga_spline,
                texture_keys=texture_keys,
            )

            rendered_images = [
                mi.render(
                    scene=scene, sensor=sensor, spp=params.rendering_samples_per_pixel
                )
                for sensor in sensors
            ]

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            show_rendered_views(
                views=[mi.util.convert_to_bitmap(image) for image in rendered_images],
                image_dpi=150,
                save_path=params.output_folder / f"views_{timestamp}_step_{it:04}.png",
            )

        # u_{n+1} = (M + 0.5*dt*K)^{-1} @ [ (M - 0.5*dt*K) @ u_n + dt * f ]
        rhs = system_matrix_low @ temp + params.time_step_size * global_vector
        temp = linalg.cg(
            system_matrix_high, rhs, x0=temp, rtol=1e-7, M=inv_lumped_mass_matrix
        )[0]

    loop_time = time.time() - start_time
    logger.info(f"Time loop time: {loop_time:.3f} seconds")
    logger.info(f"Time per time step: {loop_time / params.num_time_steps:.4f} seconds")


if __name__ == "__main__":
    parameters = tyro.cli(HeatSimulationParameters)
    mlflow.log_params(parameters.to_loggable_dict())

    simulate_heat_cn(params=parameters)
