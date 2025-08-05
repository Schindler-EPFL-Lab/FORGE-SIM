"""
This script assumes that shape optimization has already been performed and
takes in an optimized spline. It proceeds to inpaint thermal information onto the
reconstructed spline control points using the thermal images from the ThermoScenes
scene.
"""

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import drjit as dr
import jsonpickle
import mitsuba as mi
import mlflow
import tyro

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append("./")

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape import logger
from rebel_shape.data_processing.thermoscenes_loader import ThermoScenesLoader
from rebel_shape.loss_evaluators.image_loss_l1 import ImageLossL1
from rebel_shape.meshing.composite_iga_spline_surface import CompositeIGASplineSurface
from rebel_shape.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)
from rebel_shape.meshing.tessellated_spline_surface import TessellatedSplineSurface
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from rebel_shape.plugin_wrappers.prb_integrator import GuidingOptions, PRBIntegrator
from rebel_shape.plugin_wrappers.sampler import Sampler
from rebel_shape.property_optimizers.field_texture_optimizer import (
    FieldTextureOptimizer,
)
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.scene_optimizers.scene_adam_sequential import SceneAdamSequential
from rebel_shape.schedulers.constant_scheduler import ConstantScheduler


@dataclass
class ThermalSplineInpaintingParameters(ReverseCli):
    """
    Parameters used for inpainting thermal information onto mesh vertices.

    For details about the name "inpainting", see the `inpaint_thermal_information`
    function.
    """

    thermoscenes_scene_path: Path
    """
    Path to the ThermoScenes scene that is optimized. The thermal image foreground
    does not have to be extracted.
    """
    spline_path: Path
    """Path to the reconstructed spline surface mesh that will receive thermal data."""

    texture_projection_steepness: float = 1.0
    """Steepness of the sigmoid projection for texture optimization"""
    continuous_samples: int = 16
    """Number of samples per pixel used for continuous derivatives"""
    optimization_iterations: int = 20
    """How many times all images in the dataset are used for optimization"""
    texture_resolution: int = 512
    """Resolution of the texture along height and width, per patch"""
    texture_learning_rate: float = 0.01
    """Adam learning rate for texture"""
    texture_batch_size: int = 1
    """Number of images used for texture gradient computation"""

    output_folder: Path = Path("./outputs/")
    """Where to save the model and outputs (created if it does not exist)"""
    num_of_images_to_save: int = 6
    """
    Number of images to save after each optimization step. The images are taken from 
    the evaluation set.
    """

    use_gpu: bool = True
    """
    Whether to use GPU for rendering or not. Note that the compute instance
    must have GPU support for this to work. For local testing, set to False.
    """

    def __post_init__(self) -> None:
        if self.texture_projection_steepness <= 0.0:
            raise ValueError(
                "`texture_projection_steepness` must be a positive real number."
            )

        if self.texture_projection_steepness <= 0.0:
            raise ValueError(
                "`texture_projection_steepness` must be a positive real number."
            )
        if self.continuous_samples < 1:
            raise ValueError("`continuous_samples` must be a positive integer.")
        if self.optimization_iterations < 1:
            raise ValueError("`first_pass_iterations` must be a positive integer.")
        if self.texture_resolution < 1:
            raise ValueError("`texture_resolution` must be a positive integer.")
        if self.texture_learning_rate < 0.0:
            raise ValueError("`texture_learning_rate` must be non-negative.")
        if self.texture_batch_size < 1:
            raise ValueError("`batch_size` must be a positive integer")
        if self.num_of_images_to_save < 1:
            raise ValueError("`num_of_images_to_save` must be a positive integer.")


def initialize_scene(
    thermoscenes_scene_path: Path,
    spline_path: Path,
    continuous_samples: int,
    texture_resolution: int = 512,
) -> tuple[
    "mi.Scene",
    list["mi.Sensor"],
    list["mi.TensorXf"],
    list["mi.Sensor"],
    list["mi.TensorXf"],
    TessellatedIGASplineSurface,
]:
    """
    Creates and loads Mitsuba dictionary that represents the scene, as well as
    providing the reference images and sensor views.

    `thermoscenes_scene_path` is a `Path` to a scene that can be loaded
    with `ThermoScenesLoader`. `spline_path` specifies a path to a JSON file that
    contains the tessellated spline surface. The surface is then converted to an
    IGA spline surface and h_refinement is performed to increase the resolution
    of the thermal information that can be stored in the control points.

    The scene is expected to be grayscale (thermal information is a single channel
    value) so the sensors returned by method have a luminosity pixel format.

    :returns: A tuple containing:

        - `scene_source`: A Mitsuba scene object that contains the loaded scene.
        - `train_sensors`: A list of Mitsuba sensors for training images.
        - `train_images`: A list of training images loaded from the scene.
        - `eval_sensors`: A list of Mitsuba sensors for evaluation images.
        - `eval_images`: A list of evaluation images loaded from the scene.
        - `tessellated_iga_spline_surface`: A `TessellatedIGASplineSurface` object
            that represents the tessellated IGA spline surface with thermal data.
    """

    integrator = PRBIntegrator(
        max_bounces=2,
        bounces_without_russian_roulette=2,
        spp_continuous=continuous_samples,
        spp_projective=1,
        spp_indirect=1,
        guiding=GuidingOptions.NONE,
    )

    mlflow.log_param("spp_continuous", continuous_samples)
    mlflow.log_param("spp_projective", 1)
    mlflow.log_param("spp_indirect", 1)

    sampler = Sampler(type=Sampler.Type.INDEPENDENT, sample_count=continuous_samples)

    loader = ThermoScenesLoader(
        scene_path=thermoscenes_scene_path, sensor_sampler=sampler
    )

    with open(spline_path, "r") as tessellated_spline_file:
        tessellated_spline_surface: TessellatedSplineSurface = jsonpickle.decode(
            tessellated_spline_file.read()
        )

    spline_surface = tessellated_spline_surface.composite_spline

    initial_field_values = [
        0.5 * dr.ones(dtype=mi.TensorXf, shape=(spline.control_points.shape[0], 1))
        for spline in spline_surface
    ]

    iga_spline_surface = CompositeIGASplineSurface.from_spline_surface(
        spline_surface=tessellated_spline_surface.composite_spline,
        per_patch_field_values=initial_field_values,
    )
    iga_spline_surface.h_refinement(factor=4)

    scene_dict = {"type": "scene", "integrator": integrator.mitsuba_dict}

    with tempfile.TemporaryDirectory() as temp_dir:
        initial_texture_path = Path(temp_dir) / "initial_texture.exr"

        FieldTextureOptimizer.generate_empty_texture(
            base_color=mi.Color3f(0.5, 0.5, 0.5),
            texture_output_path=initial_texture_path,
            width=texture_resolution,
            height=texture_resolution,
        )

        iga_spline_surface._bsdf = DiffuseBSDF(
            DiffuseBSDF.Texture(texture_path=initial_texture_path)
        )
        tessellated_iga_spline_surface = TessellatedIGASplineSurface(
            composite_spline_surface=iga_spline_surface,
            tessellation_subdivisions_per_patch=(
                tessellated_spline_surface._triangle_subdivisions
            ),
        )

        for mesh in tessellated_iga_spline_surface:
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
                        "filename": temp_dir + "/initial_texture.exr",
                    },
                },
            }

        scene_source: "mi.Scene" = mi.load_dict(scene_dict)

    train_sensors: list["mi.Sensor"] = [
        mi.load_dict(camera.mitsuba_dict) for camera in loader.train_cameras_thermal
    ]
    eval_sensors: list["mi.Sensor"] = [
        mi.load_dict(camera.mitsuba_dict) for camera in loader.eval_cameras_thermal
    ]

    return (
        scene_source,
        train_sensors,
        loader.train_images_thermal,
        eval_sensors,
        loader.eval_images_thermal,
        tessellated_iga_spline_surface,
    )


def inpaint_thermal_information(params: ThermalSplineInpaintingParameters) -> None:
    """
    Assigns thermal values to vertices of the input mesh.

    The function first loads and prepares the spline by adding a thermal attribute
    to the control points, then optimizes the thermal data using the provided images
    as reference. The thermal information is stored in the IGA spline control points
    and is used to render the mesh with thermal information.

    In the spirit of the thermal modality, the temperature is simulated as a glow
    (area light) and the background is set to black. The optimization is done using
    the `TextureOptimizer` class. Word "inpainting" signifies that the
    thermal information is not used during mesh optimization, but is "painted"
    onto the final mesh.

    Optionally, images and meshes can be saved to `params.output_folder`. The path
    should be a directory where the images will be saved. If the directory does not
    exist, it will be created, while if a `None` value is provided, nothing is saved.

    For details on available parameters, see `ThermalInpaintingParameters`.

    :raises ValueError: If `params.thermoscenes_scene_path` is not a valid path, or
        it is not a directory.
    """
    if not params.thermoscenes_scene_path.exists():
        raise ValueError(f"Path {params.thermoscenes_scene_path} does not exist.")
    if not params.thermoscenes_scene_path.is_dir():
        raise ValueError(f"Path {params.thermoscenes_scene_path} is not a directory.")

    if params.use_gpu:
        mi.set_variant("cuda_ad_mono")
        logger.info("Using the GPU for rendering.")
    else:
        mi.set_variant("llvm_ad_mono")
        logger.info("Using the CPU for rendering.")

    # Initialize the scene, reference images and cameras
    (scene_source, sensors, ref_images, eval_sensors, eval_images, spline) = (
        initialize_scene(
            continuous_samples=params.continuous_samples,
            thermoscenes_scene_path=params.thermoscenes_scene_path,
            spline_path=params.spline_path,
        )
    )

    temperature_scheduler = ConstantScheduler(
        learning_rate=params.texture_learning_rate
    )

    optimizer = FieldTextureOptimizer(
        scene=scene_source,
        scene_params=mi.traverse(scene_source),
        optimization_events=OptimizationEvents(),
        property_key="emitter.radiance.data",
        tessellated_iga_spline=spline,
        texture_scheduler=temperature_scheduler,
        texture_loss=ImageLossL1(),
    )

    save_path = (
        params.output_folder / "Initial_state.png"
        if params.output_folder is not None
        else None
    )
    optimizer.show_comparison_with_reference(
        ref_images=eval_images,
        sensors=eval_sensors,
        samples_per_pixel=params.continuous_samples * 4,
        save_path=save_path,
    )
    logger.info("Rendered initial state")

    optimizer.optimize(
        sensors=sensors,
        ref_images=ref_images,
        scene_optimizer=SceneAdamSequential(
            batch_size=params.texture_batch_size, uniform=False
        ),
        num_iters=params.optimization_iterations,
        views_to_render=eval_sensors[: params.num_of_images_to_save],
        save_directory_path=params.output_folder,
    )
    logger.info("Finished optimization, rendering final results.")

    optimizer.show_comparison_with_reference(
        ref_images=ref_images,
        sensors=sensors,
        samples_per_pixel=params.continuous_samples * 4,
        save_path=params.output_folder / "Final_comparison.png",
    )
    # Save the final state of the scene
    optimizer.scene.shapes()[0].write_ply(
        str(params.output_folder / "thermal_mesh.ply")
    )


if __name__ == "__main__":
    parameters = tyro.cli(ThermalSplineInpaintingParameters)
    mlflow.log_params(parameters.loggable_dict)

    inpaint_thermal_information(params=parameters)
