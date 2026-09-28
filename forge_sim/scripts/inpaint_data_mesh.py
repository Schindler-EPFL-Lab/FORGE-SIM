"""
This script assumes that shape optimization has already been performed and
takes in an optimized mesh. It proceeds to inpaint information onto the
reconstructed mesh vertices using single channel images.
"""

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import mitsuba as mi
import tyro

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append("./")

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from forge_sim import logger, tracking
from forge_sim.data_processing.masked_pose_image import MaskedPoseImage
from forge_sim.data_processing.single_channel_scene_loader import (
    SingleChannelImageSceneLoader,
)
from forge_sim.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from forge_sim.plugin_wrappers.prb_integrator import GuidingOptions, PRBIntegrator
from forge_sim.plugin_wrappers.sampler import Sampler
from forge_sim.property_optimizers.attribute_texture_optimizer import (
    AttributeTextureOptimizer,
)
from forge_sim.scene_optimizers.optimization_events import OptimizationEvents
from forge_sim.scripts.configs.texture_optimization_config import (
    TextureOptimizationConfig,
)


class InpaintType(Enum):
    THERMAL = "thermal"
    WINDOWS = "windows"


@dataclass(kw_only=True)
class InpaintingParameters(ReverseCli):
    """
    Parameters used for inpainting information onto mesh vertices.

    For details about the name "inpainting", see the `inpaint_information`
    function.
    """

    thermoscenes_scene_path: Path
    """
    Path to the ThermoScenes scene that is optimized. The single channel image
    foreground does not have to be extracted.
    """
    inpainting_images_type: InpaintType
    """What data type is inpainted onto the spline `thermoscenes_scene_path`"""
    initial_color: float = 0.5
    """The initial object color for inpainting"""
    mesh_path: Path
    """Path to the reconstructed mesh that will receive the scalar vertex data."""

    continuous_samples: int = 16
    """Number of samples per pixel used for continuous derivatives"""
    optimization_iterations: int = 20
    """How many times all images in the dataset are used for optimization"""

    texture_params: TextureOptimizationConfig = field(
        default_factory=lambda: TextureOptimizationConfig(
            texture_resolution=1,  # Not used for vertex attributes
            learning_rate_start=0.01,
            learning_rate_end=0.01,
            learning_rate_steepness=4.0,
            use_lbfgs=False,
            uniform_adam=False,
            batch_size=6,
            use_msssim_loss=False,
            l1_loss_weight=1.0,
        )
    )
    """Parameters for mesh inpainting optimization"""

    output_folder: Path = Path("./outputs/")
    """Where to save the model and outputs (created if it does not exist)"""
    num_of_images_to_save: int = 3
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
        if self.continuous_samples < 1:
            raise ValueError("`continuous_samples` must be a positive integer.")
        if self.optimization_iterations < 1:
            raise ValueError("`first_pass_iterations` must be a positive integer.")
        if self.num_of_images_to_save < 1:
            raise ValueError("`num_of_images_to_save` must be a positive integer.")


def initialize_scene(
    continuous_samples: int,
    thermoscenes_scene_path: Path,
    mesh_path: Path,
    inpaint_type: InpaintType,
    initial_color: float,
) -> tuple["mi.Scene", list[MaskedPoseImage], list[MaskedPoseImage]]:
    """
    Creates and loads Mitsuba dictionary that represents the scene, as well as providing
    the reference images and sensor views.

    `thermoscenes_scene_path` is a `Path` to a scene that can be loaded with
    `ThermoScenesLoader`. `inpaint_type` specifies the data type which is inpainted. The
    initial color of the surface is `initial_color`. `mesh_path` specifies a path to the
    mesh file that will receive the scalar vertex data. `continuous_samples` is the
    number of samples per pixel used for continuous derivatives.

    Information is stored in mesh vertices, so the resolution is directly tied to the
    number of triangles.

    The scene is expected to be a single channel (information is a single channel value)
    so the sensors returned by method have a luminosity pixel format.

    :returns: A tuple containing:

        - `scene`: The Mitsuba scene dictionary representing the scene.
        - `train_images`: A list of training images loaded from ThermoScenes.
        - `evaluation_images`: A list of evaluation images loaded from ThermoScenes.

    """
    integrator = PRBIntegrator(
        max_bounces=2,
        bounces_without_russian_roulette=2,
        spp_continuous=continuous_samples,
        spp_projective=1,
        spp_indirect=1,
        guiding=GuidingOptions.NONE,
    )

    tracking.log_params(
        {
            "spp_continuous": continuous_samples,
            "spp_projective": 1,
            "spp_indirect": 1,
        }
    )

    sampler = Sampler(type=Sampler.Type.INDEPENDENT, sample_count=continuous_samples)

    if inpaint_type is InpaintType.THERMAL:
        loader = SingleChannelImageSceneLoader.thermal_scene_auto(
            scene_path=thermoscenes_scene_path, sensor_sampler=sampler
        )
    if inpaint_type is InpaintType.WINDOWS:
        loader = SingleChannelImageSceneLoader.windows_masks_scene_auto(
            scene_path=thermoscenes_scene_path, sensor_sampler=sampler
        )

    mesh = mi.load_dict(
        {
            "type": "ply",
            "filename": str(mesh_path),
            "bsdf": DiffuseBSDF(
                reflectance=DiffuseBSDF.Albedo(albedo=0.0),
            ).mitsuba_dict,
            "emitter": {
                "type": "area",
                "radiance": {
                    "type": "mesh_attribute",
                    "name": "vertex_inpainting_data",
                },
            },
        },
    )

    # 1 value per vertex (we are in the scalar domain)
    attribute_size = mesh.vertex_count()
    mesh.add_attribute("vertex_inpainting_data", 1, [initial_color] * attribute_size)

    scene_dict = {"type": "scene", "integrator": integrator.mitsuba_dict, "shape": mesh}
    scene_source: "mi.Scene" = mi.load_dict(scene_dict)

    return scene_source, loader.train_images, loader.eval_images


def inpaint_information(params: InpaintingParameters) -> None:
    """
    Assigns scalar values to vertices of the input mesh.

    The function first loads and prepares the mesh by adding a scalar attribute to the
    vertices, then optimizes the scalar data using the provided images as reference. The
    information is stored directly in mesh vertices as a temperature value per vertex.

    In the spirit of the scalar modality, the scalar data is simulated as a glow (area
    light) and the background is set to black. The optimization is done using the
    `TextureOptimizer` class. Word "inpainting" signifies that the information is not
    used during mesh optimization, but is "painted" onto the final mesh.

    Optionally, images and meshes can be saved to `params.output_folder`. The path
    should be a directory where the images will be saved. If the directory does not
    exist, it will be created, while if a `None` value is provided, nothing is saved.

    For details on available parameters, see `InpaintingParameters`.

    :raises ValueError: If `params.thermoscenes_scene_path` is not a valid path, or it
        is not a directory.
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
    scene_source, ref_images, eval_images = initialize_scene(
        continuous_samples=params.continuous_samples,
        thermoscenes_scene_path=params.thermoscenes_scene_path,
        mesh_path=params.mesh_path,
        inpaint_type=params.inpainting_images_type,
        initial_color=params.initial_color,
    )

    optimizer = AttributeTextureOptimizer(
        scene=scene_source,
        scene_params=mi.traverse(scene_source),
        optimization_events=OptimizationEvents(),
        shape_key="shape",
        attribute_key="vertex_inpainting_data",
        texture_scheduler=params.texture_params.scheduler(
            end_iteration=params.optimization_iterations,
        ),
        texture_loss=params.texture_params.image_loss,
    )

    save_path = (
        params.output_folder / "Initial_state.png"
        if params.output_folder is not None
        else None
    )
    optimizer.show_comparison_with_reference(
        ref_images=eval_images,
        samples_per_pixel=params.continuous_samples * 4,
        save_path=save_path,
    )
    logger.info("Rendered initial state.")

    optimizer.optimize(
        ref_images=ref_images,
        scene_optimizer=params.texture_params.scene_optimizer,
        num_iters=params.optimization_iterations,
        views_to_render=[
            image.sensor for image in eval_images[: params.num_of_images_to_save]
        ],
        save_directory_path=params.output_folder,
    )
    logger.info("Finished optimization, rendering final state.")

    optimizer.show_comparison_with_reference(
        ref_images=ref_images,
        samples_per_pixel=params.continuous_samples * 4,
        save_path=params.output_folder / "Final_comparison.png",
    )

    # Save the final state of the scene
    optimizer.scene.shapes()[0].write_ply(
        str(params.output_folder / "inpainted_mesh.ply")
    )


if __name__ == "__main__":
    parameters = tyro.cli(InpaintingParameters)
    with tracking.run(artifact_directory=parameters.output_folder):
        tracking.log_params(parameters.to_loggable_dict())

        inpaint_information(params=parameters)
