"""
Optimizes a ThermoScenes scene represented as a single mesh.
Performs pose, shape, and texture optimization.
"""

import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

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
from rebel_shape.data_processing.input_preprocessing import preprocess_rgb_images
from rebel_shape.data_processing.masked_pose_image import MaskedPoseImage
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from rebel_shape.plugin_wrappers.prb_integrator import GuidingOptions, PRBIntegrator
from rebel_shape.plugin_wrappers.sampler import Sampler
from rebel_shape.property_optimizers.bitmap_texture_optimizer import (
    BitmapTextureOptimizer,
)
from rebel_shape.property_optimizers.mesh_pose_optimizer import MeshPoseOptimizer
from rebel_shape.property_optimizers.mesh_texture_optimizer import MeshTextureOptimizer
from rebel_shape.scene_optimizers.composite_scene_optimizer import (
    CompositeSceneOptimizer,
)
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.scripts.configs.pose_optimization_config import PoseOptimizationConfig
from rebel_shape.scripts.configs.shape_optimization_config import (
    ShapeOptimizationConfig,
)
from rebel_shape.scripts.configs.texture_optimization_config import (
    TextureOptimizationConfig,
)


@dataclass(kw_only=True)
class ThermoSceneOptimizationParameters(ReverseCli):
    thermoscenes_scene_path: Path
    """
    Path to the ThermoScenes scene that is optimized. Assumes the foreground
    has been extracted
    """
    environment_map_path: Path | None = None
    """Path to the envionment map that is used as background in optimization."""
    initial_guess_mesh_path: Path
    """Path to the initial guess mesh."""
    optimize_initial_pose: bool = True
    """Whether to execute pose optimization before shape and texture optimization"""

    continuous_samples: int = 25
    """number of samples per pixel used for continuous derivatives"""
    visibility_samples: int = 9
    """number of samples per pixel used for direct and indirect visibility"""
    optimization_iterations: int = 100
    """Adam iterations executed on the initial mesh"""

    pose_params: PoseOptimizationConfig = field(
        default_factory=lambda: PoseOptimizationConfig(
            initial_sphere_radius=3.0,
            learning_rate_start=0.08,
            learning_rate_end=0.04,
            learning_rate_steepness=1.5,
            use_lbfgs=False,
            uniform_adam=False,
            batch_size=12,
            optimization_iterations=10,
            use_msssim_loss=False,
            l1_loss_weight=1.0,
        )
    )
    """Parameters of the pose optimization"""
    shape_params: ShapeOptimizationConfig = field(
        default_factory=lambda: ShapeOptimizationConfig(
            smoothing_strength=100.0,
            learning_rate_start=0.25,
            learning_rate_end=0.1,
            learning_rate_steepness=1.5,
            use_lbfgs=False,
            uniform_adam=True,
            batch_size=9,
        )
    )
    """Parameters for shape optimization"""
    texture_params: TextureOptimizationConfig = field(
        default_factory=lambda: TextureOptimizationConfig(
            texture_resolution=1024,
            learning_rate_start=0.02,
            learning_rate_end=0.04,
            learning_rate_steepness=2.0,
            use_lbfgs=False,
            uniform_adam=False,
            batch_size=6,
        )
    )
    """Parameters for texture optimization"""

    output_folder: Path = Path("./outputs/")
    """Where to save the meshes and images (created if it does not exist)"""
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
    mask_pixels_during_optimization: bool = False
    """
    Whether the masks are applied during the optimization process to ignore non-building
    pixels. Note that during the pose optimization the masks are not applied
    """

    def __post_init__(self) -> None:
        if self.continuous_samples < 1:
            raise ValueError("`continuous_samples` must be a positive integer.")
        if self.visibility_samples < 1:
            raise ValueError("`visibility_samples` must be a positive integer.")
        if self.optimization_iterations < 1:
            raise ValueError("`first_pass_iterations` must be a positive integer.")

        if self.num_of_images_to_save < 1:
            raise ValueError("`num_of_images_to_save` must be a positive integer.")


def initialize_scene(
    continuous_samples: int,
    visibility_samples: int,
    thermoscenes_scene_path: Path,
    initial_guess_mesh_path: Path,
    envmap_path: Path | None,
    texture_resolution: int,
) -> tuple[
    "mi.Scene",
    list[MaskedPoseImage],
    list[MaskedPoseImage],
]:
    """
    Creates and loads Mitsuba dictionary that represents the scene, as well as
    providing the reference images and sensor views.

    `continuous_samples` is the number of samples used for optimizing continuous
    derivatives (e.g. Texture), while `visibility_samples` is the number of samples
    used for direct and indirect visibility changes.

    `thermoscenes_scene_path` is a `Path` to a scene that can be loaded
    with `ThermoScenesLoader`. `initial_guess_mesh_path` specifies a path to the mesh
    file that should be used as an initial guess for optimization. `envmap_path` is the
    path to the 360 image that is used for the background during optimization.

    :returns: A tuple containing:

        - `scene_source`: A Mitsuba scene object with the initial guess mesh and
            the environment map loaded.
        - `reference_images`: List of reference images for training.
        - `evaluation_images`: List of reference images for evaluation.

    :raises ValueError: If `thermoscenes_scene_path` is not a valid path, if the
        `environment_map_path` does not exist, or if the `initial_guess_mesh_path`
        does not exist.
    """
    integrator = PRBIntegrator(
        max_bounces=4,
        bounces_without_russian_roulette=4,
        spp_continuous=continuous_samples,
        spp_projective=visibility_samples,
        spp_indirect=visibility_samples,
        guiding=GuidingOptions.NONE,
    )

    mlflow.log_param("spp_continuous", continuous_samples)
    mlflow.log_param("spp_projective", visibility_samples)
    mlflow.log_param("spp_indirect", visibility_samples)

    sampler = Sampler(type=Sampler.Type.MULTIJITTER, sample_count=continuous_samples)

    ref_images, _ = preprocess_rgb_images(
        scene_path=thermoscenes_scene_path,
        environment_map_path=envmap_path,
        spp=4,
        sampler=sampler,
    )
    eval_images, _ = preprocess_rgb_images(
        scene_path=thermoscenes_scene_path,
        environment_map_path=envmap_path,
        spp=4,
        train=False,
        sampler=sampler,
    )

    # The texture only needs to exist until mi.load_dict is called
    with tempfile.TemporaryDirectory() as temp_dir:
        initial_texture_path = Path(temp_dir) / "initial_texture.exr"
        BitmapTextureOptimizer.generate_empty_texture(
            base_color=mi.Color3f(0.5, 0.5, 0.5),
            texture_output_path=initial_texture_path,
            width=texture_resolution,
            height=texture_resolution,
        )

        environment = (
            {"type": "envmap", "filename": str(envmap_path)}
            if envmap_path is not None
            else {
                "type": "constant",
                "radiance": {
                    "type": "rgb",
                    "value": 1.0,
                },
            }
        )

        scene_dict = {
            "type": "scene",
            "integrator": integrator.mitsuba_dict,
            "emitter": environment,
            "shape": {
                "type": "ply",
                "filename": str(initial_guess_mesh_path),
                "bsdf": DiffuseBSDF(
                    reflectance=DiffuseBSDF.Texture(texture_path=initial_texture_path)
                ).mitsuba_dict,
            },
        }
        scene_source: "mi.Scene" = mi.load_dict(scene_dict)

    return scene_source, ref_images, eval_images


def optimize_thermoscenes_scene(params: ThermoSceneOptimizationParameters) -> None:
    """
    Performs shape optimization on a scene specified by `thermoscenes_scene_path`.

    The initial guess is a mesh specified by `initial_guess_mesh_path`. The guess
    is improved upon by `MeshPoseOptimizer` that optimizes the position, scale, and
    orientation of the initial mesh for `pose_optimization_iterations`. This step is
    skipped if `optimize_initial_pose` is set to False.

    Optimization is performed for `optimization_iterations`  with two phases;
    each phase goes through all the images in the dataset:

    1. Shape optimization of mesh vertex positions with `shape_batch_size` images and
       learning rate which decreases from `shape_learning_rate_start` to
       `shape_learning_rate_end` with steepness `shape_learning_rate_steepness`.
    2. Bitmap texture optimization with `texture_batch_size` images and
       learning rate which increases from `texture_learning_rate_start` to
       `texture_learning_rate_end` with steepness `texture_learning_rate_steepness`.

    Optionally, images and meshes can be saved to `params.output_folder`. The path
    should be a directory where the images will be saved. If the directory does not
    exist, it will be created, while if a `None` value is provided, nothing is saved.

    Optimization can either be executed on a GPU or CPU, depending on the`use_gpu`
    parameter.

    :raises ValueError: If `params.thermoscenes_scene_path` is not a valid path, or
        it is not a directory.
    """
    if not params.thermoscenes_scene_path.exists():
        raise ValueError(f"Path {params.thermoscenes_scene_path} does not exist.")
    if not params.thermoscenes_scene_path.is_dir():
        raise ValueError(f"Path {params.thermoscenes_scene_path} is not a directory.")

    if params.use_gpu:
        mi.set_variant("cuda_ad_rgb")
        logger.info("Using the GPU for rendering.")
    else:
        mi.set_variant("llvm_ad_rgb")
        logger.info("Using the CPU for rendering.")

    # Initialize the scene, reference images and cameras
    scene_source, ref_images, eval_images = initialize_scene(
        continuous_samples=params.continuous_samples,
        visibility_samples=params.visibility_samples,
        thermoscenes_scene_path=params.thermoscenes_scene_path,
        initial_guess_mesh_path=params.initial_guess_mesh_path,
        envmap_path=params.environment_map_path,
        texture_resolution=params.texture_params.texture_resolution,
    )
    if len(eval_images) < params.num_of_images_to_save:
        raise ValueError(
            f"Not enough evaluation images available. "
            f"Expected at least {params.num_of_images_to_save}, "
            f"but got {len(eval_images)}."
        )

    ref_images_no_mask = [
        MaskedPoseImage(image=image.original_image, camera=image.camera, mask=None)
        for image in ref_images
    ]
    eval_images_no_mask = [
        MaskedPoseImage(image=image.original_image, camera=image.camera, mask=None)
        for image in eval_images
    ]

    shape_key = "shape"
    texture_key = "shape.bsdf.reflectance.data"

    # Optimize the pose of the initial mesh
    if params.optimize_initial_pose:
        logger.info("Optimizing initial mesh pose.")

        pose_optimizer = MeshPoseOptimizer(
            scene=scene_source,
            scene_params=mi.traverse(scene_source),
            optimization_events=OptimizationEvents(),
            shape_key=shape_key,
            scheduler=params.pose_params.scheduler(
                end_iteration=params.pose_params.optimization_iterations
            ),
            image_loss=params.pose_params.image_loss,
        )

        pose_optimizer.optimize(
            ref_images=ref_images_no_mask,
            scene_optimizer=params.pose_params.scene_optimizer,
            num_iters=params.pose_params.optimization_iterations,
            views_to_render=[
                image.sensor for image in eval_images[: params.num_of_images_to_save]
            ],
            save_directory_path=params.output_folder,
        )

    # Set up the pipeline for shape + texture optimization
    shape_scheduler = params.shape_params.scheduler(
        end_iteration=params.optimization_iterations
    )
    texture_scheduler = params.texture_params.scheduler(
        end_iteration=params.optimization_iterations
    )

    # Instantiate the optimizer with high projection smoothing for stability
    optimizer = MeshTextureOptimizer(
        scene=scene_source,
        scene_params=mi.traverse(scene_source),
        optimization_events=OptimizationEvents(),
        shape_key=shape_key,
        texture_key=texture_key,
        shape_smoothing_strength=params.shape_params.smoothing_strength,
        texture_projection_steepness=params.texture_params.projection_steepness,
        shape_scheduler=shape_scheduler,
        texture_scheduler=texture_scheduler,
        shape_loss=params.shape_params.image_loss,
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
    logger.info("Rendered initial state,")

    composite_scene_optimizer = CompositeSceneOptimizer(
        optimizer_config=[
            (params.shape_params.scene_optimizer, [shape_key]),
            (params.shape_params.scene_optimizer, [texture_key]),
        ],
    )

    if not params.mask_pixels_during_optimization:
        ref_images = ref_images_no_mask
        eval_images = eval_images_no_mask

    optimizer.optimize(
        ref_images=ref_images,
        scene_optimizer=composite_scene_optimizer,
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
    logger.info("Saving final mesh and texture.")
    texture = mi.Bitmap(optimizer.scene_params["shape.bsdf.reflectance.data"])
    texture.write(str(params.output_folder / "thermoscene_rgb_texture.exr"))

    optimizer.scene.shapes()[0].write_ply(
        str(params.output_folder / "thermoscene_shape.ply")
    )


if __name__ == "__main__":
    parameters = tyro.cli(ThermoSceneOptimizationParameters)
    mlflow.log_params(parameters.to_loggable_dict())

    optimize_thermoscenes_scene(params=parameters)
