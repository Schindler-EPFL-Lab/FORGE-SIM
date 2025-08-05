"""
Optimizes a ThermoScenes scene represented as a single mesh.
Performs pose, shape, and texture optimization.
"""

import sys
import tempfile
from dataclasses import dataclass
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
from rebel_shape.loss_evaluators.composite_loss import CompositeImageLoss
from rebel_shape.loss_evaluators.image_loss_l1 import ImageLossL1
from rebel_shape.loss_evaluators.image_loss_l2 import ImageLossL2
from rebel_shape.loss_evaluators.image_loss_msssim import ImageLossMSSSIM
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
from rebel_shape.scene_optimizers.scene_adam_sequential import SceneAdamSequential
from rebel_shape.schedulers.exponential_scheduler import ExponentialScheduler


@dataclass
class ThermoSceneOptimizationParameters(ReverseCli):
    thermoscenes_scene_path: Path
    """
    Path to the ThermoScenes scene that is optimized. Assumes the foreground
    has been extracted
    """
    environment_map_path: Path
    """Path to the envionment map that is used as background in optimization."""
    initial_guess_mesh_path: Path
    """Path to the initial guess mesh."""

    shape_smoothing_strength: float = 100.0
    """Laplacian smoothing parameter for shape optimization"""
    shape_l1_loss_weight: float = 0.9
    """
    Weight of the L1 image loss for shape in composite loss. The other loss is either
    SSIM or L2, depending on `use_ssim_loss`, and is weighted by 
    1.0 - `shape_l1_loss_weight`
    """
    texture_resolution: int = 1024
    """Resolution of the texture used for optimization"""
    texture_projection_steepness: float = 1.0
    """Steepness of the sigmoid projection for texture optimization"""
    texture_l1_loss_weight: float = 0.9
    """
    Weight of the L1 image loss for texture in composite loss. The other loss is either
    SSIM or L2, depending on `use_ssim_loss`, and is weighted by 
    1.0 - `texture_l1_loss_weight`
    """
    use_msssim_loss: bool = True
    """
    If `True`, the SSIM loss is used for shape and texture optimization
    """

    continuous_samples: int = 16
    """number of samples per pixel used for continuous derivatives"""
    visibility_samples: int = 4
    """number of samples per pixel used for direct and indirect visibility"""
    optimization_iterations: int = 50
    """Adam iterations executed on the initial mesh"""

    optimize_initial_pose: bool = True
    """Whether to execute pose optimization before shape and texture optimization"""
    pose_optimization_iterations: int = 10
    """How many times the """
    pose_learning_rate_start: float = 0.1
    """Adam learning rate at the start of initial pose optimization"""
    pose_learning_rate_end: float = 0.01
    """Adam learning rate at the end of initial pose optimization"""
    pose_learning_rate_steepness: float = 2.0
    """How quickly the pose learning rate converges to the end value"""
    pose_batch_size: int = 12
    """Number of images used for pose gradient computation"""

    shape_learning_rate_start: float = 0.6
    """Adam learning rate at the start of shape optimization"""
    shape_learning_rate_end: float = 0.15
    """Adam learning rate at the end of shape optimization"""
    shape_learning_rate_steepness: float = 1.5
    """How quickly the shape learning rate converges to the end value"""
    shape_batch_size: int = 12
    """Number of images used for shape gradient computation"""

    texture_learning_rate_start: float = 0.005
    """Adam learning rate at the start of shape optimization"""
    texture_learning_rate_end: float = 0.025
    """Adam learning rate at the end of shape optimization"""
    texture_learning_rate_steepness: float = 3.0
    """How late in the optimization procedure the larger steps kick in"""
    texture_batch_size: int = 1
    """Number of images used for texture gradient computation"""

    output_folder: Path = Path("./outputs/")
    """Where to save the meshes and images (created if it does not exist)"""
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
        if self.texture_resolution < 1:
            raise ValueError("`texture_resolution` must be a positive integer.")
        if self.shape_smoothing_strength <= 0.0:
            raise ValueError(
                "`shape_projection_smoothing` must be a positive real number."
            )
        if self.texture_projection_steepness <= 0.0:
            raise ValueError(
                "`texture_projection_steepness` must be a positive real number."
            )
        if self.shape_l1_loss_weight < 0.0 or self.shape_l1_loss_weight > 1.0:
            raise ValueError(
                "`shape_l1_loss_weight` must be a real number in [0.0, 1.0]."
            )
        if self.texture_l1_loss_weight < 0.0 or self.texture_l1_loss_weight > 1.0:
            raise ValueError(
                "`texture_l1_loss_weight` must be a real number in [0.0, 1.0]."
            )

        if self.continuous_samples < 1:
            raise ValueError("`continuous_samples` must be a positive integer.")
        if self.visibility_samples < 1:
            raise ValueError("`visibility_samples` must be a positive integer.")
        if self.optimization_iterations < 1:
            raise ValueError("`first_pass_iterations` must be a positive integer.")

        if self.pose_learning_rate_start < 0.0:
            raise ValueError("`pose_learning_rate_start` must be non-negative.")
        if self.pose_learning_rate_end < 0.0:
            raise ValueError("`pose_learning_rate_end` must be non-negative.")
        if self.pose_learning_rate_end > self.pose_learning_rate_start:
            raise ValueError(
                "`pose_learning_rate_end` must be smaller than "
                "`pose_learning_rate_start`."
            )
        if self.pose_learning_rate_steepness <= 0.0:
            raise ValueError(
                "`pose_learning_rate_steepness` must be a positive real number."
            )
        if self.pose_batch_size < 1:
            raise ValueError("`batch_size` must be a positive integer")

        if self.shape_learning_rate_start < 0.0:
            raise ValueError("`shape_learning_rate_start` must be non-negative.")
        if self.shape_learning_rate_end < 0.0:
            raise ValueError("`shape_learning_rate_end` must be non-negative.")
        if self.shape_learning_rate_end > self.shape_learning_rate_start:
            raise ValueError(
                "`shape_learning_rate_end` must be smaller than "
                "`shape_learning_rate_start`."
            )
        if self.shape_learning_rate_steepness <= 0.0:
            raise ValueError(
                "`shape_learning_rate_steepness` must be a positive real number."
            )
        if self.shape_batch_size < 1:
            raise ValueError("`batch_size` must be a positive integer")

        if self.texture_learning_rate_start < 0.0:
            raise ValueError("`texture_learning_rate_start` must be non-negative.")
        if self.texture_learning_rate_end < 0.0:
            raise ValueError("`texture_learning_rate_end` must be non-negative.")
        if self.texture_learning_rate_end < self.texture_learning_rate_start:
            raise ValueError(
                "`texture_learning_rate_start` must be smaller than "
                "`texture_learning_rate_end`."
            )
        if self.texture_learning_rate_steepness <= 0.0:
            raise ValueError(
                "`texture_learning_rate_steepness` must be a positive real number."
            )
        if self.texture_batch_size < 1:
            raise ValueError("`batch_size` must be a positive integer")

        if self.num_of_images_to_save < 1:
            raise ValueError("`num_of_images_to_save` must be a positive integer.")


def initialize_scene(
    continuous_samples: int,
    visibility_samples: int,
    thermoscenes_scene_path: Path,
    initial_guess_mesh_path: Path,
    envmap_path: Path,
    texture_resolution: int,
) -> tuple[
    "mi.Scene",
    list["mi.Sensor"],
    list["mi.TensorXf"],
    list["mi.Sensor"],
    list["mi.TensorXf"],
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
        - `train_sensors`: List of loaded Mitsuba sensors used for training.
        - `reference_images`: List of reference images for training.
        - `evaluation_sensors`: List of loaded Mitsuba sensors used for evaluation.
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

    sensors, ref_images = preprocess_rgb_images(
        scene_path=thermoscenes_scene_path,
        environment_map_path=envmap_path,
        spp=4,
        sampler=sampler,
    )
    eval_sensors, eval_images = preprocess_rgb_images(
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

        scene_dict = {
            "type": "scene",
            "integrator": integrator.mitsuba_dict,
            "emitter": {
                "type": "envmap",
                "filename": str(envmap_path),
            },
            "shape": {
                "type": "ply",
                "filename": str(initial_guess_mesh_path),
                "bsdf": DiffuseBSDF(
                    reflectance=DiffuseBSDF.Texture(texture_path=initial_texture_path)
                ).mitsuba_dict,
            },
        }
        scene_source: "mi.Scene" = mi.load_dict(scene_dict)

    return scene_source, sensors, ref_images, eval_sensors, eval_images


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
    scene_source, sensors, ref_images, eval_sensors, eval_images = initialize_scene(
        continuous_samples=params.continuous_samples,
        visibility_samples=params.visibility_samples,
        thermoscenes_scene_path=params.thermoscenes_scene_path,
        initial_guess_mesh_path=params.initial_guess_mesh_path,
        envmap_path=params.environment_map_path,
        texture_resolution=params.texture_resolution,
    )
    if len(eval_images) < params.num_of_images_to_save:
        raise ValueError(
            f"Not enough evaluation images available. "
            f"Expected at least {params.num_of_images_to_save}, "
            f"but got {len(eval_images)}."
        )

    shape_key = "shape"
    texture_key = "shape.bsdf.reflectance.data"

    # Optimize the pose of the initial mesh
    if params.optimize_initial_pose:
        logger.info("Optimizing initial mesh pose.")

        pose_scheduler = ExponentialScheduler(
            start_learning_rate=params.pose_learning_rate_start,
            end_learning_rate=params.pose_learning_rate_end,
            steepness=params.pose_learning_rate_steepness,
            end_iteration=params.pose_optimization_iterations,
        )
        pose_optimizer = MeshPoseOptimizer(
            scene=scene_source,
            scene_params=mi.traverse(scene_source),
            optimization_events=OptimizationEvents(),
            shape_key=shape_key,
            scheduler=pose_scheduler,
            image_loss=ImageLossL1(),
        )

        pose_optimizer.optimize(
            sensors=sensors,
            ref_images=ref_images,
            scene_optimizer=SceneAdamSequential(
                batch_size=params.pose_batch_size, uniform=False
            ),
            num_iters=params.pose_optimization_iterations,
            views_to_render=eval_sensors[: params.num_of_images_to_save],
            save_directory_path=params.output_folder,
        )

    # Set up the pipeline for shape + texture optimization
    shape_scheduler = ExponentialScheduler(
        start_learning_rate=params.shape_learning_rate_start,
        end_learning_rate=params.shape_learning_rate_end,
        end_iteration=params.optimization_iterations,
        steepness=params.shape_learning_rate_steepness,
    )
    texture_scheduler = ExponentialScheduler(
        start_learning_rate=params.texture_learning_rate_start,
        end_learning_rate=params.texture_learning_rate_end,
        end_iteration=params.optimization_iterations,
        steepness=params.texture_learning_rate_steepness,
    )

    shape_loss = CompositeImageLoss(
        losses_and_weights=[
            (ImageLossL1(), params.shape_l1_loss_weight),
            (
                ImageLossMSSSIM() if params.use_msssim_loss else ImageLossL2(),
                1.0 - params.shape_l1_loss_weight,
            ),
        ]
    )
    texture_loss = CompositeImageLoss(
        losses_and_weights=[
            (ImageLossL1(), params.texture_l1_loss_weight),
            (
                ImageLossMSSSIM() if params.use_msssim_loss else ImageLossL2(),
                1.0 - params.texture_l1_loss_weight,
            ),
        ]
    )

    # Instantiate the optimizer with high projection smoothing for stability
    optimizer = MeshTextureOptimizer(
        scene=scene_source,
        scene_params=mi.traverse(scene_source),
        optimization_events=OptimizationEvents(),
        shape_key=shape_key,
        texture_key=texture_key,
        shape_smoothing_strength=params.shape_smoothing_strength,
        texture_projection_steepness=params.texture_projection_steepness,
        shape_scheduler=shape_scheduler,
        texture_scheduler=texture_scheduler,
        shape_loss=shape_loss,
        texture_loss=texture_loss,
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
    logger.info("Rendered initial state,")

    composite_scene_optimizer = CompositeSceneOptimizer(
        optimizer_config=[
            (
                SceneAdamSequential(batch_size=params.shape_batch_size, uniform=True),
                [shape_key],
            ),
            (
                SceneAdamSequential(
                    batch_size=params.texture_batch_size, uniform=False
                ),
                [texture_key],
            ),
        ],
    )

    optimizer.optimize(
        sensors=sensors,
        ref_images=ref_images,
        scene_optimizer=composite_scene_optimizer,
        num_iters=params.optimization_iterations,
        views_to_render=eval_sensors[: params.num_of_images_to_save],
        save_directory_path=params.output_folder,
    )
    logger.info("Finished optimization, rendering final state.")

    optimizer.show_comparison_with_reference(
        ref_images=ref_images,
        sensors=sensors,
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
    mlflow.log_params(parameters.loggable_dict)

    optimize_thermoscenes_scene(params=parameters)
