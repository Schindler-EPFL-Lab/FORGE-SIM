"""
Optimizes a ThermoScenes scene represented as a multi-patch B-spline surface.
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
sys.path.append("./src/rebel-shape")
sys.path.append("./dataclasses-reverse-cli")

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape import logger
from rebel_shape.data_processing.input_preprocessing import preprocess_rgb_images
from rebel_shape.loss_evaluators.composite_loss import CompositeImageLoss
from rebel_shape.loss_evaluators.image_loss_l1 import ImageLossL1
from rebel_shape.loss_evaluators.image_loss_l2 import ImageLossL2
from rebel_shape.loss_evaluators.image_loss_msssim import ImageLossMSSSIM
from rebel_shape.meshing.tessellated_spline_surface import TessellatedSplineSurface
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from rebel_shape.plugin_wrappers.prb_integrator import GuidingOptions, PRBIntegrator
from rebel_shape.plugin_wrappers.sampler import Sampler
from rebel_shape.property_optimizers.bitmap_texture_optimizer import (
    BitmapTextureOptimizer,
)
from rebel_shape.property_optimizers.spline_pose_optimizer import SplinePoseOptimizer
from rebel_shape.property_optimizers.spline_texture_optimizer import (
    SplineTextureOptimizer,
)
from rebel_shape.regularization.composite_regularization import CompositeRegularization
from rebel_shape.regularization.spline_g1_loss import SplineG1Loss
from rebel_shape.regularization.spline_normal_loss import SplineNormalLoss
from rebel_shape.regularization.spline_position_loss import SplinePositionLoss
from rebel_shape.scene_optimizers.composite_scene_optimizer import (
    CompositeSceneOptimizer,
)
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.scene_optimizers.scene_adam_sequential import SceneAdamSequential
from rebel_shape.schedulers.exponential_scheduler import ExponentialScheduler
from rebel_shape.splines.cube_initialization import cubed_sphere


@dataclass
class ThermoSceneSplineOptimizationParameters(ReverseCli):
    thermoscenes_scene_path: Path
    """
    Path to the ThermoScenes scene that is optimized. Assumes the foreground
    has been extracted
    """
    environment_map_path: Path
    """Path to the envionment map that is used as background in optimization."""

    shape_smoothing_strength: float = 0.0
    """Laplacian smoothing parameter for shape optimization"""
    shape_l1_loss_weight: float = 0.85
    """
    Weight of the L1 image loss for shape in composite loss. The other loss is either
    SSIM or L2, depending on `use_ssim_loss`, and is weighted by 
    1.0 - `shape_l1_loss_weight`
    """
    texture_resolution: int = 1024
    """Resolution of the texture used for optimization"""
    texture_projection_steepness: float = 1.0
    """Steepness of the sigmoid projection for texture optimization"""
    texture_l1_loss_weight: float = 0.85
    """
    Weight of the L1 image loss for texture in composite loss. The other loss is either
    SSIM or L2, depending on `use_ssim_loss`, and is weighted by 
    1.0 - `texture_l1_loss_weight`
    """
    use_msssim_loss: bool = True
    """
    If `True`, the MS-SSIM loss is used for shape and texture optimization. 
    If `False`, the L2 loss is used instead.
    """

    continuous_samples: int = 49
    """number of samples per pixel used for continuous derivatives"""
    visibility_samples: int = 9
    """number of samples per pixel used for direct and indirect visibility"""
    optimization_iterations: int = 50
    """Adam iterations executed on the initial mesh"""

    spline_degree: int = 3
    """Degree of the spline used for shape optimization"""
    spline_subdivisions: int = 2
    """Number of knot spans in each patch of the spline"""
    mesh_subdivisions: int = 256
    """Number of tessellation subdivisions used for each patch of the spline"""

    initial_sphere_radius: float = 2.0
    """Radius of the initial sphere used for optimization"""
    optimize_initial_pose: bool = True
    """Whether to execute pose optimization before shape and texture optimization"""
    pose_optimization_iterations: int = 10
    """How many times the """
    pose_learning_rate_start: float = 0.1
    """Adam learning rate at the start of initial pose optimization"""
    pose_learning_rate_end: float = 0.02
    """Adam learning rate at the end of initial pose optimization"""
    pose_learning_rate_steepness: float = 2.0
    """How quickly the pose learning rate converges to the end value"""
    pose_batch_size: int = 12
    """Number of images used for pose gradient computation"""

    shape_learning_rate_start: float = 0.15
    """Adam learning rate at the start of shape optimization"""
    shape_learning_rate_end: float = 0.002
    """Adam learning rate at the end of shape optimization"""
    shape_learning_rate_steepness: float = 1.5
    """How quickly the shape learning rate converges to the end value"""
    shape_batch_size: int = 9
    """Number of images used for shape gradient computation"""
    h_refinement_iterations: list[int] | None = None
    """
    List of iterations at which the h-refinement is applied, increasing the number
    of knot spans and thus the number of control points in the spline. The default 
    values are applied if `None` is provided: [5, 10, 15, 25].
    The values must be strictly increasing positive integers.
    """

    texture_learning_rate_start: float = 0.0025
    """Adam learning rate at the start of shape optimization"""
    texture_learning_rate_end: float = 0.025
    """Adam learning rate at the end of shape optimization"""
    texture_learning_rate_steepness: float = 2.5
    """How late in the optimization procedure the larger steps kick in"""
    texture_batch_size: int = 1
    """Number of images used for texture gradient computation"""

    output_folder: Path = Path("./outputs/")
    """Where to save the meshes and images (created if it does not exist)"""
    num_of_images_to_save: int = 4
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
        if self.shape_smoothing_strength < 0.0:
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

        if self.spline_degree < 1:
            raise ValueError("`spline_degree` must be a positive integer.")
        if self.spline_subdivisions < 1:
            raise ValueError("`spline_subdivisions` must be a positive integer.")
        if self.mesh_subdivisions < 1:
            raise ValueError("`mesh_subdivisions` must be a positive integer.")
        if self.initial_sphere_radius <= 0.0:
            raise ValueError("`initial_sphere_radius` must be a positive real number.")

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

        if self.h_refinement_iterations is None:
            self.h_refinement_iterations = [5, 10, 15, 25]
        # Check that `h_refinement_iterations` is a list of strictly increasing
        # positive integers
        prev_iteration = 0
        for iteration in self.h_refinement_iterations:
            if iteration < 1:
                raise ValueError(
                    "`h_refinement_iterations` must be a list of positive integers."
                )
            if iteration <= prev_iteration:
                raise ValueError(
                    "`h_refinement_iterations` must be a list of strictly increasing "
                    "positive integers."
                )
            prev_iteration = iteration

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
    envmap_path: Path,
    texture_resolution: int,
    spline_degree: int,
    spline_subdivisions: int,
    mesh_subdivisions: int,
    initial_sphere_radius: float,
) -> tuple[
    "mi.Scene",
    list["mi.Sensor"],
    list["mi.TensorXf"],
    list["mi.Sensor"],
    list["mi.TensorXf"],
    TessellatedSplineSurface,
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

    The initial mesh is a cubedsphere, which is created by tessellating a B-spline
    surface with `spline_degree` and `spline_subdivisions` knot spans. Each patch
    has 2 * `mesh_subdivisions` ** 2 triangles. its initial radius is
    `initial_sphere_radius`.

    :returns: Tuple of following elements:

        - `scene_source`: The Mitsuba scene with the tessellated B-spline surface
            patches and the environment map.
        - `train_sensors`: List of loaded Mitsuba sensors used for training.
        - `reference_images`: List of reference images for training.
        - `evaluation_sensors`: List of loaded Mitsuba sensors used for evaluation.
        - `evaluation_images`: List of reference images for evaluation.
        - `tessellated_spline_surface`: The tessellated spline surface used for
            optimization. Initial mesh is a cubed sphere.

    :raises ValueError: If `thermoscenes_scene_path` is not a valid path or the
        `environment_map_path` does not exist.
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

    sampler = Sampler(type=Sampler.Type.ORTHOGONAL, sample_count=continuous_samples)

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

        # Set the initial shape of the object - ico-sphere
        cubedsphere = cubed_sphere(
            name="shape",
            degree=spline_degree,
            spline_subdivisions=spline_subdivisions,
            tessellation_subdivisions=mesh_subdivisions,
            bsdf=DiffuseBSDF(DiffuseBSDF.Texture(texture_path=initial_texture_path)),
            radius=initial_sphere_radius,
        )

        scene_dict = {
            "type": "scene",
            "integrator": integrator.mitsuba_dict,
            "emitter": {
                "type": "envmap",
                "filename": str(envmap_path),
            },
        }
        for mesh in reversed(cubedsphere):
            scene_dict[mesh.name] = mesh.mitsuba_dict

        scene_source: "mi.Scene" = mi.load_dict(scene_dict)

    return scene_source, sensors, ref_images, eval_sensors, eval_images, cubedsphere


def optimize_thermoscenes_scene(
    params: ThermoSceneSplineOptimizationParameters,
) -> None:
    """
    Performs spline optimization on a scene specified by `thermoscenes_scene_path`.

    The initial guess is a cubed sphere of a specified `initial_sphere_radius. The
    guess is improved upon by `SplinePoseOptimizer` that optimizes the position,
    scale, and orientation of the initial composite spline surface for
    `pose_optimization_iterations`. This step is skipped if `optimize_initial_pose`
    is set to False.

    Optimization is performed for `optimization_iterations`  with two phases;
    each phase goes through all the images in the dataset:

    1. Shape optimization of spline global control points with `shape_batch_size` images
       and learning rate which decreases from `shape_learning_rate_start` to
       `shape_learning_rate_end` with steepness `shape_learning_rate_steepness`.
    2. Bitmap texture optimization with `texture_batch_size` images and
       learning rate which increases from `texture_learning_rate_start` to
       `texture_learning_rate_end` with steepness `texture_learning_rate_steepness`.

    Optionally, images and meshes can be saved to `params.output_folder`. The path
    should be a directory where the images will be saved. If the directory does not
    exist, it will be created, while if a `None` value is provided, nothing is saved.

    Optimization can either be executed on a GPU or CPU, depending on the
    `params.use_gpu` parameter.

    :raises ValueError: If `params.thermoscenes_scene_path` is not a valid path, or
        it is not a directory.
    """
    if not params.thermoscenes_scene_path.exists():
        raise ValueError(f"Path {params.thermoscenes_scene_path} does not exist.")
    if not params.thermoscenes_scene_path.is_dir():
        raise ValueError(f"Path {params.thermoscenes_scene_path} is not a directory.")

    if params.use_gpu:
        mi.set_variant("cuda_ad_rgb")
        logger.info("Using the GPU for rendering")
    else:
        mi.set_variant("llvm_ad_rgb")
        logger.info("Using the CPU for rendering")

    # Initialize the scene, reference images and cameras
    (scene, sensors, ref_images, eval_sensors, eval_images, tessellated_spline) = (
        initialize_scene(
            continuous_samples=params.continuous_samples,
            visibility_samples=params.visibility_samples,
            thermoscenes_scene_path=params.thermoscenes_scene_path,
            envmap_path=params.environment_map_path,
            texture_resolution=params.texture_resolution,
            spline_degree=params.spline_degree,
            spline_subdivisions=params.spline_subdivisions,
            mesh_subdivisions=params.mesh_subdivisions,
            initial_sphere_radius=params.initial_sphere_radius,
        )
    )
    scene_params = mi.traverse(scene)

    shape_key = tessellated_spline.mesh_name
    texture_key = tessellated_spline.mesh_name + "_texture"

    # Optimize the pose of the initial mesh
    if params.optimize_initial_pose:
        logger.info("Optimizing initial pose of the spline surface")

        pose_scheduler = ExponentialScheduler(
            start_learning_rate=params.pose_learning_rate_start,
            end_learning_rate=params.pose_learning_rate_end,
            steepness=params.pose_learning_rate_steepness,
            end_iteration=params.pose_optimization_iterations,
        )
        pose_optimizer = SplinePoseOptimizer(
            scene=scene,
            scene_params=scene_params,
            optimization_events=OptimizationEvents(),
            tessellated_spline=tessellated_spline,
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

    shape_regularization = CompositeRegularization(
        regularization_terms=[
            SplineG1Loss(composite_spline=tessellated_spline, weight=1.0),
            SplinePositionLoss(tessellated_spline=tessellated_spline, weight=2.0),
            SplineNormalLoss(tessellated_spline=tessellated_spline, weight=0.05),
        ],
        composite_weight=1.0e-2,
    )

    # Instantiate the optimizer with high projection smoothing for stability
    optimizer = SplineTextureOptimizer(
        scene=scene,
        scene_params=scene_params,
        optimization_events=OptimizationEvents(),
        tessellated_spline=tessellated_spline,
        shape_scheduler=shape_scheduler,
        texture_scheduler=texture_scheduler,
        shape_loss=shape_loss,
        texture_loss=texture_loss,
        shape_regularization=shape_regularization,
        parametrization_smoothing=params.shape_smoothing_strength,
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

    for iteration in params.h_refinement_iterations:
        optimizer.optimization_events.add_event(
            iteration=iteration,
            callback=lambda: optimizer.h_refinement(
                factor=2, scene_optimizer=composite_scene_optimizer
            ),
        )
        logger.info(f"Scheduling refinement by factor 2 for iteration {iteration}")

    optimizer.optimize(
        sensors=sensors,
        ref_images=ref_images,
        scene_optimizer=composite_scene_optimizer,
        num_iters=params.optimization_iterations,
        views_to_render=eval_sensors[: params.num_of_images_to_save],
        save_directory_path=params.output_folder,
    )
    logger.info("Finished optimization, rendering final images")

    optimizer.show_comparison_with_reference(
        ref_images=ref_images,
        sensors=sensors,
        samples_per_pixel=params.continuous_samples * 4,
        save_path=params.output_folder / "Final_comparison.png",
    )


if __name__ == "__main__":
    parameters = tyro.cli(ThermoSceneSplineOptimizationParameters)
    mlflow.log_params(parameters.loggable_dict)

    optimize_thermoscenes_scene(params=parameters)
