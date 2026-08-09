"""
Optimizes a ThermoScenes scene represented as a multi-patch B-spline surface.
Performs pose, shape, and texture optimization.
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
sys.path.append("./")

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape import logger, tracking
from rebel_shape.data_processing.image_scene_loader import ImageSceneLoader
from rebel_shape.data_processing.masked_pose_image import MaskedPoseImage
from rebel_shape.meshing.tessellated_spline_surface import TessellatedSplineSurface
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from rebel_shape.plugin_wrappers.prb_integrator import GuidingOptions, PRBIntegrator
from rebel_shape.plugin_wrappers.sampler import Sampler
from rebel_shape.property_optimizers.spline_pose_optimizer import SplinePoseOptimizer
from rebel_shape.property_optimizers.spline_texture_optimizer import (
    SplineTextureOptimizer,
)
from rebel_shape.regularization.composite_regularization import CompositeRegularization
from rebel_shape.regularization.spline_curvature_loss import SplineCurvatureLoss
from rebel_shape.regularization.spline_g1_loss import SplineG1Loss
from rebel_shape.scene_optimizers.composite_scene_optimizer import (
    CompositeSceneOptimizer,
)
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.scripts.configs.pose_optimization_config import PoseOptimizationConfig
from rebel_shape.scripts.configs.spline_shape_optimization_config import (
    SplineShapeOptimizationConfig,
)
from rebel_shape.scripts.configs.texture_optimization_config import (
    TextureOptimizationConfig,
)
from rebel_shape.splines.cube_initialization import cubed_sphere


@dataclass(kw_only=True)
class ThermoSceneSplineOptimizationParameters(ReverseCli):
    thermoscenes_scene_path: Path
    """
    Path to the ThermoScenes scene that is optimized. Assumes the foreground
    has been extracted.
    """
    environment_map_path: Path | None = None
    """
    Path to the envionment map that is used as background in optimization. If not
    provided, a solid white background is used instead.
    """
    optimize_initial_pose: bool = True
    """Whether to execute pose optimization before shape and texture optimization"""

    continuous_samples: int = 25
    """number of samples per pixel used for continuous derivatives"""
    visibility_samples: int = 9
    """number of samples per pixel used for direct and indirect visibility"""
    optimization_iterations: int = 250
    """Adam iterations executed on the initial mesh"""

    initial_sphere_radius: float = 3.0
    """
    Radius of the cubed-sphere proxy used as the starting point for pose
    optimization.
    """

    pose_params: PoseOptimizationConfig = field(
        default_factory=lambda: PoseOptimizationConfig(
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
    shape_params: SplineShapeOptimizationConfig = field(
        default_factory=lambda: SplineShapeOptimizationConfig(
            smoothing_strength=0.0,
            learning_rate_start=0.5,
            learning_rate_end=1.0,
            learning_rate_steepness=1.0,
            use_lbfgs=True,
            uniform_adam=True,
            batch_size=None,
            lbfgs_history_size=15,
            h_refinement_iterations=[20, 50, 80, 120, 180],
            degree=3,
            knot_spans=2,
            tessellation_subdivisions=256,
        )
    )
    """Parameters for shape optimization"""
    texture_params: TextureOptimizationConfig = field(
        default_factory=lambda: TextureOptimizationConfig(
            texture_resolution=512,
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
    Whether to masks are applied during the optimization process. Note that during
    the pose optimization the masks are not applied
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
    envmap_path: Path | None,
    texture_resolution: int,
    spline_degree: int,
    spline_subdivisions: int,
    mesh_subdivisions: int,
    initial_sphere_radius: float,
) -> tuple[
    "mi.Scene",
    list[MaskedPoseImage],
    list[MaskedPoseImage],
    TessellatedSplineSurface,
]:
    """
    Creates and loads Mitsuba dictionary that represents the scene, as well as
    providing the reference images and sensor views.

    `continuous_samples` is the number of samples used for optimizing continuous
    derivatives (e.g. Texture), while `visibility_samples` is the number of samples
    used for direct and indirect visibility changes.

    `thermoscenes_scene_path` is a `Path` to a scene that can be loaded
    with `ThermoScenesLoader`. `envmap_path` is the path to the 360 image that is
    used for the background during optimization. If it is not provided a solid white
    background is used instead.

    The initial mesh is a cubedsphere, which is created by tessellating a B-spline
    surface with `spline_degree` and `spline_subdivisions` knot spans. Each patch
    has 2 * `mesh_subdivisions` ** 2 triangles. its initial radius is
    `initial_sphere_radius`.

    :returns: Tuple of following elements:

        - `scene_source`: The Mitsuba scene with the tessellated B-spline surface
            patches and the environment map.
        - `reference_images`: List of reference images for training.
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

    tracking.log_params(
        {
            "spp_continuous": continuous_samples,
            "spp_projective": visibility_samples,
            "spp_indirect": visibility_samples,
        }
    )

    sampler = Sampler(type=Sampler.Type.ORTHOGONAL, sample_count=continuous_samples)

    loader = ImageSceneLoader.from_thermoscene(
        scene_path=thermoscenes_scene_path,
        environment_map_path=envmap_path,
        spp=continuous_samples,
        sensor_sampler=sampler,
    )

    # Set the initial shape of the object - ico-sphere
    cubedsphere = cubed_sphere(
        name="shape",
        degree=spline_degree,
        spline_subdivisions=spline_subdivisions,
        tessellation_subdivisions=mesh_subdivisions,
        bsdf=DiffuseBSDF(
            DiffuseBSDF.TextureFromTensor(
                tensor=dr.full(
                    dtype=mi.TensorXf,
                    value=0.5,
                    shape=(texture_resolution, texture_resolution, 3),
                )
            )
        ),
        radius=initial_sphere_radius,
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
    }
    for mesh in reversed(cubedsphere):
        scene_dict[mesh.name] = mesh.mitsuba_dict

    scene_source: "mi.Scene" = mi.load_dict(scene_dict)

    return scene_source, loader.train_images, loader.eval_images, cubedsphere


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

    Optimization can be done with masked pixels or with original pixels depending
    on the `params.mask_pixels_during_optimization` parameter.

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
    scene, ref_images, eval_images, tessellated_spline = initialize_scene(
        continuous_samples=params.continuous_samples,
        visibility_samples=params.visibility_samples,
        thermoscenes_scene_path=params.thermoscenes_scene_path,
        envmap_path=params.environment_map_path,
        texture_resolution=params.texture_params.texture_resolution,
        spline_degree=params.shape_params.degree,
        spline_subdivisions=params.shape_params.knot_spans,
        mesh_subdivisions=params.shape_params.tessellation_subdivisions,
        initial_sphere_radius=params.initial_sphere_radius,
    )

    ref_images_no_mask = [
        MaskedPoseImage(image=image.original_image, camera=image.camera, mask=None)
        for image in ref_images
    ]
    eval_images_no_mask = [
        MaskedPoseImage(image=image.original_image, camera=image.camera, mask=None)
        for image in eval_images
    ]

    scene_params = mi.traverse(scene)

    shape_key = tessellated_spline.mesh_name
    texture_key = tessellated_spline.mesh_name + "_texture"

    # Optimize the pose of the initial mesh
    if params.optimize_initial_pose:
        logger.info("Optimizing initial pose of the spline surface")

        pose_optimizer = SplinePoseOptimizer(
            scene=scene,
            scene_params=scene_params,
            optimization_events=OptimizationEvents(),
            tessellated_spline=tessellated_spline,
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

    shape_regularization = CompositeRegularization(
        regularization_terms=[
            SplineG1Loss(composite_spline=tessellated_spline, weight=1.0),
            SplineCurvatureLoss(tessellated_spline=tessellated_spline, weight=2e-3),
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
        shape_loss=params.shape_params.image_loss,
        texture_loss=params.texture_params.image_loss,
        shape_regularization=shape_regularization,
        parametrization_smoothing=params.shape_params.smoothing_strength,
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
    logger.info("Rendered initial state")

    composite_scene_optimizer = CompositeSceneOptimizer(
        optimizer_config=[
            (params.shape_params.scene_optimizer, [shape_key]),
            (params.texture_params.scene_optimizer, [texture_key]),
        ],
    )

    for iteration in params.shape_params.h_refinement_iterations:
        optimizer.optimization_events.add_event(
            iteration=iteration,
            callback=lambda: optimizer.h_refinement(
                factor=2, scene_optimizer=composite_scene_optimizer
            ),
        )
        logger.info(f"Scheduling refinement by factor 2 for iteration {iteration}")

    if not params.mask_pixels_during_optimization:
        ref_images = ref_images_no_mask
        eval_images = eval_images_no_mask

    optimizer.optimize(
        ref_images,
        scene_optimizer=composite_scene_optimizer,
        num_iters=params.optimization_iterations,
        views_to_render=[
            image.sensor for image in eval_images[: params.num_of_images_to_save]
        ],
        save_directory_path=params.output_folder,
    )
    logger.info("Finished optimization, rendering final images")

    optimizer.show_comparison_with_reference(
        ref_images=ref_images,
        samples_per_pixel=params.continuous_samples * 4,
        save_path=params.output_folder / "Final_comparison.png",
    )


if __name__ == "__main__":
    parameters = tyro.cli(ThermoSceneSplineOptimizationParameters)
    with tracking.run(artifact_directory=parameters.output_folder):
        tracking.log_params(
            parameters.to_loggable_dict(ignore=set(["thermoscenes_scene_path"]))
        )

        optimize_thermoscenes_scene(params=parameters)
