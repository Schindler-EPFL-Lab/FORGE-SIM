"""
Example script for optimizing an arbitrary ply shape as a multi-patch B-spline surface.
Verifies the implementation of the `SplineOptimizer` class.
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mitsuba as mi
import mlflow
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append("./")


from rebel_shape import logger
from rebel_shape.data_processing.masked_pose_image import MaskedPoseImage
from rebel_shape.meshing.tessellated_spline_surface import TessellatedSplineSurface
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from rebel_shape.plugin_wrappers.prb_integrator import GuidingOptions, PRBIntegrator
from rebel_shape.plugin_wrappers.sampler import Sampler
from rebel_shape.property_optimizers.spline_optimizer import SplineOptimizer
from rebel_shape.regularization.composite_regularization import CompositeRegularization
from rebel_shape.regularization.spline_curvature_loss import SplineCurvatureLoss
from rebel_shape.regularization.spline_g1_loss import SplineG1Loss
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.scripts.configs.spline_shape_optimization_config import (
    SplineShapeOptimizationConfig,
)
from rebel_shape.splines.cube_initialization import cubed_sphere
from rebel_shape.utils.image_utils import show_rendered_views
from rebel_shape.utils.sensor_utils import create_cameras_fibonacci


@dataclass(kw_only=True)
class PLYSplineParameters(ReverseCli):
    """Parameters used for spline optimization of an arbitrary shape"""

    shape_path: Path = Path("./data/test_scenes/meshes/armadillo.ply")
    """Path to the .ply shape"""
    environment_map_path: Path = Path("./data/test_scenes/textures/envmap2.exr")
    """
    Path to the envionment map that is used as background in optimization
    """
    sensor_count: int = 19
    """Number of camera views used in optimization"""
    reference_samples: int = 4096
    """Number of samples per pixel used for the reference/target images"""
    continuous_samples: int = 64
    """Number of samples per pixel used for continuous derivatives"""
    visibility_samples: int = 16
    """Number of samples per pixel used for direct and indirect visibility gradients"""
    resolution: int = 512
    """Number of pixels along width and height of the images"""
    optimization_iterations: int = 250
    """Number of optimization iterations with `SceneAdamBatch`"""

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
            h_refinement_iterations=[25, 50, 80, 140],  # Add 250 for armadillo
            degree=3,
            knot_spans=2,
            tessellation_subdivisions=256,
        )
    )
    """Parameters for shape optimization"""

    output_folder: Path = Path("./outputs/")
    """Where to save the meshes and images (created if it does not exist)"""

    use_gpu: bool = True
    """
    Whether to use GPU for rendering or not. Note that the compute instance
    must have GPU support for this to work. For local testing, set to False.
    """

    def __post_init__(self) -> None:
        if self.sensor_count < 1:
            raise ValueError("`sensor_count` must be a positive integer.")
        if self.reference_samples < 1:
            raise ValueError("`reference_samples` must be a positive integer.")
        if self.continuous_samples < 1:
            raise ValueError("`continuous_samples` must be a positive integer.")
        if self.visibility_samples < 1:
            raise ValueError("`visibility_samples` must be a positive integer.")
        if self.resolution < 1:
            raise ValueError("`resolution` must be a positive integer")
        if self.optimization_iterations < 1:
            raise ValueError("`optimization_iterations` must be a positive integer.")
        if not self.shape_path.exists():
            raise ValueError("`shape_path` does not exist.")
        if self.shape_path.suffix != ".ply":
            raise ValueError("`shape_path` is not a .ply file.")
        if not self.environment_map_path.exists():
            raise ValueError("`environment_map_path` does not exist.")


def initialize_scene(
    sensor_count: int,
    shape_path: Path,
    envmap_path: Path,
    continuous_samples: int,
    visibility_samples: int,
    reference_samples: int,
    resolution: int,
    spline_degree: int,
    knot_spans: int,
    tessellation_subdivisions: int,
    output_folder: Path,
) -> tuple["mi.Scene", list[MaskedPoseImage], TessellatedSplineSurface]:
    """
    Sets up the scene, sensors, reference images, and tessellated composite B-Spline
    surface for shape optimization.

    There are `sensor_count` reference images of the shape from `shape_path` using the
    environmental map from `envmap_path`. Reference images are rendered with
    `reference_samples` per pixel while `continuous_samples` controls how many samples
    per pixel are used for rendering during optimization. The resolution of images is
    given by `resolution`. Reference images are shown to the user and saved to
    `output_folder`/reference_views.png

    The number of samples used for rendering the image and continuous derivatives
    is `continuous_samples` while `visibility_samples` control the number of projective
    and indirect gradient samples.

    The initial mesh is a cubedsphere, which is created by tessellating a B-spline
    surface with `spline_degree` and `spline_subdivisions` knot spans. Each patch
    has 2 * `mesh_subdivisions` ** 2 triangles.

    :returns: A tuple containing:

        - `scene_source`: The Mitsuba scene with the tessellated B-spline surface
            patches and the environment map.
        - `ref_images`: A list of reference images rendered from the sensors with
            `reference_samples` samples per pixel. Each sensor is created using
            Fibonacci spiral distribution.
        - `tessellated_spline`: The tessellated B-spline surface used for optimization.

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

    scene_dict = {
        "type": "scene",
        "integrator": integrator.mitsuba_dict,
        "emitter": {
            "type": "envmap",
            "filename": str(envmap_path),
        },
        "shape": {
            "type": "ply",
            "filename": str(shape_path),
            "bsdf": DiffuseBSDF().mitsuba_dict,
        },
    }

    # Target images are obtained through rendering
    scene_target: "mi.Scene" = mi.load_dict(scene_dict)
    cameras = create_cameras_fibonacci(
        sensor_count=sensor_count,
        target_coords=(0.0, 0.0, 0.0),
        sensor_resolution=(resolution, resolution),
        sampler=Sampler(type=Sampler.Type.MULTIJITTER, sample_count=continuous_samples),
    )

    start_time = time.time()
    ref_images = [
        mi.render(scene_target, sensor=camera.mitsuba_sensor, spp=reference_samples)
        for camera in cameras
    ]
    show_rendered_views(
        views=[mi.util.convert_to_bitmap(image) for image in ref_images],
        save_path=output_folder / "reference_views.png",
    )
    ref_images = [
        MaskedPoseImage(
            image=ref_image,
            camera=camera,
            mask=None,
        )
        for ref_image, camera in zip(ref_images, cameras)
    ]
    logger.info(f"Rendered target images. Took {time.time() - start_time:.2f} seconds.")

    # Set the initial shape of the object - ico-sphere
    cubedsphere = cubed_sphere(
        name="spline_surface",
        degree=spline_degree,
        spline_subdivisions=knot_spans,
        tessellation_subdivisions=tessellation_subdivisions,
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

    return scene_source, ref_images, cubedsphere


def optimize_ply(params: PLYSplineParameters) -> None:
    """
    Optimizes a cubedsphere B-Spline surface to match the reference images of
    a diffuse ply mesh.

    A simple example of spline shape optimization used to explore the properties of
    spline optimization in direct comparison to mesh optimization from
    `optimize_ply_mesh.py`.

    Unlike for the mesh optimization, some regularization terms are applied to ensure
    smoothness between the patches as well as a light penalty to avoid pinching
    artifacts. One can either use a coarse-to-fine scheme or gradient preconditioning
    to optimize the spline surface; for meshes preconditioning is effectively mandatory.

    Optionally, images and meshes can be saved to `params.output_folder`. The path
    should be a directory where the images will be saved. If the directory does not
    exist, it will be created, while if a `None` value is provided, nothing is saved.

    Optimization can either be executed on a GPU or CPU, depending on the
    `params.use_gpu` parameter.
    """
    if params.use_gpu:
        mi.set_variant("cuda_ad_rgb")
        logger.info("Using the GPU for rendering.")
    else:
        mi.set_variant("llvm_ad_rgb")
        logger.info("Using the CPU for rendering.")

    params.output_folder.mkdir(parents=True, exist_ok=True)
    scene_source, ref_images, tessellated_spline = initialize_scene(
        sensor_count=params.sensor_count,
        shape_path=params.shape_path,
        envmap_path=params.environment_map_path,
        continuous_samples=params.continuous_samples,
        visibility_samples=params.visibility_samples,
        reference_samples=params.reference_samples,
        resolution=params.resolution,
        spline_degree=params.shape_params.degree,
        knot_spans=params.shape_params.knot_spans,
        tessellation_subdivisions=params.shape_params.tessellation_subdivisions,
        output_folder=params.output_folder,
    )

    regularization = CompositeRegularization(
        [
            SplineG1Loss(composite_spline=tessellated_spline, weight=1.0),
            SplineCurvatureLoss(tessellated_spline=tessellated_spline, weight=2e-3),
        ],
        composite_weight=1.0e-2,
    )

    optimizer = SplineOptimizer(
        scene=scene_source,
        scene_params=mi.traverse(scene_source),
        optimization_events=OptimizationEvents(),
        tessellated_spline=tessellated_spline,
        shape_scheduler=params.shape_params.scheduler(
            end_iteration=params.optimization_iterations,
        ),
        shape_loss=params.shape_params.image_loss,
        regularization=regularization,
        smoothing_strength=params.shape_params.smoothing_strength,
    )

    scene_optimizer = params.shape_params.scene_optimizer

    for iteration in params.shape_params.h_refinement_iterations:
        optimizer.optimization_events.add_event(
            iteration=iteration,
            callback=lambda: optimizer.h_refinement(
                factor=2, scene_optimizer=scene_optimizer
            ),
        )
        logger.info(f"Scheduling refinement by factor 2 for iteration {iteration}")

    optimizer.optimize(
        ref_images=ref_images,
        scene_optimizer=scene_optimizer,
        num_iters=params.optimization_iterations,
        views_to_render=[
            ref_images[i].sensor for i in range(0, params.sensor_count, 4)
        ],
        save_directory_path=params.output_folder,
    )
    logger.info("Finished optimization, rendering final state of the scene.")

    optimizer.show_comparison_with_reference(
        ref_images=ref_images,
        samples_per_pixel=params.reference_samples,
        save_path=params.output_folder / "Final_comparison.png",
    )


if __name__ == "__main__":
    parameters = tyro.cli(PLYSplineParameters)
    mlflow.log_params(parameters.to_loggable_dict())

    optimize_ply(params=parameters)
