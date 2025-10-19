"""
Example script for optimizing the multi-patch B-spline surface of the Suzanne mesh,
including both the shape and texture of the surface.
Verifies the correctness of the `SplineTextureOptimizer` class.
"""

import sys
import time
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
sys.path.append("./")

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape import logger
from rebel_shape.data_processing.masked_pose_image import MaskedPoseImage
from rebel_shape.meshing.tessellated_spline_surface import TessellatedSplineSurface
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from rebel_shape.plugin_wrappers.prb_integrator import GuidingOptions, PRBIntegrator
from rebel_shape.plugin_wrappers.sampler import Sampler
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
from rebel_shape.scripts.configs.spline_shape_optimization_config import (
    SplineShapeOptimizationConfig,
)
from rebel_shape.scripts.configs.texture_optimization_config import (
    TextureOptimizationConfig,
)
from rebel_shape.splines.cube_initialization import cubed_sphere
from rebel_shape.utils.image_utils import show_rendered_views
from rebel_shape.utils.sensor_utils import create_cameras_fibonacci


@dataclass(kw_only=True)
class SuzanneSplineTextureParameters(ReverseCli):
    """Parameters used for simultaneous spline and texture optimization of Suzanne"""

    sensor_count: int = 19
    """Number of camera views used in optimization"""
    reference_samples: int = 4096
    """Number of samples per pixel used for the reference/target images"""
    continuous_samples: int = 64
    """Number of samples per pixel used for continuous derivatives"""
    visibility_samples: int = 16
    """number of samples per pixel used for direct and indirect visibility gradients"""
    image_resolution: int = 512
    """Resolution of the rendered images along height and width"""
    optimization_iterations: int = 250
    """Adam learning rate in the first pass"""

    shape_params: SplineShapeOptimizationConfig = field(
        default_factory=lambda: SplineShapeOptimizationConfig(
            smoothing_strength=0.0,
            learning_rate_start=1.0,
            learning_rate_end=1.0,
            learning_rate_steepness=1.0,
            use_lbfgs=True,
            uniform_adam=True,
            batch_size=None,
            lbfgs_history_size=10,
            h_refinement_iterations=[25, 50, 80, 140],
            degree=3,
            knot_spans=2,
            tessellation_subdivisions=256,
        )
    )
    """Parameters for shape optimization"""
    texture_params: TextureOptimizationConfig = field(
        default_factory=lambda: TextureOptimizationConfig(
            texture_resolution=512,
            learning_rate_start=0.03,
            learning_rate_end=0.12,
            learning_rate_steepness=2.0,
            use_lbfgs=False,
            uniform_adam=False,
            batch_size=2,
        )
    )
    """Parameters for texture optimization"""

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
        if self.image_resolution < 1:
            raise ValueError("`resolution` must be a positive integer")
        if self.optimization_iterations < 1:
            raise ValueError("`optimization_iterations` must be a positive integer.")


def initialize_scene(
    sensor_count: int,
    continuous_samples: int,
    visibility_samples: int,
    reference_samples: int,
    image_resolution: int,
    texture_resolution: int,
    spline_degree: int,
    spline_knot_spans: int,
    tessellation_subdivisions: int,
) -> tuple["mi.Scene", list[MaskedPoseImage], TessellatedSplineSurface]:
    """
    Sets up the scene, sensors, reference images, and tessellated composite B-Spline
    surface for shape optimization.

    There are `sensor_count` cameras and reference images. Reference images are
    rendered with `reference_samples` per pixel while `continuous_samples` controls
    how many samples per pixel are used for rendering during optimization.

    `image_resolution` and `texture_resolution` specify the resolution of images and
    textures, respectively. Reference images are shown to the user.

    The number of samples used for rendering the image and continuous derivatives
    is `continuous_samples` while `visibility_samples` control the number of projective
    and indirect gradient samples.

    The initial mesh is a cubedsphere, which is created by tessellating a B-spline
    surface with `spline_degree` and `spline_subdivisions` knot spans. Each patch
    has 2 * `mesh_subdivisions` ** 2 triangles.

    Optimization can either be executed on a GPU or CPU, depending on the
    `params.use_gpu` parameter.

    :returns: The loaded scene to be optimized, a list of reference images
    and the tessellated B-spline surface.
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

    script_dir = Path(__file__).parent.resolve()
    target_texture_path = (
        script_dir / "../../data/test_scenes/textures/checkerboard_texture.jpg"
    )

    scene_dict = {
        "type": "scene",
        "integrator": integrator.mitsuba_dict,
        "emitter": {
            "type": "envmap",
            "filename": str(script_dir / "../../data/test_scenes/textures/envmap2.exr"),
        },
        "shape": {
            "type": "ply",
            "filename": str(script_dir / "../../data/test_scenes/meshes/suzanne.ply"),
            "bsdf": DiffuseBSDF(
                reflectance=DiffuseBSDF.Texture(texture_path=target_texture_path)
            ).mitsuba_dict,
        },
    }

    # Target images are obtained through rendering
    scene_target: "mi.Scene" = mi.load_dict(scene_dict)
    cameras = create_cameras_fibonacci(
        sensor_count=sensor_count,
        target_coords=(0.0, 0.0, 0.0),
        sensor_resolution=(image_resolution, image_resolution),
        sampler=Sampler(type=Sampler.Type.MULTIJITTER, sample_count=continuous_samples),
    )

    start_time = time.time()
    ref_images = [
        mi.render(scene_target, sensor=camera.mitsuba_sensor, spp=reference_samples)
        for camera in cameras
    ]
    show_rendered_views(
        views=[mi.util.convert_to_bitmap(image) for image in ref_images]
    )

    ref_images = [
        MaskedPoseImage(
            image=ref_image,
            camera=camera,
            mask=None,
        )
        for ref_image, camera in zip(ref_images, cameras)
    ]
    logger.info(f"Rendered target images. Took {time.time() - start_time:.2f} seconds")

    # Set the initial texture of the object - solid grey

    cubedsphere = cubed_sphere(
        name="suzanne",
        degree=spline_degree,
        spline_subdivisions=spline_knot_spans,
        tessellation_subdivisions=tessellation_subdivisions,
        bsdf=DiffuseBSDF(
            DiffuseBSDF.TextureFromTensor(
                tensor=dr.full(
                    dtype=mi.TensorXf,
                    value=0.5,
                    shape=(texture_resolution, texture_resolution, 3),
                )
            )
        ),
    )

    scene_dict = {
        "type": "scene",
        "integrator": integrator.mitsuba_dict,
        "emitter": {
            "type": "envmap",
            "filename": str(script_dir / "../../data/test_scenes/textures/envmap2.exr"),
        },
    }
    for mesh in reversed(cubedsphere):
        scene_dict[mesh.name] = mesh.mitsuba_dict

    scene_source: "mi.Scene" = mi.load_dict(scene_dict)

    return scene_source, ref_images, cubedsphere


def optimize_suzanne(params: SuzanneSplineTextureParameters) -> None:
    """
    Replicates the results shown in Mitsuba 3 tutorial on shape optimization.
    https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html

    A simple example of spline shape and texture optimization used to verify the
    correctness of the `SplineTextureOptimizer` class. It can also be directly
    compared to the mesh optimization example in `optimize_suzanne_mesh_texture.py`.
    Unlike for the mesh optimization, some regularization terms are applied to ensure
    smoothness between the patches as well as a light penalty to avoid pinching
    artifacts.

    Optionally, images and meshes can be saved to `params.output_folder`. The path
    should be a directory where the images will be saved. If the directory does not
    exist, it will be created, while if a `None` value is provided, nothing is saved.

    Optimization can either be executed on a GPU or CPU, depending on the
    `params.use_gpu` parameter.
    """
    if params.use_gpu:
        mi.set_variant("cuda_ad_rgb")
    else:
        mi.set_variant("llvm_ad_rgb")

    scene_source, ref_images, tessellated_spline = initialize_scene(
        sensor_count=params.sensor_count,
        continuous_samples=params.continuous_samples,
        visibility_samples=params.visibility_samples,
        reference_samples=params.reference_samples,
        image_resolution=params.image_resolution,
        texture_resolution=params.texture_params.texture_resolution,
        spline_degree=params.shape_params.degree,
        spline_knot_spans=params.shape_params.knot_spans,
        tessellation_subdivisions=params.shape_params.tessellation_subdivisions,
    )

    # Configuring the property optimizer
    shape_key = tessellated_spline.mesh_name
    texture_key = tessellated_spline.mesh_name + "_texture"

    shape_regularization = CompositeRegularization(
        regularization_terms=[
            SplineG1Loss(composite_spline=tessellated_spline, weight=1.0),
            SplineCurvatureLoss(tessellated_spline=tessellated_spline, weight=2e-3),
        ],
        composite_weight=1.0e-2,
    )

    optimizer = SplineTextureOptimizer(
        scene=scene_source,
        scene_params=mi.traverse(scene_source),
        optimization_events=OptimizationEvents(),
        tessellated_spline=tessellated_spline,
        shape_scheduler=params.shape_params.scheduler(
            end_iteration=params.optimization_iterations,
        ),
        texture_scheduler=params.texture_params.scheduler(
            end_iteration=params.optimization_iterations,
        ),
        shape_loss=params.shape_params.image_loss,
        texture_loss=params.texture_params.image_loss,
        shape_regularization=shape_regularization,
        parametrization_smoothing=params.shape_params.smoothing_strength,
    )

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

    optimizer.optimize(
        ref_images=ref_images,
        scene_optimizer=composite_scene_optimizer,
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
    parameters = tyro.cli(SuzanneSplineTextureParameters)
    mlflow.log_params(parameters.to_loggable_dict())

    optimize_suzanne(params=parameters)
