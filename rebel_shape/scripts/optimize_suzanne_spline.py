"""
Example script for optimizing Suzanne as a multi-patch B-spline surface.
Verifies the implementation of the `SplineOptimizer` class.
"""

import sys
import time
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
from rebel_shape.loss_evaluators.composite_loss import CompositeImageLoss
from rebel_shape.loss_evaluators.image_loss_l1 import ImageLossL1
from rebel_shape.loss_evaluators.image_loss_msssim import ImageLossMSSSIM
from rebel_shape.meshing.tessellated_spline_surface import TessellatedSplineSurface
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from rebel_shape.plugin_wrappers.prb_integrator import GuidingOptions, PRBIntegrator
from rebel_shape.plugin_wrappers.sampler import Sampler
from rebel_shape.property_optimizers.spline_optimizer import SplineOptimizer
from rebel_shape.regularization.composite_regularization import CompositeRegularization
from rebel_shape.regularization.spline_g1_loss import SplineG1Loss
from rebel_shape.regularization.spline_normal_loss import SplineNormalLoss
from rebel_shape.regularization.spline_position_loss import SplinePositionLoss
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.scene_optimizers.scene_adam_sequential import SceneAdamSequential
from rebel_shape.scene_optimizers.scene_lbfgs import SceneLBFGS
from rebel_shape.schedulers.exponential_scheduler import ExponentialScheduler
from rebel_shape.splines.cube_initialization import cubed_sphere
from rebel_shape.utils.image_utils import show_rendered_views
from rebel_shape.utils.sensor_utils import create_sensors_fibonacci


@dataclass
class SuzanneSplineParameters(ReverseCli):
    """Parameters used for spline optimization of Suzanne"""

    sensor_count: int = 12
    """Number of camera views used in optimization"""
    reference_samples: int = 961
    """Number of samples per pixel used for the reference/target images"""
    continuous_samples: int = 49
    """Number of samples per pixel used for continuous derivatives"""
    visibility_samples: int = 9
    """Number of samples per pixel used for direct and indirect visibility gradients"""
    resolution: int = 480
    """Number of pixels along width and height of the images"""
    optimization_iterations: int = 150
    """Number of optimization iterations with `SceneAdamBatch`"""
    start_learning_rate: float = 0.15
    """Learning rate at the beginning of the optimization"""
    end_learning_rate: float = 0.03
    """Learning rate at the end of the optimization"""
    l1_image_loss_weight: float = 0.85
    """
    The weight of L1 loss in composite loss. The other loss is MS-SSIM and it has a 
    weight 1.0 - `l1_image_loss_weight`.
    """
    use_lbfgs: bool = False
    """
    Whether to use LBFGS optimizer instead of Adam. If set to True, the
    `SceneLBFGS` optimizer is used instead of `SceneAdam`. L-BFGS typically requires
    larger learning rates to be effective (close to 1.0), while these are generally
    too large for Adam.
    """

    spline_degree: int = 3
    """Degree of the B-spline surface patches."""
    spline_knot_spans: int = 2
    """How many knot spans per patch in the B-spline surface."""
    tessellation_subdivisions: int = 256
    """How many triangles per patch in the tessellated mesh."""
    parametrization_smoothing: float = 0.0
    """The strength of gradient smoothing applied to the spline parametrization."""
    h_refinement_iterations: list[int] | None = None
    """
    List of iterations at which the h-refinement is applied, increasing the number
    of knot spans and thus the number of control points in the spline. The default 
    values are applied if `None` is provided: [5, 10, 15, 25].
    The values must be strictly increasing positive integers.
    """

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

        if self.start_learning_rate <= 0.0:
            raise ValueError("`start_learning_rate` must be positive real number.")
        if self.end_learning_rate < 0.0:
            raise ValueError("`end_learning_rate` must be non-negative real number.")
        if self.end_learning_rate >= self.start_learning_rate:
            raise ValueError(
                "`end_learning_rate` must be smaller than `start_learning_rate`."
            )
        if self.l1_image_loss_weight < 0.0 or self.l1_image_loss_weight > 1.0:
            raise ValueError("`l1_image_loss_weight` must be in the range [0.0, 1.0].")

        if self.spline_degree < 1:
            raise ValueError("`spline_degree` must be a positive integer.")
        if self.spline_knot_spans < 1:
            raise ValueError("`spline_knot_spans` must be a positive integer.")
        if self.tessellation_subdivisions < 1:
            raise ValueError("`tessellation_subdivisions` must be a positive integer.")
        if self.parametrization_smoothing < 0.0:
            raise ValueError(
                "`parametrization_smoothing` must be a non-negative real number."
            )

        if self.h_refinement_iterations is None:
            self.h_refinement_iterations = [15, 30, 50, 75]
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


def initialize_scene(
    sensor_count: int,
    continuous_samples: int,
    visibility_samples: int,
    reference_samples: int,
    resolution: int,
    spline_degree: int,
    spline_subdivisions: int,
    mesh_subdivisions: int,
) -> tuple[
    "mi.Scene", list["mi.Sensor"], list["mi.TensorXf"], TessellatedSplineSurface
]:
    """
    Sets up the scene, sensors, reference images, and tessellated composite B-Spline
    surface for shape optimization.

    There are `sensor_count` cameras and reference images. Reference images are
    rendered with `reference_samples` per pixel while `continuous_samples` controls
    how many samples per pixel are used for rendering during optimization. The
    resolution of images is given by `resolution`. Reference images are shown to the
    user.

    The number of samples used for rendering the image and continuous derivatives
    is `continuous_samples` while `visibility_samples` control the number of projective
    and indirect gradient samples.

    The initial mesh is a cubedsphere, which is created by tessellating a B-spline
    surface with `spline_degree` and `spline_subdivisions` knot spans. Each patch
    has 2 * `mesh_subdivisions` ** 2 triangles.

    :returns: A tuple containing:

        - `scene_source`: The Mitsuba scene with the tessellated B-spline surface
            patches and the environment map.
        - `sensors`: A list of sensors created using Fibonacci spiral distribution.
        - `ref_images`: A list of reference images rendered from the sensors with
            `reference_samples` samples per pixel.
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

    script_dir = Path(__file__).parent.resolve()
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
            "bsdf": DiffuseBSDF().mitsuba_dict,
        },
    }

    # Target images are obtained through rendering
    scene_target: "mi.Scene" = mi.load_dict(scene_dict)
    sensors = create_sensors_fibonacci(
        sensor_count=sensor_count,
        target_coords=(0.0, 0.0, 0.0),
        sensor_resolution=(resolution, resolution),
        sampler=Sampler(type=Sampler.Type.ORTHOGONAL, sample_count=continuous_samples),
    )

    start_time = time.time()
    ref_images = [
        mi.render(scene_target, sensor=sensors[i], spp=reference_samples)
        for i in range(sensor_count)
    ]
    show_rendered_views(
        views=[mi.util.convert_to_bitmap(image) for image in ref_images]
    )
    logger.info(f"Rendered target images. Took {time.time() - start_time:.2f} seconds.")

    # Set the initial shape of the object - ico-sphere
    cubedsphere = cubed_sphere(
        name="suzanne",
        degree=spline_degree,
        spline_subdivisions=spline_subdivisions,
        tessellation_subdivisions=mesh_subdivisions,
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

    return scene_source, sensors, ref_images, cubedsphere


def optimize_suzanne(params: SuzanneSplineParameters) -> None:
    """
    Optimizes a cubedsphere B-Spline surface to match the reference images of
    a diffuse Suzanne mesh.

    A simple example of spline shape optimization used to explore the properties of
    spline optimization in direct comparison to mesh optimization from
    `optimize_suzanne_mesh.py`.

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

    scene_source, sensors, ref_images, tessellated_spline = initialize_scene(
        sensor_count=params.sensor_count,
        continuous_samples=params.continuous_samples,
        visibility_samples=params.visibility_samples,
        reference_samples=params.reference_samples,
        resolution=params.resolution,
        spline_degree=params.spline_degree,
        spline_subdivisions=params.spline_knot_spans,
        mesh_subdivisions=params.tessellation_subdivisions,
    )

    scheduler = ExponentialScheduler(
        start_learning_rate=params.start_learning_rate,
        end_learning_rate=params.end_learning_rate,
        end_iteration=params.optimization_iterations,
    )

    loss = CompositeImageLoss(
        losses_and_weights=[
            (ImageLossL1(), params.l1_image_loss_weight),
            (ImageLossMSSSIM(), 1.0 - params.l1_image_loss_weight),
        ]
    )

    regularization = CompositeRegularization(
        [
            SplineG1Loss(composite_spline=tessellated_spline, weight=1.0),
            SplinePositionLoss(tessellated_spline=tessellated_spline, weight=2.0),
            SplineNormalLoss(tessellated_spline=tessellated_spline, weight=0.05),
        ],
        composite_weight=1.0e-2,
    )

    optimizer = SplineOptimizer(
        scene=scene_source,
        scene_params=mi.traverse(scene_source),
        optimization_events=OptimizationEvents(),
        tessellated_spline=tessellated_spline,
        shape_scheduler=scheduler,
        shape_loss=loss,
        regularization=regularization,
        smoothing_strength=params.parametrization_smoothing,
    )

    scene_optimizer = (
        SceneLBFGS() if params.use_lbfgs else SceneAdamSequential(batch_size=4)
    )

    for iteration in params.h_refinement_iterations:
        optimizer.optimization_events.add_event(
            iteration=iteration,
            callback=lambda: optimizer.h_refinement(
                factor=2, scene_optimizer=scene_optimizer
            ),
        )
        logger.info(f"Scheduling refinement by factor 2 for iteration {iteration}")

    optimizer.optimize(
        sensors=sensors,
        ref_images=ref_images,
        scene_optimizer=scene_optimizer,
        num_iters=params.optimization_iterations,
        views_to_render=[sensors[i] for i in range(0, params.sensor_count, 4)],
        save_directory_path=params.output_folder,
    )
    logger.info("Finished optimization, rendering final state of the scene.")

    optimizer.show_comparison_with_reference(
        ref_images=ref_images,
        sensors=sensors,
        samples_per_pixel=params.reference_samples,
        save_path=params.output_folder / "Final_comparison.png",
    )


if __name__ == "__main__":
    parameters = tyro.cli(SuzanneSplineParameters)
    mlflow.log_params(parameters.loggable_dict)

    optimize_suzanne(params=parameters)
