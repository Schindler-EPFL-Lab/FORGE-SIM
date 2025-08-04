"""
Simplest script to optimize the Suzanne mesh using Mitsuba 3.
Replicates the results shown in Mitsuba 3 tutorial on shape optimization and
verifies that the `MeshOptimizer` class works as expected.
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
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from rebel_shape.plugin_wrappers.prb_integrator import GuidingOptions, PRBIntegrator
from rebel_shape.plugin_wrappers.sampler import Sampler
from rebel_shape.property_optimizers.mesh_optimizer import MeshOptimizer
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.scene_optimizers.scene_adam_sequential import SceneAdamSequential
from rebel_shape.schedulers.exponential_scheduler import ExponentialScheduler
from rebel_shape.utils.image_utils import show_rendered_views
from rebel_shape.utils.mesh_utils import create_cubedsphere
from rebel_shape.utils.sensor_utils import create_sensors_fibonacci


@dataclass
class SuzanneMeshParameters(ReverseCli):
    """Parameters used for shape optimization of Suzanne"""

    sensor_count: int = 12
    """Number of camera views used in optimization"""
    reference_samples: int = 256
    """Number of samples per pixel used for the reference/target images"""
    base_samples: int = 16
    """Number of samples per pixel used for continuous derivatives"""
    resolution: int = 256
    """Number of pixels along width and height of the images"""
    optimization_iterations: int = 80
    """Number of optimization iterations with `SceneAdamBatch`"""
    start_learning_rate: float = 0.3
    """Learning rate at the beginning of the optimization"""
    end_learning_rate: float = 0.08
    """Learning rate at the end of the optimization"""
    l1_image_loss_weight: float = 0.8
    """
    The weight of L1 loss in composite loss. The other loss is MS-SSIM and it has a 
    weight 1.0 - `l1_image_loss_weight`.
    """
    initial_mesh_subdivisions: int = 5
    """How many times the cubedsphere is subdivided to create the initial mesh"""
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
        if self.base_samples < 1:
            raise ValueError("`base_samples` must be a positive integer.")
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


def initialize_scene(
    sensor_count: int,
    base_samples: int,
    reference_samples: int,
    resolution: int,
    mesh_subdivisions: int,
) -> tuple["mi.Scene", list["mi.Sensor"], list["mi.TensorXf"]]:
    """
    Sets up the scene, sensors, and reference images for shape optimization.

    There are `sensor_count` cameras and reference images. Reference images are
    rendered with `reference_samples` per pixel while `base_samples` controls
    how many samples per pixel are used during optimizations. The resolution of
    images is given by `resolution`. Reference images are shown to the user.

    The number of samples used for rendering the image and continuous derivatives
    is `base_samples` while projective and indirect ones are 4 times smaller for
    the sake of efficiency.

    The initial mesh is a cubedsphere, which is created by repeatedly subdividing
    `mesh_subdivisions` times a tessellated cube. The total number of faces is
    thus 12 * 4^`mesh_subdivisions`.

    :returns: A tuple containing:

        - `scene_source`: The Mitsuba scene with the initial cubedsphere mesh.
        - `sensors`: A list of camera sensors placed in a Fibonacci spiral around
            the origin, each with a resolution of `resolution` x `resolution`.
        - `ref_images`: A list of reference images rendered from the sensors with
            the reference mesh.
    """
    integrator = PRBIntegrator(
        max_bounces=4,
        bounces_without_russian_roulette=4,
        spp_continuous=base_samples,
        spp_projective=base_samples // 4,
        spp_indirect=base_samples // 4,
        guiding=GuidingOptions.NONE,
    )

    mlflow.log_param("spp_continuous", base_samples)
    mlflow.log_param("spp_projective", base_samples // 4)
    mlflow.log_param("spp_indirect", base_samples // 4)

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
        sampler=Sampler(type=Sampler.Type.MULTIJITTER, sample_count=base_samples),
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

    # Set the initial shape of the object - cubed sphere
    cubedsphere = create_cubedsphere(subdivisions=mesh_subdivisions)
    scene_dict["shape"] = cubedsphere
    scene_source: "mi.Scene" = mi.load_dict(scene_dict)

    return scene_source, sensors, ref_images


def optimize_suzanne(params: SuzanneMeshParameters) -> None:
    """
    Replicates the results shown in Mitsuba 3 tutorial on shape optimization.
    https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html

    Note that some aspects are slightly different (integrator settings,
    sampler type, number of samples, loss function). `params.sensor_count` sets
    how many views are optimized (default is from the tutorial).

    Optimization can either be executed on a GPU or CPU, depending on the
    `params.use_gpu` parameter.

    The Large steps smoothing strength is reduced twice during optimization, at
    0.4 * `params.optimization_iterations` and 0.7 * `params.optimization_iterations`.
    This enhances detail retention in the reconstruction.

    Optionally, images and meshes can be saved to `params.output_folder`. The path
    should be a directory where the images will be saved. If the directory does not
    exist, it will be created, while if a `None` value is provided, nothing is saved.
    """
    if params.use_gpu:
        mi.set_variant("cuda_ad_rgb")
        logger.info("Using the GPU for rendering.")
    else:
        mi.set_variant("llvm_ad_rgb")
        logger.info("Using the CPU for rendering.")

    scene_source, sensors, ref_images = initialize_scene(
        sensor_count=params.sensor_count,
        base_samples=params.base_samples,
        reference_samples=params.reference_samples,
        resolution=params.resolution,
        mesh_subdivisions=params.initial_mesh_subdivisions,
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

    optimizer = MeshOptimizer(
        scene=scene_source,
        scene_params=mi.traverse(scene_source),
        optimization_events=OptimizationEvents(),
        shape_key="shape",
        shape_scheduler=scheduler,
        shape_loss=loss,
    )

    scene_optimizer = SceneAdamSequential(batch_size=4)

    optimizer.optimization_events.add_event(
        iteration=int(params.optimization_iterations * 0.4),
        callback=lambda: optimizer.update_smoothing_strength(
            new_smoothing_strength=10.0,
            scene_optimizer=scene_optimizer,
        ),
    )
    optimizer.optimization_events.add_event(
        iteration=int(params.optimization_iterations * 0.7),
        callback=lambda: optimizer.update_smoothing_strength(
            new_smoothing_strength=5.0,
            scene_optimizer=scene_optimizer,
        ),
    )

    optimizer.optimize(
        sensors=sensors,
        ref_images=ref_images,
        scene_optimizer=scene_optimizer,
        num_iters=params.optimization_iterations,
        views_to_render=[sensors[i] for i in range(0, params.sensor_count, 2)],
        save_directory_path=params.output_folder,
    )
    logger.info("Finished optimization, rendering final views.")

    optimizer.show_comparison_with_reference(
        ref_images=ref_images,
        sensors=sensors,
        samples_per_pixel=params.reference_samples,
        save_path=params.output_folder / "Final_comparison.png",
    )

    optimizer.scene.shapes()[0].write_ply(
        str(params.output_folder / "suzanne_shape.ply")
    )


if __name__ == "__main__":
    parameters = tyro.cli(SuzanneMeshParameters)
    mlflow.log_params(parameters.loggable_dict)

    optimize_suzanne(params=parameters)
