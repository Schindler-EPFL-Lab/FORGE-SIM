"""
Example script for optimizing just the checkerboard texture of the Suzanne mesh.
Verifies the implementation of the `BitmapTextureOptimizer` class.
"""

import sys
import tempfile
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
from rebel_shape.property_optimizers.bitmap_texture_optimizer import (
    BitmapTextureOptimizer,
)
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.scene_optimizers.scene_adam_sequential import SceneAdamSequential
from rebel_shape.schedulers.constant_scheduler import ConstantScheduler
from rebel_shape.utils.image_utils import show_rendered_views
from rebel_shape.utils.sensor_utils import create_sensors_fibonacci


@dataclass
class SuzanneTextureParameters(ReverseCli):
    """Checks whether the inputted parameters are valid"""

    sensor_count: int = 8
    """Number of camera views used in optimization"""
    reference_samples: int = 256
    """number of samples per pixel used for the reference/target images"""
    base_samples: int = 16
    """number of samples per pixel used for continuous derivatives"""
    image_resolution: int = 320
    """Resolution of the rendered images along height and width"""
    texture_resolution: int = 512
    """Resolution of the texture along height and width"""
    optimization_iterations: int = 20
    """Adam iterations executed on the initial mesh"""
    learning_rate: float = 0.5
    """Adam learning rate during optimization"""
    l1_image_loss_weight: float = 0.8
    """
    The weight of L1 loss in composite loss. The other loss is MS-SSIM and it has a 
    weight 1.0 - `l1_image_loss_weight`.
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
        if self.base_samples < 1:
            raise ValueError("`base_samples` must be a positive integer.")
        if self.image_resolution < 1:
            raise ValueError("`resolution` must be a positive integer")
        if self.texture_resolution < 1:
            raise ValueError("`resolution` must be a positive integer")
        if self.optimization_iterations < 1:
            raise ValueError("`optimization_iterations` must be a positive integer.")
        if self.learning_rate <= 0.0:
            raise ValueError("`learning_rate` must be positive real number.")


def initialize_scene(
    sensor_count: int,
    base_samples: int,
    reference_samples: int,
    image_resolution: int,
    texture_resolution: int,
) -> tuple["mi.Scene", list["mi.Sensor"], list["mi.TensorXf"]]:
    """
    Sets up the scene, sensors, and reference images for shape optimization.

    There are `sensor_count` cameras and reference images. Reference images are
    rendered with `reference_samples` per pixel while `base_samples` controls
    how many samples per pixel are used during optimizations. `image_resolution` and
    `texture_resolution` specify the resolution of images and textures, respectively.
    Reference images are shown to the user.

    :returns: A tuple containing:

        - `scene_source`: The Mitsuba scene to be optimized with the initial mesh and
            texture, as well as an environment map.
        - `sensors`: A list of Mitsuba sensor instances, each representing a camera
            view in the scene.
        - `reference_images`: A list of reference images rendered from the scene
            using the sensors, each with `reference_samples` samples per pixel.
    """

    integrator = PRBIntegrator(
        max_bounces=4,
        bounces_without_russian_roulette=4,
        spp_continuous=base_samples,
        spp_projective=1,
        spp_indirect=1,
        guiding=GuidingOptions.NONE,
    )

    mlflow.log_param("spp_continuous", base_samples)

    # Reference scene initialization
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
    sensors = create_sensors_fibonacci(
        sensor_count=sensor_count,
        target_coords=(0.0, 0.0, 0.0),
        sensor_resolution=(image_resolution, image_resolution),
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

    # Modify the fields that will be optimized
    # The texture only needs to exist until mi.load_dict is called
    with tempfile.TemporaryDirectory() as temp_dir:
        initial_texture_path = Path(temp_dir) / "initial_texture.exr"
        BitmapTextureOptimizer.generate_empty_texture(
            base_color=mi.Color3f(0.5, 0.5, 0.5),
            texture_output_path=initial_texture_path,
            width=texture_resolution,
            height=texture_resolution,
        )

        scene_dict["shape"]["bsdf"] = DiffuseBSDF(
            reflectance=DiffuseBSDF.Texture(texture_path=initial_texture_path)
        ).mitsuba_dict

        scene_source: "mi.Scene" = mi.load_dict(scene_dict)

    return scene_source, sensors, ref_images


def optimize_suzanne_texture(params: SuzanneTextureParameters) -> None:
    """
    Example of texture optimization on the Suzanne mesh, used to showcase the
    `BitmapTextureOptimizer` class.

    A texture is applied to the Suzanne mesh, and it is rendered to produce reference
    images. Then, the texture is changed to a gray one, and optimization is
    carried out to recover the original texture.

    The number of samples used for rendering the image and continuous derivatives
    is `base_samples`. `reference_samples` is the number of samples used to
    render the target images.

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

    scene_source, sensors, ref_images = initialize_scene(
        sensor_count=params.sensor_count,
        base_samples=params.base_samples,
        reference_samples=params.reference_samples,
        image_resolution=params.image_resolution,
        texture_resolution=params.texture_resolution,
    )

    image_loss = CompositeImageLoss(
        losses_and_weights=[
            (ImageLossL1(), params.l1_image_loss_weight),
            (ImageLossMSSSIM(), 1.0 - params.l1_image_loss_weight),
        ]
    )

    # Optimization procedure
    optimizer = BitmapTextureOptimizer(
        scene=scene_source,
        scene_params=mi.traverse(scene_source),
        optimization_events=OptimizationEvents(),
        optimization_key="shape_texture",
        texture_keys=["shape.bsdf.reflectance.data"],
        texture_scheduler=ConstantScheduler(learning_rate=params.learning_rate),
        texture_loss=image_loss,
    )

    optimizer.optimize(
        sensors=sensors,
        ref_images=ref_images,
        scene_optimizer=SceneAdamSequential(uniform=False, batch_size=1),
        num_iters=params.optimization_iterations,
        views_to_render=[sensors[i] for i in range(0, params.sensor_count, 2)],
        save_directory_path=params.output_folder,
    )
    logger.info("Finished optimization, rendering final images.")

    optimizer.show_comparison_with_reference(
        ref_images=ref_images,
        sensors=sensors,
        samples_per_pixel=params.reference_samples,
        save_path=params.output_folder / "Final_comparison.png",
    )

    # Save the final state of the scene
    texture = mi.Bitmap(optimizer.scene_params["shape.bsdf.reflectance.data"])
    texture.write(str(params.output_folder / "suzanne_rgb_texture.exr"))


if __name__ == "__main__":
    parameters = tyro.cli(SuzanneTextureParameters)
    mlflow.log_params(parameters.loggable_dict)

    optimize_suzanne_texture(params=parameters)
