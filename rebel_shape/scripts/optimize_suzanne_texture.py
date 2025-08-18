"""
Example script for optimizing just the checkerboard texture of the Suzanne mesh.
Verifies the implementation of the `BitmapTextureOptimizer` class.
"""

import sys
import tempfile
import time
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
from rebel_shape.data_processing.masked_pose_image import MaskedPoseImage
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from rebel_shape.plugin_wrappers.prb_integrator import GuidingOptions, PRBIntegrator
from rebel_shape.plugin_wrappers.sampler import Sampler
from rebel_shape.property_optimizers.bitmap_texture_optimizer import (
    BitmapTextureOptimizer,
)
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.scripts.configs.texture_optimization_config import (
    TextureOptimizationConfig,
)
from rebel_shape.utils.image_utils import show_rendered_views
from rebel_shape.utils.sensor_utils import create_cameras_fibonacci


@dataclass(kw_only=True)
class SuzanneTextureParameters(ReverseCli):
    """Checks whether the inputted parameters are valid"""

    sensor_count: int = 12
    """Number of camera views used in optimization"""
    reference_samples: int = 1024
    """number of samples per pixel used for the reference/target images"""
    base_samples: int = 64
    """number of samples per pixel used for continuous derivatives"""
    image_resolution: int = 480
    """Resolution of the rendered images along height and width"""
    texture_resolution: int = 1024
    """Resolution of the texture along height and width"""
    optimization_iterations: int = 20
    """Adam iterations executed on the initial mesh"""

    texture_params: TextureOptimizationConfig = field(
        default_factory=lambda: TextureOptimizationConfig(
            texture_resolution=1024,
            learning_rate_start=0.5,
            learning_rate_end=0.5,
            learning_rate_steepness=1.0,
            use_lbfgs=False,
            uniform_adam=False,
            batch_size=3,
            l1_loss_weight=0.8,
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
        if self.base_samples < 1:
            raise ValueError("`base_samples` must be a positive integer.")
        if self.image_resolution < 1:
            raise ValueError("`resolution` must be a positive integer")
        if self.texture_resolution < 1:
            raise ValueError("`resolution` must be a positive integer")
        if self.optimization_iterations < 1:
            raise ValueError("`optimization_iterations` must be a positive integer.")


def initialize_scene(
    sensor_count: int,
    base_samples: int,
    reference_samples: int,
    image_resolution: int,
    texture_resolution: int,
) -> tuple["mi.Scene", list[MaskedPoseImage]]:
    """
    Sets up the scene, and reference images for shape optimization.

    There are `sensor_count` cameras and reference images. Reference images are
    rendered with `reference_samples` per pixel while `base_samples` controls
    how many samples per pixel are used during optimizations. `image_resolution` and
    `texture_resolution` specify the resolution of images and textures, respectively.
    Reference images are shown to the user.

    :returns: A tuple containing:

        - `scene_source`: The Mitsuba scene to be optimized with the initial mesh and
            texture, as well as an environment map.
        - `reference_images`: A list of reference images rendered from the scene
            using the sensors, each with `reference_samples` samples per pixel. Each
            sensor is created using Fibonacci sampling
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
    cameras = create_cameras_fibonacci(
        sensor_count=sensor_count,
        target_coords=(0.0, 0.0, 0.0),
        sensor_resolution=(image_resolution, image_resolution),
        sampler=Sampler(type=Sampler.Type.MULTIJITTER, sample_count=base_samples),
    )
    start_time = time.time()
    ref_images = [
        mi.render(scene_target, sensor=camera.mitsuba_sensor, spp=reference_samples)
        for camera in cameras
    ]
    ref_images = [
        MaskedPoseImage(image=ref_image, camera=camera, mask=None)
        for ref_image, camera in zip(ref_images, cameras)
    ]

    show_rendered_views(
        views=[mi.util.convert_to_bitmap(image.image) for image in ref_images]
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

    return scene_source, ref_images


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

    scene_source, ref_images = initialize_scene(
        sensor_count=params.sensor_count,
        base_samples=params.base_samples,
        reference_samples=params.reference_samples,
        image_resolution=params.image_resolution,
        texture_resolution=params.texture_resolution,
    )

    # Optimization procedure
    optimizer = BitmapTextureOptimizer(
        scene=scene_source,
        scene_params=mi.traverse(scene_source),
        optimization_events=OptimizationEvents(),
        optimization_key="shape_texture",
        texture_keys=["shape.bsdf.reflectance.data"],
        texture_scheduler=params.texture_params.scheduler(
            end_iteration=params.optimization_iterations,
        ),
        texture_loss=params.texture_params.image_loss,
    )

    optimizer.optimize(
        ref_images=ref_images,
        scene_optimizer=params.texture_params.scene_optimizer,
        num_iters=params.optimization_iterations,
        views_to_render=[
            ref_images[i].sensor for i in range(0, params.sensor_count, 4)
        ],
        save_directory_path=params.output_folder,
    )
    logger.info("Finished optimization, rendering final images.")

    optimizer.show_comparison_with_reference(
        ref_images=ref_images,
        samples_per_pixel=params.reference_samples,
        save_path=params.output_folder / "Final_comparison.png",
    )

    # Save the final state of the scene
    texture = mi.Bitmap(optimizer.scene_params["shape.bsdf.reflectance.data"])
    texture.write(str(params.output_folder / "suzanne_rgb_texture.exr"))


if __name__ == "__main__":
    parameters = tyro.cli(SuzanneTextureParameters)
    mlflow.log_params(parameters.to_loggable_dict())

    optimize_suzanne_texture(params=parameters)
