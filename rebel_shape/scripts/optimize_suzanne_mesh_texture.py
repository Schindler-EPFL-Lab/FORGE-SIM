"""
Example script for optimizing the shape and texture of the Suzanne mesh,
Verifies the correctness of the `RegularCompositeMeshTextureOptimizer` class.

This script is currently set up to test the new 6-patch composite-mesh path: it
initializes the source mesh as a cubed sphere whose six faces share topology via
`PatchManager`, each with its own texture, mirroring the spline cubed-sphere
setup in `optimize_suzanne_spline_texture.py`. The original single-PLY +
`MeshTextureOptimizer` path is replaced for the duration of this test; revert
the optimizer swap to compare against the old behavior.
"""

import sys
import time
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
from rebel_shape.data_processing.masked_pose_image import MaskedPoseImage
from rebel_shape.meshing.regular_composite_mesh import RegularCompositeMesh
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from rebel_shape.plugin_wrappers.prb_integrator import GuidingOptions, PRBIntegrator
from rebel_shape.plugin_wrappers.sampler import Sampler
from rebel_shape.property_optimizers.regular_composite_mesh_texture_optimizer import (
    RegularCompositeMeshTextureOptimizer,
)
from rebel_shape.scene_optimizers.composite_scene_optimizer import (
    CompositeSceneOptimizer,
)
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.scene_optimizers.scene_adam_sequential import SceneAdamSequential
from rebel_shape.scripts.configs.shape_optimization_config import (
    ShapeOptimizationConfig,
)
from rebel_shape.scripts.configs.texture_optimization_config import (
    TextureOptimizationConfig,
)
from rebel_shape.utils.image_utils import show_rendered_views
from rebel_shape.utils.sensor_utils import create_cameras_fibonacci


@dataclass(kw_only=True)
class SuzanneMeshTextureParameters(ReverseCli):
    """Parameters used for simultaneous shape and texture optimization of Suzanne"""

    sensor_count: int = 12
    """Number of camera views used in optimization"""
    reference_samples: int = 256
    """Number of samples per pixel used for the reference/target images"""
    base_samples: int = 16
    """Number of samples per pixel used for continuous derivatives"""
    image_resolution: int = 480
    """Resolution of the rendered images along height and width"""
    texture_resolution: int = 512
    """Resolution of the texture along height and width"""
    optimization_iterations: int = 100
    """Number of optimization iterations with `SceneAdamBatch`"""
    initial_subdivisions: int = 32
    """Grid resolution per cube face at the start of optimization"""
    h_refinement_iterations: list[int] = field(default_factory=lambda: [25, 50, 80])
    """Iterations at which the composite mesh is h-refined by a factor of 2"""

    shape_params: ShapeOptimizationConfig = field(
        default_factory=lambda: ShapeOptimizationConfig(
            smoothing_strength=30.0,
            learning_rate_start=0.24,
            learning_rate_end=0.06,
            learning_rate_steepness=1.5,
            use_lbfgs=False,
            uniform_adam=True,
            batch_size=4,
        ),
    )
    """Parameters for shape optimization"""
    texture_params: TextureOptimizationConfig = field(
        default_factory=lambda: TextureOptimizationConfig(
            texture_resolution=1024,
            learning_rate_start=0.03,
            learning_rate_end=0.12,
            learning_rate_steepness=2.0,
            use_lbfgs=False,
            uniform_adam=False,
            batch_size=2,
        ),
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
        if self.initial_subdivisions < 1:
            raise ValueError("`initial_subdivisions` must be a positive integer.")
        for it in self.h_refinement_iterations:
            if it < 0 or it >= self.optimization_iterations:
                raise ValueError(
                    "`h_refinement_iterations` entries must be in "
                    f"[0, {self.optimization_iterations}), got {it}.",
                )


def initialize_scene(
    sensor_count: int,
    base_samples: int,
    reference_samples: int,
    image_resolution: int,
    texture_resolution: int,
    initial_subdivisions: int,
) -> tuple["mi.Scene", list[MaskedPoseImage], RegularCompositeMesh]:
    """
    Sets up the scene, sensors, and reference images for shape optimization.

    There are `sensor_count` cameras and reference images. Reference images are
    rendered with `reference_samples` per pixel while `base_samples` controls
    how many samples per pixel are used during optimizations. `image_resolution` and
    `texture_resolution` specify the resolution of images and textures, respectively.
    Reference images are shown to the user.

    The number of samples used for rendering the image and continuous derivatives
    is `base_samples` while projective and indirect ones are 4 times smaller for
    the sake of efficiency.

    The initial source mesh is a 6-patch cubed sphere built via
    `RegularCompositeMesh.cubed_sphere` with `initial_subdivisions` grid cells
    per cube face; each face carries its own solid-grey initial texture so seams
    across patch boundaries do not require duplicated vertices. The target mesh
    is the same Suzanne PLY with a checkerboard texture as in the original
    single-mesh test setup.

    :returns: A tuple containing the source Mitsuba scene, the list of reference
        images, and the `RegularCompositeMesh` whose shape and per-patch
        textures will be optimized.
    """
    integrator = PRBIntegrator(
        max_bounces=4,
        bounces_without_russian_roulette=4,
        spp_continuous=base_samples,
        spp_projective=base_samples // 4,
        spp_indirect=base_samples // 4,
        guiding=GuidingOptions.NONE,
    )

    tracking.log_params(
        {
            "spp_continuous": base_samples,
            "spp_projective": base_samples // 4,
            "spp_indirect": base_samples // 4,
        }
    )

    script_dir = Path(__file__).parent.resolve()
    target_texture_path = script_dir / "../../data/textures/checkerboard_texture.jpg"

    target_scene_dict = {
        "type": "scene",
        "integrator": integrator.mitsuba_dict,
        "emitter": {
            "type": "envmap",
            "filename": str(script_dir / "../../data/textures/envmap2.exr"),
        },
        "shape": {
            "type": "ply",
            "filename": str(script_dir / "../../data/meshes/suzanne.ply"),
            "bsdf": DiffuseBSDF(
                reflectance=DiffuseBSDF.Texture(texture_path=target_texture_path),
            ).mitsuba_dict,
        },
    }

    # Target images are obtained through rendering
    scene_target: "mi.Scene" = mi.load_dict(target_scene_dict)
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
        views=[mi.util.convert_to_bitmap(image.original_image) for image in ref_images],
    )
    logger.info(f"Rendered target images. Took {time.time() - start_time:.2f} seconds")

    # Build the source mesh: 6-patch cubed sphere with one solid-grey texture per
    # face. This is the mesh-side analog of the cubed_sphere() call in
    # optimize_suzanne_spline_texture.py.
    composite_mesh = RegularCompositeMesh.cubed_sphere(
        name="suzanne",
        subdivisions=initial_subdivisions,
        radius=1.0,
        bsdf=DiffuseBSDF(
            DiffuseBSDF.TextureFromTensor(
                tensor=dr.full(
                    dtype=mi.TensorXf,
                    value=0.5,
                    shape=(texture_resolution, texture_resolution, 3),
                ),
            ),
        ),
    )

    source_scene_dict: dict = {
        "type": "scene",
        "integrator": integrator.mitsuba_dict,
        "emitter": {
            "type": "envmap",
            "filename": str(script_dir / "../../data/textures/envmap2.exr"),
        },
    }
    # `reversed` matches the spline_texture script's pattern; it ensures patches
    # are inserted in a stable order regardless of factory order.
    for mesh in reversed(composite_mesh):
        source_scene_dict[mesh.name] = mesh.mitsuba_dict

    scene_source: "mi.Scene" = mi.load_dict(source_scene_dict)

    return scene_source, ref_images, composite_mesh


def optimize_suzanne(params: SuzanneMeshTextureParameters) -> None:
    """
    An example of optimizing both the shape and texture of the Suzanne mesh.

    Serves to validate that `RegularCompositeMeshTextureOptimizer` works as the
    drop-in mesh-side analog of `SplineTextureOptimizer`. A reference Suzanne
    mesh is rendered, then a 6-patch cubed sphere with solid-grey per-patch
    textures is optimized to match it. The composite topology means the seam
    vertices between cube faces are shared rather than duplicated, eliminating
    the tearing failure mode of the original single-PLY `MeshTextureOptimizer`.

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

    scene_source, ref_images, composite_mesh = initialize_scene(
        sensor_count=params.sensor_count,
        base_samples=params.base_samples,
        reference_samples=params.reference_samples,
        image_resolution=params.image_resolution,
        texture_resolution=params.texture_resolution,
        initial_subdivisions=params.initial_subdivisions,
    )

    # Configuring the property optimizer
    shape_key = composite_mesh.name
    texture_key = composite_mesh.name + "_texture"

    optimizer = RegularCompositeMeshTextureOptimizer(
        scene=scene_source,
        scene_params=mi.traverse(scene_source),
        optimization_events=OptimizationEvents(),
        composite_mesh=composite_mesh,
        shape_scheduler=params.shape_params.scheduler(
            end_iteration=params.optimization_iterations,
        ),
        texture_scheduler=params.texture_params.scheduler(
            end_iteration=params.optimization_iterations,
        ),
        shape_loss=params.shape_params.image_loss,
        texture_loss=params.texture_params.image_loss,
        parametrization_smoothing=params.shape_params.smoothing_strength,
    )

    composite_scene_optimizer = CompositeSceneOptimizer(
        optimizer_config=[
            (SceneAdamSequential(batch_size=4, uniform=True), [shape_key]),
            (SceneAdamSequential(batch_size=2, uniform=False), [texture_key]),
        ],
    )

    for iteration in params.h_refinement_iterations:
        optimizer.optimization_events.add_event(
            iteration=iteration,
            callback=lambda: optimizer.h_refinement(
                factor=2,
                scene_optimizer=composite_scene_optimizer,
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
    logger.info("Finished optimization, rendering final images.")

    optimizer.show_comparison_with_reference(
        ref_images=ref_images,
        samples_per_pixel=params.reference_samples,
        save_path=params.output_folder / "Final_comparison.png",
    )

    # Save the final state of each patch's texture and geometry.
    for mesh in composite_mesh:
        texture = mi.Bitmap(
            optimizer.scene_params[f"{mesh.name}.bsdf.reflectance.data"],
        )
        texture.write(str(params.output_folder / f"{mesh.name}_texture.exr"))
        texture.write(str(params.output_folder / f"{mesh.name}_texture.png"))
        mesh.save_to_ply(
            output_dir=params.output_folder,
            save_name=f"{mesh.name}_final",
        )


if __name__ == "__main__":
    parameters = tyro.cli(SuzanneMeshTextureParameters)
    with tracking.run(artifact_directory=parameters.output_folder):
        tracking.log_params(parameters.to_loggable_dict())

        optimize_suzanne(params=parameters)
