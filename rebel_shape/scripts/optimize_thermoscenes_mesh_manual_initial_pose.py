"""
Optimizes a ThermoScenes scene represented as a 6-patch composite mesh cubed
sphere with a manually authored initial pose.

This is the mesh-side analog of `optimize_thermoscenes_spline_manual_initial_pose.py`:
the initial surface is a `RegularCompositeMesh.cubed_sphere`, the manual pose
transform is applied to the global vertices, and shape + per-patch texture are
optimized jointly via `RegularCompositeMeshTextureOptimizer`. The composite
topology shares seam vertices across patch boundaries, avoiding the
duplicate-vertex tearing failure mode of the original single-PLY
`MeshTextureOptimizer` setup.
"""

import datetime
import sys
from dataclasses import dataclass, field
from pathlib import Path

import drjit as dr
import jsonpickle
import mitsuba as mi
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append("./")

from rebel_shape import logger, tracking
from rebel_shape.data_processing.image_scene_loader import ImageSceneLoader
from rebel_shape.data_processing.manual_initialization import (
    load_bounding_bbox_transforms,
)
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
from rebel_shape.scripts.configs.shape_optimization_config import (
    ShapeOptimizationConfig,
)
from rebel_shape.scripts.configs.texture_optimization_config import (
    TextureOptimizationConfig,
)


@dataclass(kw_only=True)
class ThermoSceneOptimizationParameters(ReverseCli):
    scene_path: Path
    """
    Path to the ThermoScenes scene that is optimized. Assumes the foreground
    has been extracted
    """
    environment_map_path: Path | None = None
    """
    Path to the envionment map that is used as background in optimization.
    Defaults to checkerboard_lighter.exr.
    """
    bbox_transforms_path: Path
    """Path to the transforms of the bounding box"""

    continuous_samples: int = 25
    """number of samples per pixel used for continuous derivatives"""
    visibility_samples: int = 9
    """number of samples per pixel used for direct and indirect visibility"""
    optimization_iterations: int = 250
    """Adam iterations executed on the initial composite mesh"""
    initial_subdivisions: int = 16
    """Grid resolution per cube face at the start of optimization"""
    h_refinement_iterations: list[int] = field(
        default_factory=lambda: [20, 50, 80, 120, 180],
    )
    """Iterations at which the composite mesh is h-refined by a factor of 2"""

    shape_params: ShapeOptimizationConfig = field(
        default_factory=lambda: ShapeOptimizationConfig(
            smoothing_strength=50.0,
            learning_rate_start=0.02,
            learning_rate_end=0.01,
            learning_rate_steepness=1.0,
            use_msssim_loss=False,
            l1_loss_weight=1.0,
            use_lbfgs=False,
            uniform_adam=True,
            batch_size=9,
        ),
    )
    """Parameters for shape optimization"""
    texture_params: TextureOptimizationConfig = field(
        default_factory=lambda: TextureOptimizationConfig(
            texture_resolution=1024,
            learning_rate_start=0.04,
            learning_rate_end=0.02,
            learning_rate_steepness=1.0,
            use_msssim_loss=False,
            l1_loss_weight=1.0,
            use_lbfgs=False,
            uniform_adam=False,
            batch_size=6,
        ),
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
    Whether the masks are applied during the optimization process to ignore non-building
    pixels. Note that during the pose optimization the masks are not applied
    """

    def __post_init__(self) -> None:
        if self.continuous_samples < 1:
            raise ValueError("`continuous_samples` must be a positive integer.")
        if self.visibility_samples < 1:
            raise ValueError("`visibility_samples` must be a positive integer.")
        if self.optimization_iterations < 1:
            raise ValueError("`optimization_iterations` must be a positive integer.")

        if self.num_of_images_to_save < 1:
            raise ValueError("`num_of_images_to_save` must be a positive integer.")
        if self.initial_subdivisions < 1:
            raise ValueError("`initial_subdivisions` must be a positive integer.")
        for it in self.h_refinement_iterations:
            if it < 0 or it >= self.optimization_iterations:
                raise ValueError(
                    "`h_refinement_iterations` entries must be in "
                    f"[0, {self.optimization_iterations}), got {it}.",
                )

        if self.environment_map_path is None:
            self.environment_map_path = Path("data/textures/checkerboard_lighter.exr")


def initialize_scene(
    continuous_samples: int,
    visibility_samples: int,
    thermoscenes_scene_path: Path,
    envmap_path: Path | None,
    texture_resolution: int,
    initial_subdivisions: int,
    bbox_transforms_path: Path,
    mask_pixels_during_optimization: bool,
) -> tuple[
    "mi.Scene",
    list[MaskedPoseImage],
    list[MaskedPoseImage],
    RegularCompositeMesh,
]:
    """
    Creates and loads Mitsuba dictionary that represents the scene, as well as providing
    the reference images, evaluation views, and the composite mesh.

    `continuous_samples` is the number of samples used for optimizing continuous
    derivatives (e.g. Texture), while `visibility_samples` is the number of samples used
    for direct and indirect visibility changes.

    `thermoscenes_scene_path` is a `Path` to a scene that can be loaded with
    `ImageSceneLoader`. `envmap_path` is the path to the 360 image that is used for
    the background during optimization; if `None`, a constant white background is
    used. `texture_resolution` specifies the resolution of each per-patch texture.
    `initial_subdivisions` controls the per-cube-face grid resolution of the
    initial cubed-sphere mesh, and `bbox_transforms_path` points to a JSON
    describing the manual initial pose (scale, translation, rotation).
    MaskedPoseImage masks are populated when `mask_pixels_during_optimization` is
    True and left as `None` otherwise.

    :returns: Tuple of:
        - `scene_source`: Mitsuba scene with the 6-patch composite mesh and the
            environment map loaded.
        - `reference_images`: List of reference images for training.
        - `evaluation_images`: List of reference images for evaluation.
        - `composite_mesh`: The `RegularCompositeMesh` whose shape and per-patch
            textures will be optimized.

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

    sampler = Sampler(type=Sampler.Type.MULTIJITTER, sample_count=continuous_samples)

    loader = ImageSceneLoader.from_thermoscene(
        scene_path=thermoscenes_scene_path,
        environment_map_path=envmap_path,
        spp=4,
        sensor_sampler=sampler,
        use_foreground_masks=mask_pixels_during_optimization,
    )

    ref_images = loader.train_images
    eval_images = loader.eval_images

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

    composite_mesh = RegularCompositeMesh.cubed_sphere(
        name="shape",
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

    if bbox_transforms_path is not None:
        # The cubed sphere is already unit-radius, so we do not need a separate
        # normalization transform (unlike the original PLY-loading mesh path).
        # This matches the spline manual-initial-pose script exactly.
        tf_bbox = load_bounding_bbox_transforms(bbox_transforms_path)
        tf_initial_pose = mi.Transform4f(loader.tf_cameras) @ tf_bbox
        composite_mesh.transform_global_vertices(tf_initial_pose)

    scene_dict: dict = {
        "type": "scene",
        "integrator": integrator.mitsuba_dict,
        "emitter": environment,
    }
    for mesh in reversed(composite_mesh):
        scene_dict[mesh.name] = mesh.mitsuba_dict

    scene_source: "mi.Scene" = mi.load_dict(scene_dict)

    return scene_source, ref_images, eval_images, composite_mesh


def optimize_thermoscenes_scene(params: ThermoSceneOptimizationParameters) -> None:
    """
    Performs composite-mesh shape + texture optimization on the scene at
    `params.scene_path`.

    The initial guess is a 6-patch cubed-sphere `RegularCompositeMesh` with
    `params.initial_subdivisions` grid cells per cube face. Its global vertices
    are transformed into the scene's coordinate space using the manual bbox
    transform composed with the loader's `tf_cameras`, mirroring
    `optimize_thermoscenes_spline_manual_initial_pose.py`.

    Optimization runs for `params.optimization_iterations` with two
    sub-optimizers driven through `CompositeSceneOptimizer`:

    1. Shape: vertex positions of all 6 patches, parametrized through a Laplacian
       on the composite mesh's global vertex / face graph. Seam vertices are
       shared by construction and averaged across patches via
       `update_global_representation`.
    2. Texture: per-patch bitmap texture, optimized as a `(num_patches, W, H, C)`
       tensor through `BitmapTextureOptimizer`.

    H-refinement events are scheduled at iterations listed in
    `params.h_refinement_iterations`. Each event doubles the per-patch grid
    resolution and rebuilds the parametrization and scene optimizer state.

    Images and meshes are saved to `params.output_folder` (directory created if
    needed) using each per-optimizer's `save_callback`.

    :raises ValueError: If `params.scene_path` is not a valid path or it is not
        a directory.
    """
    if not params.scene_path.exists():
        raise ValueError(f"Path {params.scene_path} does not exist.")
    if not params.scene_path.is_dir():
        raise ValueError(f"Path {params.scene_path} is not a directory.")

    if params.use_gpu:
        mi.set_variant("cuda_ad_rgb")
        logger.info("Using the GPU for rendering.")
    else:
        mi.set_variant("llvm_ad_rgb")
        logger.info("Using the CPU for rendering.")

    # Initialize the scene, reference images and cameras
    scene_source, ref_images, eval_images, composite_mesh = initialize_scene(
        continuous_samples=params.continuous_samples,
        visibility_samples=params.visibility_samples,
        thermoscenes_scene_path=params.scene_path,
        envmap_path=params.environment_map_path,
        texture_resolution=params.texture_params.texture_resolution,
        initial_subdivisions=params.initial_subdivisions,
        bbox_transforms_path=params.bbox_transforms_path,
        mask_pixels_during_optimization=params.mask_pixels_during_optimization,
    )
    if len(eval_images) < params.num_of_images_to_save:
        raise ValueError(
            f"Not enough evaluation images available. "
            f"Expected at least {params.num_of_images_to_save}, "
            f"but got {len(eval_images)}.",
        )

    ref_images_no_mask = [
        MaskedPoseImage(image=image.original_image, camera=image.camera, mask=None)
        for image in ref_images
    ]
    eval_images_no_mask = [
        MaskedPoseImage(image=image.original_image, camera=image.camera, mask=None)
        for image in eval_images
    ]

    shape_key = composite_mesh.name
    texture_key = composite_mesh.name + "_texture"

    # Set up the pipeline for shape + texture optimization
    shape_scheduler = params.shape_params.scheduler(
        end_iteration=params.optimization_iterations,
    )
    texture_scheduler = params.texture_params.scheduler(
        end_iteration=params.optimization_iterations,
    )

    optimizer = RegularCompositeMeshTextureOptimizer(
        scene=scene_source,
        scene_params=mi.traverse(scene_source),
        optimization_events=OptimizationEvents(),
        composite_mesh=composite_mesh,
        shape_scheduler=shape_scheduler,
        texture_scheduler=texture_scheduler,
        shape_loss=params.shape_params.image_loss,
        texture_loss=params.texture_params.image_loss,
        parametrization_smoothing=params.shape_params.smoothing_strength,
        texture_projection_steepness=params.texture_params.projection_steepness,
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
    logger.info("Rendered initial state,")

    composite_scene_optimizer = CompositeSceneOptimizer(
        optimizer_config=[
            (params.shape_params.scene_optimizer, [shape_key]),
            (params.texture_params.scene_optimizer, [texture_key]),
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

    if not params.mask_pixels_during_optimization:
        ref_images = ref_images_no_mask
        eval_images = eval_images_no_mask

    optimizer.optimize(
        ref_images=ref_images,
        scene_optimizer=composite_scene_optimizer,
        num_iters=params.optimization_iterations,
        views_to_render=[
            image.sensor for image in eval_images[: params.num_of_images_to_save]
        ],
        save_directory_path=params.output_folder,
    )
    logger.info("Finished optimization, rendering final state.")

    optimizer.show_comparison_with_reference(
        ref_images=ref_images,
        samples_per_pixel=params.continuous_samples * 4,
        save_path=params.output_folder / "Final_comparison.png",
    )

    # Save the final optimized state. The full composite is written as a single
    # jsonpickle ``.json`` (reloadable with `RegularCompositeMesh.from_file` and the
    # idiomatic artifact for downstream evaluation), the welded global mesh is
    # written as one PLY (a single file for a quick Blender sanity check), and the
    # six per-patch PLYs are written alongside their optimized textures.
    logger.info("Saving final composite mesh, global mesh, patches, and textures.")

    composite_json_path = params.output_folder / "thermoscene_shape.json"
    with open(composite_json_path, "w") as composite_file:
        composite_file.write(jsonpickle.encode(composite_mesh))

    composite_mesh.global_mesh.save_to_ply(
        output_dir=params.output_folder, save_name="thermoscene_global_mesh"
    )

    for patch in composite_mesh:
        patch.save_to_ply(output_dir=params.output_folder, save_name=patch.name)

        texture = mi.Bitmap(
            optimizer.scene_params[f"{patch.name}.bsdf.reflectance.data"]
        )
        texture_path = params.output_folder / f"{patch.name}_texture.exr"
        texture.write(str(texture_path))


if __name__ == "__main__":
    parameters = tyro.cli(ThermoSceneOptimizationParameters)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    with tracking.run(
        run_name=f"large-step-{timestamp}",
        artifact_directory=parameters.output_folder,
    ):
        tracking.log_dataset_source(
            url=str(parameters.scene_path), name=str(parameters.scene_path.name)
        )
        tracking.log_params(parameters.to_loggable_dict(ignore=set(["scene_path"])))

        optimize_thermoscenes_scene(params=parameters)
