"""
Optimizes the initial pose (translation, rotation, scale) of a cubed-sphere proxy
against a ThermoScenes scene and writes the result as a `data/bbox_transforms/*.json`
style file that can be consumed by the mesh or spline shape+texture scripts.
"""

import datetime
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mitsuba as mi
import tyro

from rebel_shape.data_processing.silhouette_scene_loader import SilhouetteSceneLoader

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append("./")

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape import logger, tracking
from rebel_shape.data_processing.image_scene_loader import ImageSceneLoader
from rebel_shape.data_processing.manual_initialization import (
    pose_params_to_bbox_transforms,
    save_bounding_bbox_transforms,
)
from rebel_shape.data_processing.masked_pose_image import MaskedPoseImage
from rebel_shape.meshing.tessellated_spline_surface import TessellatedSplineSurface
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from rebel_shape.plugin_wrappers.perspective_camera import PerspectiveCamera
from rebel_shape.plugin_wrappers.prb_integrator import GuidingOptions, PRBIntegrator
from rebel_shape.plugin_wrappers.sampler import Sampler
from rebel_shape.property_optimizers.spline_pose_optimizer import SplinePoseOptimizer
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.scripts.configs.pose_optimization_config import PoseOptimizationConfig
from rebel_shape.splines.cube_initialization import cubed_sphere


@dataclass(kw_only=True)
class ThermoScenePoseOptimizationParameters(ReverseCli):
    scene_path: Path
    """
    Path to the ThermoScenes scene that is optimized. Assumes the foreground
    has been extracted.
    """
    name: str
    """
    Output name for the produced transform. The file is written to
    `bbox_transforms_dir / f"{name}.json"`.
    """

    continuous_samples: int = 4
    """
    Samples per pixel for continuous derivatives. The pose loss is a pure
    silhouette match (black proxy on white background) under the Mitsuba `mono`
    variant, so the shading payload is trivial and this can stay low.
    """
    visibility_samples: int = 4
    """Samples per pixel for direct and indirect visibility."""

    pose_params: PoseOptimizationConfig = field(
        default_factory=lambda: PoseOptimizationConfig(
            learning_rate_start=0.5,
            learning_rate_end=0.25,
            learning_rate_steepness=1.5,
            use_lbfgs=True,
            uniform_adam=False,
            batch_size=None,
            lbfgs_history_size=10,
            optimization_iterations=50,
            use_msssim_loss=False,
            l1_loss_weight=1.0,
            jitter=0.01,
            seed=0,
        ),
    )
    """Parameters of the pose optimization"""

    gss_iterations: int = 25
    """
    Maximum number of golden-section search iterations used to bracket the
    initial isotropic radius of the proxy. Each iteration renders all reference
    views once at `continuous_samples` spp.
    """
    gss_coarse_samples: int = 8
    """
    Number of evenly-spaced radii sampled across the full GSS bracket before
    the golden-section refinement. Guards against degenerate camera
    configurations (e.g. ring rigs) where the silhouette loss is locally flat
    and GSS alone would converge to a meaningless point. Raise for finicky
    scenes; lower for cheaper search.
    """

    spline_degree: int = 3
    """Degree of the cubed-sphere B-spline patches used as the pose proxy."""
    spline_knot_spans: int = 2
    """Number of knot spans per side for the cubed-sphere B-spline patches."""
    tessellation_subdivisions: int = 128
    """Triangle subdivisions per patch when tessellating the cubed sphere."""

    bbox_transforms_dir: Path = Path("./data/bbox_transforms/")
    """Directory the resulting `{name}.json` is written to (created if missing)."""
    output_folder: Path = Path("./outputs/")
    """Where to save diagnostic images and the optimization views."""
    num_of_images_to_save: int = 3
    """Number of evaluation views rendered as diagnostic images."""
    use_gpu: bool = True
    """
    Whether to use GPU for rendering. The compute instance must have GPU support
    for this to work; for local testing, set to False.
    """

    def __post_init__(self) -> None:
        if self.continuous_samples < 1:
            raise ValueError("`continuous_samples` must be a positive integer.")
        if self.visibility_samples < 1:
            raise ValueError("`visibility_samples` must be a positive integer.")
        if self.num_of_images_to_save < 1:
            raise ValueError("`num_of_images_to_save` must be a positive integer.")
        if len(self.name) == 0:
            raise ValueError("`name` must be a non-empty string.")
        if self.spline_degree < 1:
            raise ValueError("`spline_degree` must be a positive integer.")
        if self.spline_knot_spans < 1:
            raise ValueError("`spline_knot_spans` must be a positive integer.")
        if self.tessellation_subdivisions < 1:
            raise ValueError("`tessellation_subdivisions` must be a positive integer.")
        if self.gss_iterations < 1:
            raise ValueError("`gss_iterations` must be a positive integer.")
        if self.gss_coarse_samples < 3:
            raise ValueError("`gss_coarse_samples` must be at least 3.")


def initialize_scene(
    continuous_samples: int,
    visibility_samples: int,
    thermoscenes_scene_path: Path,
    spline_degree: int,
    spline_subdivisions: int,
    mesh_subdivisions: int,
) -> tuple[
    "mi.Scene",
    ImageSceneLoader,
    list[MaskedPoseImage],
    list[MaskedPoseImage],
    TessellatedSplineSurface,
]:
    """
    Loads the ThermoScenes reference images as silhouettes and builds a Mitsuba
    scene containing a unit-radius cubed-sphere B-spline proxy with a black
    diffuse BSDF on a white constant emitter background. Under the Mitsuba
    `mono` variant the rendered image is a single-channel luminance, so the
    optimization loss is a pure silhouette match between the rendered (black
    object on white) image and the reference (black object on white).

    The loader is returned alongside the scene so the caller can access
    `tf_cameras` when converting the optimized pose back to the raw input frame.
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

    loader = SilhouetteSceneLoader.from_thermoscene(
        scene_path=thermoscenes_scene_path,
        sensor_sampler=sampler,
        pixel_format=PerspectiveCamera.PixelFormat.LUMINANCE,
    )

    cubedsphere = cubed_sphere(
        name="shape",
        degree=spline_degree,
        spline_subdivisions=spline_subdivisions,
        tessellation_subdivisions=mesh_subdivisions,
        bsdf=DiffuseBSDF(DiffuseBSDF.Albedo(albedo=0.0)),
        radius=1.0,
    )

    scene_dict: dict = {
        "type": "scene",
        "integrator": integrator.mitsuba_dict,
        "emitter": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": 1.0},
        },
    }
    for mesh in reversed(cubedsphere):
        scene_dict[mesh.name] = mesh.mitsuba_dict

    scene_source: "mi.Scene" = mi.load_dict(scene_dict)
    return scene_source, loader, loader.train_images, loader.eval_images, cubedsphere


def optimize_thermoscenes_pose(
    params: ThermoScenePoseOptimizationParameters,
) -> Path:
    """
    Runs pose optimization on the ThermoScenes scene at
    `params.scene_path` using a unit-radius cubed-sphere B-spline
    proxy (black diffuse on a white constant background), under the Mitsuba
    `mono` variant. The initial radius is bracketed by a golden-section search
    before the gradient phase, and a small seeded jitter is applied to the
    initial scale/translation to break symmetry. The resulting transform is
    written to `params.bbox_transforms_dir / f"{params.name}.json"`.

    :returns: The path to the JSON file that was written.
    :raises ValueError: If `params.scene_path` is not an existing
        directory.
    """
    if not params.scene_path.exists():
        raise ValueError(f"Path {params.scene_path} does not exist.")
    if not params.scene_path.is_dir():
        raise ValueError(f"Path {params.scene_path} is not a directory.")

    if params.use_gpu:
        mi.set_variant("cuda_ad_mono")
        logger.info("Using the GPU for rendering (cuda_ad_mono)")
    else:
        mi.set_variant("llvm_ad_mono")
        logger.info("Using the CPU for rendering (llvm_ad_mono)")

    params.output_folder.mkdir(parents=True, exist_ok=True)

    scene, loader, ref_images, eval_images, tessellated_spline = initialize_scene(
        continuous_samples=params.continuous_samples,
        visibility_samples=params.visibility_samples,
        thermoscenes_scene_path=params.scene_path,
        spline_degree=params.spline_degree,
        spline_subdivisions=params.spline_knot_spans,
        mesh_subdivisions=params.tessellation_subdivisions,
    )

    if len(eval_images) < params.num_of_images_to_save:
        raise ValueError(
            f"Not enough evaluation images available. Expected at least "
            f"{params.num_of_images_to_save}, but got {len(eval_images)}.",
        )

    scene_params = mi.traverse(scene)
    pose_scene_optimizer = params.pose_params.scene_optimizer
    pose_optimizer = SplinePoseOptimizer(
        scene=scene,
        scene_params=scene_params,
        optimization_events=OptimizationEvents(),
        tessellated_spline=tessellated_spline,
        scheduler=params.pose_params.scheduler(
            end_iteration=params.pose_params.optimization_iterations,
        ),
        image_loss=params.pose_params.image_loss,
        jitter=params.pose_params.jitter,
        seed=params.pose_params.seed,
        initial_scale=1.0,
    )

    logger.info("Optimizing initial pose of the spline surface")
    pose_optimizer.optimize(
        ref_images=ref_images,
        scene_optimizer=pose_scene_optimizer,
        num_iters=params.pose_params.optimization_iterations,
        views_to_render=[
            image.sensor for image in eval_images[: params.num_of_images_to_save]
        ],
        save_directory_path=params.output_folder,
        compute_initial_scale_gss=True,
        gss_max_iters=params.gss_iterations,
        gss_samples_per_pixel=params.continuous_samples,
        gss_coarse_samples=params.gss_coarse_samples,
    )
    logger.info(f"Camera convergence point: {pose_optimizer.initial_position_guess}")
    logger.info(f"GSS initial radius: {pose_optimizer.initial_scale:.4g}")
    tracking.log_params({"gss_initial_radius": pose_optimizer.initial_scale})

    final_params = pose_scene_optimizer.parameters[tessellated_spline.mesh_name]
    # The proxy is always built with radius=1.0 here; the GSS-found radius is
    # encoded in `final_params[3:6]` via the LBFGS scaling component. Passing
    # `initial_sphere_radius=1.0` avoids double-scaling the bbox transform.
    translation, rotation, scale = pose_params_to_bbox_transforms(
        flat_params=final_params,
        tf_cameras=loader.tf_cameras,
        initial_sphere_radius=1.0,
    )

    output_path = params.bbox_transforms_dir / f"{params.name}.json"
    save_bounding_bbox_transforms(
        bbox_transforms_path=output_path,
        translation=translation,
        rotation=rotation,
        scale=scale,
    )
    logger.info(f"Wrote bbox transforms to {output_path}")

    pose_optimizer.show_comparison_with_reference(
        ref_images=eval_images[: params.num_of_images_to_save],
        samples_per_pixel=params.continuous_samples * 4,
        save_path=params.output_folder / "Pose_final_comparison.png",
    )

    return output_path


if __name__ == "__main__":
    parameters = tyro.cli(ThermoScenePoseOptimizationParameters)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    with tracking.run(
        run_name=f"pose-optimization-{timestamp}",
        artifact_directory=parameters.output_folder,
    ):
        tracking.log_dataset_source(
            url=str(parameters.scene_path), name=str(parameters.scene_path.name)
        )
        tracking.log_params(
            parameters.to_loggable_dict(ignore=set(["scene_path"])),
        )
        optimize_thermoscenes_pose(params=parameters)
