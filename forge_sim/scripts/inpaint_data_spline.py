"""
This script assumes that shape optimization has already been performed and
takes in an optimized spline. It proceeds to inpaint information onto the
reconstructed spline control points using single channel images.
"""

import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import drjit as dr
import jsonpickle
import mitsuba as mi
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

from forge_sim import logger, tracking
from forge_sim.data_processing.masked_pose_image import MaskedPoseImage
from forge_sim.data_processing.single_channel_scene_loader import (
    SingleChannelImageSceneLoader,
)
from forge_sim.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)
from forge_sim.meshing.tessellated_spline_surface import TessellatedSplineSurface
from forge_sim.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from forge_sim.plugin_wrappers.prb_integrator import GuidingOptions, PRBIntegrator
from forge_sim.plugin_wrappers.sampler import Sampler
from forge_sim.property_optimizers.field_texture_optimizer import (
    FieldTextureOptimizer,
)
from forge_sim.scene_optimizers.optimization_events import OptimizationEvents
from forge_sim.scripts.configs.texture_optimization_config import (
    TextureOptimizationConfig,
)


class InpaintType(Enum):
    THERMAL = "thermal"
    WINDOWS = "windows"


@dataclass(kw_only=True)
class SplineInpaintingParameters(ReverseCli):
    """
    Parameters used for inpainting information onto mesh vertices.

    For details about the name "inpainting", see the `inpaint_information`
    function.
    """

    thermoscenes_scene_path: Path
    """
    Path to the ThermoScenes scene that is optimized.
    """
    inpainting_images_type: InpaintType
    """What data type is inpainted onto the spline `thermoscenes_scene_path`"""
    initial_color: float = 0.1
    """The initial object color for inpainting"""
    spline_path: Path
    """
    Path to the reconstructed spline surface mesh that will receive
    data.
    """

    h_refinement_factor: int = 4
    """
    How many times to refine the mesh along each patch direction.
    Since most splines have few control points, this is used to increase the
    resolution of the information that can be stored in the control points.
    """
    continuous_samples: int = 16
    """Number of samples per pixel used for continuous derivatives"""
    optimization_iterations: int = 20
    """How many times all images in the dataset are used for optimization"""

    texture_params: TextureOptimizationConfig = field(
        default_factory=lambda: TextureOptimizationConfig(
            texture_resolution=512,
            learning_rate_start=0.01,
            learning_rate_end=0.01,
            learning_rate_steepness=4.0,
            use_lbfgs=False,
            uniform_adam=False,
            batch_size=6,
            use_msssim_loss=False,
            l1_loss_weight=1.0,
        )
    )
    """Parameters for spline inpainting optimization"""

    output_folder: Path = Path("./outputs/")
    """Where to save the model and outputs (created if it does not exist)"""
    num_of_images_to_save: int = 6
    """
    Number of images to save after each optimization step. The images are taken from
    the evaluation set.
    """

    use_gpu: bool = True
    """
    Whether to use GPU for rendering or not. Note that the compute instance
    must have GPU support for this to work. For local testing, set to False.
    """

    def __post_init__(self) -> None:
        if self.continuous_samples < 1:
            raise ValueError("`continuous_samples` must be a positive integer.")
        if self.optimization_iterations < 1:
            raise ValueError("`first_pass_iterations` must be a positive integer.")
        if self.num_of_images_to_save < 1:
            raise ValueError("`num_of_images_to_save` must be a positive integer.")


def initialize_scene(
    thermoscenes_scene_path: Path,
    spline_path: Path,
    inpaint_type: InpaintType,
    h_refinement_factor: int,
    continuous_samples: int,
    texture_resolution: int,
    initial_color: float,
) -> tuple[
    "mi.Scene",
    list[MaskedPoseImage],
    list[MaskedPoseImage],
    TessellatedIGASplineSurface,
]:
    """
    Creates and loads Mitsuba dictionary that represents the scene, as well as providing
    the reference images and sensor views.

    `thermoscenes_scene_path` is a `Path` to a scene that can be loaded with
    `ThermoScenesLoader`. `inpaint_type` specifies the data type which is inpainted.
    `spline_path` specifies a path to a JSON file that contains the tessellated spline
    surface. The surface is then converted to an IGA spline surface and h_refinement is
    performed to increase the resolution of the information that can be stored in the
    control points. The initial color of the surface is `initial_color`.

    The scene is expected to be a single channel (information is a single channel value)
    so the sensors returned by method have a luminosity pixel format.

    :returns: A tuple containing:

        - `scene_source`: A Mitsuba scene object that contains the loaded scene.
        - `train_images`: A list of training images loaded from the scene.
        - `eval_images`: A list of evaluation images loaded from the scene.
        - `tessellated_iga_spline_surface`: A `TessellatedIGASplineSurface` object
            that represents the tessellated IGA spline surface with scalar data.
    """
    integrator = PRBIntegrator(
        max_bounces=2,
        bounces_without_russian_roulette=2,
        spp_continuous=continuous_samples,
        spp_projective=1,
        spp_indirect=1,
        guiding=GuidingOptions.NONE,
    )

    tracking.log_params(
        {
            "spp_continuous": continuous_samples,
            "spp_projective": 1,
            "spp_indirect": 1,
        }
    )

    sampler = Sampler(type=Sampler.Type.INDEPENDENT, sample_count=continuous_samples)

    if inpaint_type is InpaintType.THERMAL:
        loader = SingleChannelImageSceneLoader.thermal_scene_auto(
            scene_path=thermoscenes_scene_path, sensor_sampler=sampler
        )
    if inpaint_type is InpaintType.WINDOWS:
        loader = SingleChannelImageSceneLoader.windows_masks_scene_auto(
            scene_path=thermoscenes_scene_path, sensor_sampler=sampler
        )

    tessellated_spline_surface = TessellatedSplineSurface.from_file(spline_path)

    initial_field_values = [
        initial_color
        * dr.ones(dtype=mi.TensorXf, shape=(spline.control_points.shape[0], 1))
        for spline in tessellated_spline_surface.composite_spline
    ]

    tessellated_iga_spline_surface = (
        TessellatedIGASplineSurface.from_tessellated_spline_surface(
            tessellated_spline_surface=tessellated_spline_surface,
            per_patch_field_values=initial_field_values,
        )
    )
    tessellated_iga_spline_surface.composite_spline.h_refinement(
        factor=h_refinement_factor
    )

    scene_dict = {"type": "scene", "integrator": integrator.mitsuba_dict}

    with tempfile.TemporaryDirectory() as temp_dir:
        for mesh in tessellated_iga_spline_surface:
            initial_texture_path = Path(temp_dir) / f"initial_texture_{mesh.name}.exr"

            FieldTextureOptimizer.generate_empty_texture(
                base_color=mi.Color3f(initial_color, initial_color, initial_color),
                texture_output_path=initial_texture_path,
                width=texture_resolution,
                height=texture_resolution,
            )

            mesh.save_to_ply(output_dir=Path(temp_dir), save_name=mesh.name)

            scene_dict[mesh.name] = {
                "type": "ply",
                "filename": temp_dir + f"/{mesh.name}.ply",
                "bsdf": DiffuseBSDF(
                    reflectance=DiffuseBSDF.Albedo(albedo=0.0),
                ).mitsuba_dict,
                "emitter": {
                    "type": "area",
                    "radiance": {
                        "type": "bitmap",
                        "filename": str(initial_texture_path),
                    },
                },
            }

        scene_source: "mi.Scene" = mi.load_dict(scene_dict)

    return (
        scene_source,
        loader.train_images,
        loader.eval_images,
        tessellated_iga_spline_surface,
    )


def inpaint_information(params: SplineInpaintingParameters) -> None:
    """
    Assigns scalar values to vertices of the input mesh.

    The function first loads and prepares the spline by adding a scalar attribute to the
    control points, then optimizes the data using the provided images as reference. The
    information is stored in the IGA spline control points and is used to render the
    mesh with information.

    In the spirit of the single channel modality, the data is simulated as a glow (area
    light) and the background is set to black. The optimization is done using the
    `TextureOptimizer` class. Word "inpainting" signifies that the information is not
    used during mesh optimization, but is "painted" onto the final mesh.

    Optionally, images and meshes can be saved to `params.output_folder`. The path
    should be a directory where the images will be saved. If the directory does not
    exist, it will be created, while if a `None` value is provided, nothing is saved.

    For details on available parameters, see `SplineInpaintingParameters`.

    :raises ValueError: If `params.thermoscenes_scene_path` is not a valid path, or it
        is not a directory.
    """
    if not params.thermoscenes_scene_path.exists():
        raise ValueError(f"Path {params.thermoscenes_scene_path} does not exist.")
    if not params.thermoscenes_scene_path.is_dir():
        raise ValueError(f"Path {params.thermoscenes_scene_path} is not a directory.")

    if params.use_gpu:
        mi.set_variant("cuda_ad_mono")
        logger.info("Using the GPU for rendering.")
    else:
        mi.set_variant("llvm_ad_mono")
        logger.info("Using the CPU for rendering.")

    # Initialize the scene, reference images and cameras
    scene_source, ref_images, eval_images, spline = initialize_scene(
        continuous_samples=params.continuous_samples,
        thermoscenes_scene_path=params.thermoscenes_scene_path,
        spline_path=params.spline_path,
        h_refinement_factor=params.h_refinement_factor,
        texture_resolution=params.texture_params.texture_resolution,
        inpaint_type=params.inpainting_images_type,
        initial_color=params.initial_color,
    )

    optimizer = FieldTextureOptimizer(
        scene=scene_source,
        scene_params=mi.traverse(scene_source),
        optimization_events=OptimizationEvents(),
        property_key="emitter.radiance.data",
        tessellated_iga_spline=spline,
        texture_scheduler=params.texture_params.scheduler(
            end_iteration=params.optimization_iterations,
        ),
        texture_loss=params.texture_params.image_loss,
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

    optimizer.optimize(
        ref_images=ref_images,
        scene_optimizer=params.texture_params.scene_optimizer,
        num_iters=params.optimization_iterations,
        views_to_render=[
            image.sensor for image in eval_images[: params.num_of_images_to_save]
        ],
        save_directory_path=params.output_folder,
    )
    logger.info("Finished optimization, rendering final results.")

    optimizer.show_comparison_with_reference(
        ref_images=ref_images,
        samples_per_pixel=params.continuous_samples * 4,
        save_path=params.output_folder / "Final_comparison.png",
    )

    encoded_spline = jsonpickle.encode(spline)
    with open(params.output_folder / "inpainted_data.json", "w") as spline_file:
        spline_file.write(encoded_spline)


if __name__ == "__main__":
    parameters = tyro.cli(SplineInpaintingParameters)
    with tracking.run(artifact_directory=parameters.output_folder):
        tracking.log_params(parameters.to_loggable_dict())

        inpaint_information(params=parameters)
