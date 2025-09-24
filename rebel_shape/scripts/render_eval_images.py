"""
This script assumes that shape optimization has already been performed and
takes in an optimized spline. It renders the optimized spline from different eval views
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import mitsuba as mi
import tyro

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append("./")

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape import logger
from rebel_shape.data_processing.thermoscenes_loader import ThermoScenesLoader
from rebel_shape.meshing.tessellated_spline_surface import TessellatedSplineSurface
from rebel_shape.plugin_wrappers.sampler import Sampler
from rebel_shape.render.render import Render


@dataclass(kw_only=True)
class RenderSplineParameters(ReverseCli):
    """
    Parameters used for rendering spline model from eval images.
    """

    thermoscenes_scene_path: Path
    """
    Path to the ThermoScenes scene on what the spline was optimized.
    """
    spline_path: Path
    """Path to the reconstructed spline surface"""
    textures_paths: list[Path] | None = None
    """Paths to the saved textures"""
    environment_map_path: Path | None = None
    """
    Path to the envionment map that was used as background in optimization. If not
    provided, a solid white background is used instead.
    """
    flip_y: bool = True
    """If the texture was flipped along y axis during saving"""
    samples_per_pixel: int = 289
    """Number of samples per pixel used for rendering"""
    output_folder: Path = Path("./outputs/")
    """Where to save the outputs (created if it does not exist)"""
    use_gpu: bool = True
    """
    Whether to use GPU for rendering or not. Note that the compute instance
    must have GPU support for this to work. For local testing, set to False.
    """

    def __post_init__(self) -> None:
        if self.samples_per_pixel < 1:
            raise ValueError("`samples_per_pixel` must be a positive integer.")


def initialize_scene(
    spline_path: Path,
    textures_paths: list[Path] | None,
    envmap_path: Path | None,
) -> tuple[
    "mi.Scene",
    TessellatedSplineSurface,
]:
    """
    Creates and loads Mitsuba dictionary that represents the scene.

    `spline_path` specifies a path to a JSON file that contains the tessellated spline
    surface. `textures_paths` represents paths to saved textures of the spline surface
    patches. `envmap_path` is the path to the 360 image that was used as the background
    during optimization.

    :returns: A tuple containing:
        - `scene_source`: A Mitsuba scene object that contains the loaded scene.
        - `tessellated_spline`: The tessellated spline surface
    """

    if textures_paths is None:
        tessellated_spline_surface = TessellatedSplineSurface.from_file(
            composite_spline_surface_path=spline_path
        )
    else:
        tessellated_spline_surface = TessellatedSplineSurface.from_file_with_textures(
            composite_spline_surface_path=spline_path, texture_paths=textures_paths
        )

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

    scene_dict = {
        "type": "scene",
        "integrator": {"type": "path"},
        "emitter": environment,
    }

    for mesh in reversed(tessellated_spline_surface):
        scene_dict[mesh.name] = mesh.mitsuba_dict

    scene_source: "mi.Scene" = mi.load_dict(scene_dict)

    return (
        scene_source,
        tessellated_spline_surface,
    )


def render_spline(params: RenderSplineParameters) -> None:
    """
    Render spline surface from eval cameras.
    It loads the saved spline surface from `params.spline_path`. Optionally it also
    loads spline patches textures from `params.textures_paths` if provided. It reads
    eval cameras from `params.thermoscenes_scene_path` and renders the spline surface
    based on cameras. It renders RGB images and stores the results in the
    `params.output_folder`. Names correspond to the eval RGB images from the
    thermoscenes dataset. The files are stored in .exr format.

    :raises ValueError: If `params.thermoscenes_scene_path` is not a valid path, or
        it is not a directory.
    """
    if not params.thermoscenes_scene_path.exists():
        raise ValueError(f"Path {params.thermoscenes_scene_path} does not exist.")
    if not params.thermoscenes_scene_path.is_dir():
        raise ValueError(f"Path {params.thermoscenes_scene_path} is not a directory.")

    if params.use_gpu:
        mi.set_variant("cuda_ad_rgb")
        logger.info("Using the GPU for rendering.")
    else:
        mi.set_variant("llvm_ad_rgb")
        logger.info("Using the CPU for rendering.")

    # Initialize the scene, reference images and cameras
    scene_source, tessellated_spline = initialize_scene(
        spline_path=params.spline_path,
        textures_paths=params.textures_paths,
        envmap_path=params.environment_map_path,
    )
    scene_params = mi.traverse(scene_source)

    # read eval images
    sampler = Sampler(
        type=Sampler.Type.INDEPENDENT, sample_count=params.samples_per_pixel
    )
    loader = ThermoScenesLoader(
        scene_path=params.thermoscenes_scene_path, sensor_sampler=sampler
    )

    # optionally flip along y axis:
    if params.flip_y and params.textures_paths is not None:
        for mesh in tessellated_spline:
            texture_key = f"{mesh.name}.bsdf.reflectance.data"
            scene_params[texture_key] = scene_params[texture_key][::-1, :, :]
    scene_params.update()

    renderer = Render(scene_source)

    # render views and save them
    params.output_folder.mkdir(parents=True, exist_ok=True)
    rendered_images: list[mi.Bitmap] = []
    for filename, camera in zip(loader.eval_filenames, loader.eval_cameras):
        eval_filename_exr = Path(filename).stem + ".exr"
        image_output_path = params.output_folder / eval_filename_exr

        _ = renderer.render(
            camera=camera,
            samples_per_pixel=params.samples_per_pixel,
            output_path=image_output_path,
            output_type=mi.Struct.Type.Float32,
        )

    logger.info(
        f"Saved {len(rendered_images)} rendered images to {params.output_folder}"
    )


if __name__ == "__main__":
    parameters = tyro.cli(RenderSplineParameters)

    render_spline(params=parameters)
