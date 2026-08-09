"""
This script assumes that shape optimization has already been performed and
takes in an optimized spline. It renders the optimized spline from different eval views
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import mitsuba as mi
import tyro

from rebel_shape.data_processing.vggt_pose_scene_loader import VGGTPoseSceneLoader

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append("./")

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape import logger
from rebel_shape.data_processing.pose_scene_loader import PoseSceneLoader
from rebel_shape.meshing.tessellated_spline_surface import TessellatedSplineSurface
from rebel_shape.plugin_wrappers.sampler import Sampler
from rebel_shape.render.render import Render


def discover_spline_checkpoint(result_folder: Path) -> Path:
    """
    Returns the spline checkpoint written by `SplineOptimizer.save_callback` into
    `result_folder`, picking the highest-iteration one when several are present.

    Checkpoints are named ``<spline name>_<timestamp>_iteration-<N>.json``. A result
    folder routinely holds other JSON files as well -- an evaluation metrics dump, for
    instance -- so only names carrying an ``iteration-<N>`` marker are considered.

    :returns: The path of the highest-iteration spline checkpoint.
    :raises FileNotFoundError: If `result_folder` is not a directory, or holds no file
        named like a spline checkpoint.
    """
    if not result_folder.is_dir():
        raise FileNotFoundError(f"`result_folder` {result_folder} is not a directory.")

    def iteration_of(path: Path) -> int:
        """Trailing ``iteration-NNN`` number in `path`'s stem, or -1 if absent."""
        stem = path.stem
        marker = "iteration-"
        if marker not in stem:
            return -1
        try:
            return int(stem.split(marker)[-1])
        except ValueError:
            logger.warning(f"Could not parse an iteration number from {path.name}.")
            return -1

    json_files = sorted(result_folder.glob("*.json"))
    checkpoints = [path for path in json_files if iteration_of(path) >= 0]
    if not checkpoints:
        found = ", ".join(path.name for path in json_files) or "no JSON files at all"
        raise FileNotFoundError(
            f"No spline checkpoint (<name>_<timestamp>_iteration-<N>.json) in "
            f"{result_folder}; it holds {found}."
        )
    return max(checkpoints, key=iteration_of)


@dataclass(kw_only=True)
class RenderSplineParameters(ReverseCli):
    """
    Parameters used for rendering spline model from eval images.
    """

    scene_path: Path
    """
    Path to the scene on what the spline was optimized.
    """
    spline_path: Path | None = None
    """Path to the reconstructed spline surface"""
    textures_paths: list[Path] | None = None
    """Paths to the saved textures"""
    result_folder: Path | None = None
    """
    Folder with all the texture and the spline object as json. The spline checkpoint is
    the highest-iteration ``<name>_<timestamp>_iteration-<N>.json`` in it; any other
    JSON file there is ignored. Ignored if textures_paths or spline_path is not None
    """
    environment_map_path: Path | None = None
    """
    Path to the envionment map that was used as background in optimization. If not
    provided, a solid white background is used instead.
    """
    flip_y: bool = True
    """If the texture was flipped along y axis during saving"""
    samples_per_pixel: int = 289
    """Number of samples per pixel used for rendering"""
    output_folder: Path | None = None
    """Where to save the outputs (created if it does not exist)"""
    use_gpu: bool = True
    """
    Whether to use GPU for rendering or not. Note that the compute instance
    must have GPU support for this to work. For local testing, set to False.
    """
    from_vggt: bool = False
    """
    True if the pose data is from VGGT. Revert to Thermoscene format otherwise
    """

    def __post_init__(self) -> None:
        if self.spline_path is None:
            if self.result_folder is None:
                raise RuntimeError(
                    "Provide either `spline_path`, the saved spline surface, or "
                    "`result_folder`, the optimization output directory holding it."
                )
            self.spline_path = discover_spline_checkpoint(self.result_folder)

        if self.samples_per_pixel < 1:
            raise ValueError("`samples_per_pixel` must be a positive integer.")
        if self.textures_paths is None and self.result_folder is not None:
            self.textures_paths: list[Path] = []
            for el in self.result_folder.iterdir():
                if not el.suffix == ".exr":
                    continue
                self.textures_paths.append(el.resolve())
        if self.output_folder is None:
            self.output_folder: Path = self.spline_path.parent / "renders"


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
    eval cameras from `params.scene_path` and renders the spline surface
    based on cameras. It renders RGB images and stores the results in the
    `params.output_folder`. Names correspond to the eval RGB images from the
    thermoscenes dataset. The files are stored in .exr format.

    :raises ValueError: If `params.scene_path` is not a valid path, or
        it is not a directory.
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
    if params.from_vggt:
        loader = VGGTPoseSceneLoader(
            scene_path=params.scene_path, sensor_sampler=sampler
        )
    else:
        loader = PoseSceneLoader(scene_path=params.scene_path, sensor_sampler=sampler)

    # optionally flip along y axis:
    if params.flip_y and params.textures_paths is not None:
        for mesh in tessellated_spline:
            texture_key = f"{mesh.name}.bsdf.reflectance.data"
            scene_params[texture_key] = scene_params[texture_key][::-1, :, :]
    scene_params.update()

    renderer = Render(scene_source)

    # render views and save them
    params.output_folder.mkdir(parents=True, exist_ok=True)
    for filename, camera in loader.eval_data:
        eval_filename_exr = Path(filename).stem + ".exr"
        image_output_path = params.output_folder / eval_filename_exr

        _ = renderer.render(
            camera=camera,
            samples_per_pixel=params.samples_per_pixel,
            save_path=image_output_path,
            output_type=mi.Struct.Type.Float32,
        )

    logger.info(
        f"Saved {len(loader.eval_cameras)} rendered images to {params.output_folder}"
    )


if __name__ == "__main__":
    parameters = tyro.cli(RenderSplineParameters)

    render_spline(params=parameters)
