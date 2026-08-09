"""
This script assumes that shape optimization has already been performed and takes in the
output folder of that run. It renders the optimized composite mesh from the eval views
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import drjit as dr
import mitsuba as mi
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape.data_processing.pose_scene_loader import PoseSceneLoader

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append("./")

from rebel_shape import logger
from rebel_shape.data_processing.vggt_pose_scene_loader import VGGTPoseSceneLoader
from rebel_shape.meshing.mesh import Mesh
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from rebel_shape.plugin_wrappers.sampler import Sampler
from rebel_shape.render.render import Render


CUBE_FACE_NAMES = ("left", "right", "back", "front", "bottom", "top")
"""
Patch-name suffixes of a cubed-sphere composite, as produced by
`RegularCompositeMesh.cubed_sphere`. A patch is named ``<composite name>_<face>``.
"""


def _iteration_rank(path: Path) -> int:
    """
    Ordering key over the saves of one patch: the trailing ``iteration-NNN`` number in
    `path`'s stem, or `sys.maxsize` for a stem that carries none.

    The optimization scripts write their numbered checkpoints during the run and the
    unnumbered final save once it has finished, so an unnumbered name is the latest
    state there is and must outrank every checkpoint.

    :returns: The rank of `path`; larger is later.
    """
    marker = "iteration-"
    stem = path.stem
    if marker not in stem:
        return sys.maxsize
    try:
        return int(stem.split(marker)[-1])
    except ValueError:
        logger.warning(f"Could not parse an iteration number from {path.name}.")
        return -1


def face_key(stem: str) -> str | None:
    """
    The ``<composite name>_<face>`` prefix of `stem`, or `None` if it names no cube
    face.

    Public because it is also how a caller tells a key `discover_mesh_textures`
    returned apart: one that answers non-`None` came from the per-face layout, and so
    names a single patch rather than a whole surface.

    Anchoring on the face name is what tells a patch save (``shape_left.ply``, or
    ``shape_left_<timestamp>_iteration-<N>.ply``) apart from the welded global mesh
    that the optimization scripts write alongside them, which carries no face name and
    must not enter the scene as a seventh patch.

    :returns: The patch key, or `None` when `stem` names no cube face.
    :raises ValueError: If `stem` names more than one cube face, so that no single
        patch can be meant.
    """
    tokens = stem.split("_")
    face_positions = [
        index for index, token in enumerate(tokens) if token in CUBE_FACE_NAMES
    ]
    if not face_positions:
        return None
    if len(face_positions) > 1:
        named_faces = ", ".join(tokens[index] for index in face_positions)
        raise ValueError(
            f"File name '{stem}' names several cube faces ({named_faces}), so the "
            "patch it belongs to is ambiguous."
        )
    return "_".join(tokens[: face_positions[0] + 1])


def _latest_save(candidates: list[Path], description: str) -> Path:
    """
    The single latest save among `candidates`, by `_iteration_rank`.

    `description` names what is being selected and appears in the error message.

    :returns: The highest-ranked path.
    :raises ValueError: If two or more candidates share the highest rank, since
        picking either one would silently depend on directory order.
    """
    highest_rank = max(_iteration_rank(path) for path in candidates)
    latest = [path for path in candidates if _iteration_rank(path) == highest_rank]
    if len(latest) > 1:
        names = ", ".join(path.name for path in latest)
        raise ValueError(
            f"Cannot choose the {description}: {names} are equally recent. Remove the "
            "stale ones, or point `result_path` at a directory holding a single run."
        )
    return latest[0]


def _single_mesh_textures(
    result_path: Path, meshes: list[Path], textures: list[Path]
) -> dict[str, tuple[Path, Path | None]]:
    """
    The `discover_mesh_textures` result for a result directory holding one whole-surface
    mesh rather than the patches of a composite, keyed by that mesh's stem.

    A run that optimized a single mesh names neither file after a cube face, and the
    recorded runs name the pair in unrelated styles -- ``thermoscene_shape.ply`` with
    ``thermoscene_rgb_texture.exr``, ``lion_thermoscene_shape.ply`` with
    ``lion_thermoscene_rgb_texture.exr``, ``shape_<timestamp>_iteration-<N>.ply`` with
    ``shape-bsdf-reflectance-data_<timestamp>_iteration-<N>.exr``. No name relation
    holds across all of them, so being the only candidate is the whole of the evidence
    that the two belong together. That is why a second mesh or a second texture is an
    error here rather than something to rank by recency: with nothing to match on, a
    choice between two would be a guess.

    `meshes` are the directory's ``.ply`` files (none of which names a cube face) and
    `textures` its ``.exr`` files.

    :returns: A one-entry mapping from the mesh's stem to its ``(mesh path, texture
        path or None)``.
    :raises FileNotFoundError: If `meshes` is empty.
    :raises ValueError: If `meshes` holds more than one mesh, or `textures` more than
        one texture.
    """
    if not meshes:
        raise FileNotFoundError(
            f"No mesh found in {result_path}. It must hold either the patch meshes of "
            f"a composite -- PLY files named <composite name>_<face>[_<timestamp>_"
            f"iteration-<N>].ply, with <face> one of {', '.join(CUBE_FACE_NAMES)} -- "
            f"or a single whole-surface PLY."
        )
    if len(meshes) > 1:
        names = ", ".join(path.name for path in meshes)
        raise ValueError(
            f"No PLY in {result_path} names a cube face, so it is read as a single "
            f"whole-surface reconstruction, but it holds {len(meshes)} of them "
            f"({names}). Leave only the mesh to render."
        )
    if len(textures) > 1:
        names = ", ".join(path.name for path in textures)
        raise ValueError(
            f"The single mesh {meshes[0].name} in {result_path} cannot be paired with "
            f"a texture: there are {len(textures)} of them ({names}) and no cube-face "
            f"key to match on. Leave only the texture that belongs to it."
        )

    mesh = meshes[0]
    texture = textures[0] if textures else None
    if texture is None:
        logger.warning(
            f"Single-mesh layout in {result_path}: {mesh.name}, with no texture."
        )
    else:
        logger.info(
            f"Single-mesh layout in {result_path}: {mesh.name} with texture "
            f"{texture.name}."
        )
    return {mesh.stem: (mesh.resolve(), texture.resolve() if texture else None)}


def discover_mesh_textures(result_path: Path) -> dict[str, tuple[Path, Path | None]]:
    """
    Pairs every optimized mesh in `result_path` with its optimized texture.

    Two result layouts are recognised, and which one applies is decided by whether any
    ``.ply`` in the directory carries a cube-face token.

    In the **per-face composite** layout, patch meshes are the ``.ply`` files whose
    name carries such a token, and each is keyed by its ``<composite name>_<face>``
    prefix; the welded global mesh, which names no face, is skipped. Where a patch has
    both numbered checkpoints and a final save, the latest one wins (see
    `_iteration_rank`). A patch's texture is the latest ``.exr`` whose name starts with
    the same key -- both ``<key>_texture.exr`` and the
    ``<key>-bsdf-reflectance-data_<timestamp>_iteration-<N>.exr`` written during the
    run -- or `None` when the run saved no texture for it.

    In the **single-mesh** layout, no ``.ply`` names a cube face and the directory
    holds one whole-surface mesh with at most one texture; see `_single_mesh_textures`
    for why uniqueness is the only pairing rule available there.

    Every choice is logged. `_iteration_rank`'s ordering only holds within one run, so
    two runs of the same composite dumped into one directory cannot be told apart --
    the older run's unnumbered save would outrank the newer run's checkpoints -- and
    the log is what makes that visible after the fact.

    :returns: A mapping from mesh key to its ``(mesh path, texture path or None)``.
    :raises FileNotFoundError: If `result_path` is not a directory, or holds no ``.ply``
        at all.
    :raises ValueError: If one cube face is claimed by two different composites, if a
        patch's latest mesh or texture cannot be singled out, or if the single-mesh
        layout is ambiguous.
    """
    if not result_path.is_dir():
        raise FileNotFoundError(f"`result_path` {result_path} is not a directory.")

    texture_files = sorted(result_path.glob("*.exr"))

    meshes_by_key: dict[str, list[Path]] = {}
    unkeyed_meshes: list[Path] = []
    for path in sorted(result_path.glob("*.ply")):
        key = face_key(path.stem)
        if key is None:
            unkeyed_meshes.append(path)
        else:
            meshes_by_key.setdefault(key, []).append(path)

    if not meshes_by_key:
        return _single_mesh_textures(result_path, unkeyed_meshes, texture_files)

    logger.info(
        f"Per-face composite layout in {result_path}: {len(meshes_by_key)} patches."
    )
    for path in unkeyed_meshes:
        logger.info(f"Ignoring {path.name}: its name carries no cube-face token.")

    key_by_face: dict[str, str] = {}
    for key in meshes_by_key:
        face = key.rsplit("_", 1)[-1]
        if face in key_by_face:
            raise ValueError(
                f"Cube face '{face}' is claimed by both '{key_by_face[face]}' and "
                f"'{key}' in {result_path}; the results of two runs appear to be "
                "mixed in one directory."
            )
        key_by_face[face] = key

    mesh_textures: dict[str, tuple[Path, Path | None]] = {}
    for key, meshes in meshes_by_key.items():
        mesh = _latest_save(meshes, f"mesh of patch '{key}'")
        textures = [
            path
            for path in texture_files
            if path.stem == key or path.stem.startswith((f"{key}_", f"{key}-"))
        ]
        texture = (
            _latest_save(textures, f"texture of patch '{key}'") if textures else None
        )
        if texture is None:
            logger.warning(f"No texture found for patch '{key}' in {result_path}.")
        else:
            logger.info(f"Patch '{key}': {mesh.name} with texture {texture.name}.")
        mesh_textures[key] = (mesh.resolve(), texture.resolve() if texture else None)

    return mesh_textures


@dataclass(kw_only=True)
class RenderSplineParameters(ReverseCli):
    """
    Parameters used for rendering an optimized composite mesh from the eval cameras
    of the scene it was optimized on.
    """

    scene_path: Path
    """
    Path to the ThermoScenes scene the mesh was optimized on.
    """
    result_path: Path
    """
    Path containing meshes and textures. The patch meshes and their textures are
    discovered from it by `discover_mesh_textures`; the welded global mesh and any
    other file there is ignored.
    """
    environment_map_path: Path | None = None
    """
    Path to the envionment map that was used as background in optimization. If not
    provided, a solid white background is used instead.
    """
    flip_y: bool = False
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

    def __post_init__(self) -> None:
        if self.samples_per_pixel < 1:
            raise ValueError("`samples_per_pixel` must be a positive integer.")
        if self.output_folder is None:
            self.output_folder: Path = self.result_path / "renders"

        self.mesh_textures_paths: dict[str, tuple[Path, Path | None]] = (
            discover_mesh_textures(self.result_path)
        )


def initialize_scene(
    mesh_texture_paths: dict[str, tuple[Path, Path | None]],
    envmap_path: Path | None,
) -> "mi.Scene":
    """
    Creates and loads Mitsuba dictionary that represents the scene.

    `mesh_texture_paths` maps each mesh name to a `(mesh PLY path, texture EXR
    path)` pair; a `None` texture is replaced with a flat 0.5 grey placeholder.
    `envmap_path` is the path to the 360 image that was used as the background
    during optimization; if `None`, a solid white background is used instead.

    :returns: The loaded Mitsuba scene.
    """

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

    for name, (mesh_path, texture_path) in mesh_texture_paths.items():
        if texture_path is None:
            texture = DiffuseBSDF(
                DiffuseBSDF.TextureFromTensor(
                    tensor=dr.full(
                        dtype=mi.TensorXf,
                        value=0.5,
                        shape=(256, 256, 3),
                    )
                )
            )
        else:
            texture = DiffuseBSDF(DiffuseBSDF.Texture(texture_path=texture_path))
        mesh = Mesh.from_ply_file(file_path=mesh_path, bsdf=texture)
        scene_dict[name] = mesh.mitsuba_dict

    scene_source: "mi.Scene" = mi.load_dict(scene_dict)

    return scene_source


def render_spline(params: RenderSplineParameters) -> None:
    """
    Render an optimized composite mesh from eval cameras.

    It loads each patch mesh and its texture from `params.result_path` (paired by
    `discover_mesh_textures` in `params.__post_init__`), reads the eval cameras from
    `params.scene_path`, and renders the surface from them. It renders RGB images and
    stores the results in `params.output_folder`. Names correspond to the eval RGB
    images from the thermoscenes dataset. The files are stored in .exr format.

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
    scene_source = initialize_scene(
        mesh_texture_paths=params.mesh_textures_paths,
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
    if params.flip_y:
        for name in params.mesh_textures_paths:
            texture_key = name + ".bsdf.reflectance.data"
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
