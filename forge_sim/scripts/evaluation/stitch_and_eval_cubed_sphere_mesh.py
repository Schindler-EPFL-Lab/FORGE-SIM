"""
Stitch a 6-patch cubed-sphere mesh optimization (the per-patch PLYs dumped by
`RegularCompositeMeshOptimizer.save_callback`) back into a single welded
surface, then evaluate its Chamfer distance against a reference mesh.

Background
----------
`optimize_thermoscenes_mesh_manual_initial_pose.py` optimizes a 6-patch
`RegularCompositeMesh` cubed sphere but only saves one PLY *per patch*
(`shape_left_*.ply` ... `shape_top_*.ply`): unlike the spline path it does
not jsonpickle the composite, so there is no single welded mesh. This script
rebuilds it: the patches share seam vertices by position (they are averaged
across patches during optimization), so welding coincident vertices reproduces
the watertight composite surface as one global mesh.

The optimized surface lives in Mitsuba *scene space*
(`tf_cameras @ tf_bbox @ unit_cubed_sphere`). The reference is the render-scale
BuildNet3D ground-truth mesh (e.g. `BuildNet3D_meshes/buildingN/*.ply`), which
is at the same scale as the reconstruction and differs from it by exactly the
camera transform plus the fixed nerfstudio Z-up/Y-up convention rotation. The
reference is therefore brought into the reconstruction frame deterministically
with `tf_cameras @ R_x(90 deg)` (no fitting), and the Chamfer distance is
measured there (the same frame convention as `eval_buildnet3d.py`).

Outputs (written to `output_dir`):
  - `<prefix>_stitched.ply`            welded full mesh, scene frame
  - `<reference_stem>_in_scene.ply`    reference brought into scene frame
  - `<prefix>_stitched_in_reference.ply` reconstruction in the reference frame
  - `chamfer.json`                     Chamfer distance + the applied transform
"""

import json
from dataclasses import dataclass
from pathlib import Path

import mitsuba as mi
import numpy as np
import numpy.typing as npt
import open3d as o3d
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

from forge_sim import logger
from forge_sim.metrics.shapes import (
    calculate_metrics_shape_mesh,
    load_reference_in_scene,
    reference_to_scene_transform,
)

# Cube-face suffixes in the canonical patch order used by
# `RegularCompositeMesh.cubed_sphere`.
CUBE_FACE_ORDER = ["left", "right", "back", "front", "bottom", "top"]


@dataclass(kw_only=True)
class StitchAndEvalParameters(ReverseCli):
    """Parameters for stitching the cubed-sphere patches and evaluating Chamfer."""

    patches_dir: Path
    """Directory holding the per-patch PLYs (`<prefix>_<face>_*.ply`)."""
    reference_shape_path: Path
    """
    Path to the render-scale ground-truth mesh (e.g.
    `BuildNet3D_meshes/buildingN/House.ply`), NOT the ~7x scaled-up
    `dataset/building.ply`.
    """
    scene_path: Path
    """Dataset directory containing `transforms.json` (used for `tf_cameras`)."""
    output_dir: Path = Path("./outputs/stitched_eval")
    """Where the stitched mesh, topology JSON, and metrics are written."""
    aligned_mesh_output_path: Path | None = None
    """
    Optional explicit path to also write the stitched mesh transformed into the
    reference frame (e.g. next to the reference mesh for side-by-side inspection
    in Blender).
    """

    mesh_name_prefix: str = "shape"
    """Filename prefix of the per-patch PLYs (the composite mesh name)."""
    weld_tolerance: float = 1e-5
    """Coordinates are rounded to this resolution to detect coincident seam vertices."""

    from_vggt: bool = False
    """Use the VGGT pose loader instead of the ThermoScenes one for tf_cameras."""
    align_cameras_xz: bool = True
    """
    The xz-plane camera alignment flag of the pose loader; must match the value
    used during optimization (the thermoscenes manual-pose script uses True).
    """
    n_sampled_points: int = 100000
    """Points sampled on each surface for the Chamfer distance."""

    def __post_init__(self) -> None:
        if not self.patches_dir.is_dir():
            raise FileNotFoundError(
                f"`patches_dir` {self.patches_dir} is not a directory."
            )
        if not self.reference_shape_path.exists():
            raise FileNotFoundError(
                f"`reference_shape_path` {self.reference_shape_path} does not exist."
            )
        if not (self.scene_path / "transforms.json").is_file():
            raise FileNotFoundError(
                f"`scene_path` {self.scene_path} has no transforms.json."
            )
        if self.weld_tolerance <= 0.0:
            raise ValueError("`weld_tolerance` must be positive.")


def discover_patch_files(patches_dir: Path, prefix: str) -> dict[str, Path]:
    """
    Returns one PLY per cube face, picking the highest-iteration file when several
    timestamps/iterations are present.

    `patches_dir` is searched for files matching `<prefix>_<face>_*.ply` (one per
    cube face), where `prefix` is the composite mesh name.

    :returns: A mapping from each cube-face name to its chosen PLY path.
    :raises FileNotFoundError: If any of the six faces is missing.
    """

    def iteration_of(path: Path) -> int:
        """Trailing `iteration-NNN` number in `path`'s stem, or -1 if absent."""
        stem = path.stem
        marker = "iteration-"
        if marker in stem:
            try:
                return int(stem.split(marker)[-1])
            except ValueError:
                logger.warning(f"Could not parse an iteration number from {path.name}.")
                return -1
        return -1

    patch_files: dict[str, Path] = {}
    for face in CUBE_FACE_ORDER:
        candidates = sorted(patches_dir.glob(f"{prefix}_{face}_*.ply"))
        if not candidates:
            raise FileNotFoundError(
                f"No PLY found for face '{face}' "
                f"(pattern '{prefix}_{face}_*.ply') in {patches_dir}."
            )
        patch_files[face] = max(candidates, key=iteration_of)
    return patch_files


def stitch_patches(
    patch_files: dict[str, Path], weld_tolerance: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """
    Welds the per-patch meshes into a single global mesh by merging coincident
    vertices.

    `patch_files` maps each cube face to its PLY, and `weld_tolerance` is the
    position resolution at which two vertices are treated as coincident. Seam
    vertices are shared across patches with identical positions (they are averaged
    during optimization), so rounding to `weld_tolerance` and taking unique
    positions reconstructs the watertight composite; each welded vertex position
    is the mean of its merged duplicates.

    :returns: A tuple of the welded vertices `(V, 3)` in the reconstruction
        (scene) frame and the welded triangle indices `(F, 3)`.
    """
    all_vertices: list[npt.NDArray[np.float64]] = []
    all_faces: list[npt.NDArray[np.int64]] = []
    offset = 0
    for face in CUBE_FACE_ORDER:
        mesh = o3d.io.read_triangle_mesh(str(patch_files[face]))
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.triangles, dtype=np.int64)
        all_vertices.append(vertices)
        all_faces.append(faces + offset)
        offset += vertices.shape[0]

    concatenated_vertices = np.concatenate(all_vertices, axis=0)
    concatenated_faces = np.concatenate(all_faces, axis=0)

    decimals = max(0, int(round(-np.log10(weld_tolerance))))
    _, inverse = np.unique(
        np.round(concatenated_vertices, decimals), axis=0, return_inverse=True
    )
    inverse = inverse.reshape(-1)

    welded_count = int(inverse.max()) + 1
    welded_vertices = np.zeros((welded_count, 3), dtype=np.float64)
    multiplicity = np.zeros(welded_count, dtype=np.float64)
    np.add.at(welded_vertices, inverse, concatenated_vertices)
    np.add.at(multiplicity, inverse, 1.0)
    welded_vertices /= multiplicity[:, None]
    welded_faces = inverse[concatenated_faces]

    return welded_vertices, welded_faces


def stitch_and_eval(params: StitchAndEvalParameters) -> None:
    """
    Stitches the per-patch PLYs, aligns the reference into the reconstruction
    frame deterministically, and writes the Chamfer distance.

    `params` carries the patch directory, the render-scale reference mesh, the
    scene directory (for `tf_cameras`), and the output locations; see
    `StitchAndEvalParameters` for the per-field meaning. The Chamfer distance, the
    frame it was measured in, and the reference<->scene transforms are written to
    `chamfer.json` in `params.output_dir`.
    """
    mi.set_variant("llvm_ad_rgb")
    params.output_dir.mkdir(parents=True, exist_ok=True)

    patch_files = discover_patch_files(params.patches_dir, params.mesh_name_prefix)
    logger.info(f"Stitching patches: {[p.name for p in patch_files.values()]}")

    welded_vertices, welded_faces = stitch_patches(patch_files, params.weld_tolerance)
    logger.info(
        f"Stitched {len(CUBE_FACE_ORDER)} patches into {welded_vertices.shape[0]} "
        f"welded vertices ({welded_faces.shape[0]} faces)."
    )

    stitched_path = params.output_dir / f"{params.mesh_name_prefix}_stitched.ply"
    stitched_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(welded_vertices),
        o3d.utility.Vector3iVector(welded_faces.astype(np.int32)),
    )
    stitched_mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(stitched_path), stitched_mesh)
    logger.info(f"Saved welded mesh to {stitched_path}")

    # Deterministic alignment: tf_cameras @ R_x(90 deg). No fitting.
    reference_to_scene = reference_to_scene_transform(
        params.scene_path, params.from_vggt, params.align_cameras_xz
    )
    logger.info(
        "Aligned reference into scene frame with tf_cameras @ R_x(90 deg) "
        "(nerfstudio convention)."
    )

    # Bring the reference into the scene frame and measure Chamfer there (the
    # eval_buildnet3d frame convention).
    reference_tensor = load_reference_in_scene(
        params.reference_shape_path, reference_to_scene
    )
    stitched_tensor = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(welded_vertices, o3d.core.Dtype.Float32),
        o3d.core.Tensor(welded_faces.astype(np.int32), o3d.core.Dtype.Int32),
    )

    metrics = calculate_metrics_shape_mesh(
        predicted_mesh=stitched_tensor,
        ground_truth_mesh=reference_tensor,
        n_sampled_points=params.n_sampled_points,
    )
    chamfer = metrics["chamfer_distance"]
    logger.info(f"Chamfer distance (reconstruction frame): {chamfer:.4f}")

    # The reference brought into the reconstruction frame (overlays the stitched
    # mesh at scene scale).
    reference_in_scene_path = (
        params.output_dir / f"{params.reference_shape_path.stem}_in_scene.ply"
    )
    o3d.t.io.write_triangle_mesh(str(reference_in_scene_path), reference_tensor)
    logger.info(f"Saved reference in scene frame to {reference_in_scene_path}")

    # The stitched mesh brought into the reference frame, for inspection: it
    # should overlay `reference_shape_path` at matching scale.
    scene_to_reference = np.linalg.inv(reference_to_scene)
    stitched_in_reference = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(welded_vertices),
        o3d.utility.Vector3iVector(welded_faces.astype(np.int32)),
    )
    stitched_in_reference.transform(scene_to_reference)
    stitched_in_reference.compute_vertex_normals()
    stitched_in_reference_path = (
        params.output_dir / f"{params.mesh_name_prefix}_stitched_in_reference.ply"
    )
    o3d.io.write_triangle_mesh(str(stitched_in_reference_path), stitched_in_reference)
    logger.info(
        f"Saved stitched mesh in the reference frame to {stitched_in_reference_path}."
    )

    if params.aligned_mesh_output_path is not None:
        params.aligned_mesh_output_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(
            str(params.aligned_mesh_output_path), stitched_in_reference
        )
        logger.info(
            f"Also wrote the aligned stitched mesh to {params.aligned_mesh_output_path}"
        )

    result = {
        "chamfer_distance": chamfer,
        "frame": "reconstruction_scene_space",
        "reference_to_scene_transform": reference_to_scene.tolist(),
        "stitched_to_reference_transform": scene_to_reference.tolist(),
    }
    with open(params.output_dir / "chamfer.json", "w") as chamfer_file:
        json.dump(result, chamfer_file, indent=4)


if __name__ == "__main__":
    parameters = tyro.cli(StitchAndEvalParameters)
    stitch_and_eval(parameters)
