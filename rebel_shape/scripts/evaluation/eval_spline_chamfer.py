"""
Deterministic Chamfer evaluation of a spline ("ours") reconstruction against the
render-scale BuildNet3D ground-truth mesh.

Spline counterpart of ``stitch_and_eval_cubed_sphere_mesh.py``: the optimized
spline (a jsonpickled ``TessellatedSplineSurface``) is tessellated into its
scene-frame global mesh, the reference (e.g. ``BuildNet3D_meshes/buildingN/*.ply``,
at render scale -- NOT the ~7x scaled ``dataset/building.ply``) is brought into
that frame with the deterministic ``tf_cameras @ R_x(90 deg)`` transform (no
fitting), and the Chamfer distance is measured there (the ``eval_buildnet3d.py``
frame convention).

This is torch-free (no novel-view-synthesis metrics) and applies the full
``tf_cameras @ R_x(90 deg)`` transform, whereas ``eval_buildnet3d.py`` applies
only ``tf_cameras`` to the reference (``eval_buildnet3d.py:175-178``).
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import mitsuba as mi
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append("./")

from rebel_shape import logger
from rebel_shape.meshing.tessellated_spline_surface import TessellatedSplineSurface
from rebel_shape.metrics.shapes import (
    calculate_metrics_shape_mesh,
    global_mesh_to_open3d_mesh,
    load_reference_in_scene,
    reference_to_scene_transform,
)


@dataclass(kw_only=True)
class EvalSplineChamferParameters(ReverseCli):
    """Parameters for the deterministic spline Chamfer evaluation."""

    spline_path: Path
    """Path to the optimized spline (jsonpickled ``TessellatedSplineSurface``)."""
    reference_shape_path: Path
    """
    Path to the render-scale ground-truth mesh (e.g.
    ``BuildNet3D_meshes/buildingN/House.ply``), NOT the ~7x scaled-up
    ``dataset/building.ply``.
    """
    scene_path: Path
    """Dataset directory containing ``transforms.json`` (used for ``tf_cameras``)."""
    output_file_path: Path = Path("./outputs/eval_spline_chamfer.json")
    """Where the metric is written (parent directory created if needed)."""

    tessellation_subdivisions: int = 256
    """Per-patch tessellation resolution used to sample the spline surface."""
    n_sampled_points: int = 100000
    """Points sampled on each surface for the Chamfer distance."""

    from_vggt: bool = False
    """Use the VGGT pose loader instead of the ThermoScenes one for tf_cameras."""
    align_cameras_xz: bool = True
    """
    The xz-plane camera alignment flag of the pose loader; must match the value
    used during optimization (the thermoscenes manual-pose script uses True).
    """

    def __post_init__(self) -> None:
        if not self.spline_path.exists():
            raise FileNotFoundError(f"`spline_path` {self.spline_path} does not exist.")
        if not self.reference_shape_path.exists():
            raise FileNotFoundError(
                f"`reference_shape_path` {self.reference_shape_path} does not exist."
            )
        if not (self.scene_path / "transforms.json").is_file():
            raise FileNotFoundError(
                f"`scene_path` {self.scene_path} has no transforms.json."
            )


def eval_spline_chamfer(params: EvalSplineChamferParameters) -> None:
    """
    Tessellates the spline, aligns the reference deterministically, and writes the
    Chamfer distance.

    `params` carries the spline path, the render-scale reference mesh, the scene
    directory (for `tf_cameras`), and the output file; see
    `EvalSplineChamferParameters` for the per-field meaning. The Chamfer distance,
    the frame it was measured in, and the reference -> scene transform are written
    to `params.output_file_path`.
    """
    mi.set_variant("llvm_ad_rgb")

    spline = TessellatedSplineSurface.from_file(params.spline_path)
    # Re-tessellate at the requested density and use the *returned* mesh:
    # `composite_mesh_tesselation` does not mutate the surface, so reading
    # `spline.global_*` would silently use the tessellation baked into the JSON.
    tessellated = spline.composite_spline.composite_mesh_tesselation(
        subdivisions_per_patch=params.tessellation_subdivisions
    )
    optimized_shape = global_mesh_to_open3d_mesh(
        tessellated.global_vertices, tessellated.global_faces
    )

    # Deterministic alignment: tf_cameras @ R_x(90 deg). No fitting.
    reference_to_scene = reference_to_scene_transform(
        params.scene_path, params.from_vggt, params.align_cameras_xz
    )
    logger.info(
        "Aligned reference into scene frame with tf_cameras @ R_x(90 deg) "
        "(nerfstudio convention)."
    )

    reference_tensor = load_reference_in_scene(
        params.reference_shape_path, reference_to_scene
    )

    metrics = calculate_metrics_shape_mesh(
        predicted_mesh=optimized_shape,
        ground_truth_mesh=reference_tensor,
        n_sampled_points=params.n_sampled_points,
    )
    chamfer = metrics["chamfer_distance"]
    logger.info(f"Chamfer distance (reconstruction frame): {chamfer:.4f}")

    params.output_file_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "chamfer_distance": chamfer,
        "frame": "reconstruction_scene_space",
        "reference_to_scene_transform": reference_to_scene.tolist(),
    }
    with open(params.output_file_path, "w") as output_file:
        json.dump(result, output_file, indent=4)


if __name__ == "__main__":
    eval_spline_chamfer(tyro.cli(EvalSplineChamferParameters))
