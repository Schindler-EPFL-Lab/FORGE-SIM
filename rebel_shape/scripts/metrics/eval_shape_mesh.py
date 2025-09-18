"""
This script calculates the metrics of shape reconstruction between the bspline mesh and
the reference mesh.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import mitsuba as mi
import numpy as np
import numpy.typing as npt
import open3d as o3d
import tyro

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append("./")

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape.meshing.tessellated_spline_surface import TessellatedSplineSurface


@dataclass(kw_only=True)
class EvalShapeMeshParameters(ReverseCli):
    """
    Parameters used for evaluation shape mesh reconstruction.
    """

    reference_shape_path: Path
    """Path to the target shape on what the spline was optimized"""
    spline_path: Path
    """Path to the reconstructed spline surface"""
    tessellation_subdivisions: int = 256
    """
    How well the tessellated triangle mesh approximates the original B-spline surface.
    The higher the number, the more triangles there are, and the better the
    approximation.
    """
    n_sampled_points: int = 10000
    """How many points to sample on both meshes to compute shapes similarity"""
    output_file_path: Path = Path("./outputs/eval_shape_mesh.json")
    """Where to save the metrics (created if it does not exist)"""

    def __post_init__(self) -> None:
        if not self.reference_shape_path.exists():
            raise FileNotFoundError(f"'{self.reference_shape_path}' does not exist.")
        if not self.spline_path.exists():
            raise FileNotFoundError(f"'{self.spline_path}' does not exist.")


def load_tessellated_mesh(
    spline_path: Path,
    tessellation_subdivisions: int,
) -> tuple[o3d.t.geometry.TriangleMesh,]:
    """
    Loads optimized spline surface and transforms it into a triangle mesh.

    `spline_path` specifies a path to a JSON file that contains the tessellated spline
    surface. The spline surface is tessellated into `tessellation_subdivisions` along u
    and v directions.

    :returns: a triangle mesh which approximates the optimized spline surface.
    """

    tessellated_spline_surface = TessellatedSplineSurface.from_file(
        composite_spline_surface_path=spline_path
    )
    tessellated_mesh = (
        tessellated_spline_surface._composite_spline_surface.composite_mesh_tesselation(
            subdivisions_per_patch=tessellation_subdivisions
        )
    )

    # convert to open3d TriangleMesh
    vertices: list[npt.NDArray[np.float32]] = []
    faces: list[npt.NDArray[np.uint32]] = []
    shift_by = 0
    for mesh in tessellated_mesh:
        vertices.append(mesh.vertices.numpy())
        faces.append(mesh.faces.numpy() + shift_by)
        shift_by += vertices[-1].shape[0]
    vertices = np.concatenate(vertices, axis=0)
    faces = np.concatenate(faces, axis=0)

    optimized_shape = o3d.t.geometry.TriangleMesh(
        vertex_positions=o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
        triangle_indices=o3d.core.Tensor(faces, dtype=o3d.core.Dtype.Int32),
    )

    return optimized_shape


def eval_shape_mesh(params: EvalShapeMeshParameters) -> None:
    """
    Calculates the metrics for shape reconstruction between the ground truth mesh and
    the optimized spline surface.

    It loads the saved spline surface from `params.spline_path` and tessellates it into
    a triangle mesh. It reads the original mesh from `params.reference_shape_path`. Then
    it samples `params.n_sampled_points` points on both meshes and calculates the
    Chamfer Distance between the sampled point clouds. The lower the result is the
    better the spline approximation.

    :raises ValueError: If `params.thermoscenes_scene_path` is not a valid path, or
        it is not a directory.
    """

    mi.set_variant("llvm_ad_rgb")

    # Initialize the shapes
    ref_shape = o3d.t.io.read_triangle_mesh(str(params.reference_shape_path))
    optimized_shape = load_tessellated_mesh(
        spline_path=params.spline_path,
        tessellation_subdivisions=params.tessellation_subdivisions,
    )

    # calculate the metrics
    metric_params = o3d.t.geometry.MetricParameters(
        n_sampled_points=params.n_sampled_points
    )

    metrics = ref_shape.compute_metrics(
        optimized_shape,
        (o3d.t.geometry.Metric.ChamferDistance,),
        metric_params,
    )
    metrics = metrics.numpy()

    params.output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(params.output_file_path, "w") as f:
        json.dump({"chamfer_distance": metrics[0].item()}, f)


if __name__ == "__main__":
    parameters = tyro.cli(EvalShapeMeshParameters)

    eval_shape_mesh(params=parameters)
