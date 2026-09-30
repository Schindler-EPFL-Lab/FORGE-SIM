"""
This script calculates the metrics of shape reconstruction between the bspline mesh and
the reference mesh.
"""

import json
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import mitsuba as mi
import open3d as o3d
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

from forge_sim.meshing.tessellated_spline_surface import TessellatedSplineSurface
from forge_sim.metrics.shapes import (
    calculate_metrics_shape_mesh,
    tessellated_spline_to_open3d_mesh,
)


@dataclass(kw_only=True)
class EvalShapeMeshParametersBase(ReverseCli):
    """
    Parameters used for evaluation shape mesh reconstruction.
    """

    reference_shape_path: Path
    """Path to the target shape on what the spline was optimized"""
    n_sampled_points: int = 100000
    """How many points to sample on both meshes to compute shapes similarity"""
    output_file_path: Path = Path("./outputs/eval_shape_mesh.json")
    """Where to save the metrics (created if it does not exist)"""

    def __post_init__(self) -> None:
        if not self.reference_shape_path.exists():
            raise FileNotFoundError(f"'{self.reference_shape_path}' does not exist.")

    @abstractmethod
    def get_open3d_mesh(self) -> o3d.t.geometry.TriangleMesh:
        """Returns a shape mesh converted to Open3D"""
        raise NotImplementedError("The method `get_open3d_mesh` is not implemented.")


@dataclass(kw_only=True)
class EvalShapeMeshParametersSpline(EvalShapeMeshParametersBase):
    """Parameters used for evaluation shape mesh reconstruction using splines"""

    spline_path: Path
    """Path to the reconstructed spline surface"""
    tessellation_subdivisions: int = 256
    """
    How well the tessellated triangle mesh approximates the original B-spline surface.
    The higher the number, the more triangles there are, and the better the
    approximation.
    """

    def __post_init__(self) -> None:
        if not self.spline_path.exists():
            raise FileNotFoundError(f"'{self.spline_path}' does not exist.")

    def get_open3d_mesh(self) -> o3d.t.geometry.TriangleMesh:
        """Returns a spline tessellated mesh converted to Open3D"""
        optimized_spline_surface = TessellatedSplineSurface.from_file(self.spline_path)
        optimized_spline_surface._composite_spline_surface.composite_mesh_tesselation(
            subdivisions_per_patch=self.tessellation_subdivisions
        )
        return tessellated_spline_to_open3d_mesh(optimized_spline_surface)


def eval_shape_mesh(params: EvalShapeMeshParametersBase) -> None:
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
    optimized_shape = params.get_open3d_mesh()

    # calculate the metrics
    metrics = calculate_metrics_shape_mesh(
        predicted_mesh=optimized_shape,
        ground_truth_mesh=ref_shape,
        n_sampled_points=params.n_sampled_points,
    )

    params.output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(params.output_file_path, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    parameters = tyro.cli(EvalShapeMeshParametersSpline)

    eval_shape_mesh(params=parameters)
