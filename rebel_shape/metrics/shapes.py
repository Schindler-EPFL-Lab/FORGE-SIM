"""
This file contains necessary functions to shape reconstruction metrics.
"""

import open3d as o3d

from rebel_shape.meshing.tessellated_spline_surface import TessellatedSplineSurface


def tessellated_spline_to_open3d_mesh(
    tessellated_spline_surface: TessellatedSplineSurface,
) -> o3d.t.geometry.TriangleMesh:
    """
    Transforms the spline into a open3d triangle mesh.

    :returns: a triangle mesh which approximates the optimized spline surface.
    """

    return o3d.t.geometry.TriangleMesh(
        vertex_positions=o3d.core.Tensor(
            tessellated_spline_surface.global_vertices.numpy(),
            dtype=o3d.core.Dtype.Float32,
        ),
        triangle_indices=o3d.core.Tensor(
            tessellated_spline_surface.global_faces.numpy(), dtype=o3d.core.Dtype.Int32
        ),
    )


def calculate_metrics_shape_mesh(
    predicted_mesh: o3d.t.geometry.TriangleMesh,
    ground_truth_mesh: o3d.t.geometry.TriangleMesh,
    n_sampled_points: int,
) -> dict[str, float]:
    """
    Calculates the metrics for shape reconstruction between the ground truth mesh and
    the optimized spline surface.

    It loads the saved spline surface from `params.spline_path` and tessellates it into
    a triangle mesh. It reads the original mesh from `params.reference_shape_path`. Then
    it samples `params.n_sampled_points` points on both meshes and calculates the
    Chamfer Distance between the sampled point clouds. The lower the result is the
    better the spline approximation.

    :raises ValueError: If `params.thermoscenes_scene_path` is not a valid path, or it
        is not a directory.
    """

    # calculate the metrics
    metric_params = o3d.t.geometry.MetricParameters(n_sampled_points=n_sampled_points)

    metrics = ground_truth_mesh.compute_metrics(
        predicted_mesh,
        (o3d.t.geometry.Metric.ChamferDistance,),
        metric_params,
    )
    metrics = metrics.numpy()
    return {"chamfer_distance": metrics[0].item()}
