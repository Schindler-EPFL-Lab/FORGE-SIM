"""
This file contains necessary functions to shape reconstruction metrics.
"""

from pathlib import Path

import mitsuba as mi
import numpy as np
import numpy.typing as npt
import open3d as o3d

from forge_sim.data_processing.pose_scene_loader import PoseSceneLoader
from forge_sim.data_processing.vggt_pose_scene_loader import VGGTPoseSceneLoader
from forge_sim.meshing.tessellated_spline_surface import TessellatedSplineSurface

# The nerfstudio Z-up <-> Y-up convention: a +90 deg rotation about X, applied to
# the reference mesh in the render frame (before tf_cameras). For the synthetic
# BuildNet3D scenes this is the *exact* correction between the render-scale GT mesh
# and the reconstruction (zero translation, unit scale), it reaches the same
# silhouette match as a full ICP fit, so no fitting is needed.
NERFSTUDIO_AXIS_FIX: npt.NDArray[np.float64] = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def global_mesh_to_open3d_mesh(
    global_vertices: "mi.TensorXf", global_faces: "mi.TensorXi"
) -> o3d.t.geometry.TriangleMesh:
    """
    Builds an Open3D tensor triangle mesh from a global vertex/face buffer.

    `global_vertices` is an `(V, 3)` position tensor and `global_faces` an
    `(F, 3)` index tensor, as exposed by `CompositeMesh` (`global_vertices` /
    `global_faces`), `TessellatedSplineSurface`, or `Mesh` (`vertices` /
    `faces`).

    :returns: the corresponding Open3D tensor triangle mesh.
    """
    return o3d.t.geometry.TriangleMesh(
        vertex_positions=o3d.core.Tensor(
            global_vertices.numpy(), dtype=o3d.core.Dtype.Float32
        ),
        triangle_indices=o3d.core.Tensor(
            global_faces.numpy().astype(np.int32), dtype=o3d.core.Dtype.Int32
        ),
    )


def tessellated_spline_to_open3d_mesh(
    tessellated_spline_surface: TessellatedSplineSurface,
) -> o3d.t.geometry.TriangleMesh:
    """
    Converts a tessellated spline surface into an Open3D tensor triangle mesh.

    `tessellated_spline_surface` provides the global (welded) vertices and faces of
    its current tessellation, which are copied into the returned mesh.

    :returns: a triangle mesh which approximates the optimized spline surface.
    """
    return global_mesh_to_open3d_mesh(
        tessellated_spline_surface.global_vertices,
        tessellated_spline_surface.global_faces,
    )


def calculate_metrics_shape_mesh(
    predicted_mesh: o3d.t.geometry.TriangleMesh,
    ground_truth_mesh: o3d.t.geometry.TriangleMesh,
    n_sampled_points: int,
) -> dict[str, float]:
    """
    Computes the Chamfer distance between two meshes that already share a frame.

    `predicted_mesh` is the reconstruction and `ground_truth_mesh` is the reference
    it is compared against. `n_sampled_points` points are sampled uniformly on each
    surface and the bidirectional Chamfer distance between the two point sets is
    measured (in the meshes' shared units; the lower the better).

    :returns: A dict with the single key `"chamfer_distance"`.
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


def reference_to_scene_transform(
    scene_path: Path, from_vggt: bool = False, align_cameras_xz: bool = True
) -> npt.NDArray[np.float64]:
    """
    Returns the deterministic 4x4 transform taking a render-scale reference mesh
    into the reconstruction (scene) frame: `tf_cameras @ R_x(90 deg)`.

    `scene_path` is the dataset directory holding `transforms.json` (the cameras
    `tf_cameras` is built from). `from_vggt` selects the VGGT pose loader instead
    of the ThermoScenes one. `align_cameras_xz` is the camera xz-plane alignment
    flag and must match the value the optimization used.

    `tf_cameras` is the pose-loader transform applied to the cameras during
    optimization, and `R_x(90 deg)` is the fixed nerfstudio Z-up/Y-up convention
    rotation. No scale or translation is fitted. A Mitsuba variant must already be
    set by the caller.

    :returns: The 4x4 reference -> scene transform.
    """
    loader_class = VGGTPoseSceneLoader if from_vggt else PoseSceneLoader
    loader = loader_class(scene_path=scene_path, align_cameras_xz=align_cameras_xz)
    tf_cameras = np.array(mi.Transform4f(loader.tf_cameras).matrix).reshape(4, 4)
    return tf_cameras @ NERFSTUDIO_AXIS_FIX


def load_reference_in_scene(
    reference_shape_path: Path, reference_to_scene: npt.NDArray[np.float64]
) -> o3d.t.geometry.TriangleMesh:
    """
    Loads the reference mesh and moves it into the reconstruction (scene) frame.

    `reference_shape_path` is read with the legacy reader (the tensor reader does
    not handle OBJ reliably) and converted to a tensor mesh, then transformed by
    the 4x4 `reference_to_scene` matrix.

    :returns: The reference as an Open3D tensor triangle mesh in the scene frame.
    """
    reference_mesh = o3d.io.read_triangle_mesh(str(reference_shape_path))
    reference_tensor = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(reference_mesh.vertices), o3d.core.Dtype.Float32),
        o3d.core.Tensor(
            np.asarray(reference_mesh.triangles).astype(np.int32),
            o3d.core.Dtype.Int32,
        ),
    )
    return reference_tensor.transform(
        o3d.core.Tensor(reference_to_scene, o3d.core.Dtype.Float32)
    )
