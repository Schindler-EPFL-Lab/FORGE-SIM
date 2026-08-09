import json
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np
from scipy.spatial.transform import Rotation


def unit_sphere_normalization_transform(
    vertex_positions: "mi.TensorXf",
) -> "mi.Transform4f":
    """
    Returns the transform that recenters `vertex_positions` at the origin and
    rescales them so the farthest vertex from the recentered origin lies on
    the unit sphere.

    The bbox transforms produced by `pose_params_to_bbox_transforms` (and the
    hand-authored JSONs in `data/bbox_transforms/`) assume an input that sits
    on a unit sphere centered at the origin. Composing this normalization in
    front of `tf_bbox` keeps that contract valid for an input mesh of
    arbitrary native size AND position.

    Centering uses the axis-aligned bounding-box center (not the vertex
    centroid) so the result is invariant to tessellation density.

    `vertex_positions` is a `(V, 3)` tensor as exposed by `Mesh.vertices`.

    :raises ValueError: If the bounding box has zero extent (all vertices
        coincide).
    """
    vertex_points = dr.unravel(mi.Point3f, vertex_positions.array)

    center_x = 0.5 * (
        float(dr.min(vertex_points.x)[0]) + float(dr.max(vertex_points.x)[0])
    )
    center_y = 0.5 * (
        float(dr.min(vertex_points.y)[0]) + float(dr.max(vertex_points.y)[0])
    )
    center_z = 0.5 * (
        float(dr.min(vertex_points.z)[0]) + float(dr.max(vertex_points.z)[0])
    )
    center = mi.Point3f(center_x, center_y, center_z)

    max_radius = float(dr.max(dr.norm(vertex_points - center))[0])
    if max_radius <= 0.0:
        raise ValueError(
            "Cannot normalize a mesh whose vertices all coincide with a single point."
        )
    inv_radius = 1.0 / max_radius

    # Apply translate-to-origin first, then uniform scale.
    return mi.Transform4f().scale(
        mi.Point3f(inv_radius, inv_radius, inv_radius),
    ) @ mi.Transform4f().translate(
        mi.Point3f(-center_x, -center_y, -center_z),
    )


def load_bounding_bbox_transforms(bbox_transforms_path: Path) -> "mi.Transform4f":
    """
    Creates a 4x4 transformation matrix based on scale, translation and rotation
    values from JSON file `bbox_transforms_path`.

    :returns
        a 4x4 tranformation matrix from scale, translation and rotation.

    :raises
        RuntimeError: If any of the required keys ("scale", "translation", "rotation")
        are missing in the file.
    """
    required_keys = {"scale", "translation", "rotation"}

    with open(bbox_transforms_path, "r") as f:
        bbox_transforms = json.load(f)

    if not required_keys.issubset(bbox_transforms):
        raise RuntimeError(
            f"Missing required keys in bbox_transforms: "
            f"{required_keys - set(bbox_transforms)}"
        )

    tf = np.eye(4)
    tf[:3, :3] = np.diag(bbox_transforms["scale"])
    tf[:3, :3] = (
        Rotation.from_euler(
            "xyz", bbox_transforms["rotation"], degrees=True
        ).as_matrix()
        @ tf[:3, :3]
    )
    tf[:3, 3] = bbox_transforms["translation"]
    return mi.Transform4f(tf)


def save_bounding_bbox_transforms(
    bbox_transforms_path: Path,
    translation: tuple[float, float, float],
    rotation: tuple[float, float, float],
    scale: tuple[float, float, float],
) -> None:
    """
    Writes a bounding-box transform to JSON at `bbox_transforms_path`, using the
    same schema consumed by `load_bounding_bbox_transforms`.

    `translation` is an (x, y, z) offset, `rotation` is Euler XYZ degrees
    (extrinsic), and `scale` is per-axis. The parent directory is created if
    needed.
    """
    bbox_transforms_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bbox_transforms_path, "w") as f:
        json.dump(
            {
                "translation": list(translation),
                "rotation": list(rotation),
                "scale": list(scale),
            },
            f,
            indent=4,
        )


def pose_params_to_bbox_transforms(
    flat_params: "mi.Float",
    tf_cameras: "mi.ScalarTransform4f",
    initial_sphere_radius: float,
    scale_epsilon: float = 1e-3,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """
    Converts the 9-float state produced by `PoseOptimizer` into
    (translation, rotation, scale) vectors expressed in the raw input
    (pre-alignment) camera frame, matching the schema of
    `load_bounding_bbox_transforms`.

    `flat_params` is the flattened (tx, ty, tz, sx, sy, sz, rx, ry, rz) state read
    back from the scene optimizer. Scale entries are clipped to
    [`scale_epsilon`, 1/`scale_epsilon`] and rotation entries (in radians) to
    [-π, π], mirroring `PoseOptimizer._shape_update_callback`.

    `tf_cameras` is the camera-alignment transform applied during loading
    (`ImageSceneLoader.tf_cameras`); `initial_sphere_radius` is the radius that was
    baked into the initial spline surface. The returned transform satisfies

        tf_cameras @ bbox_transforms @ unit_sphere_vertex
        == tf_new_pose @ (initial_sphere_radius * unit_sphere_vertex)

    so the produced values are drop-in replacements for hand-authored files in
    `data/bbox_transforms/`.

    :returns: `(translation, rotation_euler_xyz_deg, scale)`, each a 3-tuple.
    """
    params = np.asarray(flat_params).reshape(9)
    tx, ty, tz = params[0:3]
    sx, sy, sz = np.clip(params[3:6], scale_epsilon, 1.0 / scale_epsilon)
    rx_deg, ry_deg, rz_deg = np.clip(params[6:9], -np.pi, np.pi) * (180.0 / np.pi)

    tf_new_pose = (
        mi.Transform4f()
        .translate(mi.Point3f(float(tx), float(ty), float(tz)))
        .rotate(axis=mi.Point3f(1, 0, 0), angle=float(rx_deg))
        .rotate(axis=mi.Point3f(0, 1, 0), angle=float(ry_deg))
        .rotate(axis=mi.Point3f(0, 0, 1), angle=float(rz_deg))
        .scale(mi.Point3f(float(sx), float(sy), float(sz)))
    )
    tf_radius = mi.Transform4f().scale(
        mi.Point3f(initial_sphere_radius, initial_sphere_radius, initial_sphere_radius)
    )
    tf_bbox = mi.Transform4f(tf_cameras).inverse() @ tf_new_pose @ tf_radius

    # mi.Transform4f().matrix gives a 4x4xN drjit matrix; squeeze the trailing axis
    tf_bbox_np = np.asarray(tf_bbox.matrix).reshape(4, 4, -1)[:, :, 0]

    rs = tf_bbox_np[:3, :3]
    scale_out = np.linalg.norm(rs, axis=0)
    rotation_matrix = rs / scale_out[np.newaxis, :]

    # Fold any reflection introduced by tf_cameras into a negative scale component
    # so `rotation_matrix` stays a proper rotation for Euler extraction.
    if np.linalg.det(rotation_matrix) < 0.0:
        rotation_matrix[:, 2] *= -1.0
        scale_out[2] *= -1.0

    rotation_euler_xyz = Rotation.from_matrix(rotation_matrix).as_euler(
        "xyz", degrees=True
    )

    return (
        tuple(tf_bbox_np[:3, 3].tolist()),
        tuple(rotation_euler_xyz.tolist()),
        tuple(scale_out.tolist()),
    )
