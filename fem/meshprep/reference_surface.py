"""
Queries against a reference surface: closest points and field transfer.

The analysis mesh handed to the FEM solvers is a remeshed, coarsened stand-in for
the ground-truth surface, so its vertices lie on that surface without being shared
with it. Both helpers here answer a question of the form "what does the reference
surface say at this point?", where the surface is (:func:`project_points_to_surface`,
used to curve a P2 mesh) and what field it carries there
(:func:`transfer_vertex_field_to_points`).

Both route through Open3D's `RaycastingScene`, which casts query points to
float32 before the closest-point search; results are therefore accurate to float32,
not float64, however the inputs are typed.

Open3D is imported inside the functions, not at module scope, and must stay that
way, though not for the reason the shape suggests. Every function here needs
Open3D, so importing this module without it buys nothing directly. What the
deferral protects is the chain above it: `p2_geometry` imports
:func:`project_points_to_surface` at module scope and `fem.shells.spline_to_shell_mesh`
imports `p2_geometry` at module scope, and both carry work that needs no Open3D at
all. Moving this import out turns the fifteen currently passing tests of
`test_spline_to_shell_mesh` into collection-time import errors wherever Open3D is
absent, which is the state of the minimal conda environment.
"""

import numpy as np
import numpy.typing as npt


def _raycasting_scene(
    source_vertices: npt.NDArray[np.float64], source_faces: npt.NDArray[np.int64]
):
    """
    Open3D raycasting scene (`o3d.t.geometry.RaycastingScene`) holding a single
    triangle mesh, ready for closest-point queries.

    Left return-unannotated rather than naming the Open3D type: Open3D is
    imported only inside the functions that need it (see the module docstring),
    so the name is not in scope to annotate with, and a type-checking-only
    import is not this repo's convention.
    """
    import open3d as o3d  # heavy optional dep; see the module docstring

    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.ascontiguousarray(source_vertices, np.float64)),
        o3d.utility.Vector3iVector(np.ascontiguousarray(source_faces, np.int32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    return scene


def project_points_to_surface(
    query_points: npt.NDArray[np.float64],
    source_vertices: npt.NDArray[np.float64],
    source_faces: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """
    Closest point on a triangle mesh's surface for each query point.

    :returns: One 3D point per row of `query_points`, on the `(source_vertices,
        source_faces)` surface.
    """
    import open3d as o3d  # heavy optional dep; see the module docstring

    found = _raycasting_scene(source_vertices, source_faces).compute_closest_points(
        o3d.core.Tensor(np.ascontiguousarray(query_points, np.float32))
    )
    return found["points"].numpy().astype(np.float64)


def transfer_vertex_field_to_points(
    query_points: npt.NDArray[np.float64],
    source_vertices: npt.NDArray[np.float64],
    source_faces: npt.NDArray[np.int64],
    source_field: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Barycentric-interpolate a per-vertex scalar field from a source mesh onto query
    points lying on its surface. Each query point is snapped to its closest point on
    the source surface (the same query :func:`project_points_to_surface` runs), and
    the field is read off there with that point's barycentric weights.

    This is the honest GT field for the FEM side: the exact field the original
    (non-CAD) mesh carries, linearly interpolated onto the analysis mesh, the same
    linear interpolation the renderer applies to the original per-vertex attribute when
    producing the measured images. It deliberately does not re-sample the texture at the
    fine remesh vertices (which would hand FEM an artificially sharp field the
    measurement never had); the remesh is the FEM pipeline's tax, not a field upgrade.

    :returns: One interpolated field value per row of `query_points`.
    """
    import open3d as o3d  # heavy optional dep; see the module docstring

    ans = _raycasting_scene(source_vertices, source_faces).compute_closest_points(
        o3d.core.Tensor(query_points.astype(np.float32))
    )
    prim_ids = ans["primitive_ids"].numpy().astype(np.int64)
    bary = ans["primitive_uvs"].numpy().astype(np.float64)
    tri = source_faces[prim_ids]
    w1 = bary[:, 0]
    w2 = bary[:, 1]
    w0 = 1.0 - w1 - w2
    f = np.asarray(source_field, dtype=np.float64)
    interp = w0 * f[tri[:, 0]] + w1 * f[tri[:, 1]] + w2 * f[tri[:, 2]]
    return interp.astype(np.float64)
