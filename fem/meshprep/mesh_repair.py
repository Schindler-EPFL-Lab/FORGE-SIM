"""
Turning a hand-modelled triangle mesh into a watertight analysis surface.

Mesh repair is part of the FEM workflow: the reconstructed spline is already a
closed B-rep, so only the ground-truth side needs it, which is why these helpers
live here rather than in `forge_sim`. Nothing in this module needs DOLFINx;
Open3D and SciPy are enough.

The module mixes pure-numpy topology functions (:func:`directed_boundary_loops`,
:func:`cap_holes`, :func:`drop_degenerate_triangles`) with Open3D-backed ones
(:func:`clean_mesh`, :func:`largest_component`, `_orient_outward`). Open3D is a
declared dependency of both `fem/environment.yml` and `pyproject.toml`, so
"optional" understates it: the module simply has a half that does not need it.
Open3D therefore stays imported inside the functions that need it rather than at
module scope: importing any one name from a module runs the whole module, and
`fem/end_to_end/mesh_quality/mesh_quality_runner.py` imports :func:`clean_mesh`
while `fem/end_to_end/modal_gt/fem_shell_modal_gt_runner.py` imports
:func:`drop_degenerate_triangles`; a module-scope import would force Open3D onto
both callers regardless of which half either one meant to use.

The primitives operate on plain `(vertices, faces)` NumPy arrays, except
:func:`clean_mesh`, :func:`largest_component` and :func:`cap_holes`, which return a
:class:`fem.fem_mesh_data.FEMMeshData`. The entry point for the end-to-end
experiment is :meth:`fem.meshprep.repaired_mesh.RepairedMesh.from_raw`, which
composes several of these into the repaired, watertight mesh, since the caps it
adds have to travel with the mesh; :func:`clean_mesh` and
:meth:`fem.meshprep.repaired_mesh.RepairedMesh.smoothed` are applied separately
around the remesh that follows it.
"""

from collections import Counter

import numpy as np
import numpy.typing as npt

from fem.fem_mesh_data import FEMMeshData
from fem.meshprep.cleaned_mesh import CleanedMesh


def directed_boundary_loops(faces: npt.NDArray[np.int64]) -> list[list[int]]:
    """
    Ordered vertex-index loops for the boundary of `faces`.

    Each loop follows the surface's own boundary half-edge direction (`a -> b` as it
    appears in the single owning face), so a downstream stitch/cap can reverse it to
    stay consistent with the surface orientation.

    :returns: One list of vertex indices per boundary loop, in that walk order.
    """
    faces = np.asarray(faces, dtype=np.int64)
    edge_count: Counter = Counter()
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            edge_count[(min(int(a), int(b)), max(int(a), int(b)))] += 1
    boundary = {e for e, c in edge_count.items() if c == 1}
    succ: dict[int, int] = {}
    for f in faces:
        for a, b in (
            (int(f[0]), int(f[1])),
            (int(f[1]), int(f[2])),
            (int(f[2]), int(f[0])),
        ):
            if (min(a, b), max(a, b)) in boundary:
                succ[a] = b
    loops: list[list[int]] = []
    visited: set[int] = set()
    for start in list(succ):
        if start in visited:
            continue
        loop: list[int] = []
        u = start
        while u in succ and u not in visited:
            visited.add(u)
            loop.append(u)
            u = succ[u]
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def cap_holes(
    vertices: npt.NDArray[np.float64], faces: npt.NDArray[np.int64]
) -> FEMMeshData:
    """
    Cap every boundary loop with a centroid-fan -> watertight. Apply this to the
    welded+cleaned mesh before the per-mesh remesh, so the (initially poor) fan
    triangles get remeshed into FEM-quality triangles along with the rest of the
    surface. A genuinely closed input is returned unchanged.

    The loops come from :func:`directed_boundary_loops`, so the two functions agree
    on where the boundary is and how it is walked, including that walk's own
    pinch-vertex defect, which `test_mesh_fields` records as an expected failure.
    Each loop gets one new vertex at its centroid and a triangle fan wound to stay
    manifold (the cap edge runs opposite to the owning face's half-edge). The raw
    fan is intentionally crude (centroid spoke -> high aspect ratio); that is fine
    because Botsch remeshes it afterwards. Robust and self-contained: open3d's
    `fill_holes` proved unreliable on these meshes (spurious far vertices / partial
    fills).

    :returns: The capped mesh, with one new centroid vertex and its fan appended
        per loop.
    """
    faces = np.asarray(faces, dtype=np.int64)
    loops = directed_boundary_loops(faces)
    if not loops:
        return FEMMeshData(
            vertices=np.asarray(vertices, dtype=np.float64),
            faces=faces.astype(np.uint32),
        )

    verts = list(np.asarray(vertices, dtype=np.float64))
    new_faces = faces.tolist()
    for loop in loops:
        centre_idx = len(verts)
        verts.append(np.mean([verts[i] for i in loop], axis=0))
        for a, b in zip(loop, loop[1:] + loop[:1]):
            new_faces.append([b, a, centre_idx])  # opposite winding -> manifold
    return FEMMeshData(
        vertices=np.asarray(verts, dtype=np.float64),
        faces=np.asarray(new_faces, dtype=np.uint32),
    )


def clean_mesh(
    vertices: npt.NDArray[np.float64], faces: npt.NDArray[np.int64], merge_eps: float
) -> FEMMeshData:
    """
    Collapse near-coincident vertices (short edges -> slivers) and drop
    degenerate/duplicate triangles.

    `merge_eps` is the coincidence radius, and 0 disables welding entirely:
    the right choice before a remesh, where only the degenerate/duplicate pass is
    wanted and a weld could collapse a legitimately short edge. Callers pass it
    explicitly because its safe magnitude is set by the target edge length of the
    remesh it brackets, not by anything this function can see.

    :returns: The cleaned mesh.
    """
    import open3d as o3d  # deferred import; see the module docstring

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.ascontiguousarray(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.ascontiguousarray(faces, dtype=np.int32)),
    )
    if merge_eps > 0.0:
        mesh = mesh.merge_close_vertices(merge_eps)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    return FEMMeshData(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.triangles, dtype=np.uint32),
    )


def drop_degenerate_triangles(
    vertices: npt.NDArray[np.float64],
    faces: npt.NDArray[np.int64],
    min_area_ratio: float,
) -> CleanedMesh:
    """
    Drops near-zero-area triangles and reindexes, removing any orphaned vertices.

    Manifold-preserving: only faces are removed; vertex positions are untouched.
    A triangle is dropped when its area falls below `min_area_ratio` times the
    mean triangle area, a soft, scale-free threshold, unlike `clean_mesh`'s
    exact-zero check via Open3D. NumPy only, so it needs no optional dependency.

    :returns: The kept vertices, the reindexed faces, and how many triangles
        were dropped.
    """
    p0, p1, p2 = (vertices[faces[:, i]] for i in range(3))
    areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    mean_area = float(areas.mean()) if areas.size else 0.0
    keep = areas > (min_area_ratio * max(mean_area, 1e-30))
    n_dropped = int((~keep).sum())
    if n_dropped == 0:
        return CleanedMesh(vertices=vertices, faces=faces, num_dropped=0)
    faces = faces[keep]
    used = np.unique(faces)
    remap = -np.ones(vertices.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    return CleanedMesh(
        vertices=vertices[used],
        faces=remap[faces].astype(np.int64),
        num_dropped=n_dropped,
    )


def largest_component(
    vertices: npt.NDArray[np.float64], faces: npt.NDArray[np.int64]
) -> FEMMeshData:
    """
    Keep only the largest connected component of a triangle mesh.

    Suzanne (and many assets) ships as several disconnected surfaces: the head
    plus separate eyeballs. Each free-floating body contributes its own 6
    rigid-body modes, so a modal analysis on the raw mesh has 18 (not 6) zero
    modes. The reconstructed spline is a single closed surface, so the GT analysis
    mesh must be reduced to its dominant connected component for a fair comparison.

    :returns: The largest connected component only.
    """
    import open3d as o3d  # deferred import; see the module docstring

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.ascontiguousarray(vertices, np.float64)),
        o3d.utility.Vector3iVector(np.ascontiguousarray(faces, np.int32)),
    )
    labels = np.asarray(mesh.cluster_connected_triangles()[0])
    keep = labels == np.bincount(labels).argmax()
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    return FEMMeshData(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.triangles, dtype=np.uint32),
    )


def _orient_outward(
    vertices: npt.NDArray[np.float64], faces: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """
    Make the winding globally consistent (open3d `orient_triangles`) and flip the
    whole mesh if its signed volume is negative, so normals point outward.

    :returns: `faces`, re-wound so every triangle normal points outward.
    """
    import open3d as o3d  # deferred import; see the module docstring

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.ascontiguousarray(vertices, np.float64)),
        o3d.utility.Vector3iVector(np.ascontiguousarray(faces, np.int32)),
    )
    mesh.orient_triangles()
    oriented = np.asarray(mesh.triangles, dtype=np.int64)
    tv = vertices[oriented]
    signed_vol = float(np.sum(np.cross(tv[:, 0], tv[:, 1]) * tv[:, 2]) / 6.0)
    if signed_vol < 0.0:
        oriented = oriented[:, ::-1]
    return np.ascontiguousarray(oriented, dtype=np.int64)
