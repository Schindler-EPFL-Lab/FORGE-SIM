"""
Build a curved (isoparametric) DOLFINx surface mesh from a rebel-shape spline.

The geometrically-exact shell solver needs a mesh whose geometry lies exactly on
the spline mid-surface (flat facets only degrade accuracy). This module turns a
serialized `TessellatedSplineSurface` JSON into an order-1 (flat) or order-2
(curved) conforming DOLFINx triangle mesh embedded in 3D, applying the same
geometry rescaling as the IGA modal scripts so the two solvers operate on
identically scaled geometry.

Strategy (order 2): reuse `composite_mesh_tesselation` for the conforming P1
vertex topology (it already merges shared vertices across patch boundaries), then
add one mid-edge node per global mesh edge, evaluated directly on the spline at the
edge's midpoint parametric coordinate. Keying mid-edge nodes by the global vertex pair
keeps patch boundaries conforming. All P2 geometry nodes therefore sit on the
spline surface; only the intra-element quadratic interpolation approximates it
(an O(h^3) geometry error vs O(h^2) for flat facets).
"""

from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np
import numpy.typing as npt
from dolfinx import mesh
from mpi4py import MPI

from fem.dolfinx_mesh import create_manifold_mesh
from fem.meshprep.p2_geometry import basix_edge_table
from rebel_shape import logger
from rebel_shape.meshing.tessellated_spline_surface import TessellatedSplineSurface


def load_and_rescale_spline(
    spline_path: Path,
    target_length_scale: float | None = None,
) -> TessellatedSplineSurface:
    """
    Loads a spline surface and applies the IGA scripts' geometry rescaling.

    Mirrors `reissner_mindlin_modal_runner.prepare_modal_geometry`: if
    `target_length_scale` is given, scale uniformly so the longest
    control-point bounding-box extent equals it. Modal frequencies scale as
    1/length, so this must match the IGA run exactly.

    `spline_path` is the path to the serialized `TessellatedSplineSurface`
    JSON. `target_length_scale` is the target longest-bbox extent, or
    `None` to skip rescaling.

    :returns: The loaded (and rescaled) tessellated spline surface.
    :raises ValueError: If the longest control-point bbox extent is near zero.
    """
    tessellated = TessellatedSplineSurface.from_file(
        composite_spline_surface_path=spline_path
    )
    composite = tessellated.composite_spline

    if target_length_scale is not None:
        cps = composite.global_control_points
        extent = dr.max(cps, axis=0) - dr.min(cps, axis=0)
        longest = float(dr.max(extent).numpy())
        if longest <= 1.0e-12:
            raise ValueError("Longest bbox extent is near zero; cannot rescale.")
        factor = target_length_scale / longest
        composite.global_control_points = cps * factor
        logger.info(
            f"Applied target-length scaling factor {factor:.6g} so the longest "
            f"control-point bbox extent equals {target_length_scale} "
            f"(was {longest:.6g})."
        )

    return tessellated


def _as_texcoords(values: npt.NDArray[np.float64]) -> "mi.TensorXf":
    """
    Builds a Mitsuba `TensorXf` texcoord array (shape (n, 2)) from a numpy array.

    Matches the type returned by `Mesh.texcoords` that the spline patch
    consumes. `values` is the texcoord array, shape `(n, 2)`.

    :returns: The values as a Mitsuba `TensorXf`.
    """
    return mi.TensorXf(np.ascontiguousarray(values, dtype=np.float32))


def build_shell_mesh(
    tessellated_spline: TessellatedSplineSurface,
    subdivisions_per_patch: int,
    order: int = 2,
    comm: MPI.Comm = MPI.COMM_SELF,
) -> mesh.Mesh:
    """
    Builds a conforming DOLFINx surface mesh on the spline mid-surface, from
    the (already rescaled) `tessellated_spline`.

    `subdivisions_per_patch` is the quad subdivisions per patch and direction;
    each patch yields `2 * subdivisions_per_patch**2` triangles. `order` is
    the geometry order: 1 for flat facets, 2 for curved with nodes on the
    spline. `comm` is the MPI communicator; the shell pipeline runs serially
    (the SLEPc solve refuses more than one rank), so the single-rank path is
    the one exercised, and the rank-0 gating in `create_manifold_mesh` is
    what keeps a multi-rank run from silently building one mesh per rank.

    :returns: A DOLFINx triangle mesh (topological dim 2, geometric dim 3).
    :raises ValueError: If `order` is not 1 or 2.
    :raises RuntimeError: If a mesh edge did not receive a curved node.
    """
    if order not in (1, 2):
        raise ValueError("`order` must be 1 (flat) or 2 (curved).")

    composite = tessellated_spline.composite_spline
    tessellation = composite.composite_mesh_tesselation(
        subdivisions_per_patch=subdivisions_per_patch
    )
    p1_vertices = tessellation.global_vertices.numpy().astype(np.float64)
    p1_faces = tessellation.global_faces.numpy().astype(np.int64)
    n_vertices = p1_vertices.shape[0]

    if order == 1:
        return create_manifold_mesh(comm, p1_faces, p1_vertices, degree=1)

    # Order 2: one curved mid-edge node per unique global mesh edge.
    unique_edges, edge_index = basix_edge_table(p1_faces)
    n_edges = unique_edges.shape[0]
    edge_lookup = {(int(a), int(b)): i for i, (a, b) in enumerate(unique_edges)}

    edge_positions = np.zeros((n_edges, 3), dtype=np.float64)
    edge_filled = np.zeros(n_edges, dtype=bool)

    for patch_idx, (patch_mesh, patch_mapping) in enumerate(
        tessellation.patches_with_mapping
    ):
        spline_patch = composite[patch_idx]
        texcoords = patch_mesh.texcoords.numpy().astype(np.float64)
        local_faces = patch_mesh.faces.numpy().astype(np.int64)
        local_to_global = np.asarray(patch_mapping)

        # unique local edges still needing a node, with their midpoint texcoords
        local_pairs = np.concatenate(
            [local_faces[:, [1, 2]], local_faces[:, [0, 2]], local_faces[:, [0, 1]]],
            axis=0,
        )
        local_pairs.sort(axis=1)
        local_unique = np.unique(local_pairs, axis=0)

        mid_texcoords: list[npt.NDArray[np.float64]] = []
        target_edges: list[int] = []
        for la, lb in local_unique:
            ga, gb = int(local_to_global[la]), int(local_to_global[lb])
            global_edge = edge_lookup[(min(ga, gb), max(ga, gb))]
            if edge_filled[global_edge]:
                continue
            mid_texcoords.append(0.5 * (texcoords[la] + texcoords[lb]))
            target_edges.append(global_edge)

        if not target_edges:
            continue

        mids = np.asarray(mid_texcoords)
        parametric = spline_patch.texcoords_to_parametric_coords(
            _as_texcoords(mids)
        )
        positions = np.asarray(spline_patch.evaluate_positions(parametric)).astype(
            np.float64
        )
        if positions.shape[0] != mids.shape[0]:  # (3, n) -> (n, 3) guard
            positions = positions.T
        for j, global_edge in enumerate(target_edges):
            edge_positions[global_edge] = positions[j]
            edge_filled[global_edge] = True

    if not edge_filled.all():
        raise RuntimeError(
            f"{(~edge_filled).sum()} mesh edges did not receive a curved node; "
            "patch/topology mapping is inconsistent."
        )

    points = np.vstack([p1_vertices, edge_positions])
    p2_cells = np.column_stack([p1_faces, n_vertices + edge_index])

    return create_manifold_mesh(comm, p2_cells, points, degree=2)
