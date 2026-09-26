"""
Order-2 (curved) triangle geometry: the basix edge layout and the mid-edge nodes.

A P2 triangle carries one node per edge in addition to its three vertices, and
every producer and consumer of such a mesh has to agree on which edge is which.
:func:`basix_edge_table` is the single definition of that agreement; see its
docstring for why it is worth centralising.

Pure NumPy, and deliberately no `basix` import: `basix` exists only in the conda
environment, while `fem/tests/test_mesh_fields.py` characterises this module from
both interpreters (it documents a run under each, and the virtualenv run is the one
that exercises every case). The convention is therefore written out here, with the
check against the real `basix` recorded in the docstring below.
"""

import numpy as np
import numpy.typing as npt

from fem.meshprep.reference_surface import project_points_to_surface


def basix_edge_table(
    faces: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Unique undirected mesh edges, plus each face's three edges in basix order.

    `unique_edges` is `(n_edges, 2)`, each row a vertex pair sorted ascending;
    `edge_index` is `(n_faces, 3)`, column `j` giving the `unique_edges` row
    of local edge `j`. A P2 cell is then
    `[v0, v1, v2, *(n_vertices + edge_index[face])]`, i.e. the layout
    `[v0, v1, v2, e12, e02, e01]`.

    That ordering is basix's, not a choice made here: `basix.topology` reports
    `[[1, 2], [0, 2], [0, 1]]` for the triangle's edges, and the equispaced P2
    Lagrange element's points are `[(0,0), (1,0), (0,1), (.5,.5), (0,.5), (.5,0)]`,
    the last three being the midpoints of `(v1,v2)`, `(v0,v2)`, `(v0,v1)`
    in that order (verified against basix 0.9.0).

    The convention is centralised because it is not checkable at run time. It is
    shared by the spline-side mesh builder and the ground-truth baking stage, and the
    latter writes `edge_index` into `modal_gt.npz` for a DOLFINx driver running in
    a wholly separate process to read back, so the column order is the whole of what
    carries the meaning across, with no shared object to disagree with. A producer
    and a consumer that disagreed about it would still assemble, still solve, and
    return silently wrong frequencies: the cells stay valid, only the geometry is
    scrambled.

    :returns: `(unique_edges, edge_index)`, as described above.
    """
    faces = np.asarray(faces, dtype=np.int64)
    edge_pairs = np.concatenate(
        [faces[:, [1, 2]], faces[:, [0, 2]], faces[:, [0, 1]]], axis=0
    )
    edge_pairs.sort(axis=1)
    unique_edges, edge_index = np.unique(edge_pairs, axis=0, return_inverse=True)
    return unique_edges, edge_index.reshape(3, -1).T.astype(np.int64)


def curved_edge_nodes(
    vertices: npt.NDArray[np.float64],
    faces: npt.NDArray[np.int64],
    source_vertices: npt.NDArray[np.float64],
    source_faces: npt.NDArray[np.int64],
    max_offset_ratio: float = 0.25,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], int]:
    """
    Mid-edge nodes for an order-2 (curved) mesh, projected onto a reference surface.

    A flat analysis mesh chords across the geometry, losing surface area, and shell
    mass is `rho * h * area`, so that loss is a physical mass deficit, not a cosmetic
    one. Curving recovers it: each mid-edge node moves from the straight midpoint onto
    the reference surface, so the element follows the geometry quadratically. It costs
    no field DOFs (geometry order is independent of field degree).

    The reference surface is the original, un-remeshed mesh: the analysis mesh is a
    coarsened stand-in for it, not the geometry itself. Note this recovers chord error
    only: a curved triangle spanning a feature the remesh skipped entirely (a
    collapsed fence) still does not represent it.

    Nodes whose projection would move further than `max_offset_ratio` times the edge
    length are left straight: an over-displaced mid-edge node inverts the element
    (negative Jacobian), which is worse than the chord error it fixes.

    The quarter is the classical limit for a quadratic edge: its Jacobian stays
    positive throughout exactly while the mid-side node lies in the middle half of
    the edge, and fracture mechanics places the node at the quarter point
    deliberately to produce the degeneracy. That limit is for displacement along the
    edge, which is where a node inverts an element most readily; a projection onto a
    reference surface is mostly transverse to it, for which the true limit is more
    permissive. So a quarter is the conservative bound applied in any direction
    rather than a tuned one.

    :returns: `(edge_positions, edge_index, n_left_straight)`: the mid-edge
        positions (straight where left so), `edge_index` as
        :func:`basix_edge_table` defines it, and how many nodes were left straight.
    """
    unique_edges, edge_index = basix_edge_table(faces)

    start, end = vertices[unique_edges[:, 0]], vertices[unique_edges[:, 1]]
    straight = 0.5 * (start + end)
    projected = project_points_to_surface(straight, source_vertices, source_faces)

    offset = np.linalg.norm(projected - straight, axis=1)
    too_far = offset > max_offset_ratio * np.linalg.norm(end - start, axis=1)
    projected[too_far] = straight[too_far]

    return projected, edge_index, int(too_far.sum())
