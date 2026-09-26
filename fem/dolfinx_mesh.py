"""
Hands a serial triangle mesh to DOLFINx as a 2-manifold embedded in 3-space.

Both solver families build their geometry outside DOLFINx: the heat side
tessellates an IGA spline, and the shell side additionally evaluates curved mid-edge
nodes on it. Both then need the same handover, a Lagrange coordinate element with
a 3-vector value shape (that shape is what makes the mesh a surface in 3D rather
than a planar one), and the rank-0 gating described on `create_manifold_mesh`.
This is the single shared implementation of that handover.
"""

import basix.ufl
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx import mesh
from mpi4py import MPI


def create_manifold_mesh(
    comm: MPI.Comm,
    cells: npt.NDArray[np.integer],
    points: npt.NDArray[np.float64],
    degree: int = 1,
) -> mesh.Mesh:
    """
    Distributes a serially-built triangle mesh over `comm`, gated on rank 0.

    The geometry is built identically on every rank (it is evaluated from a spline,
    not read from a partitioned file), so every rank holds the same complete mesh.
    Handing that whole copy to `dolfinx.mesh.create_mesh` everywhere would make each
    rank contribute the full topology, and an N-rank run would build N duplicates of
    the mesh with no error raised. Only rank 0 therefore supplies the cells and
    points; the rest supply empty arrays and receive their share from the
    partitioner. This mirrors what a `gmshio.model_to_mesh` route does by passing
    the gmsh model on rank 0 alone.

    `cells` is the connectivity, shape `(n_cells, nodes_per_cell)`: 3 nodes per
    cell at `degree` 1 and 6 (vertices then mid-edge nodes) at `degree` 2. Any
    integer dtype is accepted; DOLFINx wants int64, and the conversion happens
    here rather than in each caller so that a rank supplying no cells never pays
    for it. Must be the same on every rank; only rank 0's copy is read. `points`
    is the node coordinates, shape `(n_points, 3)`, same on every rank. `degree`
    is the Lagrange degree of the coordinate element: 1 for flat facets, 2 for a
    curved (isoparametric) surface.

    :returns: DOLFINx mesh (topological dim 2, geometric dim 3).
    """
    coordinate_element = basix.ufl.element("Lagrange", "triangle", degree, shape=(3,))
    if comm.rank == 0:
        cells = np.asarray(cells, dtype=np.int64)
    else:
        cells = np.empty((0, cells.shape[1]), dtype=np.int64)
        points = np.empty((0, 3), dtype=points.dtype)
    return mesh.create_mesh(comm, cells, points, ufl.Mesh(coordinate_element))
