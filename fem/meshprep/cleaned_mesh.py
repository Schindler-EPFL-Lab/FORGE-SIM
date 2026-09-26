from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, kw_only=True)
class CleanedMesh:
    """
    What :func:`fem.meshprep.mesh_repair.drop_degenerate_triangles` returns: a
    mesh with its near-zero-area triangles dropped, and how many were.
    """

    vertices: npt.NDArray[np.float64]
    """Vertex positions, shape `(n, 3)`; unchanged from the input when
    `num_dropped` is 0."""
    faces: npt.NDArray[np.int64]
    """Triangle vertex indices, shape `(m, 3)`, renumbered if any were dropped."""
    num_dropped: int
    """Degenerate triangles removed."""
