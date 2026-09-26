from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, kw_only=True)
class FEMMeshData:
    """
    Triangle-mesh geometry extracted from a tessellated spline surface, in numpy
    format suitable for DOLFINx mesh creation.
    """

    vertices: npt.NDArray[np.float64]
    """Mesh vertex positions, shape (n_vertices, 3)."""
    faces: npt.NDArray[np.uint32]
    """Triangle connectivity, shape (n_faces, 3)."""
