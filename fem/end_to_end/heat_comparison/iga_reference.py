from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, kw_only=True)
class IGAReference:
    """The IGA reference run a scene's FEM solve is compared against."""

    vertices: npt.NDArray[np.float64]
    """`(n_vertices, 3)` node coordinates of the IGA export mesh."""
    faces: npt.NDArray[np.int32]
    """`(n_faces, 3)` triangle connectivity."""
    times: tuple[float, ...]
    """Snapshot times [s], in file order."""
    snapshots: tuple[npt.NDArray[np.float64], ...]
    """One `(n_vertices,)` temperature field per entry of `times`."""
