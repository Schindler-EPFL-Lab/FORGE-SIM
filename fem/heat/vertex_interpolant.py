from functools import partial

import numpy as np
import numpy.typing as npt
from dolfinx import fem
from scipy.spatial import cKDTree


class VertexInterpolant:
    """
    Maps vertex-based field values onto a DOLFINx function by nearest-vertex
    matching.

    A single KD-tree of the original tessellation vertices is built once, at
    construction, and reused by every call to `__call__`. For each function DOF
    coordinate, the value of the nearest original vertex is assigned. For P1
    this is exact (DOFs are vertices); for higher degrees the extra DOFs take
    their nearest vertex value.
    """

    def __init__(self, original_vertices: npt.NDArray[np.float64]) -> None:
        """
        Builds the KD-tree every `__call__` matches against.

        `original_vertices` is the original tessellation vertices, shape
        (n_vertices, 3).
        """
        self._tree = cKDTree(original_vertices)

    def __call__(
        self, V: fem.FunctionSpace, values: npt.NDArray[np.float64]
    ) -> fem.Function:
        """
        Interpolates `values`, one per original vertex, onto `V` by nearest match.

        :returns: The interpolated function.
        """
        function = fem.Function(V)
        function.interpolate(partial(self._nearest_values, values))
        return function

    def _nearest_values(
        self,
        values: npt.NDArray[np.float64],
        coords: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Looks up the nearest original vertex's value for each of `coords`.

        `values` is bound at the call site by `functools.partial`, since DOLFINx
        only ever calls the interpolation callback with `coords`, shape
        (gdim, n).

        :returns: One value per column of `coords`.
        """
        _, indices = self._tree.query(coords.T, k=1)
        return values[indices]
