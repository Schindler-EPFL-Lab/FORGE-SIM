from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, kw_only=True)
class ModePayload:
    """
    One eigenmode reduced to the arrays the `eigenmodes` format writes for it.

    Built by `eigenmode_xdmf.export_mode_shapes_xdmf` and consumed by the HDF5 and
    XDMF writers beside it, which is the whole of its life: it exists so the two
    writers agree on what a mode carries, rather than one of them reading a key the
    other forgot to set. Every array is per geometry vertex and float64, matching
    the on-disk contract stated in `eigenmode_xdmf`'s module docstring.
    """

    mode_index: int
    """
    Index of this mode in the solve, i.e. its frequency rank, distinct from
    the export position `k` the HDF5 groups are named by; a caller may
    export any subset. Appears in the XDMF grid name and comment.
    """
    eigenvalue: float
    """Squared angular frequency omega^2 of this mode."""
    frequency: float
    """Natural frequency of this mode in Hz."""
    displacement: npt.NDArray[np.float64]
    """Translational mode shape at the vertices, shape `(n_vertices, 3)`."""
    rotation: npt.NDArray[np.float64]
    """Rotational mode shape at the vertices, shape `(n_vertices, 3)`."""
