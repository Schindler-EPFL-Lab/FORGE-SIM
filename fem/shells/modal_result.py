from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, kw_only=True)
class ModalResult:
    """Eigenpairs of one shell modal solve, as returned by both solve paths."""

    eigenvalues: npt.NDArray[np.float64]
    """Squared angular frequencies omega^2, ascending, shape (num_modes,)."""

    frequencies: npt.NDArray[np.float64]
    """Natural frequencies in Hz, sqrt(|eigenvalue|) / (2 pi)."""

    eigenvectors: npt.NDArray[np.float64]
    """
    Mode shapes in the mixed (displacement, rotation) space, shape
    (ndofs, num_modes), column i belonging to `eigenvalues[i]`.
    """

    mode_energies: npt.NDArray[np.float64]
    """
    Per-mode strain-energy split (membrane, bending, shear, drilling), shape
    (num_modes, 4). Left all-zero by `solve_modes_slepc`, which does not
    assemble the split blocks.
    """

    ndofs: int
    """Global number of degrees of freedom of the mixed space."""
