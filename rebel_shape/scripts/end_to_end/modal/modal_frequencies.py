from pathlib import Path

import numpy as np
import numpy.typing as npt

RIGID_BODY_FREQ_HZ: float = 1.0
"""
Frequency below which a mode is one of the six rigid-body modes.

A free-floating body's six rigid-body modes are exact zeros in theory; numerically
they land at most at 3.6e-3 Hz across the recorded runs, while the lowest elastic
mode of these thin steel shells is 15.1 Hz. 1 Hz sits inside that gap with roughly
two decades of margin on the rigid-body side and one on the elastic side, on either
solver. (Measured over the four recorded targets, not derived.)
"""


def modal_frequencies(npz_path: Path) -> npt.NDArray[np.float64]:
    """
    Natural frequencies [Hz] of a modal run, in the order the solver stored them.

    Reads either of the two schemas the solvers write: a ready `frequencies` array,
    or `eigenvalues` (omega^2), converted by `f = sqrt(|lambda|) / (2 pi)`. The
    absolute value covers the rigid-body eigenvalues, which land a hair below zero
    rather than exactly on it.

    Deliberately not sorted: `align_modal_xdmf` pairs these frequencies with the
    mode order stored alongside them in the eigenmode HDF5, so a sort there would
    silently mislabel the modes. Callers comparing two spectra sort explicitly.
    """
    data = np.load(npz_path)
    if "frequencies" in data.files:
        return np.asarray(data["frequencies"], dtype=float)
    eigenvalues = np.asarray(data["eigenvalues"], dtype=float)
    return np.sqrt(np.abs(eigenvalues)) / (2.0 * np.pi)
