"""
Locating the rigid-body band of a computed modal spectrum.

An unsupported (closed or free-free) surface has exactly six zero-energy rigid
body motions, and the mass-shifted shift-invert both solve paths use returns
them as round-off about zero rather than as exact zeros. Finding where they end
is therefore a question about the spectrum's shape, not about any absolute
threshold, which is what keeps the answer independent of the geometry's
scale and of the material.

Both shell drivers report the band after solving, and the ground-truth-mesh one
also records the count it found in its run summary, so the detection lives here
rather than in either of them.

`_NEAR_ZERO_FLOOR` only has to sit between the rigid-body residuals (round-off about
zero) and the elastic spectrum (whose smallest eigenvalue is within a decade or two
of the largest scanned one), so anything across a very wide range works and 1e-6
sits near the middle of it in log terms. Measured on the steel sphere at h = 0.5,
the largest rigid-body residual is 3.7e-15 of the largest scanned eigenvalue on
curved geometry and 9.3e-15 on flat facets (nine orders below this floor), while
the first elastic eigenvalue sits at 0.7 and 0.5 of that same scale respectively,
six orders above it.
"""

import numpy as np
import numpy.typing as npt

from fem.shells.rigid_body_mode_detection import RigidBodyModeDetection
from rebel_shape import logger

_NEAR_ZERO_FLOOR: float = 1.0e-6
"""
Floor under the scanned eigenvalues, as a fraction of the largest of them.

The rigid-body eigenvalues are post-shift residuals: tiny, noisy and of either sign,
so unfloored ratios inside that band can spuriously exceed the real rigid-body-to-
elastic jump. Flooring collapses the intra-band ratios to ~1, so only that jump
stands out; see the module docstring for the measured margin that makes 1e-6 a safe
choice.
"""


def detect_rigid_body_modes(
    eigenvalues: npt.NDArray[np.float64], expected_count: int
) -> RigidBodyModeDetection:
    """
    Locates the rigid-body band as the largest relative gap in the low spectrum.

    The scan covers the rigid-body band plus a comparable stretch of elastic
    modes, `2 * expected_count + 4` eigenvalues, so that the jump out of the
    band is inside the window and later, larger gaps higher in the spectrum
    cannot outrank it; the `+ 4` keeps the window meaningful when
    `expected_count` is small. Note that `expected_count` sizes the window
    and nothing else: the count returned is read off the spectrum.

    `eigenvalues` are the eigenvalues of the generalized eigenvalue problem,
    ascending. `expected_count` is the rigid-body modes the geometry should
    have (6 for an unsupported surface).

    :returns: The number of modes before the largest relative gap and the ratio
        across it, or `count=0, gap=nan` when fewer than two eigenvalues were
        computed and no ratio can be formed.
    """
    n_scan = min(len(eigenvalues), 2 * expected_count + 4)
    if n_scan < 2:
        return RigidBodyModeDetection(count=0, gap=float("nan"))

    scanned = np.abs(eigenvalues[:n_scan])
    capped = np.maximum(scanned, _NEAR_ZERO_FLOOR * float(scanned.max()))
    ratios = capped[1:] / capped[:-1]

    detected = int(np.argmax(ratios)) + 1
    return RigidBodyModeDetection(count=detected, gap=float(ratios[detected - 1]))


def report_rigid_body_modes(
    eigenvalues: npt.NDArray[np.float64],
    frequencies: npt.NDArray[np.float64],
    expected_count: int,
) -> None:
    """
    Logs the rigid-body-mode test, warning when the detected count disagrees.

    This is the only automatic check a production run performs on its own
    output: the spectrum must show `expected_count` near-zero modes separated
    from the elastic ones by a large gap, and a disagreement means either the
    geometry is not closed or the eigensolver settings are wrong.

    `eigenvalues` are the eigenvalues of the generalized eigenvalue problem,
    and `frequencies` are the natural frequencies (Hz) corresponding to them.
    `expected_count` is the expected number of rigid-body modes.
    """
    detection = detect_rigid_body_modes(eigenvalues, expected_count)
    if detection.count == 0:
        logger.warning(
            f"Rigid-body-mode test skipped: only {len(eigenvalues)} eigenvalue(s) "
            f"computed, at least 2 are needed to locate the spectral gap.")
        return

    logger.info(
        f"Rigid-body-mode test: largest spectral gap after mode {detection.count} "
        f"(expected {expected_count}), gap ratio = {detection.gap:.3e}.")
    logger.info(
        f"  RB eigenvalues: "
        f"{np.array2string(eigenvalues[:expected_count], precision=3)}")
    if len(eigenvalues) > expected_count:
        logger.info(
            f"  first elastic mode: lambda = {eigenvalues[expected_count]:.4e}, "
            f"f = {frequencies[expected_count]:.3f} Hz")
    if detection.count != expected_count:
        logger.warning(
            f"Detected {detection.count} rigid-body modes but expected "
            f"{expected_count}; check geometry closure or eigensolver settings.")
