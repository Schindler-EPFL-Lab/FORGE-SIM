import numpy as np
import numpy.typing as npt

_RELATIVE_GAP_FLOOR: float = 1e-9
"""
Absolute floor added under the reference frequency in the relative-gap test, so a
pair of literally-zero frequencies still compares instead of dividing by zero.
Every frequency the two callers of `degenerate_mode_runs` group, rigid-body or
elastic, sits many orders of magnitude above it (see `RIGID_BODY_FREQ_HZ`'s
docstring for the measured rigid-body range), so its exact magnitude is
inconsequential; it only ever matters for an exact zero.
"""


def degenerate_mode_runs(
    rtol: float, *freqs: npt.NDArray[np.float64]
) -> list[list[int]]:
    """
    Maximal runs of consecutive (sorted-ascending) indices that are near-degenerate
    on every array in `freqs`.

    Two indices are linked only when their relative gap is within `rtol` on *every*
    array passed: one array asks whether that spectrum alone is self-degenerate,
    two (FEM and IGA, say) additionally require the pairing to hold on both
    solvers, so agreement on one side alone does not merge modes that genuinely
    differ on the other.

    :returns: Each run as the list of consecutive indices it groups, in ascending
        order. A mode with no near neighbour is a run of length 1.
    """
    n = len(freqs[0])

    def close(a: float, b: float) -> bool:
        """Relative agreement, with an absolute floor so a near-zero pair compares."""
        return abs(a - b) <= rtol * max(abs(b), _RELATIVE_GAP_FLOOR)

    runs, start = [], 0
    for i in range(1, n + 1):
        linked = i < n and all(close(freq[i], freq[i - 1]) for freq in freqs)
        if not linked:
            runs.append(list(range(start, i)))
            start = i
    return runs
