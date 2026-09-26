from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, kw_only=True)
class ErrorNorms:
    """
    Per-step L2 and Linf difference norms between two temperature time series.

    The two factories differ only in whether they normalize. `absolute` reports
    the norms of the difference itself, in the temperatures' own units;
    `relative` is those norms divided by the IGA field's own norms at the same
    step, which is what makes runs at different temperature scales comparable.
    Both take the two (n_steps, n_vertices) arrays, the FEM one already gathered
    onto the IGA vertex ordering, and both record in `kind` which of them ran,
    so a consumer never has to be told separately.
    """

    class Kind(Enum):
        """Whether the norms carried here were normalized by the IGA field."""

        ABSOLUTE = "absolute"
        """Norms of the difference itself, in the temperatures' own units."""
        RELATIVE = "relative"
        """Norms divided by the IGA field's norms at the same step."""

    REFERENCE_NORM_FLOOR: ClassVar[float] = 1e-10
    """
    IGA reference L2 norm at or below which `relative` reports absolute norms.

    Heuristic rather than derived from an error bound: it separates a step whose
    IGA field is identically zero (a run started from zero temperature) from
    a physical one, and any threshold between the float64 round-off of such a
    field and the smallest physical norm serves. Being an absolute norm in the
    field's units, it follows neither a rescaling nor a refinement.
    """

    difference: npt.NDArray[np.float64]
    """`(n_steps, n_vertices)` IGA minus FEM temperatures."""
    l2_errors: npt.NDArray[np.float64]
    """One L2 norm per step, normalized as `kind` says."""
    linf_errors: npt.NDArray[np.float64]
    """One Linf norm per step, normalized as `kind` says."""
    kind: "ErrorNorms.Kind"
    """Which of the two factories produced `l2_errors` and `linf_errors`."""

    @classmethod
    def absolute(
        cls,
        iga_temps: npt.NDArray[np.float64],
        fem_temps: npt.NDArray[np.float64],
    ) -> "ErrorNorms":
        """
        Norms of the difference between `iga_temps` and `fem_temps`, unnormalized.

        The row-wise reduction is bit-identical to reducing each step separately
        only while the rows are contiguous: NumPy then runs the same pairwise
        summation over the same memory it would for a 1-D row. Reducing a
        non-contiguous array changes that order and moves the norms by a few ulp
        (observed, not a documented NumPy guarantee). Subtracting two
        C-contiguous arrays gives another one, so the copy below is a no-op for
        this package's own callers and pins the order for anyone else.

        :returns: The difference and its per-step norms, tagged `Kind.ABSOLUTE`.
        """
        difference = iga_temps - fem_temps
        rows = np.ascontiguousarray(difference)
        return cls(
            difference=difference,
            l2_errors=np.sqrt(np.sum(rows**2, axis=1)),
            linf_errors=np.max(np.abs(rows), axis=1),
            kind=cls.Kind.ABSOLUTE,
        )

    @classmethod
    def relative(
        cls,
        iga_temps: npt.NDArray[np.float64],
        fem_temps: npt.NDArray[np.float64],
    ) -> "ErrorNorms":
        """
        The same norms, each step divided by the IGA norms of `iga_temps` there.

        Falls back to the absolute norms at any step whose IGA reference L2 norm
        is at or below `REFERENCE_NORM_FLOOR`, since dividing by a near-zero
        reference would report a spurious huge (or NaN) error instead of the
        field's true, small one.

        :returns: The difference and its per-step norms, tagged `Kind.RELATIVE`.
        """
        unnormalized = cls.absolute(iga_temps, fem_temps)

        l2_errors = np.zeros(len(iga_temps))
        linf_errors = np.zeros(len(iga_temps))
        for i in range(len(iga_temps)):
            ref_norm = np.sqrt(np.sum(iga_temps[i] ** 2))
            if ref_norm > cls.REFERENCE_NORM_FLOOR:
                ref_peak = np.max(np.abs(iga_temps[i]))
                l2_errors[i] = unnormalized.l2_errors[i] / ref_norm
                linf_errors[i] = unnormalized.linf_errors[i] / ref_peak
            else:
                l2_errors[i] = unnormalized.l2_errors[i]
                linf_errors[i] = unnormalized.linf_errors[i]

        return cls(
            difference=unnormalized.difference,
            l2_errors=l2_errors,
            linf_errors=linf_errors,
            kind=cls.Kind.RELATIVE,
        )
