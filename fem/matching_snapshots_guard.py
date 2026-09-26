from collections.abc import Sequence
from typing import ClassVar

import numpy as np


class MatchingSnapshotsGuard:
    """
    Rejects two solution series that were not saved at the same instants.

    `ComparisonReport.require_matching_time_steps`
    (`fem/iga_fem_comparison/comparison_report.py`) and
    `SceneHeatComparison.require_matching_snapshot_counts`
    (`fem/end_to_end/heat_comparison/scene_heat_comparison.py`) both pair an
    IGA and a FEM time series index by index, so unequal counts, or equal
    counts at different times, both mean the two series do not describe the
    same instants and every error a per-snapshot comparison would report is
    between fields that are not simultaneous. `require` is the one place that
    check is made; both callers delegate to it, each with the `remediation`
    sentence naming the flag its own CLI exposes to fix the mismatch, and,
    where a batch touches more than one run, the `context` that says which one
    failed.
    """

    _TIME_RTOL: ClassVar[float] = 1e-6
    """
    Relative tolerance `require` allows between paired times. Each side's
    times are `step * dt`, so float64 rounding alone bounds their disagreement
    at roughly `num_steps * eps ~ 1e3 * 2.2e-16 ~ 2e-13`, seven orders below
    this value, while a mismatched `final_time` moves times by a physical
    scale, not by ulps.
    """

    @classmethod
    def require(
        cls,
        *,
        iga_times: Sequence[float],
        fem_times: Sequence[float],
        remediation: str,
        context: str | None = None,
    ) -> None:
        """
        Raises unless `iga_times` and `fem_times` describe the same instants.

        Snapshot `i` of one series is compared against snapshot `i` of the
        other, so this rejects unequal counts outright and, for equal counts,
        checks the paired times themselves: equal counts over different
        spans pair snapshots that are not the same instant, which the count
        alone cannot see. `remediation` is appended to the raised message and
        should name the concrete fix; `context` is prepended, quoted, when the
        caller has a name for the run or scene under check.

        :raises ValueError: If the two series carry different snapshot
            counts, or the same count at times that disagree by more than
            `_TIME_RTOL` (relative to the IGA time) at some snapshot.
        """
        label = f"'{context}' " if context is not None else ""
        if len(iga_times) != len(fem_times):
            raise ValueError(
                f"{label}IGA and FEM solutions have different snapshot counts "
                f"({len(iga_times)} vs {len(fem_times)}); they were not saved "
                f"at the same instants, so a per-snapshot comparison is "
                f"meaningless. {remediation}"
            )
        if not np.allclose(fem_times, iga_times, rtol=cls._TIME_RTOL, atol=0.0):
            raise ValueError(
                f"{label}IGA and FEM solutions have {len(iga_times)} "
                f"snapshots each, but the paired times disagree by more than "
                f"a relative {cls._TIME_RTOL:g}: IGA times {list(iga_times)} "
                f"vs FEM times {list(fem_times)}. Equal counts over different "
                f"spans pair snapshots that are not the same instant. "
                f"{remediation}"
            )
