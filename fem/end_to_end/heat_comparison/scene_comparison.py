from dataclasses import dataclass

import numpy as np

from fem.end_to_end.heat_comparison.snapshot_comparison import SnapshotComparison


@dataclass(frozen=True, kw_only=True)
class SceneComparison:
    """One scene's finished comparison: its per-snapshot rows, digested on demand."""

    name: str
    """
    Directory this scene's artifacts and IGA reference run live under, relative to
    `CompareSceneParameters.dataset_root` (`HeatComparisonSceneConfig.directory`).
    """
    snapshots: tuple[SnapshotComparison, ...]
    """Per-snapshot agreement rows, in time order."""

    def summary(self) -> str:
        """
        One-line digest of `snapshots`' relative L2 errors, in percent.

        Computed on demand rather than stored, so it cannot go stale against
        `snapshots`. The transient mean drops the t=0 row, where both fields are the
        same projected initial condition and the error therefore measures the
        projection, not the solve.
        """
        errors = np.array([row.relative_l2 for row in self.snapshots])
        return (
            f"mean_relL2={100 * errors.mean():.4f}%  "
            f"mean_transient={100 * errors[1:].mean():.4f}%  "
            f"max={100 * errors.max():.4f}%"
        )
