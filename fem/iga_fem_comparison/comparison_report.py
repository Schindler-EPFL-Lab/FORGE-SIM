from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fem.iga_fem_comparison.error_norms import ErrorNorms
from fem.iga_fem_comparison.xdmf_solution import XDMFSolution
from fem.matching_snapshots_guard import MatchingSnapshotsGuard
from fem.xdmf_time_series import XDMFTimeSeries
from forge_sim import logger


@dataclass(frozen=True, kw_only=True)
class ComparisonReport:
    """
    One finished IGA-vs-FEM comparison, and the two ways it is reported.

    Three phases of one report. `require_matching_time_steps` rejects, before
    anything is computed, a pair the comparison cannot pair up at all; `log`
    writes the per-step error table and its summary statistics to the log; and
    `write_difference_xdmf` saves the difference field for ParaView. The report
    holds the IGA solution itself rather than a copy of its mesh, because the
    whole comparison lives in IGA vertex ordering.
    """

    iga: XDMFSolution
    """IGA solution the FEM run was compared against; it supplies mesh and times."""
    norms: ErrorNorms
    """Per-step errors between that solution and the gathered FEM one."""

    @staticmethod
    def require_matching_time_steps(
        *, iga_times: Sequence[float], fem_times: Sequence[float]
    ) -> None:
        """
        Rejects two solutions that were not saved at the same instants.

        Delegates to `MatchingSnapshotsGuard.require`, the check this class
        shares with `SceneHeatComparison.require_matching_snapshot_counts`.
        That is a structural mismatch of the two runs, not a magnitude a
        unit conversion could explain, so it is rejected rather than
        papered over. Match the runs instead: give both the same
        `save_interval` and `num_time_steps`, or pin both runs' end with the
        same `--final-time`.

        :raises ValueError: If the two solutions carry different snapshot
            counts, or the same count at times that disagree by more than
            `MatchingSnapshotsGuard._TIME_RTOL` (relative to the IGA time) at
            some snapshot.
        """
        MatchingSnapshotsGuard.require(
            iga_times=iga_times,
            fem_times=fem_times,
            remediation=(
                "Rerun the two with matching `save_interval` / "
                "`num_time_steps`, or pin both runs' end with the same "
                "`--final-time`."
            ),
        )

    def log(self) -> None:
        """
        Logs the per-step error table and the summary statistics beneath it.
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Error Analysis ({self.norms.kind.value})")
        logger.info(f"{'=' * 60}")
        logger.info(f"{'Step':>6} {'Time':>10} {'L2 Error':>14} {'L∞ Error':>14}")
        logger.info(f"{'-' * 60}")

        for i, (time, l2, linf) in enumerate(
            zip(self.iga.times, self.norms.l2_errors, self.norms.linf_errors)
        ):
            logger.info(f"{i:>6} {time:>10.2f} {l2:>14.6e} {linf:>14.6e}")

        logger.info(f"{'-' * 60}")
        logger.info(f"{'Summary Statistics':^60}")
        logger.info(f"{'-' * 60}")
        logger.info(f"Mean L2 error:  {np.mean(self.norms.l2_errors):.6e}")
        logger.info(f"Max L2 error:   {np.max(self.norms.l2_errors):.6e}")
        logger.info(f"Mean L∞ error:  {np.mean(self.norms.linf_errors):.6e}")
        logger.info(f"Max L∞ error:   {np.max(self.norms.linf_errors):.6e}")
        logger.info(f"{'=' * 60}")

    def write_difference_xdmf(self, output_xdmf: Path) -> None:
        """
        Writes the IGA-minus-FEM difference time series to `output_xdmf`.

        The layout / precision arguments pin this writer's cosmetic formatting,
        so its output is byte-for-byte the form the recorded sidecars carry; they
        say nothing about the data itself (see `xdmf_time_series`).

        The whole comparison lives in IGA vertex ordering: the difference has
        one column per IGA vertex (the FEM field was gathered onto that ordering
        already) and the IGA faces index the full IGA vertex array. So the
        geometry written here is the complete IGA mesh: truncating to
        `min(n_iga, n_fem)` would produce geometry inconsistent with both the
        topology and the attribute whenever the FEM mesh is not at least as fine.
        """
        XDMFTimeSeries.write(
            output_xdmf,
            self.iga.vertices,
            self.iga.faces,
            self.iga.times,
            self.norms.difference,
            attribute_name="Difference",
            layout=XDMFTimeSeries.Layout.INDENTED,
            declare_float_precision=False,
            create_parent_dirs=True,
        )
        logger.info(f"Wrote difference XDMF to {output_xdmf}")
