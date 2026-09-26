"""
Align IGA modal eigenmodes to the FEM ones for side-by-side ParaView comparison.

An eigenvector is defined only up to sign (simple eigenvalue) or up to an orthogonal
transform within the eigenspace (repeated/degenerate eigenvalue). FEM and IGA pick these
arbitrarily, so the exported mode shapes look flipped (or rotated within a degenerate
pair) when overlaid. This step fixes that without reordering the spectrum: modes are
compared in frequency rank (mode k <-> mode k, exactly as the paper's modal comparison
pairs them), so the i-th elastic mode of each solver always stays the i-th.

For each `eigenmodes.h5` pair (both carry undeformed `/Mesh/vertices` + per-vertex
`displacement`/`rotation`), it:

  1. samples the IGA displacement at the FEM vertices (nearest surface vertex; the two
     surfaces coincide to <1%),
  2. flips the sign of each IGA mode if its area-weighted signed correlation with the
     FEM mode of the same index is negative,
  3. for any run of adjacent modes that is frequency-degenerate on both solvers (a true
     repeated eigenvalue, e.g. a symmetric panel pair), solves the orthogonal Procrustes
     problem within that run so the arbitrary in-subspace rotation is matched too,
  4. rewrites the (in-place, sign/subspace-aligned) IGA modes as
     `eigenmodes_aligned.xdmf`/`.h5` next to the IGA export.

A mode is never paired with one of a different frequency. A global Hungarian/MAC
assignment does pair across frequencies, and on weakly-correlated spectra (Spot, whose
thin legs reconstruct poorly, so the shapes genuinely diverge) it shuffles modes across
the whole spectrum, dropping a far-spectrum mode into a slot. Where the shapes genuinely
differ, the aligned mode simply will not overlay perfectly: that is the honest result,
not something a permutation can fix.

Run:

    python -m rebel_shape.scripts.end_to_end.modal.align_modal_xdmf [--name <dataset>]
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape import logger
from rebel_shape.scripts.end_to_end.end_to_end_config import EndToEndConfig
from rebel_shape.scripts.end_to_end.metrics.recorded_runs_parameters import (
    RecordedRunsParameters,
)
from rebel_shape.scripts.end_to_end.modal.align_modal_xdmf_runner import (
    AlignModalXdmfRunner,
)
from rebel_shape.scripts.end_to_end.modal.modal_frequencies import modal_frequencies


def _fem_frequencies(fem_log: Path) -> npt.NDArray[np.float64]:
    """
    FEM modal frequencies [Hz], read from the run's own log sidecar (no re-solve).

    The FEM driver reports them directly, so unlike the IGA side there is no
    eigenvalue conversion to do.
    """
    return np.asarray(json.loads(fem_log.read_text())["frequencies_hz"], float)


@dataclass(kw_only=True)
class AlignModalXdmfParameters(ReverseCli):
    """Which end-to-end runs to align the exported eigenmodes of."""

    name: str | None = None
    """Dataset to align. Defaults to every dataset the recorded runs cover."""
    output_root: Path | None = None
    """
    Run output root, holding `07_fem_modal` and `08_iga_modal`. Defaults to
    `outputs/<name>_e2e` for each dataset in turn.
    """
    degeneracy_rtol: float = 0.02
    """
    Relative frequency agreement, required on both solvers, below which two adjacent
    modes count as one repeated eigenvalue and get the in-subspace Procrustes
    treatment rather than a plain sign flip. Observed, provenance unrecorded: a
    mis-grouping widens the alignment within a run, never moving a mode out of rank.
    """
    def __post_init__(self) -> None:
        if self.degeneracy_rtol <= 0.0:
            raise ValueError(
                f"`degeneracy_rtol` must be a positive relative tolerance, got "
                f"{self.degeneracy_rtol}."
            )

    def config_for(self, name: str) -> EndToEndConfig:
        """The stage configuration whose artifact paths locate `name`'s run."""
        root = self.output_root or Path("outputs") / f"{name}_e2e"
        return EndToEndConfig(name=name, output_root=root)

    @property
    def names(self) -> Sequence[str]:
        """The datasets to process: the one named, or every recorded one."""
        if self.name is not None:
            return [self.name]
        return RecordedRunsParameters().meshes


def main(params: AlignModalXdmfParameters) -> None:
    """Align every selected dataset's IGA eigenmodes, skipping incomplete runs."""
    for name in params.names:
        config = params.config_for(name)
        # Guard before touching the stage-directory properties: those create the
        # directory they name, so reaching them for a dataset that was never run
        # would leave an empty run tree behind instead of just skipping it.
        if not config.output_root.is_dir():
            logger.warning(f"{name}: no run directory {config.output_root}, skip")
            continue
        fem_h5 = config.fem_modal_dir / "eigenmodes.h5"
        fem_log = config.fem_modal_dir / "fem_modal_gt_log.json"
        iga = list(config.iga_modal_dir.glob("**/eigenmodes.h5"))
        iga_npz = config.iga_modal_dir / "modal_analysis_results.npz"
        if not (fem_h5.exists() and fem_log.exists() and iga and iga_npz.exists()):
            logger.warning(
                f"{name}: missing inputs (fem_h5={fem_h5.exists()}, "
                f"fem_log={fem_log.exists()}, iga_h5={bool(iga)}, "
                f"iga_npz={iga_npz.exists()}), skip"
            )
            continue
        info = AlignModalXdmfRunner.align(
            fem_h5=fem_h5,
            iga_h5=iga[0],
            out_stem=iga[0].parent / "eigenmodes_aligned",
            fem_freq=_fem_frequencies(fem_log),
            iga_freq=modal_frequencies(iga_npz),
            degeneracy_rtol=params.degeneracy_rtol,
        )
        logger.info(
            f"{name}: {info.n_modes} modes (identity pairing), "
            f"{info.n_degenerate_runs} degenerate run(s); sign-flipped "
            f"{info.neg_before} modes, {info.neg_after} still negative; "
            f"mean signed cos {info.mean_cos_after:.3f}"
        )
        logger.info(f"   -> {info.xdmf_path}")


if __name__ == "__main__":
    main(tyro.cli(AlignModalXdmfParameters))
