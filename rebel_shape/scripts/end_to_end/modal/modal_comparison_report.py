import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class ModalComparisonReport:
    """
    The elastic spectra of the FEM-GT and IGA-spline modal runs, compared by rank.

    Both sides drop their rigid-body modes and the remaining elastic modes are
    paired by frequency rank, so mode k of one solver always meets mode k of the
    other. Serialised as `modal_comparison.json`, which the recompute stage
    reads back for its parity fit.
    """

    @dataclass(frozen=True, kw_only=True)
    class Mode:
        """One elastic mode as the two solvers see it."""

        elastic_mode: int
        """One-based elastic mode index, rigid-body modes excluded."""
        fem_gt_hz: float
        """Frequency on the FEM ground-truth mesh [Hz]."""
        iga_spline_hz: float
        """Frequency on the reconstructed IGA spline [Hz]."""
        rel_error: float
        """`|iga - fem| / fem`; NaN when the FEM frequency vanishes."""

    num_rb_modes_fem: int
    """Rigid-body modes the FEM side dropped; 6 for a free-floating closed shell."""
    num_rb_modes_iga: int
    """Rigid-body modes the IGA side dropped; must match `num_rb_modes_fem`."""
    num_compared_elastic_modes: int
    """Elastic modes compared, the shorter of the two spectra."""
    mean_rel_error: float
    """Mean relative frequency error over the compared modes."""
    median_rel_error: float
    """Median relative frequency error: the headline number, robust to reordering."""
    max_rel_error: float
    """Largest relative frequency error; inflated by near-degenerate mode swaps."""
    modes: list[Mode]
    """Every compared mode, in frequency rank."""

    @classmethod
    def from_json(cls, path: Path) -> "ModalComparisonReport":
        """
        Read back the `modal_comparison.json` `run_modal_comparison` wrote.

        The summary stage reports this table without recomputing it, and reading it
        as the record it was written from is what keeps the two in step: a renamed
        field cannot pass the constructor.
        """
        raw = json.loads(path.read_text())
        return cls(
            num_rb_modes_fem=raw["num_rb_modes_fem"],
            num_rb_modes_iga=raw["num_rb_modes_iga"],
            num_compared_elastic_modes=raw["num_compared_elastic_modes"],
            mean_rel_error=raw["mean_rel_error"],
            median_rel_error=raw["median_rel_error"],
            max_rel_error=raw["max_rel_error"],
            modes=[cls.Mode(**mode) for mode in raw["modes"]],
        )
