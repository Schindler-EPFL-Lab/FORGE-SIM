"""
Step 8 (compare): Independent modal eigenvalue comparison: FEM-GT vs IGA-spline.

Loads the two `modal_analysis_results.npz` files (DOLFINx FEM on the ground-truth
mesh; IGA on the reconstructed spline) and compares the elastic eigenfrequencies.
The first 6 (rigid-body) modes are dropped on each side. Produces a table (JSON +
Markdown) and a side-by-side plot. Closeness of the two independent spectra is the
central-thesis evidence: the recovered geometry reproduces the true modal response.

Run (after steps 7 & 8):  python -m forge_sim.scripts.end_to_end.modal.compare_modal
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tyro

from forge_sim import logger
from forge_sim.scripts.end_to_end.end_to_end_config import EndToEndConfig
from forge_sim.scripts.end_to_end.modal.modal_comparison_report import (
    ModalComparisonReport,
)
from forge_sim.scripts.end_to_end.modal.modal_frequencies import (
    RIGID_BODY_FREQ_HZ,
    modal_frequencies,
)

_EXPECTED_RIGID_BODY_MODES: int = 6
"""
Rigid-body modes a free-floating closed shell has: three translations and three
rotations. Both solvers must drop exactly this many, or elastic mode k on one side
is lined up against a different mode on the other.
"""


def run_modal_comparison(config: EndToEndConfig) -> ModalComparisonReport:
    """
    Compare the two elastic spectra and write the table, Markdown and plot.

    :returns: The per-mode rows plus the mean/median/max relative frequency error.
    """
    fem_freq = np.sort(
        modal_frequencies(config.fem_modal_dir / "modal_analysis_results.npz")
    )
    iga_freq = np.sort(
        modal_frequencies(config.iga_modal_dir / "modal_analysis_results.npz")
    )

    fem_el = fem_freq[fem_freq > RIGID_BODY_FREQ_HZ]
    iga_el = iga_freq[iga_freq > RIGID_BODY_FREQ_HZ]
    num_rb_modes_fem = int((fem_freq <= RIGID_BODY_FREQ_HZ).sum())
    num_rb_modes_iga = int((iga_freq <= RIGID_BODY_FREQ_HZ).sum())
    # Both sides must drop the same number of rigid-body modes, otherwise elastic
    # mode k on one side is aligned against the wrong mode on the other.
    if (
        num_rb_modes_fem != num_rb_modes_iga
        or num_rb_modes_fem != _EXPECTED_RIGID_BODY_MODES
    ):
        logger.warning(
            f"[compare modal] rigid-body mode count: FEM {num_rb_modes_fem}, IGA "
            f"{num_rb_modes_iga} (expected {_EXPECTED_RIGID_BODY_MODES} each). "
            f"Elastic-mode alignment by index may be unreliable."
        )
    n = min(len(fem_el), len(iga_el))
    fem_el, iga_el = fem_el[:n], iga_el[:n]

    modes = [
        ModalComparisonReport.Mode(
            elastic_mode=i + 1,
            fem_gt_hz=float(fem_el[i]),
            iga_spline_hz=float(iga_el[i]),
            rel_error=float(abs(iga_el[i] - fem_el[i]) / fem_el[i]),
        )
        for i in range(n)
    ]

    rel_errors = np.array([mode.rel_error for mode in modes])
    summary = ModalComparisonReport(
        num_rb_modes_fem=num_rb_modes_fem,
        num_rb_modes_iga=num_rb_modes_iga,
        num_compared_elastic_modes=n,
        mean_rel_error=float(np.mean(rel_errors)),
        median_rel_error=float(np.median(rel_errors)),
        max_rel_error=float(np.max(rel_errors)),
        modes=modes,
    )

    out = config.modal_compare_dir
    (out / "modal_comparison.json").write_text(json.dumps(asdict(summary), indent=2))

    lines = [
        "# Modal comparison: FEM (ground-truth mesh) vs IGA (reconstructed spline)",
        "",
        f"- RB modes: FEM {summary.num_rb_modes_fem}, IGA "
        f"{summary.num_rb_modes_iga} (expected 6 each)",
        f"- Elastic modes compared: {n}",
        f"- Mean / median / max relative error: "
        f"{summary.mean_rel_error * 100:.2f}% / "
        f"{summary.median_rel_error * 100:.2f}% / "
        f"{summary.max_rel_error * 100:.2f}%",
        "",
        "| elastic mode | FEM-GT (Hz) | IGA-spline (Hz) | rel. error |",
        "|---:|---:|---:|---:|",
    ]
    for mode in modes:
        lines.append(
            f"| {mode.elastic_mode} | {mode.fem_gt_hz:.3f} | "
            f"{mode.iga_spline_hz:.3f} | {mode.rel_error * 100:.2f}% |"
        )
    (out / "modal_comparison.md").write_text("\n".join(lines))

    _plot(fem_el, iga_el, out / "modal_comparison.png")
    logger.info(
        f"[compare modal] {n} elastic modes, mean rel err "
        f"{summary.mean_rel_error * 100:.2f}%, max "
        f"{summary.max_rel_error * 100:.2f}% -> {out}"
    )
    return summary


def _plot(
    fem_el: npt.NDArray[np.float64], iga_el: npt.NDArray[np.float64], path: Path
) -> None:
    """The two spectra side by side, with the per-mode relative error beside them."""
    # Deferred: the headless Agg backend must be selected before pyplot is imported,
    # which a module-level import cannot express.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(fem_el)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    idx = np.arange(1, n + 1)
    axes[0].plot(idx, fem_el, "o-", label="FEM (GT mesh)", color="tab:blue")
    axes[0].plot(idx, iga_el, "s--", label="IGA (recon spline)", color="tab:orange")
    axes[0].set_xlabel("elastic mode #")
    axes[0].set_ylabel("frequency (Hz)")
    axes[0].set_title("Independent modal spectra")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    rel = 100.0 * np.abs(iga_el - fem_el) / fem_el
    axes[1].bar(idx, rel, color="tab:green")
    axes[1].set_xlabel("elastic mode #")
    axes[1].set_ylabel("relative error (%)")
    axes[1].set_title("FEM vs IGA frequency relative error")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run_modal_comparison(tyro.cli(EndToEndConfig))
