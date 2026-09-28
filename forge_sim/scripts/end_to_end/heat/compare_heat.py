"""
Step 6 (compare): Independent heat comparison: FEM-GT vs IGA-spline.

A direct field diff is not meaningful across two different geometries, so the
comparison uses geometry-robust physical scalars logged by both solvers:

* total thermal energy at the initial and final time: a proxy for how well the
  reconstruction recovered the (per-material-weighted) surface area;
* heat-generation rate: `integral(source dA)` over the source material, a proxy
  for that material's recovered area;
* peak surface temperature gradient over time (best-effort, from the XDMF): a
  proxy for how well the sharp checkerboard / interface transients are resolved.

The two XDMF series remain as the side-by-side visual deliverable (ParaView).

Run (after steps 5 & 6):  python -m forge_sim.scripts.end_to_end.heat.compare_heat
"""

import json
from dataclasses import asdict
from pathlib import Path

import tyro

from forge_sim import logger
from forge_sim.scripts.end_to_end.end_to_end_config import EndToEndConfig
from forge_sim.scripts.end_to_end.heat.compare_heat_runner import CompareHeatRunner
from forge_sim.scripts.end_to_end.heat.heat_comparison_report import (
    HeatComparisonReport,
)
from forge_sim.scripts.end_to_end.heat.heat_energy_log import HeatEnergyLog


def _require_matching_source_region(fem: HeatEnergyLog, iga: HeatEnergyLog) -> None:
    """
    Rejects two runs that put the heat source on different materials.

    Every quantity compared below assumes the two solvers integrated one scenario, and
    which material carries the source is part of that scenario: heating the active
    material on one side and the passive one on the other makes the energies describe
    different physics, and their agreement number nothing at all. That is a structural
    mismatch of the two runs (not a magnitude two discretisations could explain), so
    it is rejected rather than reported. Rerun both with the same
    `thermal.heat_source_region`.

    A log written before the region was recorded carries none; that is warned about
    rather than rejected, since a recorded artifact cannot be re-stamped with the
    physics it actually integrated.

    :raises ValueError: If both runs recorded a source region and the two disagree.
    """
    if fem.heat_source_region is None or iga.heat_source_region is None:
        logger.warning(
            f"[compare heat] heat source region recorded on only one side (FEM "
            f"{fem.heat_source_region}, IGA {iga.heat_source_region}); the two runs "
            f"cannot be checked for having integrated the same scenario."
        )
        return
    if fem.heat_source_region is not iga.heat_source_region:
        raise ValueError(
            f"The two runs put the heat source on different materials (FEM "
            f"{fem.heat_source_region.value}, IGA {iga.heat_source_region.value}), so "
            f"they integrated different physics and comparing their energies is "
            f"meaningless. Rerun both with the same `thermal.heat_source_region`."
        )


def run_heat_comparison(config: EndToEndConfig) -> HeatComparisonReport:
    """
    Compare the two runs' logged energy scalars; write the table, Markdown and plot.

    :returns: Each compared quantity with both sides' value and their relative error.
    :raises ValueError: If the two runs recorded different heat source regions (see
        `_require_matching_source_region`).
    """
    fem = HeatEnergyLog.from_json(config.fem_heat_dir / "fem_heat_gt_log.json")
    iga = HeatEnergyLog.from_json(config.iga_heat_dir / "iga_heat_log.json")
    _require_matching_source_region(fem, iga)

    def compared(fem_value: float, iga_value: float) -> HeatComparisonReport.Quantity:
        """One quantity on both geometries, with the IGA side's relative error."""
        return HeatComparisonReport.Quantity(
            fem_gt=float(fem_value),
            iga_spline=float(iga_value),
            rel_error=float(
                abs(fem_value - iga_value) / abs(fem_value)
                if fem_value != 0
                else float("nan")
            ),
        )

    metrics = HeatComparisonReport(
        total_heat_initial=compared(fem.total_heat[0], iga.total_heat[0]),
        total_heat_final=compared(fem.total_heat[-1], iga.total_heat[-1]),
        heat_generation_rate=compared(fem.gen_rate[-1], iga.gen_rate[-1]),
        relaxation_time=HeatComparisonReport.RelaxationTimes(
            fem_gt=fem.relaxation_time, iga_spline=iga.relaxation_time
        ),
    )

    out = config.heat_compare_dir
    (out / "heat_comparison.json").write_text(json.dumps(asdict(metrics), indent=2))

    lines = [
        "# Heat comparison: FEM (ground-truth mesh) vs IGA (reconstructed spline)",
        "",
        "| quantity | FEM-GT | IGA-spline | rel. error |",
        "|---|---:|---:|---:|",
    ]
    for quantity, label, unit in [
        (metrics.total_heat_initial, "Total energy @ t0", "J"),
        (metrics.total_heat_final, "Total energy @ tN", "J"),
        (metrics.heat_generation_rate, "Heat generation rate", "W"),
    ]:
        lines.append(
            f"| {label} ({unit}) | {quantity.fem_gt:.6g} | {quantity.iga_spline:.6g} "
            f"| {quantity.rel_error * 100:.2f}% |"
        )
    (out / "heat_comparison.md").write_text("\n".join(lines))

    _plot(fem, iga, config, out / "heat_comparison.png")
    logger.info(
        f"[compare heat] energy@t0 rel {metrics.total_heat_initial.rel_error * 100:.2f}"
        f"%, @tN rel {metrics.total_heat_final.rel_error * 100:.2f}%, gen-rate rel "
        f"{metrics.heat_generation_rate.rel_error * 100:.2f}% -> {out}"
    )
    return metrics


def _plot(
    fem: HeatEnergyLog, iga: HeatEnergyLog, config: EndToEndConfig, path: Path
) -> None:
    """Total energy and peak gradient against physical time, for both solvers."""
    # Deferred: the headless Agg backend must be selected before pyplot is imported,
    # which a module-level import cannot express.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(fem.time, fem.total_heat, "o-", label="FEM (GT mesh)")
    axes[0].plot(iga.time, iga.total_heat, "s--", label="IGA (recon spline)")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("total thermal energy (J)")
    axes[0].set_title("Total energy vs time")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    fem_grad = CompareHeatRunner.peak_gradient_series(
        config.fem_heat_dir / "fem_heat_gt.h5"
    )
    iga_grad = CompareHeatRunner.peak_gradient_series(
        config.iga_heat_dir / "iga_heat.h5"
    )
    if fem_grad is not None and iga_grad is not None:
        axes[1].plot(fem_grad[0], fem_grad[1], "o-", label="FEM (GT mesh)")
        axes[1].plot(iga_grad[0], iga_grad[1], "s--", label="IGA (recon spline)")
        axes[1].set_xlabel("snapshot index")
        axes[1].set_ylabel("peak surface |grad T|")
        axes[1].set_title("Peak temperature gradient vs time")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(
            0.5,
            0.5,
            "peak gradient unavailable\n(h5py / XDMF missing)",
            ha="center",
            va="center",
        )
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run_heat_comparison(tyro.cli(EndToEndConfig))
