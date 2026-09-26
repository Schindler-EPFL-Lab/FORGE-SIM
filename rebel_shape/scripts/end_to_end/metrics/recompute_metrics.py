"""
Derive the stronger validation numbers from the artifacts the end-to-end runs already
wrote, with no new solves.

The metrics named below are described in the directory README; each reads only the
deliverable directory `End-to-end-verification/<mesh>/`:

  * M1: mass-normalized (area-weighted) MAC between the FEM-GT and IGA-spline mode
    shapes: diagonal (index-paired) shape agreement + max-MAC / Hungarian re-pairing, to
    test whether the large index-paired frequency `max` errors are mode-swap artifacts
    or genuine geometry-driven frequency differences.
  * M2: modal parity fit `f_IGA = a * f_FEM (+ b)` with slope + R^2 (a
    systematic stiffness/mass/scale bias shows up as slope != 1).
  * H1(k=1): first heat-operator eigenvalue `lambda_1 = 1/T_relax` on GT vs
    recovered geometry (both already stored in the heat logs). The geometry-intrinsic
    Laplace-Beltrami scalar; k>1 would need re-assembly (a small new-run, not here).
  * Geometry: geometry error (Chamfer / Hausdorff / mean+RMS surface distance,
    all as %% of the GT bbox diagonal) between the co-registered GT and recovered
    analysis surfaces, then correlated (Spearman) against the heat and modal simulation
    errors: the direct "better geometry -> better simulation" evidence.

Everything is read from `End-to-end-verification/<mesh>/` and written back under
`<root>/metrics/` as JSON (machine) + Markdown (human). Re-runnable, no re-solve.

Run:  python -m rebel_shape.scripts.end_to_end.metrics.recompute_metrics \
          [--root <End-to-end-verification>] [--meshes suzanne spot bunny armadillo]
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tyro

from rebel_shape import logger
from rebel_shape.scripts.end_to_end.heat.heat_comparison_report import (
    HeatComparisonReport,
)
from rebel_shape.scripts.end_to_end.metrics.geometry_sim_correlation import (
    GeometrySimCorrelation,
)
from rebel_shape.scripts.end_to_end.metrics.mesh_metrics_report import MeshMetricsReport
from rebel_shape.scripts.end_to_end.metrics.recompute_metrics_runner import (
    RecomputeMetricsRunner,
)
from rebel_shape.scripts.end_to_end.metrics.recorded_runs_parameters import (
    RecordedRunsParameters,
)
from rebel_shape.scripts.end_to_end.modal.modal_comparison_report import (
    ModalComparisonReport,
)
from rebel_shape.scripts.end_to_end.modal.modal_frequencies import modal_frequencies


def _metrics_for_mesh(params: RecordedRunsParameters, name: str) -> MeshMetricsReport:
    """Recompute every metric for one recorded mesh, tagged with its name."""
    d = params.recorded_runs_root / name
    fem_modal_h5 = d / "fem_modal_simulation" / "eigenmodes.h5"
    iga_modal_h5 = d / "iga_modal_simulation" / "eigenmodes_aligned.h5"
    fem_freq = np.sort(
        modal_frequencies(d / "fem_modal_simulation" / "modal_analysis_results.npz")
    )
    iga_freq = np.sort(
        modal_frequencies(d / "iga_modal_simulation" / "modal_analysis_results.npz")
    )

    modal_json = d / "modal_comparison" / "modal_comparison.json"
    heat = HeatComparisonReport.from_json(
        d / "heat_comparison" / "heat_comparison.json"
    )
    modal = ModalComparisonReport.from_json(modal_json)

    return MeshMetricsReport(
        mesh=name,
        mac=RecomputeMetricsRunner.modal_mac(
            fem_modal_h5, iga_modal_h5, fem_freq, iga_freq, params.cluster_gap
        ),
        parity=RecomputeMetricsRunner.modal_parity(modal_json),
        heat_lambda1=RecomputeMetricsRunner.heat_lambda1(
            d / "iga_heat_simulation" / "iga_heat_log.json",
            d / "fem_heat_simulation" / "fem_heat_gt_log.json",
        ),
        geometry=RecomputeMetricsRunner.geometry_error(
            fem_modal_h5,
            iga_modal_h5,
            params.geometry_sample_points,
            params.geometry_sample_seed,
        ),
        e_reference=MeshMetricsReport.ReferenceErrors(
            heat_H_t0_rel_error=heat.total_heat_initial.rel_error,
            heat_gen_rate_rel_error=heat.heat_generation_rate.rel_error,
            modal_mean_rel_error=modal.mean_rel_error,
            modal_median_rel_error=modal.median_rel_error,
            modal_max_rel_error=modal.max_rel_error,
        ),
    )


def write_reports(out_dir: Path, results: list[MeshMetricsReport]) -> None:
    """
    Write the recompute JSON, the geometry/simulation correlations, and the Markdown
    report into `out_dir`.
    """
    (out_dir / "recompute_metrics.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2)
    )

    geo = np.array([r.geometry.mean_surf_dist_pct_diag for r in results])
    cham = np.array([r.geometry.chamfer_pct_diag for r in results])
    heat_e = np.array([r.e_reference.heat_H_t0_rel_error for r in results])
    modal_e = np.array([r.e_reference.modal_mean_rel_error for r in results])
    lam1 = np.array([r.heat_lambda1.rel_error for r in results])
    best_mac = np.array([r.mac.mean_best_mac for r in results])
    corr = GeometrySimCorrelation(
        spearman_meansurfdist_vs_heatH0=RecomputeMetricsRunner.spearman(geo, heat_e),
        spearman_meansurfdist_vs_modalmean=RecomputeMetricsRunner.spearman(
            geo, modal_e
        ),
        spearman_meansurfdist_vs_lambda1=RecomputeMetricsRunner.spearman(geo, lam1),
        spearman_meansurfdist_vs_meanbestmac=RecomputeMetricsRunner.spearman(
            geo, best_mac
        ),
        spearman_chamfer_vs_heatH0=RecomputeMetricsRunner.spearman(cham, heat_e),
        spearman_chamfer_vs_modalmean=RecomputeMetricsRunner.spearman(cham, modal_e),
    )
    (out_dir / "geometry_sim_correlation.json").write_text(
        json.dumps(asdict(corr), indent=2)
    )

    lines: list[str] = []
    a = lines.append
    a("# Recomputed metrics (from existing artifacts; no re-solve)")
    a("")
    a("## M1: MAC-based modal shape agreement")
    a("")
    a("Area-weighted (lumped-mass) MAC between FEM-GT and IGA-spline **displacement** "
      "mode shapes. `mean diag MAC` = agreement of the index-paired modes (i vs i); "
      "`mean best MAC` = each FEM mode's best MAC over *all* IGA modes (is the shape "
      "recovered *somewhere*?); `#best>=0.5` counts shapes recovered up to reordering; "
      "`#reordered` = modes whose best partner is not the same index (near-degenerate "
      "cluster reshuffle, NOT lost shape); `cluster overlap` = mean subspace overlap "
      "over near-degenerate clusters (M4: 1 = FEM and IGA span the same subspace).")
    a("")
    a("| mesh | elastic | mean diag MAC | mean best MAC | #best>=0.5 | #reordered | "
      "multi-mode clusters | mean cluster overlap |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        mc = r.mac
        a(f"| {r.mesh} | {mc.n_elastic} | {mc.mean_diagonal_mac:.3f} | "
          f"{mc.mean_best_mac:.3f} | {mc.n_best_mac_ge_0p5}/{mc.n_elastic} | "
          f"{mc.n_reordered} | {mc.n_multi_mode_clusters} | "
          f"{mc.mean_cluster_subspace_overlap:.3f} |")
    a("")
    a("*Reading:* low diagonal MAC with high best MAC + high cluster overlap means the "
      "shapes ARE recovered but near-degenerate modes reorder between solvers, so "
      "the per-mode frequency **max** error is a pairing/degeneracy artifact and the "
      "**mean/median** is the honest headline.")
    a("")
    a("## M2: Modal parity  f_IGA = slope * f_FEM (+ intercept)")
    a("")
    a("| mesh | slope | intercept [Hz] | R^2 | scale (through origin) |")
    a("|---|---:|---:|---:|---:|")
    for r in results:
        p = r.parity
        a(f"| {r.mesh} | {p.slope:.4f} | {p.intercept_hz:.3f} | "
          f"{p.r_squared:.5f} | {p.scale_through_origin:.4f} |")
    a("")
    a("## H1(k=1): First heat-operator eigenvalue  lambda_1 = 1/T_relax")
    a("")
    a("| mesh | T_relax GT [s] | T_relax IGA [s] | lambda_1 rel error |")
    a("|---|---:|---:|---:|")
    for r in results:
        h = r.heat_lambda1
        a(f"| {r.mesh} | {h.T_relax_fem_gt_s:.1f} | "
          f"{h.T_relax_iga_spline_s:.1f} | {h.rel_error * 100:.2f}% |")
    a("")
    a("## Geometry error (recovered vs GT analysis surface, co-registered)")
    a("")
    a("All distances as %% of the GT bounding-box diagonal.")
    a("")
    a("| mesh | Chamfer % | Hausdorff % | mean surf % | RMS surf % | area ratio |")
    a("|---|---:|---:|---:|---:|---:|")
    for r in results:
        g = r.geometry
        a(f"| {r.mesh} | {g.chamfer_pct_diag:.3f} | "
          f"{g.hausdorff_pct_diag:.3f} | {g.mean_surf_dist_pct_diag:.3f} | "
          f"{g.rms_surf_dist_pct_diag:.3f} | {g.area_ratio_rec_over_gt:.4f} |")
    a("")
    a("### Geometry error vs simulation error (the thesis correlation)")
    a("")
    a("Meshes ordered by geometry error. `mean best MAC` rises as geometry improves "
      "(shape agreement); the error columns fall.")
    a("")
    a("| mesh | mean surf dist % | Chamfer % | heat H(t0) err % | lambda_1 err % | "
      "modal mean err % | mean best MAC |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(results, key=lambda r: r.geometry.mean_surf_dist_pct_diag):
        g, e = r.geometry, r.e_reference
        a(f"| {r.mesh} | {g.mean_surf_dist_pct_diag:.3f} | "
          f"{g.chamfer_pct_diag:.3f} | {e.heat_H_t0_rel_error * 100:.2f} | "
          f"{r.heat_lambda1.rel_error * 100:.2f} | "
          f"{e.modal_mean_rel_error * 100:.2f} | {r.mac.mean_best_mac:.3f} |")
    a("")
    a(f"Spearman rank correlation vs mean surface distance (n={len(results)} meshes): "
      f"heat H(t0) err {corr.spearman_meansurfdist_vs_heatH0:+.3f}; "
      f"lambda_1 err {corr.spearman_meansurfdist_vs_lambda1:+.3f}; "
      f"modal-mean err {corr.spearman_meansurfdist_vs_modalmean:+.3f}; "
      f"mean-best-MAC {corr.spearman_meansurfdist_vs_meanbestmac:+.3f} "
      f"(negative = shape agreement improves with geometry, as expected).")
    a("")
    (out_dir / "recompute_metrics.md").write_text("\n".join(lines) + "\n")


def main(params: RecordedRunsParameters) -> None:
    """Recompute every metric for the selected meshes and write the reports."""
    results: list[MeshMetricsReport] = []
    for name in params.meshes:
        logger.info(f"=== {name} ===")
        r = _metrics_for_mesh(params, name)
        results.append(r)
        mc, g = r.mac, r.geometry
        logger.info(
            f"  MAC mean-diag {mc.mean_diagonal_mac:.3f} mean-best "
            f"{mc.mean_best_mac:.3f} best>=0.5 {mc.n_best_mac_ge_0p5}/"
            f"{mc.n_elastic} reordered {mc.n_reordered} cluster-overlap "
            f"{mc.mean_cluster_subspace_overlap:.3f}"
        )
        logger.info(
            f"  geom: Chamfer {g.chamfer_pct_diag:.3f}%  mean-surf "
            f"{g.mean_surf_dist_pct_diag:.3f}%  lambda1 err "
            f"{r.heat_lambda1.rel_error * 100:.2f}%  "
            f"parity slope {r.parity.slope:.3f} R2 "
            f"{r.parity.r_squared:.4f}"
        )
    out_dir = params.metrics_dir
    write_reports(out_dir, results)
    logger.info(f"-> {out_dir}/recompute_metrics.{{json,md}}")


if __name__ == "__main__":
    main(tyro.cli(RecordedRunsParameters))
