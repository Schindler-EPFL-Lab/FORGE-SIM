import json

from rebel_shape.scripts.end_to_end.end_to_end_config import EndToEndConfig
from rebel_shape.scripts.end_to_end.heat.heat_comparison_report import (
    HeatComparisonReport,
)
from rebel_shape.scripts.end_to_end.modal.modal_comparison_report import (
    ModalComparisonReport,
)


class SummarizeRunner:
    """
    The four deliverable sections `summarize` collates into `SUMMARY.md`: mesh
    quality, reconstruction, heat and modal. Each reads one stage's own artifacts and
    reports "(not run)" when that stage has not produced them, so the summary is safe
    to call at any point in the pipeline.
    """

    _NOT_RUN: list[str] = ["", "(not run)", ""]
    """What a section reduces to when the stage that feeds it has not been run."""

    _MAX_LISTED_MODES: int = 10
    """How many elastic modes the modal table shows before it is truncated."""

    @classmethod
    def mesh_quality_section(cls, config: EndToEndConfig) -> list[str]:
        """
        Step 1: the remesh's effect on triangle quality, before against after.

        The only stage read as a raw mapping rather than through its record, because
        `MeshQualityReport` lives in `fem/` and nothing under `rebel_shape/` may
        import it.
        """
        path = config.mesh_quality_dir / "mesh_quality_report.json"
        lines = ["## 1. Mesh quality (before -> after remesh)"]
        if not path.exists():
            return lines + cls._NOT_RUN

        report = json.loads(path.read_text())
        b, a = report["before"], report["after"]
        rows = [
            ("triangles", b["num_triangles"], a["num_triangles"], "d"),
            ("skewness mean", b["skewness"]["mean"], a["skewness"]["mean"], ".3f"),
            ("skewness max", b["skewness"]["max"], a["skewness"]["max"], ".3f"),
            (
                "aspect mean",
                b["aspect_ratio"]["mean"],
                a["aspect_ratio"]["mean"],
                ".2f",
            ),
            (
                "min angle (deg)",
                b["min_angle_deg"]["min"],
                a["min_angle_deg"]["min"],
                ".1f",
            ),
            (
                "% skew > 0.5",
                b["fraction_skewness_gt_0p5"] * 100,
                a["fraction_skewness_gt_0p5"] * 100,
                ".1f",
            ),
            ("edge len min", b["edge_length"]["min"], a["edge_length"]["min"], ".4g"),
            ("edge len max", b["edge_length"]["max"], a["edge_length"]["max"], ".4g"),
        ]
        lines += ["", "| metric | before | after |", "|---|---:|---:|"]
        lines += [
            f"| {label} | {bv:{fmt}} | {av:{fmt}} |" for label, bv, av, fmt in rows
        ]
        lines.append("")
        return lines

    @classmethod
    def reconstruction_section(cls, config: EndToEndConfig) -> list[str]:
        """Step 2: the reconstructed spline and how many patch textures came with it."""
        lines = ["## 2. Reconstruction"]
        spline = config.reconstructed_spline_path
        if not spline.exists():
            return lines + cls._NOT_RUN

        tex_dir = config.reconstruction_dir / "final_patch_textures"
        n_tex = len(list(tex_dir.glob("*.exr"))) if tex_dir.exists() else 0
        return lines + [
            "",
            f"- Spline: `{spline.name}`",
            f"- Patch textures: {n_tex} EXR",
            "",
        ]

    @classmethod
    def heat_section(cls, config: EndToEndConfig) -> list[str]:
        """Steps 5/6: the geometry-robust heat scalars on both solvers."""
        lines = ["## 5/6. Heat: FEM (ground truth) vs IGA (recovered)"]
        path = config.heat_compare_dir / "heat_comparison.json"
        if not path.exists():
            return lines + cls._NOT_RUN

        heat = HeatComparisonReport.from_json(path)
        lines += [
            "",
            "| quantity | FEM-GT | IGA-spline | rel. error |",
            "|---|---:|---:|---:|",
        ]
        for quantity, label in [
            (heat.total_heat_initial, "Total energy @ t0 (J)"),
            (heat.total_heat_final, "Total energy @ tN (J)"),
            (heat.heat_generation_rate, "Generation rate (W)"),
        ]:
            lines.append(
                f"| {label} | {quantity.fem_gt:.6g} | {quantity.iga_spline:.6g} "
                f"| {quantity.rel_error * 100:.2f}% |"
            )
        lines.append("")
        return lines

    @classmethod
    def modal_section(cls, config: EndToEndConfig) -> list[str]:
        """Steps 7/8: the two elastic spectra and the leading per-mode errors."""
        lines = ["## 7/8. Modal: FEM (ground truth) vs IGA (recovered)"]
        path = config.modal_compare_dir / "modal_comparison.json"
        if not path.exists():
            return lines + cls._NOT_RUN

        modal = ModalComparisonReport.from_json(path)
        lines += [
            "",
            f"- RB modes: FEM {modal.num_rb_modes_fem}, IGA "
            f"{modal.num_rb_modes_iga} (expected 6)",
            f"- Elastic modes compared: {modal.num_compared_elastic_modes}",
            f"- Mean relative frequency error: {modal.mean_rel_error * 100:.2f}%",
            f"- Max relative frequency error: {modal.max_rel_error * 100:.2f}%",
            "",
            "| elastic mode | FEM-GT (Hz) | IGA-spline (Hz) | rel. error |",
            "|---:|---:|---:|---:|",
        ]
        for mode in modal.modes[: cls._MAX_LISTED_MODES]:
            lines.append(
                f"| {mode.elastic_mode} | {mode.fem_gt_hz:.3f} | "
                f"{mode.iga_spline_hz:.3f} | {mode.rel_error * 100:.2f}% |"
            )
        lines.append("")
        return lines
