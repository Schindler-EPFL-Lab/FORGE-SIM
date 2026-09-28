"""
Collate every stage's outputs into one paper-ready SUMMARY.md.

Reads the JSON/log deliverables produced by the pipeline and writes a single
Markdown summary (mesh quality before/after, reconstruction artifacts, heat energy
comparison, modal eigenvalue comparison) plus an inventory of all output files.
Missing stages are reported as "(not run)" so it is safe to call at any point.

Run:  python -m forge_sim.scripts.end_to_end.metrics.summarize
"""

import tyro

from forge_sim import logger
from forge_sim.scripts.end_to_end.end_to_end_config import EndToEndConfig
from forge_sim.scripts.end_to_end.metrics.summarize_runner import SummarizeRunner


def summarize(config: EndToEndConfig) -> None:
    """Collate every stage's deliverables into `SUMMARY.md` under the output root."""
    lines = [
        "# End-to-end synthetic digital-twin validation: summary",
        "",
        "Simulation on the *recovered* geometry (IGA spline + inpainted fields) vs. "
        "the *ground-truth* geometry (DOLFINx FEM mesh + exact fields), independently.",
        "",
    ]
    lines += SummarizeRunner.mesh_quality_section(config)
    lines += SummarizeRunner.reconstruction_section(config)
    lines += SummarizeRunner.heat_section(config)
    lines += SummarizeRunner.modal_section(config)

    out = config.output_root / "SUMMARY.md"
    out.write_text("\n".join(lines))
    logger.info(f"[summarize] wrote {out}")


if __name__ == "__main__":
    summarize(tyro.cli(EndToEndConfig))
