"""
Compares IGA and FEM heat simulation results loaded from XDMF files.

Reports per-step L2 and Linf error norms between the two temperature time
series and, optionally, writes the IGA-minus-FEM difference series to XDMF
for visualization in ParaView.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

# Repository root: makes `forge_sim` and this script's own `fem` package
# importable when it is run by path rather than as `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fem.iga_fem_comparison.comparison_report import ComparisonReport
from fem.iga_fem_comparison.error_norms import ErrorNorms
from fem.iga_fem_comparison.vertex_mapping import VertexMapping
from fem.iga_fem_comparison.xdmf_solution import XDMFSolution
from forge_sim import logger


@dataclass(kw_only=True)
class CompareIGAFEMParameters(ReverseCli):
    """
    Parameters for comparing IGA and FEM heat simulation results.
    """

    iga_xdmf: Path
    """Path to the IGA solution XDMF file."""
    fem_xdmf: Path
    """Path to the FEM solution XDMF file."""
    output_xdmf: Path | None = None
    """Optional path for output difference XDMF file."""
    relative_errors: bool = True
    """Whether to compute relative errors (vs absolute)."""

    def __post_init__(self) -> None:
        if not self.iga_xdmf.exists():
            raise ValueError(f"IGA XDMF file not found: {self.iga_xdmf}")
        if not self.fem_xdmf.exists():
            raise ValueError(f"FEM XDMF file not found: {self.fem_xdmf}")


def compare_solutions(params: CompareIGAFEMParameters) -> None:
    """
    Compares two XDMF temperature series and reports the error between them.

    Loads both solutions, gathers the FEM field onto the IGA vertex ordering by
    nearest-neighbour matching, computes the per-step error norms, and logs the
    per-step table plus summary statistics. Writes the IGA-minus-FEM difference
    series to `params.output_xdmf` when given.

    :raises ValueError: If the two solutions carry different snapshot counts (see
        `ComparisonReport.require_matching_time_steps`).
    """
    logger.info(f"Loading IGA solution from {params.iga_xdmf}")
    iga_solution = XDMFSolution.from_file(params.iga_xdmf)
    logger.info(f"Loading FEM solution from {params.fem_xdmf}")
    fem_solution = XDMFSolution.from_file(params.fem_xdmf)
    logger.info(
        f"IGA: {iga_solution.vertices.shape[0]} vertices, "
        f"{len(iga_solution.times)} time steps"
    )
    logger.info(
        f"FEM: {fem_solution.vertices.shape[0]} vertices, "
        f"{len(fem_solution.times)} time steps"
    )

    ComparisonReport.require_matching_time_steps(
        iga_times=iga_solution.times, fem_times=fem_solution.times
    )
    mapping = VertexMapping.iga_to_fem(iga_solution.vertices, fem_solution.vertices)
    fem_temperatures = mapping.gather(fem_solution.temperatures)

    logger.info("Computing errors...")
    norms = (
        ErrorNorms.relative(iga_solution.temperatures, fem_temperatures)
        if params.relative_errors
        else ErrorNorms.absolute(iga_solution.temperatures, fem_temperatures)
    )

    report = ComparisonReport(iga=iga_solution, norms=norms)
    report.log()

    if params.output_xdmf is not None:
        report.write_difference_xdmf(params.output_xdmf)


if __name__ == "__main__":
    parameters = tyro.cli(CompareIGAFEMParameters)
    compare_solutions(params=parameters)
