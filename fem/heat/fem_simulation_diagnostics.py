import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

from fem.heat.snapshot_diagnostics import SnapshotDiagnostics
from rebel_shape import logger


@dataclass(frozen=True, kw_only=True)
class FEMSimulationDiagnostics:
    """
    What a completed FEM heat run reports about itself.

    Returned by `FEMHeatRun.run_with_xdmf` so a caller can log or record the run's
    characterization without reaching into the solver (or re-deriving the source
    scaling). Values are global, so every MPI rank receives the same record.
    """

    JSON_NAME: ClassVar[str] = "fem_heat_diagnostics.json"
    """
    Name of the run-diagnostics sidecar written beside the XDMF series.

    Deliberately not the ground-truth driver's `fem_heat_gt_log.json`: that name
    carries a schema the end-to-end comparison stages read back, whereas this is a
    plain dump of the record, so a finished run stays legible to whoever did not
    watch its log.
    """

    dt: float
    """Physical time step actually used [s], however it was resolved."""
    relaxation_time: float | None
    """
    Natural relaxation time T_relax [s] when the step was resolved from
    `dt_over_t_relax`, else None; the other paths never compute it, and it is
    deliberately not triggered here just to report it (the eigensolve is
    expensive).
    """
    heat_generation_rate: float
    """Total source power integral(Q) dA [W]."""
    applied_heat_source: float | None
    """
    The uniform source Q [W/m^2] imposed by `target_delta_temp` scaling, or None
    when no such scaling was applied.
    """
    snapshots: tuple[SnapshotDiagnostics, ...]
    """Per-snapshot energy diagnostics, in time order, starting at t = 0."""

    def write_json(self, output_folder: Path) -> None:
        """
        Writes this record to `JSON_NAME` in `output_folder`, beside the series it
        describes.

        `asdict` flattens the nested per-snapshot records as well, and every field is
        a JSON scalar or None: the assembled energies come back from
        `assemble_scalar` as plain floats, so no encoder hook is needed.

        Every rank holds the same record, so a parallel run calls this on rank 0
        alone rather than having each rank overwrite the same file.
        """
        path = output_folder / self.JSON_NAME
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        logger.info(f"Run diagnostics written to {path}.")
