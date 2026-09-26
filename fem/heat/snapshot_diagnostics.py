from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class SnapshotDiagnostics:
    """
    Energy bookkeeping for one saved snapshot of the run.

    `total_heat` is the assembled thermal energy H = integral(rho c d u) dA and
    `relative_drift` is its departure from the energy the source alone accounts
    for, (H - H_expected) / H_expected with H_expected = H(0) + t * Q_total. Both
    are global (MPI-reduced) values.
    """

    time: float
    """Snapshot time [s], measured from the start of the run."""
    total_heat: float
    """Total thermal energy at that time [J]."""
    relative_drift: float
    """
    Relative energy drift, dimensionless (0.0 when the expected energy is
    exactly zero).
    """
