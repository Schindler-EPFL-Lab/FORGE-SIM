from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class SnapshotComparison:
    """
    Agreement between the two solvers at one snapshot time.

    Both errors are normalized on the IGA field: `relative_l2` is
    `||u_IGA - u_FEM||_2 / ||u_IGA||_2`, which is reported as 0.0 when the IGA norm
    vanishes, and `relative_linf` is `max|u_IGA - u_FEM| / max(max|u_IGA|, floor)`,
    which is not: on a vanishing IGA field it reports a huge ratio rather than 0.0.
    The two guards are inconsistent, and stay so because the published numbers were
    produced with them; the floor is `SceneHeatComparison._PEAK_FLOOR`, a bare guard
    against dividing by an all-zero reference field, not a derived tolerance.
    `fem/compare_iga_fem.py` gates both norms on one derived tolerance and falls back
    to absolute norms, which is the better design.
    """

    time: float
    """Snapshot time [s]."""
    relative_l2: float
    """Relative L2 error, dimensionless."""
    relative_linf: float
    """Relative Linf error, dimensionless."""
    fem_peak: float
    """Peak FEM value at this time, for a sanity read on the magnitudes."""
    iga_peak: float
    """Peak IGA value at this time."""
