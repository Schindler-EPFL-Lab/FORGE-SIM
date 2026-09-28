from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class HeatFieldReport:
    """
    H2: the cross-geometry space-time heat field discrepancy for one mesh.

    The raw cross-surface relative L2 of a sharp field is dominated by a transfer
    artifact rather than by any physical disagreement, so each step also carries
    its own symmetric round-trip floor and the floor-corrected
    `excess = sqrt(cross^2 - floor^2)`, which is the part that is a genuine field
    discrepancy. Serialised as one entry of `metrics/heat_field_l2.json`.
    """

    @dataclass(frozen=True, kw_only=True)
    class ProjectionDistance:
        """
        How far the closest-point transfer had to travel between the two surfaces.

        Reported as a percentage of the bounding-box diagonal, because the
        floor-corrected excess still carries a `|grad u| * d_proj` term and this
        is the `d_proj` in it.
        """

        fem_to_iga_mean: float
        """Mean distance from an IGA vertex to its footpoint on the FEM mesh [%]."""
        fem_to_iga_max: float
        """Largest such distance [%]."""
        iga_to_fem_mean: float
        """Mean distance from a FEM vertex to its footpoint on the IGA mesh [%]."""

    mesh: str
    """Name of the recorded mesh this comparison belongs to."""
    n_steps: int
    """Saved timesteps compared, the shorter of the two series."""
    bbox_diag: float
    """Bounding-box diagonal of the IGA surface, in model units."""
    d_proj_pct_diag: ProjectionDistance
    """Projection distances of the two transfer operators."""
    hot_fraction_fem: float
    """Area fraction of the FEM surface above the hot threshold at t0."""
    hot_fraction_iga: float
    """Area fraction of the IGA surface above the hot threshold at t0."""
    hot_fraction_abs_error: float
    """Absolute difference of the two hot fractions: an inpainting-quality proxy."""
    t0_cross_relL2: float
    """
    Cross-field relative L2 at t0. Transfer-limited on a near-binary field, so it
    measures the edge artifact rather than the inpainting error.
    """
    final_excess_relL2: float
    """Floor-corrected excess at the final (diffused) step: the cleanest number."""
    transient_mean_excess: float
    """Mean floor-corrected excess over every step after t0."""
    spacetime_excess: float
    """Time-weighted RMS of the excess curve over the whole run."""
    max_cross_relL2: float
    """Largest cross-field relative L2 over the run."""
    cross_relL2_curve: list[float]
    """Per-step symmetric cross-field relative L2."""
    floor_relL2_curve: list[float]
    """Per-step symmetric round-trip transfer floor."""
    excess_relL2_curve: list[float]
    """Per-step floor-corrected excess."""
    times: list[float]
    """Physical time of each compared step [s]."""
