from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class GeometrySimCorrelation:
    """
    How strongly the simulation errors track the geometry error, over the meshes.

    Each field is a Spearman rank correlation taken across the recorded meshes
    between one geometry measure and one simulation error, which is the direct
    "better geometry -> better simulation" evidence. Rank correlation rather than
    Pearson because four meshes cannot support a functional fit, and the claim is
    only about ordering. Serialised as `metrics/geometry_sim_correlation.json`.
    """

    spearman_meansurfdist_vs_heatH0: float
    """Mean surface distance against the total-heat-at-t0 relative error."""
    spearman_meansurfdist_vs_modalmean: float
    """Mean surface distance against the mean relative elastic-frequency error."""
    spearman_meansurfdist_vs_lambda1: float
    """Mean surface distance against the first heat eigenvalue's relative error."""
    spearman_meansurfdist_vs_meanbestmac: float
    """
    Mean surface distance against the mean best MAC.

    Expected strongly negative: shape agreement improves as geometry improves,
    while the error columns fall.
    """
    spearman_chamfer_vs_heatH0: float
    """Chamfer distance against the total-heat-at-t0 relative error."""
    spearman_chamfer_vs_modalmean: float
    """Chamfer distance against the mean relative elastic-frequency error."""
