from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class ModalAlignmentReport:
    """
    What aligning one run's IGA eigenmodes onto the FEM ones did.

    The alignment only removes each mode's arbitrary sign and, within a genuinely
    degenerate run, its arbitrary in-subspace rotation; it never reorders the
    spectrum. `neg_after` is therefore the honest residual: a mode still anti-
    correlated after alignment is one whose shape really differs.
    """

    xdmf_path: Path
    """The rewritten `eigenmodes_aligned.xdmf`."""
    n_modes: int
    """Modes aligned, the shorter of the two exports."""
    n_degenerate_runs: int
    """
    Runs of adjacent modes degenerate on both solvers, which got the orthogonal-
    Procrustes treatment rather than a plain sign flip.
    """
    neg_before: int
    """Modes whose signed correlation with their FEM partner was negative before."""
    neg_after: int
    """Modes still negatively correlated afterwards: genuinely differing shapes."""
    mean_cos_after: float
    """Mean signed area-weighted correlation after alignment; 1 is a perfect overlay."""
