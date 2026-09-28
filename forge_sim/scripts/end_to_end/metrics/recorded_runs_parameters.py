import os
from dataclasses import dataclass, field
from pathlib import Path

from dataclasses_reverse_cli.reverse_cli import ReverseCli

RECORDED_RUNS_ENV_VAR: str = "FORGE_SIM_RECORDED_RUNS"
"""Environment variable pointing at the `End-to-end-verification` root."""


@dataclass(kw_only=True)
class RecordedRunsParameters(ReverseCli):
    """
    Selects which recorded end-to-end runs a metrics script reads.

    The metrics scripts derive stronger numbers from artifacts the pipeline already
    wrote, so they are pointed at a deliverable tree (`<root>/<mesh>/...`) rather
    than at a live `EndToEndConfig` output root, and they re-solve nothing.

    The three clustering and sampling knobs below belong to `recompute_metrics`;
    `heat_field_metrics` shares this class for the tree and the mesh list alone.
    """

    root: Path | None = None
    """
    Root of the recorded `End-to-end-verification` tree, holding one directory per
    mesh. Defaults to the `FORGE_SIM_RECORDED_RUNS` environment variable.
    """
    meshes: list[str] = field(
        default_factory=lambda: ["suzanne", "spot", "bunny", "armadillo"]
    )
    """
    Which meshes under `root` to process, in the order they are reported. The
    default is the four targets the recorded end-to-end runs cover.
    """
    cluster_gap: float = 0.10
    """
    Relative frequency gap below which consecutive elastic modes count as one
    near-degenerate cluster, where only the spanned subspace is comparable (metric M4).
    Observed, provenance unrecorded: it groups the report rather than deciding a
    number, since only the `subspace_overlap` column changes with it.
    """
    geometry_sample_seed: int = 0
    """
    Seed handed to Open3D's global RNG before the geometry metrics sample the surfaces.
    Unseeded they wobble run to run (~0.4 % on Chamfer, more on Hausdorff, an
    extremum); seeded, a re-run is bit-identical across fresh processes, so a change
    in these numbers means a change in geometry. Any fixed value would do.
    """
    geometry_sample_points: int = 400_000
    """
    Surface samples per mesh for the geometry metrics. Large enough that the sampling
    noise (see `geometry_sample_seed`) sits well below the geometry differences
    reported, which run from 0.1 % to several % of the bounding-box diagonal.
    """

    def __post_init__(self) -> None:
        if not self.meshes:
            raise ValueError("`meshes` must name at least one mesh, got an empty list.")
        if self.cluster_gap <= 0.0:
            raise ValueError(
                f"`cluster_gap` must be a positive relative gap, got "
                f"{self.cluster_gap}."
            )
        if self.geometry_sample_points < 1:
            raise ValueError(
                f"`geometry_sample_points` must be a positive integer, got "
                f"{self.geometry_sample_points}."
            )

    @property
    def recorded_runs_root(self) -> Path:
        """
        Resolve `root`, falling back to `RECORDED_RUNS_ENV_VAR`.

        :raises ValueError: If `root` was not given and the environment variable is
            not set either, leaving the tree unlocated.
        """
        if self.root is not None:
            return self.root
        value = os.environ.get(RECORDED_RUNS_ENV_VAR)
        if value is None:
            raise ValueError(
                f"--root not given and {RECORDED_RUNS_ENV_VAR} is not set; point it "
                "at the End-to-end-verification directory."
            )
        return Path(value)

    @property
    def metrics_dir(self) -> Path:
        """Return (and create) `<root>/metrics`, where both scripts write."""
        path = self.recorded_runs_root / "metrics"
        path.mkdir(exist_ok=True)
        return path
