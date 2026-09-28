import dataclasses
from dataclasses import dataclass
from pathlib import Path

from forge_sim.scripts.end_to_end.end_to_end_config import EndToEndConfig


@dataclass(frozen=True, kw_only=True)
class DatasetOverrides:
    """
    The four dataset-identity fields a caller may override on the command line.

    `EndToEndConfig`'s defaults for these four are real values, not `None`, so an
    unset field here must be left alone rather than passed through and overwriting a
    meaningful default. `apply` is the one path that reaches the config, and it
    guards every field on that basis. `flags` is the matching argv, so the same
    overrides reach both `run_pipeline`'s own config and every stage it spawns.
    """

    name: str | None
    """Dataset identifier override, or `None` to keep `EndToEndConfig`'s default."""
    mesh_path: Path | None
    """Ground-truth source mesh override, or `None` to keep the default."""
    prebuilt_spline_path: Path | None
    """Pre-optimised recovered spline override, or `None` to keep the default."""
    output_root: Path | None
    """Root directory for stage outputs override, or `None` to keep the default."""

    def apply(self, cfg: EndToEndConfig) -> EndToEndConfig:
        """Replaces `cfg`'s four dataset-identity fields with whichever are set."""
        if self.name is not None:
            cfg = dataclasses.replace(cfg, name=self.name)
        if self.mesh_path is not None:
            cfg = dataclasses.replace(cfg, mesh_path=self.mesh_path)
        if self.prebuilt_spline_path is not None:
            cfg = dataclasses.replace(
                cfg, prebuilt_spline_path=self.prebuilt_spline_path
            )
        if self.output_root is not None:
            cfg = dataclasses.replace(cfg, output_root=self.output_root)
        return cfg

    @property
    def flags(self) -> list[str]:
        """CLI flags forwarding whichever fields are set to every spawned stage."""
        forwarded: list[str] = []
        if self.name is not None:
            forwarded += ["--name", self.name]
        if self.mesh_path is not None:
            forwarded += ["--mesh-path", str(self.mesh_path)]
        if self.prebuilt_spline_path is not None:
            forwarded += ["--prebuilt-spline-path", str(self.prebuilt_spline_path)]
        if self.output_root is not None:
            forwarded += ["--output-root", str(self.output_root)]
        return forwarded
