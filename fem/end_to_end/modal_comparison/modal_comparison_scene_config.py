from dataclasses import dataclass
from pathlib import Path

from forge_sim.scripts.configs.structural_material_config import (
    StructuralMaterialConfig,
)


@dataclass(frozen=True, kw_only=True)
class ModalComparisonSceneConfig:
    """
    One scene's own facts a modal comparison needs: where it lives, its
    reconstruction spline, and the shell material and length scale both
    solvers integrate.

    Carries one scene's parameters, not a table of which scenes exist: a
    caller that knows several scenes resolves each one into a config of its
    own and calls `SceneModalComparison` once per scene; the registry that
    does that resolving is `main_scripts`' own subject (`CLAUDE.md` SS3)
    and is unknown here.
    """

    directory: Path
    """Scene's directory under the dataset root, relative to it."""
    spline_path: Path
    """
    Reconstruction spline (`TessellatedSplineSurface` JSON) both the FEM solve
    and the already-computed IGA modal run used. Which of a scene's several
    candidate reconstructions this is, is a dataset-layout question that is
    `main_scripts`' own subject; this class takes the resolved path.
    """
    target_length_scale: float | None
    """
    Longest control-point bbox axis [m] to rescale the scene to, or None to
    keep it at the scale the reconstruction came out at. Must match the IGA
    run's own value for the two spectra to compare.
    """
    structural_material: StructuralMaterialConfig
    """Shell properties (E, nu, rho, thickness, shear correction) both solvers use."""

    def __post_init__(self) -> None:
        if self.target_length_scale is not None and self.target_length_scale <= 0.0:
            raise ValueError(
                f"`target_length_scale` must be positive if provided, got "
                f"{self.target_length_scale}."
            )
