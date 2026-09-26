from dataclasses import dataclass
from pathlib import Path

from rebel_shape.scripts.configs.thermal_material_config import ThermalMaterialConfig


@dataclass(frozen=True, kw_only=True)
class HeatComparisonSceneConfig:
    """
    One scene's own facts a heat comparison needs: where it lives, which IGA run
    it is compared against, and the materials and length scale both solvers
    integrate.

    Carries one scene's parameters, not a table of which scenes exist: a caller
    that knows several scenes resolves each one into a config of its own and calls
    `SceneHeatComparison` once per scene; the registry that does that resolving is
    `main_scripts`' own subject (`CLAUDE.md` SS3) and is unknown here.
    """

    directory: Path
    """Scene's directory under the dataset root, relative to it."""
    iga_base_name: str = "heat_simulation"
    """Stem of the IGA reference heat run's `.h5` / `.xdmf` pair."""
    target_length_scale: float | None
    """
    Longest control-point bbox axis [m] to rescale the scene to, or None to keep
    it at the scale the reconstruction came out at.
    """
    base_material: ThermalMaterialConfig
    """Thermal properties of the material covering the whole surface."""
    window_material: ThermalMaterialConfig | None = None
    """
    Thermal properties of the source-carrying window material, or None for a
    scene that is genuinely single-material and is solved with no heat source.
    """

    def __post_init__(self) -> None:
        if self.target_length_scale is not None and self.target_length_scale <= 0.0:
            raise ValueError(
                f"`target_length_scale` must be positive if provided, got "
                f"{self.target_length_scale}."
            )

    @property
    def is_two_material(self) -> bool:
        """Whether a facade mask splits the surface into base and window materials."""
        return self.window_material is not None
