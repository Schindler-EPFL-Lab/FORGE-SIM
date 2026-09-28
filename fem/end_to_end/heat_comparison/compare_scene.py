"""
IGA-vs-FEM heat validation for one reconstructed VGGT / BuildNet3D scene.

This is the per-scene cross-validator behind the paper's "same geometry" (V) heat
numbers. It re-runs the DOLFINx FEM heat solve on the same inpainted spline the IGA
run used, snapshots the solver at the IGA save times, and reports the per-timestep
relative L2 / Linf against the IGA reference field. Because both solvers integrate
the same scenario on the same surface, the residual is a measure of the two
discretizations' agreement rather than of the reconstruction, which is what makes
the comparison self-validating.

`SceneHeatComparison` (`fem/end_to_end/heat_comparison/`) does the work; this module
is its entry point, taking one scene's paths and materials as CLI arguments rather
than a scene identity: the registry of which scenes exist and their recorded
materials is `main_scripts`' own subject (`CLAUDE.md` SS3), not this library's;
`main_scripts/compare_all_scenes.py` is what resolves a registered
scene into the arguments below and calls this once per scene.

Artifacts land in `<dataset-root>/<directory>/<scene-subdir>/fem_comparison/`: a
per-timestep table (`comparison_L2.txt`), the FEM solution (`fem_solution.xdmf`)
and the IGA-minus-FEM difference field (`difference.xdmf`).

Run from the repo root in the `dolfinx-forge-sim` conda environment::

    python fem/end_to_end/heat_comparison/compare_scene.py \\
        --dataset-root <DATASET_ROOT> --directory buildnet-1 \\
        --two-material --window-material-config.conductivity 0.9 \\
        --window-material-config.density 2500.0 \\
        --window-material-config.specific-heat 840.0 \\
        --window-material-config.thickness 0.01
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import mitsuba as mi
import tyro

# Repository root: makes `forge_sim` and this script's own `fem` package
# importable when it is run by path rather than as `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from fem.end_to_end.heat_comparison.heat_comparison_scene_config import (
    HeatComparisonSceneConfig,
)
from fem.end_to_end.heat_comparison.scene_heat_comparison import SceneHeatComparison
from forge_sim import logger
from forge_sim.scripts.configs.thermal_material_config import ThermalMaterialConfig


@dataclass(kw_only=True)
class CompareSceneParameters(ReverseCli):
    """
    One scene's paths, materials and FEM discretization.

    The defaults are the configuration the paper reports for the VGGT scenes: P1
    FEM at 512x512 quads per patch with a direct MUMPS factorization, 500
    Crank-Nicolson steps and 11 snapshots.
    """

    dataset_root: Path
    """Root directory holding one subdirectory per scene (e.g. a `VGGT-SAM3` tree)."""
    directory: Path
    """This scene's subdirectory under `dataset_root`."""
    scene_subdir: Path = Path("heat_simulation")
    """
    Per-scene subdirectory holding the inpainted splines and the IGA reference run.
    """
    iga_base_name: str = "heat_simulation"
    """Stem of the IGA reference heat run's `.h5` / `.xdmf` pair."""

    tessellation_subdivisions: int = 512
    """Subdivisions per patch per direction (2 * value^2 triangles per patch)."""
    num_time_steps: int = 500
    """Number of Crank-Nicolson steps spanning the IGA run's final time."""
    save_interval: int = 50
    """Steps between snapshots; must divide the IGA run's own snapshot cadence."""

    target_length_scale: float | None = None
    """Longest control-point bbox axis [m] to rescale the scene to, or None."""

    base_material: ThermalMaterialConfig = field(default_factory=ThermalMaterialConfig)
    """Thermal properties of the material covering the whole surface."""
    two_material: bool = False
    """Whether the scene carries a facade mask splitting base and window materials."""
    window_material_config: ThermalMaterialConfig = field(
        default_factory=ThermalMaterialConfig
    )
    """Thermal properties of the window material; read only when `two_material`."""
    window_target_delta_temp: float | None = 0.5
    """
    Temperature rise [K] the window material is scaled to reach; must match what
    the compared IGA run used, and is read only when `two_material`.
    """

    def __post_init__(self) -> None:
        if not self.dataset_root.exists():
            raise ValueError(f"Dataset root {self.dataset_root} does not exist.")
        if not self.scene_subdir.name:
            raise ValueError("`scene_subdir` must be non-empty.")
        if self.tessellation_subdivisions < 1:
            raise ValueError(
                f"`tessellation_subdivisions` must be a positive integer, got "
                f"{self.tessellation_subdivisions}."
            )
        if self.num_time_steps < 1:
            raise ValueError(
                f"`num_time_steps` must be a positive integer, got "
                f"{self.num_time_steps}."
            )
        if self.save_interval < 1:
            raise ValueError(
                f"`save_interval` must be a positive integer, got {self.save_interval}."
            )

    @property
    def scene(self) -> HeatComparisonSceneConfig:
        """This scene's facts, as `SceneHeatComparison` reads them."""
        return HeatComparisonSceneConfig(
            directory=self.directory,
            iga_base_name=self.iga_base_name,
            target_length_scale=self.target_length_scale,
            base_material=self.base_material,
            window_material=self.window_material_config if self.two_material else None,
        )


def main(params: CompareSceneParameters) -> None:
    """Re-solves the scene under the FEM and logs its agreement with the IGA run."""
    # The Mitsuba variant is a process-global; the launching script sets it once, before
    # any spline is loaded, and no module sets it on import.
    mi.set_variant("llvm_ad_mono")

    comparison = SceneHeatComparison(
        scene=params.scene,
        dataset_root=params.dataset_root,
        scene_subdir=params.scene_subdir,
        tessellation_subdivisions=params.tessellation_subdivisions,
        num_time_steps=params.num_time_steps,
        save_interval=params.save_interval,
        window_target_delta_temp=params.window_target_delta_temp,
    ).run()
    logger.info(f"{comparison.name}: {comparison.summary()}")


if __name__ == "__main__":
    main(params=tyro.cli(CompareSceneParameters))
