"""
FEM-vs-IGA modal validation for one reconstructed VGGT / BuildNet3D scene, on the
SAME spline (the modal analog of the heat comparison).

Runs the DOLFINx geometrically-exact FEM shell modal solver on the scene's
reconstruction spline, at the shell material and geometry scale the compared IGA
run used, then diffs the elastic spectrum against the already-computed IGA modal
result. This isolates pure code agreement (both solvers on the same geometry)
from the geometry-recovery gap that a ground-truth-mesh-vs-spline comparison
conflates.

`SceneModalComparison` (`fem/end_to_end/modal_comparison/`) does the work; this
module is its entry point, taking one scene's paths and material as CLI
arguments rather than a scene identity: the registry of which scenes exist and
their recorded shell material is `main_scripts`' own subject (`CLAUDE.md`
SS3), not this library's: `main_scripts/compare_modal_fem_iga.py` is what
resolves a registered scene into the arguments below and calls this once per
scene.

Artifacts land in `<dataset-root>/<directory>/modal_analysis/fem_comparison/`:
the JSON and markdown comparison reports (`modal_comparison_fem_iga.json`,
`modal_comparison_fem_iga.md`) and the FEM solver's own
`modal_analysis_results.npz` and mode-shape XDMF.

Run from the repo root in the `dolfinx-forge-sim` conda environment::

    python fem/end_to_end/modal_comparison/compare_scene.py \\
        --dataset-root <DATASET_ROOT> --directory buildnet-1 \\
        --spline-path <DATASET_ROOT>/buildnet-1/results/shape_0100.json \\
        --material-config.youngs-modulus 40e9 --material-config.poisson-ratio 0.2 \\
        --material-config.density 2400.0 --material-config.thickness 0.2 \\
        --material-config.shear-correction-factor 0.8333
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

from fem.end_to_end.modal_comparison.modal_comparison_scene_config import (
    ModalComparisonSceneConfig,
)
from fem.end_to_end.modal_comparison.scene_modal_comparison import SceneModalComparison
from forge_sim import logger
from forge_sim.scripts.configs.structural_material_config import (
    StructuralMaterialConfig,
)


@dataclass(kw_only=True)
class CompareSceneParameters(ReverseCli):
    """
    One scene's paths, shell material and FEM discretization.

    The defaults are the configuration the paper reports for the VGGT scenes:
    P3 fields on a P2 (curved) mesh at 64 subdivisions per patch, 20 modes.
    """

    dataset_root: Path
    """Root directory holding one subdirectory per scene (e.g. a `VGGT-SAM3` tree)."""
    directory: Path
    """This scene's subdirectory under `dataset_root`."""
    spline_path: Path
    """Reconstruction spline both this FEM solve and the compared IGA run used."""

    target_length_scale: float | None = None
    """Longest control-point bbox axis [m] to rescale the scene to, or None."""

    num_modes: int = 20
    """Modes the FEM solver is asked for, rigid-body modes included."""
    geometry_order: int = 2
    """Lagrange degree of the FEM mesh geometry, so the shell keeps its curvature."""
    field_degree: int = 3
    """Lagrange degree of the FEM displacement and rotation fields."""
    subdivisions_per_patch: int = 64
    """Tessellation of the FEM mesh, per patch per direction."""
    rb_freq_hz: float = 1.0
    """Modes at or below this Hz are rigid-body; dropped before the elastic compare."""

    material_config: StructuralMaterialConfig = field(
        default_factory=StructuralMaterialConfig
    )
    """Uniform shell material (E, nu, rho, thickness, shear correction)."""

    def __post_init__(self) -> None:
        if not self.dataset_root.exists():
            raise ValueError(f"Dataset root {self.dataset_root} does not exist.")
        if not self.directory.name:
            raise ValueError("`directory` must be non-empty.")
        if not self.spline_path.exists():
            raise ValueError(f"Spline path {self.spline_path} does not exist.")
        if self.num_modes < 1:
            raise ValueError(
                f"`num_modes` must be a positive integer, got {self.num_modes}."
            )
        if self.geometry_order < 1:
            raise ValueError(
                f"`geometry_order` must be a positive integer, got "
                f"{self.geometry_order}."
            )
        if self.field_degree < 1:
            raise ValueError(
                f"`field_degree` must be a positive integer, got {self.field_degree}."
            )
        if self.subdivisions_per_patch < 1:
            raise ValueError(
                f"`subdivisions_per_patch` must be a positive integer, got "
                f"{self.subdivisions_per_patch}."
            )
        if self.rb_freq_hz <= 0.0:
            raise ValueError(
                f"`rb_freq_hz` must be a positive real number, got {self.rb_freq_hz}."
            )

    @property
    def scene(self) -> ModalComparisonSceneConfig:
        """This scene's facts, as `SceneModalComparison` reads them."""
        return ModalComparisonSceneConfig(
            directory=self.directory,
            spline_path=self.spline_path,
            target_length_scale=self.target_length_scale,
            structural_material=self.material_config,
        )


def main(params: CompareSceneParameters) -> None:
    """Re-solves the scene under the FEM and logs its agreement with the IGA run."""
    # The Mitsuba variant is a process-global; the launching script sets it once,
    # before any spline is loaded, and no module sets it on import.
    mi.set_variant("llvm_ad_mono")

    report = SceneModalComparison(
        scene=params.scene,
        dataset_root=params.dataset_root,
        num_modes=params.num_modes,
        geometry_order=params.geometry_order,
        field_degree=params.field_degree,
        subdivisions_per_patch=params.subdivisions_per_patch,
        rb_freq_hz=params.rb_freq_hz,
    ).run()
    logger.info(
        f"{report.scene}: {report.num_compared_elastic_modes} elastic, mean "
        f"{report.mean_rel_error * 100:.3f}% max {report.max_rel_error * 100:.3f}%"
    )


if __name__ == "__main__":
    main(params=tyro.cli(CompareSceneParameters))
