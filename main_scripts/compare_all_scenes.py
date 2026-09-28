"""
IGA-vs-FEM heat validation across the reconstructed VGGT / BuildNet3D scenes.

This is the batch cross-validator behind the paper's "same geometry" (V) heat numbers.
For each registered scene it spawns `fem/end_to_end/heat_comparison/compare_scene.py`
under the DOLFINx conda environment, which re-runs the FEM heat solve on the same
inpainted spline the IGA run used and reports the per-timestep relative L2 / Linf
against the IGA reference field. Because both solvers integrate the same scenario on
the same surface, the residual is a measure of the two discretizations' agreement
rather than of the reconstruction, which is what makes the comparison
self-validating.

The FEM solver runs in the conda `dolfinx-forge-sim` env (subprocess, one per
scene); this driver runs in the .venv, mirroring `compare_modal_fem_iga.py` beside it
on the modal side. `SceneHeatComparisonRun`
(`main_scripts/heat_comparison/scene_heat_comparison_run.py`) resolves each scene's
`VGGTSceneProfile` and builds the spawned command; the solve, the diff and the write
are `SceneHeatComparison`'s, entirely inside `fem/`.

Per-scene artifacts land in `<dataset_root>/<scene>/heat_simulation/fem_comparison/`:
a per-timestep table (`comparison_L2.txt`), the FEM solution (`fem_solution.xdmf`)
and the IGA-minus-FEM difference field (`difference.xdmf`).

Run from the repo root in the .venv. Every scene::

    python main_scripts/compare_all_scenes.py --dataset-root <DATASET_ROOT>

A subset, by scene name (repeat the value, do not comma-separate; tyro renders a
`VGGTScene` argument by its member NAME, not its value)::

    python main_scripts/compare_all_scenes.py \\
        --dataset-root <DATASET_ROOT> --scenes BUILDNET_1 BUILDNET_2

`<DATASET_ROOT>` is the directory holding one subdirectory per scene.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main_scripts.scenes.vggt_scene import VGGTScene  # noqa: E402
from main_scripts.scenes.vggt_scene_profile import VGGTSceneProfile  # noqa: E402


@dataclass(kw_only=True)
class CompareAllScenesParameters(ReverseCli):
    """
    Which scenes to compare, and at what discretization.

    The defaults are the configuration the paper reports for the VGGT scenes: P1 FEM at
    512x512 quads per patch with a direct MUMPS factorization, 500 Crank-Nicolson steps
    and 11 snapshots.
    """

    dataset_root: Path
    """Root directory holding one subdirectory per scene (e.g. a `VGGT-SAM3` tree)."""
    scenes: list[VGGTScene] = field(default_factory=list)
    """Scenes to compare; empty selects all of them."""
    scene_subdir: Path = Path("heat_simulation")
    """
    Per-scene subdirectory holding the inpainted splines and the IGA reference run.
    Every recorded comparison used `heat_simulation`; a later stage that reruns the
    IGA side into `heat_simulation_fixed` will need this pointed there instead.
    """

    tessellation_subdivisions: int = 512
    """Subdivisions per patch per direction (2 * value^2 triangles per patch)."""
    num_time_steps: int = 500
    """Number of Crank-Nicolson steps spanning the IGA run's final time."""
    save_interval: int = 50
    """Steps between snapshots; must divide the IGA run's own snapshot cadence."""

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
    def selected_scenes(self) -> list[VGGTScene]:
        """The registered scenes filtered by `scenes`, in the registry's own order."""
        if not self.scenes:
            return list(VGGTSceneProfile.SCENE_PROFILES)
        selected = set(self.scenes)
        return [scene for scene in VGGTSceneProfile.SCENE_PROFILES if scene in selected]


def main(params: CompareAllScenesParameters) -> None:
    """
    Compares every selected scene.

    A scene whose inputs are missing or malformed aborts the batch rather than being
    skipped: a summary that silently omits a scene reads exactly like one where that
    scene was never asked for.

    :raises ValueError: If `CondaPython.ENV_VAR` is unset or names a path that does
        not exist, or if a selected scene has no `SCENE_PROFILES` entry.
    :raises subprocess.CalledProcessError: If a scene's FEM comparison exits non-zero.
    """
    # Deferred: `SceneHeatComparisonRun` type-hints its `params` argument with
    # `CompareAllScenesParameters`, so importing it at module scope here would cycle.
    from main_scripts.heat_comparison.scene_heat_comparison_run import (
        SceneHeatComparisonRun,
    )

    for scene in params.selected_scenes:
        SceneHeatComparisonRun(scene, params=params).run()


if __name__ == "__main__":
    main(params=tyro.cli(CompareAllScenesParameters))
