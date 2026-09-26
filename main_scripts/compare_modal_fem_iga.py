"""FEM-vs-IGA modal validation on the SAME spline (the modal analog of the heat study).

For each VGGT scene, spawns `fem/end_to_end/modal_comparison/compare_scene.py` under
the DOLFINx conda environment, which re-runs the geometrically-exact FEM shell modal
solver on the scene's reconstruction spline, matched to the IGA run via
`VGGTSceneProfile.SCENE_PROFILES` (the shell material, and the geometry scale it
carries in its shared `VGGTSceneThermalProfile`), then diffs the elastic spectrum
against the already-computed IGA modal result. This isolates pure CODE agreement (both
solvers on the same geometry) from the geometry-recovery gap that the
GT-mesh-vs-spline comparison conflates.

The FEM solver runs in the conda `dolfinx-rebel-shape` env (subprocess, one per
scene); this driver runs in the .venv, mirroring `compare_all_scenes.py` beside it on
the heat side. `SceneModalComparisonRun`
(`main_scripts/modal_comparison/scene_modal_comparison_run.py`) resolves each
scene's `VGGTSceneProfile` and builds the spawned command; the solve, the diff and the
write are `SceneModalComparison`'s, entirely inside `fem/`.

Output per scene: `<scene>/modal_analysis/fem_comparison/`. There is no aggregate
table across scenes here, matching `compare_all_scenes.py`: each scene's one-line
summary, logged by its own subprocess, is the record of the run.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main_scripts.scenes.vggt_scene import VGGTScene  # noqa: E402
from main_scripts.scenes.vggt_scene_profile import VGGTSceneProfile  # noqa: E402
from rebel_shape import logger  # noqa: E402


@dataclass(kw_only=True)
class ModalComparisonParameters(ReverseCli):
    """
    Which scenes to compare, and the FEM discretisation to compare them at.

    The IGA side is never re-solved: its spectrum is read from the modal run already
    sitting in the scene directory, so every knob here describes the FEM run alone.
    The defaults are the ones the published modal table was produced with.
    """

    DATASET_ROOT_ENV_VAR: ClassVar[str] = "REBEL_SHAPE_VGGT_DATASET_ROOT"
    """
    Environment variable pointing at the VGGT dataset root, the directory holding one
    folder per scene. Used as the `--dataset-root` default, so the tree this driver
    reads and writes into is named by the environment rather than by one machine's
    layout.
    """

    dataset_root: Path | None = None
    """
    Root directory holding one folder per scene. Defaults to the
    `REBEL_SHAPE_VGGT_DATASET_ROOT` environment variable.
    """
    scenes: list[VGGTScene] = field(default_factory=list)
    """Scenes to compare; empty selects all of them."""

    subdivisions_per_patch: int = 64
    """Tessellation of the FEM mesh, per patch per direction."""
    field_degree: int = 3
    """Lagrange degree of the FEM displacement and rotation fields."""
    geometry_order: int = 2
    """Lagrange degree of the FEM mesh geometry, so the shell keeps its curvature."""
    num_modes: int = 20
    """Modes the FEM solver is asked for, rigid-body modes included."""
    rb_freq_hz: float = 1.0
    """Modes at or below this Hz are rigid-body; dropped before the elastic compare."""

    def __post_init__(self) -> None:
        """
        Validates the selection and the FEM discretisation knobs.

        :raises ValueError: If any of the discretisation, mode-count or cutoff knobs
            is not positive.
        """
        if self.subdivisions_per_patch < 1:
            raise ValueError(
                f"`subdivisions_per_patch` must be a positive integer, got "
                f"{self.subdivisions_per_patch}."
            )
        if self.field_degree < 1:
            raise ValueError(
                f"`field_degree` must be a positive integer, got {self.field_degree}."
            )
        if self.geometry_order < 1:
            raise ValueError(
                f"`geometry_order` must be a positive integer, got "
                f"{self.geometry_order}."
            )
        if self.num_modes < 1:
            raise ValueError(
                f"`num_modes` must be a positive integer, got {self.num_modes}."
            )
        if self.rb_freq_hz <= 0.0:
            raise ValueError(
                f"`rb_freq_hz` must be a positive real number, got {self.rb_freq_hz}."
            )

    @property
    def selected_scenes(self) -> list[VGGTScene]:
        """The registered scenes filtered by `scenes`, in the registry's own order."""
        if not self.scenes:
            return list(VGGTSceneProfile.SCENE_PROFILES)
        selected = set(self.scenes)
        return [scene for scene in VGGTSceneProfile.SCENE_PROFILES if scene in selected]

    @property
    def resolved_dataset_root(self) -> Path:
        """
        Resolve `dataset_root`, falling back to `DATASET_ROOT_ENV_VAR`.

        :raises ValueError: If `dataset_root` was not given and the environment
            variable is not set either, leaving the dataset unlocated; or if the
            resolved path is not a directory.
        """
        root = self.dataset_root
        if root is None:
            value = os.environ.get(self.DATASET_ROOT_ENV_VAR)
            if value is None:
                raise ValueError(
                    f"--dataset-root not given and {self.DATASET_ROOT_ENV_VAR} is "
                    "not set; point it at the directory holding one folder per "
                    "VGGT scene."
                )
            root = Path(value)
        if not root.is_dir():
            raise ValueError(f"`dataset_root` is not a directory: {root}")
        return root


def main(params: ModalComparisonParameters) -> None:
    """
    Compares every selected scene.

    A scene whose profile or inputs are missing aborts the run rather than being
    skipped: a summary that silently omits a scene reads exactly like one where that
    scene was never asked for. This is the same rule its heat-side counterpart
    `main_scripts/compare_all_scenes.py` follows.

    :raises ValueError: If the dataset root is neither given nor named by the
        environment; if `CondaPython.ENV_VAR` does not resolve to an interpreter; or
        if a selected scene has no `SCENE_PROFILES` entry, or its entry names no
        `structural_material`.
    :raises subprocess.CalledProcessError: If a scene's FEM comparison exits non-zero.
    """
    # Deferred: `SceneModalComparisonRun` type-hints its `params` argument with
    # `ModalComparisonParameters`, so importing it at module scope here would cycle.
    from main_scripts.modal_comparison.scene_modal_comparison_run import (
        SceneModalComparisonRun,
    )

    dataset_root = params.resolved_dataset_root

    logger.info("===== FEM-vs-IGA modal validation (same spline) =====")
    for scene in params.selected_scenes:
        SceneModalComparisonRun(scene, dataset_root=dataset_root, params=params).run()


if __name__ == "__main__":
    main(tyro.cli(ModalComparisonParameters))
