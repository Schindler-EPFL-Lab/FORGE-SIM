import subprocess
from pathlib import Path
from typing import ClassVar

from main_scripts.compare_modal_fem_iga import ModalComparisonParameters
from main_scripts.conda_python import CondaPython
from main_scripts.scenes.vggt_scene import VGGTScene
from main_scripts.scenes.vggt_scene_layout import VGGTSceneLayout
from main_scripts.scenes.vggt_scene_profile import VGGTSceneProfile
from rebel_shape import logger
from rebel_shape.scripts.end_to_end.data_paths import REPO_ROOT


class SceneModalComparisonRun:
    """
    One scene's sweep step: resolve its registered profile, run the FEM
    comparison for it under the DOLFINx conda environment.

    Construction resolves `scene`'s `VGGTSceneProfile` and rejects a scene
    whose entry names no shell material, before anything is written; an
    archive surviving from an earlier run is then reported rather than passed
    over, the same order `run_vggt_pipeline._run_heat_sim` checks its own
    window-material requirement in. `run` spawns
    `fem/end_to_end/modal_comparison/compare_scene.py`, which does the solve,
    the diff and the write; this class only resolves the scene and builds the
    argv, mirroring `SceneHeatComparisonRun`
    (`main_scripts/heat_comparison/scene_heat_comparison_run.py`) on the
    heat side.
    """

    COMPARISON_SCRIPT: ClassVar[Path] = (
        REPO_ROOT / "fem" / "end_to_end" / "modal_comparison" / "compare_scene.py"
    )
    """The one-scene FEM entry point every scene's comparison is spawned from."""

    def __init__(
        self, scene: VGGTScene, *, dataset_root: Path, params: ModalComparisonParameters
    ) -> None:
        """
        Resolves `scene`'s `VGGTSceneProfile`; touches no disk.

        The profile names the scene's directory, which is how a nested layout
        such as `buildnet-4/buildnet-4` is reached without this class knowing
        the exception exists.

        :raises ValueError: If `scene` has no `SCENE_PROFILES` entry, or its
            entry names no `structural_material`; either way the shell and the
            length scale its IGA modal run used are unknown.
        """
        profile = VGGTSceneProfile.for_scene(scene)
        if profile.structural_material is None:
            raise ValueError(
                f"Scene '{scene.value}' names no `structural_material` in its "
                f"`SCENE_PROFILES` entry, so the shell its IGA modal run used is "
                f"unknown."
            )

        self._scene = scene
        self._params = params
        self._profile = profile
        self._dataset_root = dataset_root
        self._scene_dir = dataset_root / profile.directory

    def run(self) -> None:
        """
        Runs the FEM comparison for this scene under the conda interpreter.

        The comparison's own artifacts, including its summary log line, land
        under `<dataset_root>/<directory>/modal_analysis/fem_comparison/`; see
        `fem/end_to_end/modal_comparison/scene_modal_comparison.py`.

        :raises ValueError: If `CondaPython.ENV_VAR` is unset or names a path
            that does not exist.
        :raises FileNotFoundError: If the scene holds no reconstruction.
        :raises subprocess.CalledProcessError: If the FEM comparison exits
            non-zero.
        """
        logger.info(f"==== {self._profile.directory} ====")
        subprocess.run(self._command(), check=True, cwd=str(REPO_ROOT))

    def _command(self) -> list[str]:
        """The argv `run` spawns: this scene's paths, material and discretization."""
        reconstruction = VGGTSceneLayout(self._scene_dir).reconstruction
        material = self._profile.structural_material
        return [
            str(CondaPython.resolve_from_environment()),
            str(self.COMPARISON_SCRIPT),
            "--dataset-root", str(self._dataset_root),
            "--directory", str(self._profile.directory),
            "--spline-path", str(reconstruction),
            "--target-length-scale",
            self._profile.thermal.target_length_scale_argument,
            "--num-modes", str(self._params.num_modes),
            "--geometry-order", str(self._params.geometry_order),
            "--field-degree", str(self._params.field_degree),
            "--subdivisions-per-patch", str(self._params.subdivisions_per_patch),
            "--rb-freq-hz", str(self._params.rb_freq_hz),
            "--material-config.youngs-modulus", str(material.youngs_modulus),
            "--material-config.poisson-ratio", str(material.poisson_ratio),
            "--material-config.density", str(material.density),
            "--material-config.thickness", str(material.thickness),
            "--material-config.shear-correction-factor",
            str(material.shear_correction_factor),
        ]
