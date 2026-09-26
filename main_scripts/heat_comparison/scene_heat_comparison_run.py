import subprocess
from pathlib import Path
from typing import ClassVar

from main_scripts.compare_all_scenes import CompareAllScenesParameters
from main_scripts.conda_python import CondaPython
from main_scripts.scenes.vggt_scene import VGGTScene
from main_scripts.scenes.vggt_scene_profile import VGGTSceneProfile
from rebel_shape import logger
from rebel_shape.scripts.end_to_end.data_paths import REPO_ROOT


class SceneHeatComparisonRun:
    """
    One scene's sweep step: resolve its registered profile, run the FEM
    comparison for it under the DOLFINx conda environment.

    Construction resolves `scene`'s `VGGTSceneProfile`; it touches no disk. `run`
    spawns `fem/end_to_end/heat_comparison/compare_scene.py`, which does the
    solve, the diff and the write; this class only resolves the scene and builds
    the argv, mirroring `SceneModalComparisonRun`
    (`main_scripts/modal_comparison/scene_modal_comparison_run.py`) on the
    modal side.
    """

    COMPARISON_SCRIPT: ClassVar[Path] = (
        REPO_ROOT / "fem" / "end_to_end" / "heat_comparison" / "compare_scene.py"
    )
    """The one-scene FEM entry point every scene's comparison is spawned from."""

    def __init__(
        self, scene: VGGTScene, *, params: CompareAllScenesParameters
    ) -> None:
        """
        Resolves `scene`'s `VGGTSceneProfile`; touches no disk.

        :raises ValueError: If `scene` has no `SCENE_PROFILES` entry.
        """
        self._scene = scene
        self._profile = VGGTSceneProfile.for_scene(scene)
        self._params = params

    def run(self) -> None:
        """
        Runs the FEM comparison for this scene under the conda interpreter.

        The comparison's own artifacts, including its summary line, land under
        `<dataset_root>/<directory>/<scene_subdir>/fem_comparison/`; see
        `fem/end_to_end/heat_comparison/scene_heat_comparison.py`.

        :raises ValueError: If `CondaPython.ENV_VAR` is unset or names a path
            that does not exist.
        :raises subprocess.CalledProcessError: If the FEM comparison exits
            non-zero.
        """
        logger.info(f"==== {self._profile.directory} ====")
        subprocess.run(self._command(), check=True, cwd=str(REPO_ROOT))

    def _command(self) -> list[str]:
        """The argv `run` spawns: this scene's paths, materials and discretization."""
        thermal = self._profile.thermal
        command = [
            str(CondaPython.resolve_from_environment()),
            str(self.COMPARISON_SCRIPT),
            "--dataset-root", str(self._params.dataset_root),
            "--directory", str(self._profile.directory),
            "--scene-subdir", str(self._params.scene_subdir),
            "--iga-base-name", self._profile.iga_base_name,
            "--tessellation-subdivisions",
            str(self._params.tessellation_subdivisions),
            "--num-time-steps", str(self._params.num_time_steps),
            "--save-interval", str(self._params.save_interval),
            "--target-length-scale", thermal.target_length_scale_argument,
            "--base-material.conductivity", str(thermal.base_material.conductivity),
            "--base-material.density", str(thermal.base_material.density),
            "--base-material.specific-heat", str(thermal.base_material.specific_heat),
            "--base-material.thickness", str(thermal.base_material.thickness),
            "--window-target-delta-temp", str(thermal.window_target_delta_temp),
        ]
        if thermal.window_material is not None:
            window = thermal.window_material
            command += [
                "--two-material",
                "--window-material-config.conductivity", str(window.conductivity),
                "--window-material-config.density", str(window.density),
                "--window-material-config.specific-heat", str(window.specific_heat),
                "--window-material-config.thickness", str(window.thickness),
            ]
        return command
