"""
Batch novel-view-synthesis evaluation of meshes reconstructed by the
LargeSteps method, across the real (Thermoscenes) scenes.

`--scenes-names` takes `RealScene` members; tyro renders them by member NAME, not
by value, so a command line spells `--scenes-names CAR` where it used to spell
`--scenes-names extracted-car`.
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forge_sim import logger  # noqa: E402
from main_scripts.evaluation.eval_all_common import (  # noqa: E402
    prepare_metrics_output_folder,
)
from main_scripts.evaluation.real_scene import RealScene  # noqa: E402


@dataclass(kw_only=True)
class EvalAllRealLargeStepsParameters(ReverseCli):
    """
    Which real scenes to evaluate, and at what environment map.

    Each scene's reconstruction folder holds one reconstruction, whose mesh and
    texture `render_eval_images_mesh.py` discovers for itself, so the folder
    is handed over whole, and each scene is evaluated once. The scene's metrics
    are written to
    `validation_inputs_root`/`output_folder_name`/<scene>_metrics.json.
    """

    scenes_names: list[RealScene]
    """Real scenes to validate on."""
    validation_inputs_root: Path
    """
    Root directory holding the reconstructions to validate. Must contain a
    `data/` subdirectory with one folder per scene (e.g. downloaded from
    Azure using `download_dataset`) and a `large-steps/` subdirectory with
    one folder per scene, holding that scene's reconstruction.
    """
    environment_map_path: Path = Path("./data/textures/overcast_soil_2_2k.exr")
    """Environment map used as background during training."""
    output_folder_name: str = "large-steps-metrics"
    """Name of the output folder created under `validation_inputs_root`."""

    def __post_init__(self) -> None:
        """
        :raises FileNotFoundError: If `validation_inputs_root`/data,
            `validation_inputs_root`/large-steps, or `environment_map_path`
            does not exist.
        """
        dataset_root = self.validation_inputs_root / "data"
        large_steps_root = self.validation_inputs_root / "large-steps"
        if not large_steps_root.exists():
            raise FileNotFoundError(
                f"The `large_steps_root` {large_steps_root} does not exist."
            )
        if not dataset_root.exists():
            raise FileNotFoundError(
                f"The `dataset_root` {dataset_root} does not exist."
            )
        if not self.environment_map_path.exists():
            raise FileNotFoundError(
                f"The `environment_map_path` {self.environment_map_path} does "
                "not exist."
            )


def main(params: EvalAllRealLargeStepsParameters) -> None:
    """
    Evaluates every selected scene, writing each scene's metrics.

    :raises FileExistsError: If a scene's output folder already holds the
        metrics of an earlier run.
    :raises subprocess.CalledProcessError: If a render, evaluation or
        aggregation subprocess fails.
    """
    dataset_root = params.validation_inputs_root / "data"
    large_steps_root = params.validation_inputs_root / "large-steps"
    output_root = params.validation_inputs_root / params.output_folder_name
    output_root.mkdir(exist_ok=True, parents=True)

    for scene in params.scenes_names:
        result_path = large_steps_root / scene.value
        thermoscenes_path = dataset_root / scene.value / scene.value

        logger.info(f"Evaluating the reconstruction in {result_path}")
        output_folder = output_root / scene.value
        prepare_metrics_output_folder(output_folder, "eval_nvs.json")
        output_images_folder = output_folder / "images"
        output_images_folder.mkdir(exist_ok=True, parents=True)

        cmd = [
            "python3",
            "forge_sim/scripts/render_eval_images_mesh.py",
            "--scene-path",
            str(thermoscenes_path),
            "--environment-map-path",
            str(params.environment_map_path),
            "--result-path",
            str(result_path),
            "--output-folder",
            str(output_images_folder),
            "--no-use-gpu",
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        cmd = [
            "python3",
            "forge_sim/scripts/evaluation/eval_real.py",
            "--scene-path",
            str(thermoscenes_path),
            "--environment-map-path",
            str(params.environment_map_path),
            "--rendered-images-folder",
            str(output_images_folder),
            "--output-folder",
            str(output_folder),
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        cmd = [
            "python3",
            "forge_sim/scripts/evaluation/aggregate_metrics.py",
            "--metrics-folders",
            str(output_folder),
            "--metrics-file-names",
            "eval_nvs.json",
            "--output-file-path",
            str(output_root / f"{scene.value}_metrics.json"),
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main(params=tyro.cli(EvalAllRealLargeStepsParameters))
