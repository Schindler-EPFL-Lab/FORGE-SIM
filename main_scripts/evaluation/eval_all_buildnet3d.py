"""
Batch novel-view-synthesis and Chamfer evaluation of splines reconstructed by
the proposed forge-sim method, across the BuildNet3D scenes.

`--scenes-ids` takes `BuildNet3DSceneID` members; tyro renders them by member
NAME, not by value, so a command line spells `--scenes-ids SCENE_1` where it
used to spell `--scenes-ids 1`.
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forge_sim import logger  # noqa: E402
from forge_sim.scripts.evaluation.spline_texture_paths import (  # noqa: E402
    SplineTexturePaths,
)
from main_scripts.evaluation.buildnet3d_scene_id import (  # noqa: E402
    BuildNet3DSceneID,
)
from main_scripts.evaluation.buildnet3d_scene_target import (  # noqa: E402
    BuildNet3DSceneTarget,
)
from main_scripts.evaluation.eval_all_common import (  # noqa: E402
    prepare_metrics_output_folder,
)


@dataclass(kw_only=True)
class EvalAllBuildNet3DParameters(ReverseCli):
    """
    Which BuildNet3D scenes to evaluate, and at what environment map.

    For each scene, spawns `forge_sim/scripts/render_eval_images.py` then
    `forge_sim/scripts/evaluation/eval_buildnet3d.py` once per reconstructed
    spline under `validation_inputs_root`/splines/<scene>/, then
    `forge_sim/scripts/evaluation/aggregate_metrics.py` once to average the
    per-spline metrics into that scene's mean.
    """

    scenes_ids: list[BuildNet3DSceneID]
    """BuildNet3D scenes to validate on."""
    validation_inputs_root: Path
    """
    Root directory holding the reconstructions to validate. Must contain a
    `data/` subdirectory with one folder per scene and its own
    `BuildNet3DCorrectedMeshes/` subdirectory of ground-truth meshes, and a
    `splines/` subdirectory with one folder per scene, holding that scene's
    spline reconstructions and their cube-face textures.
    """
    environment_map_path: Path = Path("./data/textures/checkerboard_lighter.exr")
    """Environment map used as background during training."""
    output_folder_name: str = "splines-metrics"
    """Name of the output folder created under `validation_inputs_root`."""

    def __post_init__(self) -> None:
        """
        :raises FileNotFoundError: If `validation_inputs_root`/data,
            `validation_inputs_root`/splines,
            `validation_inputs_root`/data/BuildNet3DCorrectedMeshes, or
            `environment_map_path` does not exist.
        """
        dataset_root = self.validation_inputs_root / "data"
        splines_root = self.validation_inputs_root / "splines"
        ground_truth_meshes_root = dataset_root / "BuildNet3DCorrectedMeshes"
        if not splines_root.exists():
            raise FileNotFoundError(
                f"The `splines_root` {splines_root} does not exist."
            )
        if not dataset_root.exists():
            raise FileNotFoundError(
                f"The `dataset_root` {dataset_root} does not exist."
            )
        if not ground_truth_meshes_root.exists():
            raise FileNotFoundError(
                f"The `ground_truth_meshes_root` {ground_truth_meshes_root} "
                "does not exist."
            )
        if not self.environment_map_path.exists():
            raise FileNotFoundError(
                f"The `environment_map_path` {self.environment_map_path} does "
                "not exist."
            )


def main(params: EvalAllBuildNet3DParameters) -> None:
    """
    Evaluates every selected scene, writing each scene's mean metrics.

    :raises RuntimeError: If a scene's reconstruction directory holds a
        different number of splines than of texture sets.
    :raises ValueError: If a scene's cube-face textures match different
        numbers of files per face (from `SplineTexturePaths.discover_all`).
    :raises FileExistsError: If a scene's output folder already holds the
        metrics of an earlier run.
    :raises subprocess.CalledProcessError: If a render, evaluation or
        aggregation subprocess fails.
    """
    dataset_root = params.validation_inputs_root / "data"
    ground_truth_meshes_root = dataset_root / "BuildNet3DCorrectedMeshes"
    splines_root = params.validation_inputs_root / "splines"
    output_root = params.validation_inputs_root / params.output_folder_name
    output_root.mkdir(exist_ok=True, parents=True)

    for scene_id in params.scenes_ids:
        target = BuildNet3DSceneTarget.for_id(scene_id)

        splines_paths = sorted((splines_root / target.directory).glob("*.json"))
        splines_textures_paths = SplineTexturePaths.discover_all(
            splines_root / target.directory
        )

        if len(splines_textures_paths) != len(splines_paths):
            raise RuntimeError(
                "The number of textures and splines must be the same but found "
                + f"{len(splines_paths)} splines {splines_paths} and "
                + f"{len(splines_textures_paths)} textures {splines_textures_paths}."
            )

        thermoscenes_path = dataset_root / target.directory / target.directory

        output_folders: list[Path] = []
        for idx, spline_path, spline_texture_path in zip(
            range(len(splines_paths)), splines_paths, splines_textures_paths
        ):
            logger.info(f"Evaluating {spline_path} {spline_texture_path.as_list_str()}")
            output_folder = output_root / f"{target.directory}_{idx}"
            prepare_metrics_output_folder(
                output_folder, "eval_nvs.json", "eval_shape.json"
            )
            output_folders.append(output_folder)
            output_images_folder = output_folder / "images"
            output_images_folder.mkdir(exist_ok=True, parents=True)

            cmd = [
                "python3",
                "forge_sim/scripts/render_eval_images.py",
                "--scene-path",
                str(thermoscenes_path),
                "--environment-map-path",
                str(params.environment_map_path),
                "--spline-path",
                str(spline_path),
                "--textures-paths",
                *spline_texture_path.as_list_str(),
                "--output-folder",
                str(output_images_folder),
                "--no-use-gpu",
            ]
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

            cmd = [
                "python3",
                "forge_sim/scripts/evaluation/eval_buildnet3d.py",
                "--thermoscenes-scene-path",
                str(thermoscenes_path),
                "--environment-map-path",
                str(params.environment_map_path),
                "--rendered-images-folder",
                str(output_images_folder),
                "--spline-path",
                str(spline_path),
                "--reference-shape-path",
                str(ground_truth_meshes_root / target.ground_truth_mesh_filename),
                "--output-folder",
                str(output_folder),
            ]
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

        cmd = [
            "python3",
            "forge_sim/scripts/evaluation/aggregate_metrics.py",
            "--metrics-folders",
            *map(str, output_folders),
            "--metrics-file-names",
            "eval_nvs.json",
            "eval_shape.json",
            "--output-file-path",
            str(output_root / f"{target.directory}_aggregated.json"),
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main(params=tyro.cli(EvalAllBuildNet3DParameters))
