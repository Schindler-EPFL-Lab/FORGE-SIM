"""
Batch novel-view-synthesis and Chamfer evaluation of meshes reconstructed by
the LargeSteps method, across the BuildNet3D scenes.

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
from forge_sim.scripts.render_eval_images_mesh import (  # noqa: E402
    discover_mesh_textures,
    face_key,
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
class EvalAllBuildNet3DLargeStepsParameters(ReverseCli):
    """
    Which BuildNet3D scenes to evaluate, and at what environment map.

    Each scene's reconstruction folder holds one reconstruction, whose mesh
    and texture `render_eval_images_mesh.py` discovers for itself, so the
    folder is handed over whole, and each scene is evaluated once. The
    scene's metrics are written to
    `validation_inputs_root`/`output_folder_name`/<scene>_metrics.json.
    """

    scenes_ids: list[BuildNet3DSceneID]
    """BuildNet3D scenes to validate on."""
    validation_inputs_root: Path
    """
    Root directory holding the reconstructions to validate. Must contain a
    `data/` subdirectory with one folder per scene and its own
    `BuildNet3DCorrectedMeshes/` subdirectory of ground-truth meshes, and a
    `large-steps/` subdirectory with one folder per scene, holding that
    scene's reconstruction.
    """
    environment_map_path: Path = Path("./data/textures/overcast_soil_2_2k.exr")
    """Environment map used as background during training."""
    output_folder_name: str = "large-steps-metrics"
    """Name of the output folder created under `validation_inputs_root`."""

    def __post_init__(self) -> None:
        """
        :raises FileNotFoundError: If `validation_inputs_root`/data,
            `validation_inputs_root`/large-steps,
            `validation_inputs_root`/data/BuildNet3DCorrectedMeshes, or
            `environment_map_path` does not exist.
        """
        dataset_root = self.validation_inputs_root / "data"
        large_steps_root = self.validation_inputs_root / "large-steps"
        ground_truth_meshes_root = dataset_root / "BuildNet3DCorrectedMeshes"
        if not large_steps_root.exists():
            raise FileNotFoundError(
                f"The `large_steps_root` {large_steps_root} does not exist."
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


def main(params: EvalAllBuildNet3DLargeStepsParameters) -> None:
    """
    Evaluates every selected scene, writing each scene's metrics.

    The Chamfer step reads one mesh file, so its path is resolved here rather
    than in the subprocess: a composite has no single file to hand it and
    must be caught before anything is rendered.

    :raises FileNotFoundError: If a scene's reconstruction folder holds no
        mesh (from `discover_mesh_textures`).
    :raises ValueError: If a scene's reconstruction folder is ambiguous (from
        `discover_mesh_textures`).
    :raises RuntimeError: If a scene's reconstruction folder holds cube-face
        patches rather than a single mesh, which the Chamfer step cannot
        read.
    :raises FileExistsError: If a scene's output folder already holds the
        metrics of an earlier run.
    :raises subprocess.CalledProcessError: If a render, evaluation or
        aggregation subprocess fails.
    """
    dataset_root = params.validation_inputs_root / "data"
    ground_truth_meshes_root = dataset_root / "BuildNet3DCorrectedMeshes"
    large_steps_root = params.validation_inputs_root / "large-steps"
    output_root = params.validation_inputs_root / params.output_folder_name
    output_root.mkdir(exist_ok=True, parents=True)

    for scene_id in params.scenes_ids:
        target = BuildNet3DSceneTarget.for_id(scene_id)
        result_path = large_steps_root / target.directory
        thermoscenes_path = dataset_root / target.directory / target.directory

        mesh_textures = discover_mesh_textures(result_path)
        patch_keys = [key for key in mesh_textures if face_key(key) is not None]
        if patch_keys:
            raise RuntimeError(
                f"{result_path} holds the cube-face patches of a composite "
                f"({', '.join(patch_keys)}), but `eval_buildnet3d_mesh.py` measures "
                "the Chamfer distance of a single mesh file. Weld the patches and "
                "evaluate with `stitch_and_eval_cubed_sphere_mesh.py`, which takes "
                "the patch directory instead."
            )
        mesh_path, _ = next(iter(mesh_textures.values()))

        logger.info(f"Evaluating the reconstruction in {result_path}")
        output_folder = output_root / target.directory
        prepare_metrics_output_folder(output_folder, "eval_nvs.json", "eval_shape.json")
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
            "forge_sim/scripts/evaluation/eval_buildnet3d_mesh.py",
            "--thermoscenes-scene-path",
            str(thermoscenes_path),
            "--environment-map-path",
            str(params.environment_map_path),
            "--rendered-images-folder",
            str(output_images_folder),
            "--mesh-path",
            str(mesh_path),
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
            str(output_folder),
            "--metrics-file-names",
            "eval_nvs.json",
            "eval_shape.json",
            "--output-file-path",
            str(output_root / f"{target.directory}_metrics.json"),
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main(params=tyro.cli(EvalAllBuildNet3DLargeStepsParameters))
