import json
import subprocess
from pathlib import Path

import numpy as np
import tyro

from rebel_shape import logger


def eval_all_buildnet3d_large_steps(
    scenes_ids: list[str],
    validation_inputs_root: Path,
    environment_map_path: Path = Path("./data/textures/overcast_soil_2_2k.exr"),
    output_folder_name: str = "large-steps-metrics",
):
    """
    Runs evaluation of BuildNet3D reconstructed using the LargeSteps method. The
    `scenes_ids` specifies the ids of BuildNet3D scenes to validate on, the
    `validation_inputs_root` specifies data related to rebel-shape validation, such as
    scenes, reconstruction meshes and textures. The `environment_map_path` specifies the
    path to the environmental map used during training. The `output_folder_name` defines
    the name of the output folder.

    The `validation_inputs_root` must contain folders:
        - data/ containing all BuildNet3D scenes (e.g. downloaded from Azure using
          download_dataset),
        - large-steps/ containing all BuildNet3D reconstructions with folders names
          match scenes in data/,
        - data/BuildNet3DCorrectedMeshes/ containing ground truth meshes.

    :raise:
        FileNotFoundError if:
            - The `validation_inputs_root`/data does not exist,
            - The `validation_inputs_root`/large-steps does not exist,
            - The `validation_inputs_root`/data/BuildNet3DCorrectedMeshes/ does not
            exist.
            - The `environment_map_path` does not exist,
        ValueError if `scenes_ids` is not a subset of {"1", "2", "3", "4"}.
        RuntimeError if the number of found meshes differs from the number of
        found textures.
    """

    dataset_root = validation_inputs_root / "data"
    ground_truth_meshes_root = dataset_root / "BuildNet3DCorrectedMeshes"
    large_steps_root = validation_inputs_root / "large-steps"
    output_root = validation_inputs_root / output_folder_name

    mapping_to_data = {
        "1": "extracted-buildnet3d-realistic-1",
        "2": "extracted-buildnet3d-realistic-2",
        "3": "extracted-buildnet3d-realistic-3",
        "4": "extracted-buildnet3d-realistic-4",
    }

    mapping_to_meshes = {
        "1": "building1",
        "2": "building2",
        "3": "building3",
        "4": "building4",
    }

    if not large_steps_root.exists():
        raise FileNotFoundError(
            f"The `large_steps_root` {large_steps_root} does not exist."
        )
    if not dataset_root.exists():
        raise FileNotFoundError(f"The `dataset_root` {dataset_root} does not exist.")
    if not ground_truth_meshes_root.exists():
        raise FileNotFoundError(
            f"The `ground_truth_meshes_root` {ground_truth_meshes_root} does not exist."
        )
    if not environment_map_path.exists():
        raise FileNotFoundError(
            f"The `environment_map_path` {environment_map_path} does not exist."
        )

    output_root.mkdir(exist_ok=True, parents=True)

    if not set(scenes_ids).issubset(mapping_to_data.keys()):
        raise ValueError(
            f"The `scenes_ids` must be a subset of {list(mapping_to_data.keys())}, "
            + f"but got {scenes_ids}, extra scenes are "
            + f"{set(scenes_ids) - set(list(mapping_to_data.keys()))}"
        )

    for scene_id in scenes_ids:
        scene_name_data = mapping_to_data[scene_id]
        ground_truth_mesh_name = mapping_to_meshes[scene_id]
        meshes_paths = list((large_steps_root / scene_name_data).glob("*.ply"))
        meshes_paths = sorted(meshes_paths)
        textures_paths = list((large_steps_root / scene_name_data).glob("*.exr"))
        textures_paths = sorted(textures_paths)
        if len(meshes_paths) != len(textures_paths):
            raise RuntimeError(
                "The number of textures and meshes must be the same but found "
                + f"{len(meshes_paths)} meshes {meshes_paths} and {len(textures_paths)}"
                + f" textures {textures_paths}."
            )

        thermoscenes_path = dataset_root / scene_name_data / scene_name_data

        aggregated_metrics: dict[str, list[float]] = {}
        for idx, mesh_path, texture_path in zip(
            range(len(meshes_paths)), meshes_paths, textures_paths
        ):
            logger.info(f"Evaluating {mesh_path} {texture_path}")
            output_folder = output_root / f"{scene_name_data}_{idx}"
            output_folder.mkdir(exist_ok=True)
            output_images_folder = output_folder / "images"
            output_images_folder.mkdir(exist_ok=True, parents=True)

            # render reconstruction from eval poses
            cmd = [
                "python3",
                "rebel_shape/scripts/render_eval_images_mesh.py",
                "--thermoscenes-scene-path",
                str(thermoscenes_path),
                "--environment-map-path",
                str(environment_map_path),
                "--mesh-path",
                str(mesh_path),
                "--texture-path",
                str(texture_path),
                "--output-folder",
                str(output_images_folder),
                "--no-use-gpu",
            ]
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd)

            # calculate NVS and Chamfer distance metrics
            cmd = [
                "python3",
                "rebel_shape/scripts/evaluation/eval_buildnet3d_mesh.py",
                "--thermoscenes-scene-path",
                str(thermoscenes_path),
                "--environment-map-path",
                str(environment_map_path),
                "--rendered-images-folder",
                str(output_images_folder),
                "--mesh-path",
                str(mesh_path),
                "--reference-shape-path",
                str(ground_truth_meshes_root / f"{ground_truth_mesh_name}.ply"),
                "--output-folder",
                str(output_folder),
            ]
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd)

            current_metrics: dict[str, float] = {}
            with open(output_folder / "eval_nvs.json") as f:
                current_metrics.update(json.load(f))
            with open(output_folder / "eval_shape.json") as f:
                current_metrics.update(json.load(f))

            for k in current_metrics:
                if k not in aggregated_metrics:
                    aggregated_metrics[k] = []
                aggregated_metrics[k].append(current_metrics[k])

        mean_aggregated: dict[str, float] = {}
        for k in aggregated_metrics:
            mean_aggregated[k] = np.mean(aggregated_metrics[k]).item()

        with open(output_root / f"{scene_name_data}_aggregated.json", "w") as f:
            json.dump(mean_aggregated, f, indent=4)


if __name__ == "__main__":
    tyro.cli(eval_all_buildnet3d_large_steps)
