import json
import subprocess
from pathlib import Path

import numpy as np
import tyro

from rebel_shape import logger


def eval_all_real_large_steps(
    scenes_names: list[str],
    validation_inputs_root: Path,
    environment_map_path: Path = Path("./data/textures/overcast_soil_2_2k.exr"),
    output_folder_name: str = "large-steps-metrics",
):
    """
    Runs evaluation of Real Scenes reconstructed using the LargeSteps method. The
    `scenes_names` specifies the names of Real scenes to validate on, the
    `validation_inputs_root` specifies data related to rebel-shape validation, such as
    scenes, reconstruction meshes and textures. The `environment_map_path` specifies the
    path to the environmental map used during training. The `output_folder_name` defines
    the name of the output folder.

    The `validation_inputs_root` must contain folders:
        - data/ containing all Real scenes (e.g. downloaded from Azure using
          download_dataset),
        - large-steps/ containing all Real reconstructions with folders names
          match scenes in data/,

    :raise:
        FileNotFoundError if:
            - The `validation_inputs_root`/data does not exist,
            - The `validation_inputs_root`/large-steps does not exist,
            - The `environment_map_path` does not exist,
        ValueError if `scenes_ids` is not a subset of { "extracted-buildingA-spring",
            "extracted-car", "extracted-woodshed-sun", "extracted_heater_water_kettle"}.
        RuntimeError if the number of found meshes differs from the number of
        found textures.
    """
    dataset_root = validation_inputs_root / "data"
    large_steps_root = validation_inputs_root / "large-steps"
    output_root = validation_inputs_root / output_folder_name

    possible_scenes = {
        "extracted-buildingA-spring",
        "extracted-car",
        "extracted-woodshed-sun",
        "extracted_heater_water_kettle",
    }

    if not large_steps_root.exists():
        raise FileNotFoundError(
            f"The `large_steps_root` {large_steps_root} does not exist."
        )
    if not dataset_root.exists():
        raise FileNotFoundError(f"The `dataset_root` {dataset_root} does not exist.")
    if not environment_map_path.exists():
        raise FileNotFoundError(f"The `dataset_root` {dataset_root} does not exist.")

    output_root.mkdir(exist_ok=True, parents=True)

    if not set(scenes_names).issubset(possible_scenes):
        raise ValueError(
            f"The `scenes_names` must be a subset of {possible_scenes}, "
            + f"but got {scenes_names}, extra scenes are "
            + f"{set(scenes_names) - possible_scenes}"
        )

    for scene_name in scenes_names:
        meshes_paths = list((large_steps_root / scene_name).glob("*.ply"))
        meshes_paths = sorted(meshes_paths)
        textures_paths = list((large_steps_root / scene_name).glob("*.exr"))
        textures_paths = sorted(textures_paths)
        if len(meshes_paths) != len(textures_paths):
            raise RuntimeError(
                "The number of textures and meshes must be the same but found "
                + f"{len(meshes_paths)} meshes {meshes_paths} and {len(textures_paths)}"
                + f" textures {textures_paths}."
            )

        thermoscenes_path = dataset_root / scene_name / scene_name

        aggregated_metrics: dict[str, list[float]] = {}
        for idx, mesh_path, texture_path in zip(
            range(len(meshes_paths)), meshes_paths, textures_paths
        ):
            logger.info(f"Evaluating {mesh_path} {texture_path}")
            output_folder = output_root / f"{scene_name}_{idx}"
            output_folder.mkdir(exist_ok=True)
            output_images_folder = output_folder / "images"
            output_images_folder.mkdir(exist_ok=True, parents=True)

            # render images
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

            # calculate NVS metrics
            cmd = [
                "python3",
                "rebel_shape/scripts/evaluation/eval_real.py",
                "--thermoscenes-scene-path",
                str(thermoscenes_path),
                "--environment-map-path",
                str(environment_map_path),
                "--rendered-images-folder",
                str(output_images_folder),
                "--output_folder",
                str(output_folder),
            ]
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd)

            current_metrics: dict[str, float] = {}
            with open(output_folder / "eval_nvs.json") as f:
                current_metrics.update(json.load(f))

            for k in current_metrics:
                if k not in aggregated_metrics:
                    aggregated_metrics[k] = []
                aggregated_metrics[k].append(current_metrics[k])

        mean_aggregated: dict[str, float] = {}
        for k in aggregated_metrics:
            mean_aggregated[k] = np.mean(aggregated_metrics[k]).item()

        with open(output_root / f"{scene_name}_aggregated.json", "w") as f:
            json.dump(mean_aggregated, f, indent=4)


if __name__ == "__main__":
    tyro.cli(eval_all_real_large_steps)
