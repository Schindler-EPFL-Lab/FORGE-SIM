import json
import subprocess
from pathlib import Path

import tyro

from rebel_shape import logger
from rebel_shape.scripts.evaluation.eval_all_common import prepare_metrics_output_folder


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

    Each scene's reconstruction folder holds one reconstruction, whose meshes and
    textures `render_eval_images_mesh.py` discovers for itself -- so the folder is
    handed over whole, and each scene is evaluated once. The scene's metrics are
    written to `validation_inputs_root`/`output_folder_name`/<scene>_metrics.json.

    :raise:
        FileNotFoundError if:
            - The `validation_inputs_root`/data does not exist,
            - The `validation_inputs_root`/large-steps does not exist,
            - The `environment_map_path` does not exist,
        ValueError if `scenes_ids` is not a subset of { "extracted-buildingA-spring",
            "extracted-car", "extracted-woodshed-sun", "extracted_heater_water_kettle"}.
        FileExistsError if a scene's output folder already holds the metrics of an
        earlier run.
        subprocess.CalledProcessError if a render or evaluation subprocess fails.
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
        raise FileNotFoundError(
            f"The `environment_map_path` {environment_map_path} does not exist."
        )

    output_root.mkdir(exist_ok=True, parents=True)

    if not set(scenes_names).issubset(possible_scenes):
        raise ValueError(
            f"The `scenes_names` must be a subset of {possible_scenes}, "
            + f"but got {scenes_names}, extra scenes are "
            + f"{set(scenes_names) - possible_scenes}"
        )

    for scene_name in scenes_names:
        result_path = large_steps_root / scene_name
        thermoscenes_path = dataset_root / scene_name / scene_name

        logger.info(f"Evaluating the reconstruction in {result_path}")
        output_folder = output_root / scene_name
        prepare_metrics_output_folder(output_folder, "eval_nvs.json")
        output_images_folder = output_folder / "images"
        output_images_folder.mkdir(exist_ok=True, parents=True)

        # render images
        cmd = [
            "python3",
            "rebel_shape/scripts/render_eval_images_mesh.py",
            "--scene-path",
            str(thermoscenes_path),
            "--environment-map-path",
            str(environment_map_path),
            "--result-path",
            str(result_path),
            "--output-folder",
            str(output_images_folder),
            "--no-use-gpu",
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # calculate NVS metrics
        cmd = [
            "python3",
            "rebel_shape/scripts/evaluation/eval_real.py",
            "--scene-path",
            str(thermoscenes_path),
            "--environment-map-path",
            str(environment_map_path),
            "--rendered-images-folder",
            str(output_images_folder),
            "--output-folder",
            str(output_folder),
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        with open(output_folder / "eval_nvs.json") as f:
            metrics: dict[str, float] = json.load(f)

        with open(output_root / f"{scene_name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    tyro.cli(eval_all_real_large_steps)
