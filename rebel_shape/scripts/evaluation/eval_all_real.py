import json
import subprocess
from pathlib import Path

import numpy as np
import tyro

from rebel_shape import logger
from rebel_shape.scripts.evaluation.eval_all_buildnet3d import SplineTexturePaths
from rebel_shape.scripts.evaluation.eval_all_common import prepare_metrics_output_folder


def eval_all_real(
    scenes_names: list[str],
    validation_inputs_root: Path,
    environment_map_path: Path = Path("./data/textures/overcast_soil_2_2k.exr"),
    output_folder_name: str = "splines-metrics",
):
    """
    Runs evaluation of Real Scenes reconstructed using the proposed Splines method. The
    `scenes_names` specifies the names of Real scenes to validate on, the
    `validation_inputs_root` specifies data related to rebel-shape validation, such as
    scenes, reconstruction meshes and textures. The `environment_map_path` specifies the
    path to the environmental map used during training. The `output_folder_name` defines
    the name of the output folder.

    The `validation_inputs_root` must contain folders:
        - data/ containing all Real scenes (e.g. downloaded from Azure using
          download_dataset),
        - splines/ containing all Real reconstructions with folders names
          match scenes in data/,

    :raise:
        FileNotFoundError if:
            - The `validation_inputs_root`/data does not exist,
            - The `validation_inputs_root`/splines does not exist,
            - The `environment_map_path` does not exist,
        ValueError if `scenes_ids` is not a subset of { "extracted-buildingA-spring",
            "extracted-car", "extracted-woodshed-sun", "extracted_heater_water_kettle"}.
        RuntimeError if the number of found meshes differs from the number of
        found textures.
        FileExistsError if a scene's output folder already holds the metrics of an
        earlier run.
        subprocess.CalledProcessError if a render or evaluation subprocess fails.
    """
    dataset_root = validation_inputs_root / "data"
    splines_root = validation_inputs_root / "splines"
    output_root = validation_inputs_root / output_folder_name

    possible_scenes = {
        "extracted-buildingA-spring",
        "extracted-car",
        "extracted-woodshed-sun",
        "extracted_heater_water_kettle",
    }

    if not splines_root.exists():
        raise FileNotFoundError(f"The `splines_root` {splines_root} does not exist.")
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
        splines_paths = list((splines_root / scene_name).glob("*.json"))
        splines_paths = sorted(splines_paths)

        def get_face_paths(face_name: str) -> list[Path]:
            textures_paths = list(
                (splines_root / scene_name).glob(f"*{face_name}*.exr")
            )
            return sorted(textures_paths)

        back_textures_paths = get_face_paths("back")
        bottom_textures_paths = get_face_paths("bottom")
        front_textures_paths = get_face_paths("front")
        left_textures_paths = get_face_paths("left")
        right_textures_paths = get_face_paths("right")
        top_textures_paths = get_face_paths("top")

        splines_textures_paths: list[SplineTexturePaths] = []
        for back, bottom, front, left, right, top in zip(
            back_textures_paths,
            bottom_textures_paths,
            front_textures_paths,
            left_textures_paths,
            right_textures_paths,
            top_textures_paths,
        ):
            splines_textures_paths.append(
                SplineTexturePaths(
                    back=back,
                    bottom=bottom,
                    front=front,
                    left=left,
                    right=right,
                    top=top,
                )
            )

        if len(splines_paths) != len(splines_textures_paths):
            raise RuntimeError(
                "The number of textures and meshes must be the same but found "
                + f"{len(splines_paths)} meshes {splines_paths} and "
                + f"{len(splines_textures_paths)} textures {splines_textures_paths}."
            )

        thermoscenes_path = dataset_root / scene_name / scene_name

        aggregated_metrics: dict[str, list[float]] = {}
        for idx, spline_path, spline_texture_path in zip(
            range(len(splines_paths)), splines_paths, splines_textures_paths
        ):
            logger.info(f"Evaluating {spline_path} {spline_texture_path.as_list_str()}")
            output_folder = output_root / f"{scene_name}_{idx}"
            prepare_metrics_output_folder(output_folder, "eval_nvs.json")
            output_images_folder = output_folder / "images"
            output_images_folder.mkdir(exist_ok=True, parents=True)

            # render images
            cmd = [
                "python3",
                "rebel_shape/scripts/render_eval_images.py",
                "--scene-path",
                str(thermoscenes_path),
                "--environment-map-path",
                str(environment_map_path),
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
    tyro.cli(eval_all_real)
