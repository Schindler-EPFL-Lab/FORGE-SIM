import json
import subprocess
from dataclasses import astuple, dataclass
from pathlib import Path

import numpy as np
import tyro

from rebel_shape import logger


@dataclass
class SplineTexturePaths:
    """Contains paths to spline shape faces textures"""

    back: Path
    """Path to the .exr back texture"""
    bottom: Path
    """Path to the .exr bottom texture"""
    front: Path
    """Path to the .exr front texture"""
    left: Path
    """Path to the .exr left texture"""
    right: Path
    """Path to the .exr right texture"""
    top: Path
    """Path to the .exr top texture"""

    def as_list_str(self) -> list[str]:
        """Converts the inner fields into list of strings"""
        return [str(path) for path in astuple(self)]


def eval_all_buildnet3d(
    scenes_ids: list[str],
    validation_inputs_root: Path,
    environment_map_path: Path = Path("./data/textures/checkerboard_lighter.exr"),
    output_folder_name: str = "splines-metrics",
):
    """
    Runs evaluation of BuildNet3D reconstructed using the proposed Splines method. The
    `scenes_ids` specifies the ids of BuildNet3D scenes to validate on, the
    `validation_inputs_root` specifies data related to rebel-shape validation, such as
    scenes, reconstruction meshes and textures. The `environment_map_path` specifies the
    path to the environmental map used during training. The `output_folder_name` defines
    the name of the output folder.

    The `validation_inputs_root` must contain folders:
        - data/ containing all BuildNet3D scenes (e.g. downloaded from Azure using
          download_dataset),
        - splines/ containing all BuildNet3D reconstructions with folders names
          match scenes in data/,
        - data/BuildNet3DCorrectedMeshes/ containing ground truth meshes.

    :raise:
        FileNotFoundError if:
            - The `validation_inputs_root`/data does not exist,
            - The `validation_inputs_root`/splines does not exist,
            - The `validation_inputs_root`/data/BuildNet3DCorrectedMeshes/ does not
            exist.
            - The `environment_map_path` does not exist,
        ValueError if `scenes_ids` is not a subset of {"1", "2", "3", "4"}.
        RuntimeError if the number of found meshes differs from the number of
        found textures.
    """

    dataset_root = validation_inputs_root / "data"
    ground_truth_meshes_root = dataset_root / "BuildNet3DCorrectedMeshes"
    splines_root = validation_inputs_root / "splines"
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

    if not splines_root.exists():
        raise FileNotFoundError(f"The `splines_root` {splines_root} does not exist.")
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

        splines_paths = list((splines_root / scene_name_data).glob("*.json"))
        splines_paths = sorted(splines_paths)

        def get_face_paths(face_name: str) -> list[Path]:
            textures_paths = list(
                (splines_root / scene_name_data).glob(f"*{face_name}*.exr")
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

        if len(splines_textures_paths) != len(splines_paths):
            raise RuntimeError(
                "The number of textures and splines must be the same but found "
                + f"{len(splines_paths)} splines {splines_paths} and "
                + f"{len(splines_textures_paths)} textures {splines_textures_paths}."
            )

        thermoscenes_path = dataset_root / scene_name_data / scene_name_data

        aggregated_metrics: dict[str, list[float]] = {}
        for idx, spline_path, spline_texture_path in zip(
            range(len(splines_paths)), splines_paths, splines_textures_paths
        ):
            logger.info(f"Evaluating {spline_path} {spline_texture_path.as_list_str()}")
            output_folder = output_root / f"{scene_name_data}_{idx}"
            output_folder.mkdir(exist_ok=True)
            output_images_folder = output_folder / "images"
            output_images_folder.mkdir(exist_ok=True, parents=True)

            # render reconstruction from eval poses
            cmd = [
                "python3",
                "rebel_shape/scripts/render_eval_images.py",
                "--thermoscenes-scene-path",
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
            subprocess.run(cmd)

            # calculate NVS and Chamfer distance metrics
            cmd = [
                "python3",
                "rebel_shape/scripts/evaluation/eval_buildnet3d.py",
                "--thermoscenes-scene-path",
                str(thermoscenes_path),
                "--environment-map-path",
                str(environment_map_path),
                "--rendered-images-folder",
                str(output_images_folder),
                "--spline-path",
                str(spline_path),
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
    tyro.cli(eval_all_buildnet3d)
