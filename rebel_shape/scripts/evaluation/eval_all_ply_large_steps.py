import json
import subprocess
from pathlib import Path

import numpy as np
import tyro

from rebel_shape import logger


def eval_all_ply_large_steps(
    scenes_names: list[str],
    validation_inputs_root: Path,
    output_folder_name: str = "large-steps-metrics",
):
    """
    Runs evaluation of PLY Meshes reconstructed using the LargeSteps method. The
    `scenes_names` specifies the names of PLY meshes to validate on, the
    `validation_inputs_root` specifies data related to rebel-shape validation, such as
    scenes, reconstruction meshes and textures. The `output_folder_name` defines
    the name of the output folder.

    The `validation_inputs_root` must contain folders:
        - large-steps/ containing all PLY Mesh reconstructions with folders names
          match PLY meshes names.

    :raise:
        FileNotFoundError if the `validation_inputs_root`/large-steps does not exist,
        ValueError if `scenes_ids` is not a subset of { "armadillo", "bunny", "car_1",
        "car_24", "spot", "suzanne" }.
    """
    dataset_root = Path("data/meshes")
    large_steps_root = validation_inputs_root / "large-steps"
    output_root = validation_inputs_root / output_folder_name

    possible_scenes = {"armadillo", "bunny", "car_1", "car_24", "spot", "suzanne"}

    if not large_steps_root.exists():
        raise FileNotFoundError(
            f"The `large_steps_root` {large_steps_root} does not exist."
        )
    if not dataset_root.exists():
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

        aggregated_metrics: dict[str, list[float]] = {}
        for idx, mesh_path in enumerate(meshes_paths):
            logger.info(f"Evaluating {mesh_path}")
            output_folder = output_root / f"{scene_name}_{idx}"
            output_folder.mkdir(exist_ok=True)

            reference_shape_path = dataset_root / mesh_path.name

            # calculate PLY metrics
            cmd = [
                "python3",
                "rebel_shape/scripts/evaluation/eval_ply_mesh.py",
                "--reference-shape-path",
                str(reference_shape_path),
                "--mesh-path",
                str(mesh_path),
                "--output-file-path",
                str(output_folder / "eval_shape.json"),
            ]
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd)

            current_metrics: dict[str, float] = {}
            with open(output_folder / "eval_shape.json") as f:
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
    tyro.cli(eval_all_ply_large_steps)
