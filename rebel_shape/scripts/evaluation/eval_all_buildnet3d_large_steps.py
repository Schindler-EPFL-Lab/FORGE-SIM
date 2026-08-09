import json
import subprocess
from pathlib import Path

import tyro

from rebel_shape import logger
from rebel_shape.scripts.evaluation.eval_all_common import prepare_metrics_output_folder
from rebel_shape.scripts.render_eval_images_mesh import (
    discover_mesh_textures,
    face_key,
)


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

    Each scene's reconstruction folder holds one reconstruction, whose mesh and texture
    `render_eval_images_mesh.py` discovers for itself -- so the folder is handed over
    whole, and each scene is evaluated once. The scene's metrics are written to
    `validation_inputs_root`/`output_folder_name`/<scene>_metrics.json.

    :raise:
        FileNotFoundError if:
            - The `validation_inputs_root`/data does not exist,
            - The `validation_inputs_root`/large-steps does not exist,
            - The `validation_inputs_root`/data/BuildNet3DCorrectedMeshes/ does not
            exist.
            - The `environment_map_path` does not exist,
            - A scene's reconstruction folder holds no mesh (from
            `discover_mesh_textures`),
        ValueError if `scenes_ids` is not a subset of {"1", "2", "3", "4"}, or a
        scene's reconstruction folder is ambiguous (from `discover_mesh_textures`).
        RuntimeError if a scene's reconstruction folder holds cube-face patches
        rather than a single mesh, which the Chamfer step cannot read.
        FileExistsError if a scene's output folder already holds the metrics of an
        earlier run.
        subprocess.CalledProcessError if a render or evaluation subprocess fails.
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
        result_path = large_steps_root / scene_name_data
        thermoscenes_path = dataset_root / scene_name_data / scene_name_data

        # The Chamfer step reads one mesh file, so resolve it here rather than in the
        # subprocess: a composite has no single file to hand it and must be caught
        # before anything is rendered. The test is whether any key names a cube face,
        # not how many keys there are -- a run that saved one patch and then died
        # leaves a single key that is still one sixth of a surface, and measuring that
        # against the whole building would report a wrong number rather than fail.
        mesh_textures = discover_mesh_textures(result_path)
        patch_keys = [key for key in mesh_textures if face_key(key) is not None]
        if patch_keys:
            raise RuntimeError(
                f"{result_path} holds the cube-face patches of a composite "
                f"({', '.join(patch_keys)}), but `eval_buildnet3d_mesh.py` measures "
                "the Chamfer distance of a single mesh file. Weld the patches and "
                "evaluate with `stitch_and_eval_cubed_sphere_mesh.py`, which takes the "
                "patch directory instead."
            )
        mesh_path, _ = next(iter(mesh_textures.values()))

        logger.info(f"Evaluating the reconstruction in {result_path}")
        output_folder = output_root / scene_name_data
        prepare_metrics_output_folder(output_folder, "eval_nvs.json", "eval_shape.json")
        output_images_folder = output_folder / "images"
        output_images_folder.mkdir(exist_ok=True, parents=True)

        # render reconstruction from eval poses
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
        subprocess.run(cmd, check=True)

        metrics: dict[str, float] = {}
        with open(output_folder / "eval_nvs.json") as f:
            metrics.update(json.load(f))
        with open(output_folder / "eval_shape.json") as f:
            metrics.update(json.load(f))

        with open(output_root / f"{scene_name_data}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    tyro.cli(eval_all_buildnet3d_large_steps)
