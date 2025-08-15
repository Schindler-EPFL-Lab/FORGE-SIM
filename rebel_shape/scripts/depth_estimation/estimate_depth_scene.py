"""
Estimates depth of eval and train images of the provided scene using MoGe-2
"""

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torchvision
import tyro
from moge.model.v2 import MoGeModel
from PIL import Image

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append("./")

from dataclasses_reverse_cli.reverse_cli import ReverseCli


@dataclass
class SceneDepthEstimation(ReverseCli):
    """
    A configure file to run depth estimation on a training and eval images of a
    provided scene located in `input_scene_path` and save the
    results in `output_scene_path`. The output scene folder has all the files that
    input scene has with a `depths/` folder which contains estimated depths of training
    and eval images. Moreover the file `transforms.json` contains paths to depths
    images named `depth_file`.
    Before saving new files in `output_scene_path` the script optionally removes files
    in that directory depending on `clear_existing_files_in_output`.

    The model checkpoint is located in `model_ckpt_path` and the processing will take
    place on gpu if it is available.

    The input scene must have the following structure:
    ```
    input_scene_path/
    ├── images/
    ├── transforms.json
    │...
    ```
    :raises ValueError
        if `input_scene_path` does not exist or
        if `output_scene_path` does not exist or
        if `model_ckpt_path` does not exist.
    """

    model_ckpt_path: Path
    """Specifies the path to the model checkpoint"""

    input_scene_path: Path
    """Path to the scene whose image are inferenced by MoGe-2"""

    output_scene_path: Path
    """Path to a folder where to store the scene data with a new depth folder"""

    clear_existing_files_in_output: bool = False
    """
    Whether to clear the existing `output_scene_path` 
    directory before saving new files
    """

    def __post_init__(self) -> None:
        if not self.input_scene_path.exists():
            raise ValueError(
                f"The provided input_scene_path {self.input_scene_path} does not exist."
            )

        if not self.output_scene_path.exists():
            raise ValueError(
                f"The provided output_scene_path "
                f"{self.output_scene_path} does not exist."
            )

        if not self.model_ckpt_path.exists():
            raise ValueError(f"Model checkpoint {self.model_ckpt_path} does not exist.")


def load_scene_images_and_corresponding_names(
    scene_path: Path, device: torch.device = torch.device("cpu")
) -> dict[str, torch.Tensor]:
    """
    Loads images from a given `scene_path` and returns them as torch tensors on the
    specified `device`, along with their corresponding names.

    The scene mush have files and directories as below:
    ```
    scene_path/
    ├── images/
    ├── transforms.json
    │...
    ```

    If "file_path" is present in any "frame" entry in `scene_path / transforms.json`,
    the function loads images using these file paths. Otherwise, it scans the
    `images/` folder and processes files with extensions: ".JPG", ".jpg", ".JPEG",
    ".jpeg", ".PNG", or ".png".

    :returns dict where values are torch tensors of scene images and keys are
    the corresponding images
    """
    transform_path = scene_path / "transforms.json"

    with open(transform_path, "r") as f:
        transform_data = json.load(f)

    if len(transform_data["frames"]) == 0:
        return {}

    if "file_path" in transform_data["frames"][0]:
        image_rgb_filenames = [
            scene_path / frame["file_path"] for frame in transform_data["frames"]
        ]
    else:
        image_extensions = (".JPG", ".jpg", ".JPEG", ".jpeg", ".PNG", ".png")
        image_rgb_filenames = sorted(
            [
                image
                for image in (scene_path / "images").iterdir()
                if image.suffix in image_extensions
            ]
        )

    images = [
        Image.open(image_rgb_filename).convert("RGB")
        for image_rgb_filename in image_rgb_filenames
    ]

    images_transformation = torchvision.transforms.ToTensor()
    images = [images_transformation(image) for image in images]

    return {
        image_filename.stem: image.to(device)
        for image_filename, image in zip(image_rgb_filenames, images)
    }


def estimate_depth_scene(params: SceneDepthEstimation) -> None:
    """
    Runs depth estimation on a scene located in `params.input_scene_path` and stores
    the results in `params.output_scene_path`.

    It loads MoGe-2 model and runs depth estimation on train and eval images of the
    provided input scene. Then it saves depth estimations result along with the initial
    scene data at `params.output_scene_path`. Before saving new files to
    `params.output_scene_path` it optionally removes files which exist in the folder.

    The depth values are stored in `.npy` format in `depths/` folder
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    images_and_names = load_scene_images_and_corresponding_names(
        params.input_scene_path,
        device=device,
    )

    # load MoGe-2 model
    model = MoGeModel.from_pretrained(params.model_ckpt_path)
    model = model.to(device)
    model.eval()

    # estimates depths of all images
    depths: dict[str, npt.NDArray[np.float32]] = {}
    with torch.inference_mode():
        for image_name, image in images_and_names.items():
            model_outputs = model.infer(image)
            depth_torch = model_outputs["depth"]
            depths_numpy = depth_torch.detach().cpu().numpy()
            depths[image_name] = depths_numpy

    # optionally remove existing files before saving new files
    if params.clear_existing_files_in_output:
        for file_or_dir in params.output_scene_path.rglob("*"):
            if file_or_dir.exists():
                if file_or_dir.is_file():
                    os.remove(str(file_or_dir))
                else:
                    shutil.rmtree(file_or_dir)

    shutil.copytree(
        params.input_scene_path, params.output_scene_path, dirs_exist_ok=True
    )
    output_depth_directory = params.output_scene_path / "depths"
    output_depth_directory.mkdir(exist_ok=True)

    for image_name, depth in depths.items():
        np.save(output_depth_directory / f"{image_name}.npy", depth)

    # update transforms.json with new depth files
    with open(params.output_scene_path / "transforms.json", "r") as f:
        transforms_data = json.load(f)
    for frame in transforms_data["frames"]:
        image_filename = Path(frame["file_path"])
        frame["depth_file_path"] = f"depths/{image_filename.stem}.npy"
    with open(params.output_scene_path / "transforms.json", "w") as f:
        json.dump(transforms_data, f, indent=4)


if __name__ == "__main__":
    parameters = tyro.cli(SceneDepthEstimation)

    estimate_depth_scene(params=parameters)
