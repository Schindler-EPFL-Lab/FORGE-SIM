"""
Segments foreground and background masks of a scene based on depths and objects masks
"""

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import tyro

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append("./")

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape.scripts.segmentation.extract_foreground_and_background import (  # noqa: E501
    extract_foreground_and_background,
)


@dataclass
class SceneForegroundBackgroundSegmentation(ReverseCli):
    """
    Configuration to run foreground and background extraction on training and
    eval images of a provided scene located in `input_scene_path`, and save the
    results in `output_scene_path`. The output scene folder has all the files that
    input scene has with extra `foreground_masks/` and `background_masks/` folders
    which contain extracted foreground and background masks respectively. Moreover, the
    file `transforms.json` in output_scene_path contains paths to foreground and
    background masks named `foreground_mask_file_path` and `background_mask_file_path`.
    Before saving new files in `output_scene_path` the script optionally removes files
    in that directory depending on `clear_existing_files_in_output`.

    The input scene must have the following structure:
    ```
    input_scene_path/
    ├── images/
    ├── depths/
    ├── transforms.json
    │...
    ```
    :raises ValueError
        if `input_scene_path` does not exist or
        if `output_scene_path` does not exist
    """

    input_scene_path: Path
    """Path to the input scene"""

    output_scene_path: Path
    """
    Path to a folder where to store the scene data with a new foreground_masks and
    background_masks folders
    """

    dilation: int = 7
    """
    Expands the region of pixels that belong to the object's convex hull mask but not to
    the object mask using morphological operations. A larger `dilation` value produces
    a bigger mask.
    """

    clear_existing_files_in_output: bool = False
    """
    Whether to clear the existing `output_scene_path` directory before saving new files
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


def load_scene_images_depths_and_corresponding_names(
    scene_path: Path,
) -> tuple[dict[str, npt.NDArray[np.uint8]], dict[str, npt.NDArray[np.float32]]]:
    """
    Loads images and depths from a given `scene_path` and returns them as numpy arrays
    along with their corresponding names.

    The scene must have files and directories in a structure as below:
    ```
    scene_path/
    ├── images/
    ├── depths/
    ├── transforms.json
    │...
    ```

    If "file_path" is present in any "frame" entry in `scene_path / transforms.json`,
    the function loads images using these file paths. Otherwise, it scans the
    `images/` folder and processes files with extensions: ".JPG", ".jpg", ".JPEG",
    ".jpeg", ".PNG", or ".png".

    :returns: Two dictionaries:
    - The first contains images as values and their corresponding names as keys.
    - The second contains depth maps as values and their corresponding image names
    as keys.
    """

    transform_path = scene_path / "transforms.json"

    with open(transform_path, "r") as f:
        transform_data = json.load(f)

    if len(transform_data["frames"]) == 0:
        return {}, {}

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

    depth_filenames = [
        scene_path / "depths" / (image_filename.stem + ".npy")
        for image_filename in image_rgb_filenames
    ]

    images = {
        image_rgb_filename.stem: cv2.imread(
            str(image_rgb_filename), cv2.IMREAD_UNCHANGED
        )
        for image_rgb_filename in image_rgb_filenames
    }
    depths = {
        depth_filename.stem: np.load(depth_filename)
        for depth_filename in depth_filenames
    }

    return images, depths


def extract_foreground_background_scene(
    params: SceneForegroundBackgroundSegmentation,
) -> None:
    """
    Runs extraction of foreground and background on a scene located in
    `params.input_scene_path` and stores the results in `params.output_scene_path`.

    It runs foreground and background extraction on train and eval images of the
    provided input scene using depths and object masks. Then it saves masks results
    along with the initial scene data at `params.output_scene_path`. Before saving new
    files to `params.output_scene_path` it optionally removes files which exist in the
    folder.

    The masks values are stored in `.png` format in `/` folder

    :raises RuntimeError
        if "images/" folder does not exist in `input_scene_path` or
        if "depths/" folder does not exist in `input_scene_path` or
        if "transforms.json" does not exist in `input_scene_path`.
    """
    if not (params.input_scene_path / "images").exists():
        raise RuntimeError(
            f"""{str(params.input_scene_path / "images")} folder does not exist"""
        )
    if not (params.input_scene_path / "depths").exists():
        raise RuntimeError(
            f"""{str(params.input_scene_path / "depths")} folder does not exist"""
        )
    if not (params.input_scene_path / "transforms.json").exists():
        raise RuntimeError(
            f"""{str(params.input_scene_path / "transforms.json")} folder does "
            + "not exist"""
        )

    images_and_names, depths_and_names = (
        load_scene_images_depths_and_corresponding_names(params.input_scene_path)
    )

    # extract foreground and background masks of all images
    foreground_masks: dict[str, npt.NDArray[np.uint8]] = {}
    background_masks: dict[str, npt.NDArray[np.uint8]] = {}
    for image_name in images_and_names.keys():
        image = images_and_names[image_name]
        depth = depths_and_names[image_name]

        foreground_mask, background_mask = extract_foreground_and_background(
            image=image,
            depth=depth,
            dilation=params.dilation,
        )

        foreground_masks[image_name] = foreground_mask.astype(np.uint8) * 255
        background_masks[image_name] = background_mask.astype(np.uint8) * 255

    # optionally remove existing files before saving new files
    if params.clear_existing_files_in_output:
        for file_or_dir in params.output_scene_path.rglob("*"):
            if file_or_dir.exists():
                if file_or_dir.is_file():
                    file_or_dir.unlink()
                else:
                    shutil.rmtree(file_or_dir)

    # save previous files along with foreground and background masks
    shutil.copytree(
        params.input_scene_path, params.output_scene_path, dirs_exist_ok=True
    )
    output_foreground_mask_directory = params.output_scene_path / "foreground_masks"
    output_foreground_mask_directory.mkdir(exist_ok=True)

    output_background_mask_directory = params.output_scene_path / "background_masks"
    output_background_mask_directory.mkdir(exist_ok=True)

    for image_name, foreground_mask in foreground_masks.items():
        cv2.imwrite(
            str(output_foreground_mask_directory / f"{image_name}.png"), foreground_mask
        )

    for image_name, background_mask in background_masks.items():
        cv2.imwrite(
            str(output_background_mask_directory / f"{image_name}.png"), background_mask
        )

    # update transforms.json with new depth files
    with open(params.output_scene_path / "transforms.json", "r") as f:
        transforms_data = json.load(f)
    for frame in transforms_data["frames"]:
        image_filename = Path(frame["file_path"])
        frame["foreground_mask_file_path"] = (
            f"foreground_masks/{image_filename.stem}.png"
        )
        frame["background_mask_file_path"] = (
            f"background_masks/{image_filename.stem}.png"
        )
    with open(params.output_scene_path / "transforms.json", "w") as f:
        json.dump(transforms_data, f, indent=4)


if __name__ == "__main__":
    parameters = tyro.cli(SceneForegroundBackgroundSegmentation)

    extract_foreground_background_scene(params=parameters)
