"""
This script calculates the metrics of novel view synthesis and shape reconstruction for
on thermoscenes dataset.
"""

import os

import torchvision

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import mitsuba as mi
import torch
import tyro

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append("./")

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape.data_processing.image_scene_loader import ImageSceneLoader
from rebel_shape.metrics.novel_view_synthesis import (
    calculate_metrics_novel_view_synthesis,
    mitsuba_to_torch,
)


@dataclass(kw_only=True)
class EvalNVSParameters(ReverseCli):
    """
    Parameters used for evaluation of novel view synthesis.
    """

    thermoscenes_scene_path: Path
    """Path to the ThermoScenes scene on what the spline was optimized."""
    rendered_images_folder: Path
    """Path to the rendered images folder of the optimized spline surface."""
    environment_map_path: Path | None = None
    """
    Path to the envionment map that was used as background in optimization. If not
    provided, a solid white background is used instead.
    """
    samples_per_pixel: int = 169
    """Number of samples per pixel used for rendering"""
    output_file_path: Path = Path("./outputs/eval_nvs.json")
    """Where to save the metrics (created if does not exist)"""

    def __post_init__(self) -> None:
        if not self.thermoscenes_scene_path.exists():
            raise FileNotFoundError(f"'{self.thermoscenes_scene_path}' does not exist.")
        if not self.rendered_images_folder.exists():
            raise FileNotFoundError(f"'{self.rendered_images_folder}' does not exist.")


def eval_novel_view_synthesis(params: EvalNVSParameters) -> None:
    """
    Calculates the metrics for novel view synthesis between the ground truth images and
    the optimized spline surface rendered images.

    It loads the ThermoScenes scene from `params.thermoscenes_scene_path` along with
    rendered images from `params.rendered_images_folder`. Then it calculates novel view
    synthesis metrics between ground truth (GT) and rendered images.
    It assumes that stems of rendered images and grouth truth images
    are the same for the corresponding images. It saves the result metrics in json file
    at `params.output_file_path`

    :raises ValueError: If `params.thermoscenes_scene_path` is not a valid path, or
        it is not a directory.
    """

    mi.set_variant("llvm_ad_rgb")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    loader = ImageSceneLoader(
        scene_path=params.thermoscenes_scene_path,
        environment_map_path=params.environment_map_path,
        use_foreground_masks=True,
    )

    rendered_images_paths = sorted(list(params.rendered_images_folder.glob("*")))
    rendered_images_paths = {
        image_path.stem: image_path for image_path in rendered_images_paths
    }

    eval_images_torch: list[torch.Tensor] = []
    rendered_images_torch: list[torch.Tensor] = []

    for filename, eval_image in loader.eval_data:
        eval_image_torch = mitsuba_to_torch(eval_image.image, device=device)
        eval_images_torch.append(eval_image_torch)

        rendered_image_path = rendered_images_paths[Path(filename).stem]
        mitsuba_rendered_image = mi.TensorXf(
            mi.Bitmap(str(rendered_image_path)).convert(
                component_format=mi.Struct.Type.Float32, srgb_gamma=False
            )
        )
        rendered_image = eval_image.apply_mask(mitsuba_rendered_image)
        rendered_image_torch = mitsuba_to_torch(rendered_image, device=device)
        rendered_images_torch.append(rendered_image_torch)

    torchvision.utils.save_image(
        eval_images_torch, params.output_file_path.parent / "eval_images.png"
    )
    torchvision.utils.save_image(
        rendered_images_torch, params.output_file_path.parent / "rendered_images.png"
    )

    metrics = calculate_metrics_novel_view_synthesis(
        rendered_images=rendered_images_torch,
        ground_truth_images=eval_images_torch,
        device=device,
    )

    params.output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(params.output_file_path, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    parameters = tyro.cli(EvalNVSParameters)
    eval_novel_view_synthesis(params=parameters)
