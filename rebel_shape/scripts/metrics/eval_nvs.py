"""
This script calculates the metrics of novel view synthesis based on ground truth images
and rendered images from bspline.
"""

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import drjit as dr
import mitsuba as mi
import numpy as np
import numpy.typing as npt
import torch
import torchmetrics
import tyro

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append("./")

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape.data_processing.image_scene_loader import ImageSceneLoader
from rebel_shape.data_processing.input_preprocessing import (
    MaskStrategy,
    preprocess_rgb_images,
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
    loader = ImageSceneLoader(scene_path=params.thermoscenes_scene_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load ground truth images
    eval_images, _ = preprocess_rgb_images(
        scene_path=params.thermoscenes_scene_path,
        environment_map_path=params.environment_map_path,
        spp=params.samples_per_pixel,
        train=False,
        mask_strategy=MaskStrategy.FOREGROUND,
    )

    # load rendered images
    rendered_images_paths = sorted(list(params.rendered_images_folder.glob("*")))
    rendered_images: dict[str, npt.NDArray[np.float32]] = {
        image_path.stem: cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        for image_path in rendered_images_paths
    }

    metric_calculators: dict[str, torchmetrics.metric.Metric] = {
        "psnr": torchmetrics.image.PeakSignalNoiseRatio((0.0, 1.0)).to(device),
        "ssim": torchmetrics.image.StructuralSimilarityIndexMeasure().to(device),
        "lpips": torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity().to(
            device
        ),
    }
    metrics = {metric_name: [] for metric_name in metric_calculators}

    for real_image_name, real_image in zip(loader.eval_filenames, eval_images):
        rendered_image = rendered_images[Path(real_image_name).stem]
        rendered_image = torch.from_numpy(rendered_image).to(device)

        real_image_rgb = dr.linear_to_srgb(real_image.original_image)
        real_image_rgb = real_image_rgb.numpy().astype(np.float32)
        real_image_rgb = torch.from_numpy(real_image_rgb).to(device)

        inverse_foreground_mask = real_image.mask.numpy().astype(np.float32)
        inverse_foreground_mask = torch.from_numpy(inverse_foreground_mask).to(device)

        # mask out foreground on real and rendered images
        real_image_rgb = real_image_rgb * inverse_foreground_mask
        rendered_image = rendered_image * inverse_foreground_mask

        for metric_name in metric_calculators:
            metric_result = metric_calculators[metric_name](
                rendered_image[None, :, :, :].permute(0, 3, 1, 2),
                real_image_rgb[None, :, :, :].permute(0, 3, 1, 2),
            )
            metrics[metric_name].append(metric_result)

    metrics_average: dict[str, float] = {}
    for metric_name in metrics:
        concated_metric_tensors = torch.stack(metrics[metric_name])
        metrics_average[metric_name] = torch.mean(concated_metric_tensors).item()

    params.output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(params.output_file_path, "w") as f:
        json.dump(metrics_average, f, indent=4)


if __name__ == "__main__":
    parameters = tyro.cli(EvalNVSParameters)

    eval_novel_view_synthesis(params=parameters)
