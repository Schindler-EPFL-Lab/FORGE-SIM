"""
This script calculates the metrics of novel view synthesis and shape reconstruction for
on thermoscenes dataset.
"""

import os

import torchvision

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import json
from dataclasses import dataclass
from pathlib import Path

import mitsuba as mi
import torch
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

from forge_sim.data_processing.image_scene_loader import ImageSceneLoader
from forge_sim.metrics.novel_view_synthesis import (
    calculate_metrics_novel_view_synthesis,
    mitsuba_to_torch,
)


@dataclass(kw_only=True)
class EvalNVSParameters(ReverseCli):
    """
    Parameters used for evaluation of novel view synthesis.
    """

    scene_path: Path
    """Path to the scene on what the spline was optimized."""
    rendered_images_folder: Path
    """Path to the rendered images folder of the optimized spline surface."""
    environment_map_path: Path | None = None
    """
    Path to the envionment map that was used as background in optimization. If not
    provided, a solid white background is used instead.
    """
    samples_per_pixel: int = 169
    """Number of samples per pixel used for rendering"""
    output_folder: Path | None = None
    """Where to save the metrics (created if does not exist)"""
    from_vggt: bool = False
    """
    True of pose are estimatedusing VGGT. Reverts to Thermoscene otherwise
    """
    ignore_background: bool = False
    """
    If True, white pixels of the eval images (considered white here) are ignored in the
    evaluation of the rendered images. This is used for real world dataset to avoid
    low scores due to modeling of occluding elements which will be removed from the
    eval images but present in the render. if we kept them, a "bad" render of a missing
    part in the eval images could lead to a higher score for a method that performs
    badly.
    """

    def __post_init__(self) -> None:
        if not self.scene_path.exists():
            raise FileNotFoundError(f"'{self.scene_path}' does not exist.")
        if not self.rendered_images_folder.exists():
            raise FileNotFoundError(f"'{self.rendered_images_folder}' does not exist.")
        if self.output_folder is None:
            self.output_folder: Path = self.rendered_images_folder / "nvs"


def eval_novel_view_synthesis(params: EvalNVSParameters) -> None:
    """
    Calculates the metrics for novel view synthesis between the ground truth images and
    the optimized spline surface rendered images.

    It loads the ThermoScenes scene from `params.scene_path` along with
    rendered images from `params.rendered_images_folder`. Then it calculates novel view
    synthesis metrics between ground truth (GT) and rendered images.
    It assumes that stems of rendered images and grouth truth images
    are the same for the corresponding images. It saves the result metrics in json file
    at `params.output_file_path`

    :raises ValueError: If `params.scene_path` is not a valid path, or
        it is not a directory.
    """

    mi.set_variant("llvm_ad_rgb")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if params.from_vggt:
        loader = ImageSceneLoader.from_vggt(
            scene_path=params.scene_path,
            environment_map_path=params.environment_map_path,
            use_foreground_masks=True,
        )
    else:
        loader = ImageSceneLoader.from_thermoscene(
            scene_path=params.scene_path,
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

        # Ignore white background pixel to avoid occlusion in real datasets
        # The trick is we make all white pixels in eval_image_torch a white pixel in
        # rendered_image_torch
        if params.ignore_background:
            white_value = 1.0
            white_mask = (eval_image_torch == white_value).all(dim=0, keepdim=True)
            rendered_image_torch = torch.where(
                white_mask, white_value, rendered_image_torch
            )

        rendered_images_torch.append(rendered_image_torch)

    params.output_folder.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(
        eval_images_torch, params.output_folder / "eval_images.png"
    )
    torchvision.utils.save_image(
        rendered_images_torch, params.output_folder / "rendered_images.png"
    )

    metrics = calculate_metrics_novel_view_synthesis(
        rendered_images=rendered_images_torch,
        ground_truth_images=eval_images_torch,
        device=device,
    )

    with open(params.output_folder / "eval_nvs.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    parameters = tyro.cli(EvalNVSParameters)
    eval_novel_view_synthesis(params=parameters)
