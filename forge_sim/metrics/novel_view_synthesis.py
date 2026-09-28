"""
This file contains necessary functions to calculate novel view synthesis metrics.
"""

import mitsuba as mi
import numpy as np
import torch
import torchmetrics


def mitsuba_to_torch(
    image: "mi.TensorXf", device: torch.device | None = None
) -> torch.Tensor:
    """
    Converts an `image` to a pytorch tensor and loads it on `device`. If `device` is
    not specified then it loads on "cuda" if possible.

    :returns: the image as a pytorch tensor with shapes (C, H, W).
    """

    if not image.ndim == 3:
        raise RuntimeError(
            f"Input image should be (H, W, C), but image.ndim = {image.ndim}"
        )

    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    image_np = image.numpy().astype(np.float32)
    image_torch = torch.from_numpy(image_np).to(device)
    return image_torch.permute(2, 0, 1)


def calculate_metrics_novel_view_synthesis(
    rendered_images: list[torch.Tensor],
    ground_truth_images: list[torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    """
    Computes novel view synthesis metrics (PSNR, SSIM, LPIPS) between `rendered_images`
    and `ground_truth_images`. The computation is performed on `device`.

    :returns A dictionary where keys are metric names ('psnr', 'ssim', 'lpips') and
        values are average values over all image pairs.

    :raises ValueError if the number of rendered_images differs from the number of
        ground_truth_images
    """

    if len(rendered_images) != len(ground_truth_images):
        raise ValueError(
            f"Number of rendered images ({len(rendered_images)}) does not match number"
            + f" of ground truth images ({len(ground_truth_images)})."
        )

    metric_calculators = {
        "psnr": torchmetrics.image.PeakSignalNoiseRatio((0.0, 1.0)).to(device),
        "ssim": torchmetrics.image.StructuralSimilarityIndexMeasure().to(device),
        "lpips": torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity().to(
            device
        ),
    }

    metrics = {metric_name: [] for metric_name in metric_calculators}
    for rendered_image, ground_truth_image in zip(rendered_images, ground_truth_images):
        for metric_name in metric_calculators:
            metric_result = metric_calculators[metric_name](
                rendered_image.unsqueeze(0), ground_truth_image.unsqueeze(0)
            )
            metrics[metric_name].append(metric_result)

    metrics_average: dict[str, float] = {}
    for metric_name in metrics:
        concated_metric_tensors = torch.stack(metrics[metric_name])
        metrics_average[metric_name] = torch.mean(concated_metric_tensors).item()

    return metrics_average
