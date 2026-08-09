"""
This script automates the generation of segmentation masks for 3D reconstruction
datasets, replacing the previous GroundingDINO-based pipeline.

It leverages SAM 3 (Segment Anything Model 3) to identify and isolate specific
objects within an image sequence based on an open-vocabulary text prompt.

Main Features:
- Open-Vocabulary Segmentation: Segment any object via text prompts.
- ThermoScenes Format Support: Reads 'transforms.json' and generates a masked
  dataset compatible with rebel-shape optimization scripts.
- RGBA Output: Saves processed frames as .png files with the mask.
"""

import json
import logging
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


class Prompts(Enum):
    WINDOWS = "windows"
    """used to extract windows masks for inpainting"""
    LAPTOP = "laptop"


@dataclass(kw_only=True)
class Sam3ThermoScenesParams(ReverseCli):
    """Configuration parameters for SAM 3 segmentation"""

    scene_dir: Path
    """Input scene folder (must contain transforms.json and images/)"""
    sam3_checkpoint_dir: Path
    """Folder containing sam3.pt (e.g. mounted AzureML data asset)"""
    text_prompt: Prompts | str
    """Prompt for SAM3 open-vocabulary segmentation"""
    confidence_threshold: float = 0.5
    """Min score for masks to be kept in union mode"""
    mask_mode: str = "best"
    """
    best  -> use the highest-score mask only
    union -> union all masks with score >= confidence_threshold
    """
    bpe_path: Path | None = None
    """
    Path to bpe_simple_vocab_16e6.txt.gz. If None it is set to the same value as
    `self.sam3_checkpoint_dir`
    """
    output_folder: str = "images_sam3"
    """Output folder for the segmented images"""
    is_mask: bool = False
    """If True, the images saved on binary masks"""
    output_frame_name: str = "file_path"
    """In the transform.json, the new field will be saved under this name.
    Can overwrite an existing field."""

    from_vggt: bool = False
    """True if dataset from vggt. False if thermoscene"""

    def __post_init__(self) -> None:
        if isinstance(self.text_prompt, Prompts):
            self.text_prompt: str = self.text_prompt.value
        if self.bpe_path is None:
            self.bpe_path: Path = self.sam3_checkpoint_dir

    @property
    def bpe_file(self) -> Path:
        return self.bpe_path / "bpe_simple_vocab_16e6.txt.gz"


def build_processor(params: Sam3ThermoScenesParams) -> Sam3Processor:
    """
    Initializes the SAM 3 model and processor.
    Loads the checkpoint and sets up the inference device.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = params.sam3_checkpoint_dir / "sam3.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"sam3.pt not found at: {ckpt}")

    model = build_sam3_image_model(
        bpe_path=params.bpe_file,
        load_from_HF=False,
        checkpoint_path=str(ckpt),
    )
    model.to(device)
    model.eval()

    return Sam3Processor(
        model=model,
        device=device,
        confidence_threshold=params.confidence_threshold,
    )


def load_transforms(scene_dir: Path) -> dict[str, Any]:
    """
    Loads the transforms.json camera metadata.
    """
    tpath = scene_dir / "transforms.json"
    if not tpath.exists():
        raise FileNotFoundError(f"transforms.json not found: {tpath}")
    with tpath.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_transforms(data: dict[str, Any], out_path: Path) -> None:
    """
    Saves the updated metadata to the output directory.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def pil_to_rgba_with_alpha(rgb: Image.Image, alpha01: np.ndarray) -> Image.Image:
    """
    Merges an RGB PIL image with a numpy alpha mask [0,1].
    """
    if rgb.mode != "RGB":
        rgb = rgb.convert("RGB")

    alpha = (alpha01.astype(np.float32).clip(0.0, 1.0) * 255.0).astype(np.uint8)
    rgba = rgb.copy().convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha, mode="L"))
    return rgba


def choose_mask(
    masks: torch.Tensor, scores: torch.Tensor, params: Sam3ThermoScenesParams
) -> np.ndarray | None:
    """
    Selects the mask to apply based on the 'best' or 'union' strategy.
    """
    if masks.numel() == 0:
        return None
    if masks.ndim != 4 and masks.ndim != 3:
        raise ValueError(f"Unexpected masks ndim: {masks.ndim}")

    masks_nhw = masks
    if masks.ndim == 4:
        masks_nhw = masks[:, 0, :, :]

    if scores.ndim != 1:
        scores = scores.reshape(-1)

    if params.mask_mode.lower() == "best":
        best_idx = int(torch.argmax(scores).item())
        m = masks_nhw[best_idx]
        return (m > 0.5).to(torch.uint8).cpu().numpy().astype(np.float32)

    if params.mask_mode.lower() != "union":
        raise ValueError("mask_mode must be 'best' or 'union'")

    keep = scores >= float(params.confidence_threshold)

    if keep.sum().item() == 0:
        best_idx = int(torch.argmax(scores).item())
        m = masks_nhw[best_idx]
        return (m > 0.5).to(torch.uint8).cpu().numpy().astype(np.float32)

    union = torch.zeros_like(masks_nhw[0], dtype=torch.bool)

    for m in masks_nhw[keep]:
        union |= m > 0.5

    return union.to(torch.uint8).cpu().numpy().astype(np.float32)


def run(params: Sam3ThermoScenesParams) -> None:
    """
    Executes the end-to-end segmentation pipeline: processes each frame,
    generates SAM 3 masks, and exports the final RGBA dataset.
    """
    logger = logging.getLogger("sam3-thermoscenes")
    logger.setLevel(logging.INFO)

    out_images_dir = params.scene_dir / params.output_folder
    out_images_dir.mkdir(parents=True, exist_ok=True)

    transforms = load_transforms(params.scene_dir)
    frames = transforms.get("frames", [])
    if not isinstance(frames, list) or len(frames) == 0:
        raise ValueError("transforms.json has no frames[]")

    processor = build_processor(params)

    n_ok = 0
    n_fail = 0

    for i, frame in enumerate(frames, start=1):
        if params.from_vggt:
            frame = frame["rgb"]
        relative_path: str = frame.get("file_path", None)
        image_path = params.scene_dir / relative_path

        rgba_image = Image.open(image_path).convert("RGBA")
        rgba_array = np.array(rgba_image)
        rgb_array = rgba_array[:, :, :3]
        alpha = rgba_array[:, :, 3]
        rgb_array[alpha == 0] = [255, 255, 255]
        rgb = Image.fromarray(rgb_array, "RGB")

        image_state = processor.set_image(rgb)

        state = processor.set_text_prompt(prompt=params.text_prompt, state=image_state)

        masks = state.get("masks", None)
        scores = state.get("scores", None)
        if masks is None or scores is None:
            n_fail += 1
            logger.warning("SAM3 returned no masks/scores for %s", image_path.name)
            continue

        alpha01 = choose_mask(masks=masks, scores=scores, params=params)
        if alpha01 is None:
            n_fail += 1
            logger.warning("No mask found for %s", image_path.name)
            continue

        rgba = pil_to_rgba_with_alpha(rgb, alpha01)
        stem = Path(relative_path).stem
        out_path = out_images_dir / f"{stem}.png"
        rgba.save(out_path)

        if params.is_mask:
            rgba_image_masked = Image.open(out_path).convert("RGBA")
            rgba_array_masked = np.array(rgba_image_masked)
            rgb_array_masked = rgba_array_masked[:, :, :3]
            alpha_masked = rgba_array_masked[:, :, 3]
            rgb_array_masked[alpha_masked == 0] = [0, 0, 0]
            rgb_array_masked[alpha_masked != 0] = [255, 255, 255]
            rgb_masked = Image.fromarray(rgb_array_masked, "RGB")
            out_path = out_images_dir / f"{stem}.png"
            rgb_masked.save(out_path)

        frame[params.output_frame_name] = str(Path(out_path.parent.name, out_path.name))
        n_ok += 1

        if i % 25 == 0 or i == len(frames):
            logger.info(
                "Processed %d/%d (ok=%d, fail=%d)", i, len(frames), n_ok, n_fail
            )
    shutil.copyfile(
        params.scene_dir / "transforms.json", params.scene_dir / "transforms_old.json"
    )
    save_transforms(transforms, params.scene_dir / "transforms.json")
    logger.info("Done. ok=%d fail=%d", n_ok, n_fail)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(tyro.cli(Sam3ThermoScenesParams))
