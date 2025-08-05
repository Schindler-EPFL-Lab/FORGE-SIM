"""
Performs facade window extraction from textures obtained by optimizing the spline
shape and texture from images of the ThermoScenes dataset.
This is a post-processing step to obtain facade material property masks.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tyro
from groundingdino.util.inference import annotate, load_image, load_model, predict
from segment_anything import SamPredictor, sam_model_registry

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append("./")

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape import logger
from rebel_shape.scripts.segmentation.groundingdino_config import GroundingDINOConfig
from rebel_shape.scripts.segmentation.sam_config import SAMConfig
from rebel_shape.scripts.segmentation.utils import (
    convert_boxes,
    get_image_extension,
    overlay_segmentation_mask,
)


@dataclass
class FacadeSegmentationParameters(ReverseCli):
    """
    Configuration and settings for extracting facade windows from optimized textures.

    `image_input_dir_path` should contain the optimized textures, while
    `image_output_dir_path` is where the extracted facade windows will be saved.
    These two directories can in principle match as there is a safeguard against
    processing the same image twice in `extract_facade` function.
    """

    image_input_dir_path: Path
    """Path to the folder that contains input RGB images"""
    image_output_dir_path: Path
    """Path to the folder to which the extracted foregrounds are saved as RGBA images"""

    sam_config: SAMConfig
    """
    Configuration structure for Segment Anything Model containing paths to the model
    weights and the type of the model.
    """
    groundingdino_config: GroundingDINOConfig
    """
    Configuration structure for GroundingDINO containing paths to the model
    weights and the type of the model.
    """

    use_cpu: bool = True
    """Which device to use, if `True` then CPU is used, otherwise it uses CUDA"""
    text_prompt: str = "window"
    """Text prompt used by GroundingDINO to find objects of interest"""

    export_annotated_images: bool = False
    """
    Whether to save the intermediate result produced by GroundingDINO that has
    a bounding box
    """
    export_segmented_overlay: bool = False
    """Whether to show the segmentation produced by SAM using `plt`"""

    def __post_init__(self) -> None:
        if not self.image_input_dir_path.exists():
            raise ValueError(
                f"Image input directory {self.image_input_dir_path} does not exist"
            )
        if not self.image_output_dir_path.exists():
            raise ValueError(
                f"Image output directory {self.image_output_dir_path} does not exist"
            )


def segment_facade(params: FacadeSegmentationParameters) -> None:
    """
    Segments facade windows from optimized texture images using GroundingDINO and
    Segment Anything Model (SAM).

    This function processes all compatible images in the input directory, using
    GroundingDINO to detect objects matching the text prompt, and then applies SAM to
    generate precise segmentation masks. The results are saved as RGBA images with
    transparent backgrounds for areas that are not part of the detected objects.

    The function automatically filters out previously processed images (those with
    "masked_" or "annotated_" prefixes) to prevent reprocessing.

    Detected bounding boxes larger than 25% of the image area are rejected to filter
    out false positives. It was noted that GroundingDINO can produce multiple bounding
    boxes for the same object, in particular a big bounding box encapsulating a
    collection of smaller bounding boxes.

    :raises ValueError: If input or output directories specified in params do not exist.
    :raises FileNotFoundError: If required model checkpoint files for GroundingDINO
        or SAM are not found at the specified paths.
    :raises RuntimeError: If CUDA is requested but not available on the system.
    """

    # Load GroundingDINO and Segment Anything Model models
    device = "cpu" if params.use_cpu else "cuda"

    sam_model_checkpoint = str(params.sam_config.checkpoint_path)
    groundingdino_checkpoint = str(params.groundingdino_config.checkpoint_path)
    groundingdino_script = str(params.groundingdino_config.script_path)

    sam_model = sam_model_registry[params.sam_config.model_type](
        checkpoint=sam_model_checkpoint
    )
    sam_model.to(device=device)

    groundingdino_model = load_model(
        model_config_path=groundingdino_script,
        model_checkpoint_path=groundingdino_checkpoint,
        device=device,
    )

    image_suffix = get_image_extension(params.image_input_dir_path)

    image_files: list[Path] = [
        f
        for f in params.image_input_dir_path.iterdir()
        if f.is_file() and f.suffix == image_suffix
    ]

    for image_path in image_files:
        name = image_path.stem

        # Filter out images that might exist from the previous runs
        if (
            name.startswith("masked_")
            or name.startswith("annotated_")
            or name.startswith("overlay_")
        ):
            logger.warning(
                f"Skipping {image_path} as it is a side effect of a previous run."
            )
            # Skip to the next image
            continue

        image_source, image = load_image(str(image_path))

        # Use GroundingDINO to find bounding boxes for the input prompt
        boxes, logits, phrases = predict(
            model=groundingdino_model,
            image=image,
            caption=params.text_prompt,
            box_threshold=params.groundingdino_config.box_threshold,
            text_threshold=params.groundingdino_config.text_threshold,
            device=device,
        )

        # Optionally save the intermediate result produced by GroundingDINO
        if params.export_annotated_images:
            annotated_frame = annotate(
                image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
            )
            save_path = params.image_output_dir_path / f"annotated_{name}{image_suffix}"

            cv2.imwrite(filename=str(save_path), img=annotated_frame)
            logger.info(f"Saved annotated image with bounding boxes to {save_path}")

        logger.info(f"Detected {len(boxes)} boxes in image {image_path}:\n {boxes} ")

        height, width, _ = image_source.shape
        converted_boxes = convert_boxes(boxes=boxes, height=height, width=width)

        # Reject boxes that are larger than a quarter of the image size
        converted_boxes = [
            box
            for box in converted_boxes
            if ((box[2] - box[0]) * (box[3] - box[1])) <= 0.25 * height * width
        ]

        # Use Segment Anything Model to generate a mask of the object
        predictor = SamPredictor(sam_model)

        sam_image = cv2.imread(str(image_path))
        sam_image = cv2.cvtColor(sam_image, cv2.COLOR_BGR2RGB)
        predictor.set_image(sam_image)

        # Multiple boxes can contain multiple objects of interest from the prompt
        # So all have to be fed one at a time
        full_mask = np.zeros(sam_image.shape[:2], dtype=np.bool_)
        for box in converted_boxes:
            masks, _, _ = predictor.predict(box=box, multimask_output=False)
            full_mask |= masks[0]

        if params.export_segmented_overlay and len(converted_boxes) > 0:
            overlaid_image = overlay_segmentation_mask(image=sam_image, mask=full_mask)
            save_path = params.image_output_dir_path / f"overlay_{name}{image_suffix}"

            cv2.imwrite(
                filename=str(save_path),
                img=cv2.cvtColor(overlaid_image, cv2.COLOR_RGBA2BGRA),
            )
            logger.info(f"Saved overlay image with segmentation mask to {save_path}")

        # Create an RGBA image from the original RGB image and save it
        segmented_image = np.dstack(
            [sam_image, np.full(full_mask.shape, 255, dtype=np.uint8)]
        )
        segmented_image[full_mask == 0, 3] = 0

        cv2.imwrite(
            filename=str(params.image_output_dir_path / f"masked_{name}{image_suffix}"),
            img=cv2.cvtColor(segmented_image, cv2.COLOR_RGBA2BGRA),
        )


if __name__ == "__main__":
    parameters = tyro.cli(FacadeSegmentationParameters)

    segment_facade(params=parameters)
