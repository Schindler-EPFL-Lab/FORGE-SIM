"""
Performs foreground extraction from images in the ThermoScenes dataset
using GroundingDINO and Segment Anything Model (SAM).
This is a preprocessing step for shape optimization with meshes or splines.
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
class ExtractForegroundParameters(ReverseCli):
    """
    Configuration and settings needed to extract the foreground of images from
    the ThermoScenes compatible dataset.

    `image_input_dir_path` should contain the RGB images of the ThermoScenes dataset,
    while `image_output_dir_path` is where the extracted foreground objects will be
    saved. These two directories must be different to ensure no images are processed
    twice.

    Since thermoscenes can have many images, `start_image_index` and `end_image_index`
    are used to specify the range of images to process. This is especially useful if
    segmentation fails on a few images, and only they have to be reprocessed with
    adapted parameters.
    """

    start_image_index: int
    """
    Which thermoscenes image index to start at 
    (inclusive, typically the train/eval dataset starts with index 1)
    """
    end_image_index: int
    """
    Which thermoscenes image index to end at 
    (exclusive, the image with this index not included)
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

    use_train_images: bool = True
    """
    If `True` then the training images ('train' in the name) are extracted,
    otherwise the evaluation images ('eval' in the name) are extracted
    """
    text_prompt: str = "a building"
    """Text prompt used by GroundingDINO to find objects of interest"""

    use_cpu: bool = True
    """Which device to use, if `True` then CPU is used, otherwise it uses CUDA"""

    export_annotated_images: bool = False
    """
    Whether to save the intermediate result produced by GroundingDINO that has
    a bounding box. Useful for debugging and visualizing results.
    """
    export_segmented_overlay: bool = False
    """
    Whether to save the the image with the segmentation mask overlaid on it.
    Useful for debugging and visualizing results.
    """

    def __post_init__(self) -> None:
        if self.end_image_index < self.start_image_index:
            raise ValueError(
                "At least one image should be converted, "
                "i.e. `end_image_index` has to be >= `start_image_index`"
            )

        if not self.image_input_dir_path.exists():
            raise ValueError(
                f"Image input directory {self.image_input_dir_path} does not exist"
            )
        if not self.image_output_dir_path.exists():
            raise ValueError(
                f"Image output directory {self.image_output_dir_path} does not exist"
            )
        if self.image_input_dir_path == self.image_output_dir_path:
            raise ValueError(
                "`image_input_dir_path` and `image_output_dir_path` "
                "cannot be the same directory. Please provide different paths."
            )


def extract_thermoscenes_foreground(params: ExtractForegroundParameters) -> None:
    """
    Extracts foregrounds for a specified range of images and processes them
    according to specifications in `params`. Further information is provided
    in the class.

    :raises ValueError: If `params.end_image_index` is smaller than
        `params.start_image_index` and if one of the 4 input directory
        paths do not exist.

    :raises FileNotFoundError: if the specific files needed by `GroundingDINO` or
        `Segment Anything Model` are missing, or if images are not found.
    """

    # Load GroundingDINO and Segment Anything Model models
    device = "cpu" if params.use_cpu else "cuda"
    train_or_eval = "train" if params.use_train_images else "eval"

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

    file_extension = get_image_extension(params.image_input_dir_path)

    for image_index in range(params.start_image_index, params.end_image_index + 1):
        image_path = (
            params.image_input_dir_path
            / f"frame_{train_or_eval}_{image_index:05d}{file_extension}"
        )
        if not image_path.exists():
            raise FileNotFoundError(
                f"Could not find {image_path.name} in {params.image_input_dir_path}"
            )

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

        # Optionally save the intermediate result produced by GroundingDINO to .png
        if params.export_annotated_images:
            annotated_frame = annotate(
                image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
            )
            save_path = (
                params.image_output_dir_path
                / f"annotated_{train_or_eval}_{image_index:05d}.png"
            )

            cv2.imwrite(filename=str(save_path), img=annotated_frame)
            logger.info(f"Saved annotated image with bounding boxes to {save_path}")

        logger.info(f"Detected {len(boxes)} boxes in image {image_path}:\n {boxes} ")
        converted_boxes = convert_boxes(
            boxes=boxes, height=image_source.shape[0], width=image_source.shape[1]
        )

        # Use SAM to generate a mask of the object
        predictor = SamPredictor(sam_model)

        sam_image = cv2.imread(str(image_path))
        sam_image = cv2.cvtColor(sam_image, cv2.COLOR_BGR2RGB)
        predictor.set_image(sam_image)

        # Multiple boxes can contain multiple objects of interest from the prompt
        # So all have to be fed one at a time and the resulting mask is a union
        full_mask = np.zeros(sam_image.shape[:2], dtype=np.bool_)
        for box in converted_boxes:
            masks, _, _ = predictor.predict(box=box, multimask_output=False)
            full_mask |= masks[0]

        if params.export_segmented_overlay:
            overlaid_image = overlay_segmentation_mask(image=sam_image, mask=full_mask)
            save_path = (
                params.image_output_dir_path
                / f"overlay_{train_or_eval}_{image_index:05d}.png"
            )

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
            str(
                params.image_output_dir_path
                / f"frame_{train_or_eval}_{image_index:05d}.png"
            ),
            cv2.cvtColor(segmented_image, cv2.COLOR_RGBA2BGRA),
        )


if __name__ == "__main__":
    parameters = tyro.cli(ExtractForegroundParameters)

    extract_thermoscenes_foreground(params=parameters)
