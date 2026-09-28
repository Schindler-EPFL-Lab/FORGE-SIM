from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch


def get_image_extension(
    image_input_dir_path: Path, supported_extensions: list[str] | None = None
) -> str:
    """
    Determines the file extension of images in the specified input directory.

    Gets all files in the directory provided in `image_input_dir_path`
    and retrieves the extension of the first file.
    Ensures that the extension is one of the supported image formats.

    :returns: A string representing the image file extension

    :raises FileNotFoundError: If the `image_input_dir_path` does not exist, if it
        is not a directory, or if the directory is empty.
    :raises ValueError: If the identified extension is of an unsupported type or if
        files in the directory have different extensions.
    """
    if not image_input_dir_path.exists():
        raise FileNotFoundError(
            f"Image input directory {image_input_dir_path} does not exist"
        )
    if not image_input_dir_path.is_dir():
        raise ValueError(f"Provided path {image_input_dir_path} is not a directory")

    # If no supported extensions are provided, use the default ones
    if supported_extensions is None:
        supported_extensions = [".jpeg", ".jpg", ".png", ".PNG", ".JPG", ".JPEG"]

    for extension in supported_extensions:
        if not extension.startswith("."):
            raise ValueError(
                f"Supported extensions must start with a dot, got {extension} instead."
            )

    directory_files = list(image_input_dir_path.glob("*"))
    if len(directory_files) == 0:
        raise FileNotFoundError(f"Image directory {image_input_dir_path} is empty")

    # Get the extension from one file to use for all images
    sample_file = directory_files[0]
    file_extension = sample_file.suffix

    if file_extension not in supported_extensions:
        raise ValueError(
            f"Found file with unsupported extension {file_extension}. "
            f"Supported extensions are: {', '.join(supported_extensions)}"
        )

    for file in directory_files:
        if file.suffix != file_extension:
            raise ValueError(
                f"All files in the directory {image_input_dir_path} should have the"
                f"same extension. Found {file.suffix} instead of {file_extension}."
            )

    return file_extension


def convert_boxes(
    boxes: torch.tensor, height: int, width: int
) -> npt.NDArray[np.float32]:
    """
    Convert boxes from normalized coordinates to absolute pixel coordinates
    used by the Segment Anything Model (SAM).

    Takes a `torch` tensor of `boxes` and an `image` tensor (to obtain height and width)
    and converts normalized coordinates of the box of kind
    (center_x, center_y, width_x, width_h) used by GroundingDINO and coverts them into
    (left_bottom_x_pixel, left_bottom_y_pixel, right_top_x_pixel, right_top_y_pixel).

    :returns: A numpy array of boxes with absolute pixel coordinates; shape (N, 4)
        where N is the number of boxes. Each box is represented as [x1, y1, x2, y2]
        with (x1, y1) being the coordinates of the left bottom corner and
        (x2, y2) being the coordinates of the right top corner of the box.

    :raises ValueError: If `boxes` is not a 2D tensor with shape (N, 4) or if
        `height` or `width` are not positive integers.
    """
    # Convert normalized coordinates to absolute pixel coordinates:
    if len(boxes.shape) != 2 or boxes.shape[1] != 4:
        raise ValueError(
            f"Expected boxes to have shape (N, 4), got {boxes.shape} instead."
        )
    if height <= 0 or width <= 0:
        raise ValueError(
            f"Height and width must be positive integers, got height={height}, "
            f"width={width} instead."
        )

    boxes_np = boxes.cpu().numpy()
    converted_boxes: list[list[float]] = []

    for box in boxes_np:
        cx, cy, w, h = box

        cx_abs = cx * width
        cy_abs = cy * height
        width_abs = w * width
        height_abs = h * height

        x1 = cx_abs - 0.5 * width_abs
        y1 = cy_abs - 0.5 * height_abs
        x2 = cx_abs + 0.5 * width_abs
        y2 = cy_abs + 0.5 * height_abs
        converted_boxes.append([x1, y1, x2, y2])

    return np.array(converted_boxes, dtype=np.float32)


def overlay_segmentation_mask(
    image: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_],
    transparency: float = 0.35,
) -> npt.NDArray[np.uint8]:
    """
    Returns an image with the segmented areas highlighted in red.

    The `mask` is applied to the original `image`, where the segmented areas are
    highlighted in red. The pixels at the masked locations are blended with a solid
    red color and`transparency` controls the intensity of the red overlay.
    The `mask` is defined by a boolean 2D array where `True` indicates the pixels to be
    highlighted.

    :returns: A numpy array of shape (H, W, 3) representing the highlighted image.
    :raises ValueError: If the `mask` is not a 2D array, if `image` does not have 3
        channels, if `image` and `mask` width and height do not match, or if the
        `transparency` value is not in the range [0, 1].
    """
    if len(mask.shape) != 2:
        raise ValueError(
            f"Expected mask to be a 2D array, got shape {mask.shape} instead."
        )
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected image to be a 3D array with 3 channels, got shape {image.shape}."
        )
    if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {image.shape}."
        )
    if transparency < 0 or transparency > 1.0:
        raise ValueError(
            f"Transparency value {transparency} must be in the range [0, 1.0]."
        )

    # Create solid red overlay of the same shape as the image and set red channel to 255
    red_overlay = np.zeros_like(image, dtype=np.uint8)
    red_overlay[:, :, 0] = 255

    # Blend only where mask is True
    blended = image.astype(np.float32)
    blended[mask] = (1.0 - transparency) * blended[mask] + transparency * red_overlay[
        mask
    ]

    return blended.astype(np.uint8)
