import numpy as np
import numpy.typing as npt
from skimage import morphology
from skimage.morphology import convex_hull_image
from skimage.restoration import inpaint


def _not_masked_in_convex_hull(
    binary_mask: npt.NDArray[np.bool],
) -> npt.NDArray[np.bool]:
    """
    Returns the pixels within the convex hull of an `binary_mask` that are not part of
    the `binary_mask`.

    :returns a boolean array of shape (H, W) where:
        - True indicates a pixel is inside the convex hull but not part of the
        `image` mask.
        - False indicates a pixel is either outside the convex hull or part of the
        `image` mask.

    :raises ValueError if:
        1. `image` number of dimentions is not equal to 3 or
        2. `image` does not contain the object mask as the last channel
        3. `image` mask has other values except 0 and 255
    """

    if binary_mask.ndim != 2:
        raise ValueError("Input mask must have 2 dimensions (H, W).")

    hull_image = convex_hull_image(binary_mask)
    return hull_image & (~binary_mask)


def extract_foreground_and_background(
    image: npt.NDArray[np.uint8], depth: npt.NDArray[np.float32], dilation: int = 7
) -> tuple[npt.NDArray[np.bool], npt.NDArray[np.bool]]:
    """
    Extracts foreground and background masks from an RGBA image and its corresponding
    depth map. Given an input `image` (with an alpha channel as the fourth channel) and
    a `depth` map, this function identifies the foreground and background regions based
    on depth inpainting and morphological operations. Input masks are not precise
    therefore the `dilation` is applied to expand input masks.

    This method is used to identify foreground pixels located between the object and
    the camera plane. These pixels should be ignored during optimization.
    To find them, we proceed as follows:

    1. Find pixels within the convex hull of the object mask that do not belong to the
    object mask itself. These pixels are candidates for being foreground pixels.
    However, they may also be background pixels, so we need to determine their correct
    classification.
    2. Use depth information to decide. First, remove the depth values of the pixels
    identified in step 1. Then, fill in the missing depth values using
    `inpaint_biharmonic` from `skimage`.

    3) Compare the inpainted depth with the real depth:
        - If the inpainted depth is smaller than the real depth, the pixel belongs to
        the background.
        - Otherwise, it belongs to the foreground.

    Here is a small illustration of why this indicates foreground pixels:

    (the view is from above, the big rectangle is the object, and the small rectangle
    area is a foreground between the object and the camera)

    real image    |      depth     |    removed depth    |    inpainted depth
    .----------.  |                |                     |
    |  object  |  |  |          |  |    |          |     |      |          |
    '----------'  |  '---.  .---'  |    '---????---'     |      '---++++---'
        .--.      |      |  |      |   "?" - highlights  |    "+" - highlights
        '--'      |      '--'      |    missing values   |    inpainted values
                  |                |                     |
       camera     |                |                     |

    We can see that the inpainted depth values are larger, which means that foreground
    pixels were blocking those regions.

    :returns: a tuple of two boolean numpy arrays of shape (H, W):
        - the first is foreground: True indicates foreground pixels and False all other
        pixels
        - the second is background: True indicates background pixels and False all
        other pixels

    :raises ValueError: if
        1. `image` does not have 3 dimensions or the last dimension is not 4.
        2. `depth` does not have 2 dimensions or does not match the spatial shape of
        `image`.
        3. The alpha channel of `image` contains values other than {0, 255}.
    """

    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(
            "`image` must have 3 dimensions and the last dimension must be 4 (RGBA)."
        )
    if depth.ndim != 2 or depth.shape != image.shape[:2]:
        raise ValueError(
            "`depth` must have 2 dimensions and match the spatial shape of `image`."
        )
    alpha_channel = image[:, :, 3]
    unique_alpha = np.unique(alpha_channel)
    if not np.all(np.isin(unique_alpha, [0, 255])):
        raise ValueError(
            "The alpha channel of `image` must only contain values 0 and 255."
        )

    # preprocess depth
    depth = np.copy(depth)
    depth[depth > 1e2] = 1e2

    binary_mask = image[:, :, 3] == 255
    not_masked = _not_masked_in_convex_hull(binary_mask)
    not_masked_dilated = morphology.dilation(
        image=not_masked, footprint=morphology.footprint_rectangle((dilation, dilation))
    )

    depth_inpainted = inpaint.inpaint_biharmonic(image=depth, mask=not_masked_dilated)
    foreground = (depth_inpainted > depth) & not_masked
    background = ~binary_mask & ~foreground
    return foreground, background
