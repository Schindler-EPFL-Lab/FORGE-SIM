import drjit as dr
import mitsuba as mi

from forge_sim import logger
from forge_sim.loss_evaluators.image_loss import ImageLoss
from forge_sim.loss_evaluators.image_loss_ssim import ImageLossSSIM


class ImageLossMSSSIM(ImageLoss):
    """
    Multi-Scale SSIM (MS-SSIM) loss.

    Extends single-scale SSIM by computing SSIM and sensitivity map at multiple scales
    and combining them with predefined weights. The loss is defined as 1 - MS-SSIM,
    which is minimized when images match across all scales.

    The formula for MS-SSIM is:
        MS-SSIM(x, y) = prod_{i}^{M} {(sens(x_i, y_i) ^ w_i)} * lum(x_M, y_M) ^ w_M
                      = prod_{i}^{M - 1} {(sens(x_i, y_i) ^ w_i)} * SSIM(x_M, y_M) ^ w_M
    where:
        - M is the number of scales (i starts from 1 and goes to M)
        - x_i and y_i are the images at scale i (scale 1 is the original image)
        - sens(x_i, y_i) is the sensitivity map at scale i
        - lum(x_i, y_i) is the luminance map at scale i
        - SSIM(x_i, y_i) is the SSIM value at scale i
        - w_i are the weights for each scale

    For details on how to compute SSIM, see `ImageLossSSIM`.
    """

    DEFAULT_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1332)
    """
    Default weights from the original MS-SSIM paper
    https://www.cns.nyu.edu/pub/eero/wang03b.pdf
    """

    def __init__(
        self,
        weights: list[float] | None = None,
        k1: float = 0.01,
        k2: float = 0.03,
        window_size: int = 11,
        sigma: float = 1.5,
        dynamic_range: float = 1.0,
    ) -> None:
        """
        All default values provided by the constructor are used in the literature as
        well as the PyTorch implementation.

        `weights` are the per-scale coefficients for the MS-SSIM loss and balance
        the contribution of each scale. If they are not provided, the default weights
        are: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333] (5 scales), the same ones
        as in the original MS-SSIM paper. The weights must sum to 1.0 and be in the
        range [0.0, 1.0].

        `k1` and `k2` are constants used in the SSIM formula, specifically for the
        constants `c1` and `c2` that are used to stabilize the division in the SSIM
        formula. The default values are typically used in the literature.

        `window_size` determines the size of the sliding window used to compute
        local statistics. It should be an odd number. `sigma` is the standard deviation
        of the Gaussian window used for weighting the contributions from pixels in the
        sliding window.

        `dynamic_range` is the "native" range of the pixel values. For example, for
        low dynamic range images this is usually 1.0 (or 255 if they use an 8-bit
        encoding), while for HDR images, it should be the maximum value of all pixels.

        Errors propagate from `check_ssim_settings` in `ImageLossSSIM` and
        `_check_weights` in this class.
        """
        if weights is None:
            weights = list(ImageLossMSSSIM.DEFAULT_WEIGHTS)

        self._check_weights(weights)

        ImageLossSSIM.check_ssim_settings(
            k1=k1,
            k2=k2,
            window_size=window_size,
            sigma=sigma,
            dynamic_range=dynamic_range,
        )

        self._weights = weights
        self._k1 = k1
        self._k2 = k2
        self._window_size = window_size
        self._sigma = sigma
        self._dynamic_range = dynamic_range

    @staticmethod
    def _check_weights(weights: list[float]) -> None:
        """
        Check if the weights are valid.
        """
        if len(weights) < 2:
            raise ValueError("At least 2 scales are required for MS-SSIM")
        if not all(0.0 <= w <= 1.0 for w in weights):
            raise ValueError("`weights` must be in the range [0.0, 1.0]")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("`weights` must sum to 1.0")

    def evaluate(self, image: "mi.TensorXf", ref_image: "mi.TensorXf") -> "mi.TensorXf":
        """
        Compute MS-SSIM loss between `image` and `ref_image`.
        Returns a scalar loss TensorXf.
        """
        msssim_val = self.compute_msssim(
            image_1=image,
            image_2=ref_image,
            weights=self._weights,
            k1=self._k1,
            k2=self._k2,
            window_size=self._window_size,
            sigma=self._sigma,
            dynamic_range=self._dynamic_range,
        )
        return 1.0 - msssim_val

    @staticmethod
    def compute_msssim(
        image_1: "mi.TensorXf",
        image_2: "mi.TensorXf",
        weights: list[float] | None = None,
        k1: float = 0.01,
        k2: float = 0.03,
        window_size: int = 11,
        sigma: float = 1.5,
        dynamic_range: float = 1.0,
    ) -> "mi.TensorXf":
        """
        Compute MS-SSIM between two images across multiple scales.

        Default parameter values are the same as in the original MS-SSIM paper and the
        PyTorch implementation.

        MS-SSIM extends single-scale SSIM by computing SSIM and contrast sensitivity
        at multiple scales and combining them with weights. At each scale, the images
        are down-sampled by a factor of 2 in both dimensions.

        The formula for MS-SSIM is:
        MS-SSIM(x, y) = prod_{i}^{M-1} {(sens(x_i, y_i) ^ w_i)} * SSIM(x_M, y_M) ^ w_M
        where:
            - M is the number of scales (first scale has index 1, like in the paper)
            - x_i and y_i are the images at scale i
            - sens(x_i, y_i) is the contrast sensitivity map at scale i
            - SSIM(x_M, y_M) is the full SSIM value at the final scale M
            - w_i are the weights for each scale

        Inputs `image_1` and `image_2` are the images to compare and must have the
        same shape. The `weights` are the per-scale coefficients for the MS-SSIM loss
        and balance the contribution of each scale. The weights must sum to 1.0 and
        be in the range [0.0, 1.0]. The `k1`, `k2`, `window_size`, `sigma`, and
        `dynamic_range` parameters are the same as in the SSIM formula.

        Checks are performed on `weights` and other parameters to ensure they are valid.
        See `_check_weights` and `check_ssim_settings` for details.

        Images whose height or width is not divisible by `2^(length(weights) - 1)`
        cannot be down-sampled the required number of times, so both are cropped to
        the nearest compatible size. The crop is reported as a warning naming the
        discarded rows and columns. An image smaller than the divisor along either
        dimension has no compatible size to crop to and is rejected.

        :returns: Single MS-SSIM value as a TensorXf

        :raises ValueError: If the images do not have the same shape, if the images
            have an incorrect shape, or if either dimension is smaller than
            `2^(length(weights) - 1)`.
        """
        if weights is None:
            weights = list(ImageLossMSSSIM.DEFAULT_WEIGHTS)

        ImageLossMSSSIM._check_weights(weights)
        ImageLossSSIM.check_ssim_settings(
            k1=k1,
            k2=k2,
            window_size=window_size,
            sigma=sigma,
            dynamic_range=dynamic_range,
        )

        if len(image_1.shape) != 3 or len(image_2.shape) != 3:
            raise ValueError(
                "Input images must be 3 dimensional tensors of shape "
                "(height, width, channels)"
            )
        if image_1.shape != image_2.shape:
            raise ValueError("Input images must have the same shape")

        scales = len(weights)
        # Crop images to the nearest size compatible with the number of scales
        # The required divisor is 2^(scales - 1)
        divisor = 2 ** (scales - 1)
        target_height = image_1.shape[0] - (image_1.shape[0] % divisor)
        target_width = image_1.shape[1] - (image_1.shape[1] % divisor)

        if target_height <= 0 or target_width <= 0:
            raise ValueError(
                f"Input images of shape {image_1.shape[0]}x{image_1.shape[1]} are "
                f"smaller than the divisor {divisor} = 2^({scales} - 1) along at "
                f"least one dimension, so cropping them to a size that can be "
                f"down-sampled {scales - 1} times leaves nothing to compare. Use "
                "larger images or fewer scales."
            )

        if target_height != image_1.shape[0] or target_width != image_1.shape[1]:
            # Crop both images to the compatible size
            logger.warning(
                f"MS-SSIM inputs of shape {image_1.shape[0]}x{image_1.shape[1]} are "
                f"not divisible by {divisor} = 2^({scales} - 1); cropping both images "
                f"to {target_height}x{target_width}, discarding the bottom "
                f"{image_1.shape[0] - target_height} row(s) and the rightmost "
                f"{image_1.shape[1] - target_width} column(s)."
            )
            image_1 = image_1[:target_height, :target_width, :]
            image_2 = image_2[:target_height, :target_width, :]

        ssim_vals = []
        sensitivity_maps = []
        current_image_1, current_image_2 = image_1, image_2

        for scale in range(scales):
            # Single-scale SSIM map
            ssim_map, sensitivity_map = ImageLossSSIM.compute_ssim(
                image_1=current_image_1,
                image_2=current_image_2,
                window_size=window_size,
                sigma=sigma,
                k1=k1,
                k2=k2,
                dynamic_range=dynamic_range,
            )
            ssim_vals.append(dr.mean(ssim_map))
            sensitivity_maps.append(dr.mean(sensitivity_map))

            # Downsample for next scale except last
            if scale < scales - 1:
                height, width, channels = current_image_1.shape

                new_height = height // 2
                new_width = width // 2

                # Reshape and compute mean across 2x2 blocks
                current_image_1 = dr.reshape(
                    dtype=mi.TensorXf,
                    value=current_image_1,
                    shape=[new_height, 2, new_width, 2, channels],
                )
                current_image_1 = dr.mean(current_image_1, axis=1)
                current_image_1 = dr.mean(current_image_1, axis=2)

                current_image_2 = dr.reshape(
                    dtype=mi.TensorXf,
                    value=current_image_2,
                    shape=[new_height, 2, new_width, 2, channels],
                )
                current_image_2 = dr.mean(current_image_2, axis=1)
                current_image_2 = dr.mean(current_image_2, axis=2)

        # Last scale also contributes the luminance as well as sensitivity map
        # SSIM = luminance_map * sensitivity_map
        sensitivity_maps[-1] = ssim_vals[-1]

        # Combine SSIMs via weighted geometric product
        msssim = dr.prod([val**w for val, w in zip(sensitivity_maps, weights)])
        return msssim
