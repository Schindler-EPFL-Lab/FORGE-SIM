import drjit as dr
import mitsuba as mi

from forge_sim.loss_evaluators.image_loss import ImageLoss


class ImageLossSSIM(ImageLoss):
    """
    SSIM loss with a sliding Gaussian window.

    This implementation follows the standard SSIM approach which uses a Gaussian window
    to compute local statistics around each pixel. The formula used is:
        SSIM(x, y) = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2) /
                     (mu_x^2 + mu_y^2 + c1) * (sigma_x^2 + sigma_y^2 + c2)

    where mu_x and mu_y are the local means of the two images, sigma_x^2 and sigma_y^2
    are the local variances, and sigma_xy is the local covariance. `c1` and `c2` are
    small constants used to stabilize the division if the denominator is close to zero.
    The first fraction (mean values) is the luminance comparison, the second fraction
    (variance) is the contrast comparison.

    The final loss value is 1.0 minus an average over all pixels and creates a loss
    function that is minimized when the images are identical.
    """

    def __init__(
        self,
        k1: float = 0.01,
        k2: float = 0.03,
        window_size: int = 11,
        sigma: float = 1.5,
        dynamic_range: float = 1.0,
    ) -> None:
        """
        All default values provided by the constructor are used in the literature as
        well as the PyTorch implementation.

        `k1` and `k2` are constants used in the SSIM formula, specifically for the
        constants `c1` and `c2` that are used to stabilize the division in the SSIM
        formula. The default values are typically used in the literature.

        `window_size` determines the size of the sliding window used to compute
        local statistics. It should be an odd number.

        `sigma` is the standard deviation of the Gaussian filter that weighs the
        contributions from pixels in the window. The window should be approximately
        as large as 2 * ceil(3 * `sigma`) + 1; larger sizes waste computational
        resources.

        `dynamic_range` is the "native" range of the pixel values. For example, for
        low dynamic range images this is usually 1.0 (or 255 if they use an 8-bit
        encoding), while for HDR images, it should be the maximum value of all pixels.

        Inputs are checked by `check_ssim_settings` for validity.
        """
        ImageLossSSIM.check_ssim_settings(
            k1=k1,
            k2=k2,
            window_size=window_size,
            sigma=sigma,
            dynamic_range=dynamic_range,
        )

        self._k1 = k1
        self._k2 = k2
        self._window_size = window_size
        self._sigma = sigma
        self._dynamic_range = dynamic_range

    def evaluate(self, image: "mi.TensorXf", ref_image: "mi.TensorXf") -> "mi.TensorXf":
        ssim, _ = self.compute_ssim(
            image_1=image,
            image_2=ref_image,
            k1=self._k1,
            k2=self._k2,
            window_size=self._window_size,
            sigma=self._sigma,
            dynamic_range=self._dynamic_range,
        )

        ssim_mean = dr.mean(ssim)
        return 1.0 - ssim_mean

    @staticmethod
    def check_ssim_settings(
        k1: float,
        k2: float,
        window_size: int,
        sigma: float,
        dynamic_range: float,
    ) -> None:
        """
        Check the SSIM settings.

        :raises ValueError: If `k1` or `k2` are not in the range (0.0, 1.0], if
            `window_size` is not a positive odd integer, if `sigma` is not a positive
            real number, or if `dynamic_range` is not a positive real number.
        """
        if k1 <= 0.0 or k1 > 1.0:
            raise ValueError("`k1` must be in the range (0.0, 1.0]")
        if k2 <= 0.0 or k2 > 1.0:
            raise ValueError("`k2` must be in the range (0.0, 1.0]")
        if window_size % 2 == 0:
            raise ValueError("`window_size` must be an odd number")
        if window_size < 1:
            raise ValueError("`window_size` must be positive integer")
        if sigma <= 0.0:
            raise ValueError("`sigma` must be positive real number")
        if dynamic_range <= 0.0:
            raise ValueError("`dynamic_range` must be positive real number")

    @staticmethod
    def compute_ssim(
        image_1: "mi.TensorXf",
        image_2: "mi.TensorXf",
        window_size: int = 11,
        sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        dynamic_range: float = 1.0,
    ) -> tuple["mi.TensorXf", "mi.TensorXf"]:
        """
        Computes the SSIM value between `image_1` and `image_2` (pixel values are float)

        `window_size` is the size of the sliding window and should be an odd number.
        `sigma` is the standard deviation of the Gaussian filter that weighs the
        contributions from pixels in the window. The window should be approximately
        as large as 2 * ceil(3 * `sigma`) + 1. Larger sizes waste computational
        resources.

        `k1` and `k2` are constants used to avoid dividing by a small denominator.
        Together with `dynamic_range` they define `c1` and `c2` in the SSIM formula as
        c1 = (k1 * dynamic_range) ** 2, c2 = (k2 * dynamic_range) ** 2. Note that
        `dynamic_range` of LDR images represented with floats is `1.0`, while for HDR
        images it should be set to the value of the brightest pixel.

        The formula
        SSIM(x, y) = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2) /
                     (mu_x^2 + mu_y^2 + c1) * (sigma_x^2 + sigma_y^2 + c2)
        is used on a per pixel value. The second fraction (variance) is the contrast
        comparison that is returned as a second value.

        Checks are performed on the input parameters to ensure they are valid.
        See `check_ssim_settings` for details.

        :returns: Tensor of per-pixel, per-channel SSIM values in range [-1.0, 1.0] and
            Tensor of per-pixel contrast sensitivity values

        :raises ValueError: If the images do not have the same shape, or if the images
            have an incorrect shape
        """
        if len(image_1.shape) != 3 or len(image_2.shape) != 3:
            raise ValueError(
                "Input images must be 3 dimensional tensors of shape "
                "(height, width, channels)"
            )
        if image_1.shape != image_2.shape:
            raise ValueError("Input images must have the same shape")

        ImageLossSSIM.check_ssim_settings(
            k1=k1,
            k2=k2,
            window_size=window_size,
            sigma=sigma,
            dynamic_range=dynamic_range,
        )

        image_shape = image_1.shape
        height, width, _ = image_shape
        aperture_width = window_size // 2

        coords = dr.arange(size=window_size, dtype=mi.TensorXf) - aperture_width
        gaussian_values = dr.exp(-(coords * coords) / (2.0 * sigma**2))
        kernel_1d = gaussian_values / dr.sum(gaussian_values)

        height_ids = dr.ravel(dr.arange(size=height, dtype=mi.TensorXi))
        width_ids = dr.ravel(dr.arange(size=width, dtype=mi.TensorXi))

        # Exploiting the tensor product of the 2D Gaussian kernel, the 2D convolution
        # can be computed as a sequence of two 1D convolutions. This is more
        # efficient and saves a lot of Mitsuba compilation time.

        temp_mean_1 = dr.zeros(shape=image_shape, dtype=mi.TensorXf)
        temp_mean_2 = dr.zeros(shape=image_shape, dtype=mi.TensorXf)

        for kernel_height_offset in range(-aperture_width, aperture_width + 1):
            local_height_ids = dr.ravel(
                dr.clip(height_ids + kernel_height_offset, 0, height - 1)
            )
            temp_mean_1 += (
                kernel_1d[kernel_height_offset + aperture_width]
                * image_1[local_height_ids, :]
            )
            temp_mean_2 += (
                kernel_1d[kernel_height_offset + aperture_width]
                * image_2[local_height_ids, :]
            )

        mean_1 = dr.zeros(shape=image_shape, dtype=mi.TensorXf)
        mean_2 = dr.zeros(shape=image_shape, dtype=mi.TensorXf)

        for kernel_width_offset in range(-aperture_width, aperture_width + 1):
            local_width_ids = dr.ravel(
                dr.clip(width_ids + kernel_width_offset, 0, width - 1)
            )
            mean_1 += (
                kernel_1d[kernel_width_offset + aperture_width]
                * temp_mean_1[:, local_width_ids]
            )
            mean_2 += (
                kernel_1d[kernel_width_offset + aperture_width]
                * temp_mean_2[:, local_width_ids]
            )

        temp_second_1 = dr.zeros(shape=image_shape, dtype=mi.TensorXf)
        temp_second_2 = dr.zeros(shape=image_shape, dtype=mi.TensorXf)
        temp_cross_moment = dr.zeros(shape=image_shape, dtype=mi.TensorXf)

        for kernel_height_offset in range(-aperture_width, aperture_width + 1):
            local_height_ids = dr.ravel(
                dr.clip(height_ids + kernel_height_offset, 0, height - 1)
            )
            delta_1 = image_1[local_height_ids, :]
            delta_2 = image_2[local_height_ids, :]

            weight = kernel_1d[kernel_height_offset + aperture_width]

            temp_second_1 += weight * delta_1 * delta_1
            temp_second_2 += weight * delta_2 * delta_2
            temp_cross_moment += weight * delta_1 * delta_2

        second_1 = dr.zeros(shape=image_shape, dtype=mi.TensorXf)
        second_2 = dr.zeros(shape=image_shape, dtype=mi.TensorXf)
        cross_moment = dr.zeros(shape=image_shape, dtype=mi.TensorXf)

        for kernel_width_offset in range(-aperture_width, aperture_width + 1):
            local_width_ids = dr.ravel(
                dr.clip(width_ids + kernel_width_offset, 0, width - 1)
            )
            weight = kernel_1d[kernel_width_offset + aperture_width]

            second_1 += weight * temp_second_1[:, local_width_ids]
            second_2 += weight * temp_second_2[:, local_width_ids]
            cross_moment += weight * temp_cross_moment[:, local_width_ids]

        variance_1 = second_1 - mean_1 * mean_1
        variance_2 = second_2 - mean_2 * mean_2
        covariance = cross_moment - mean_1 * mean_2

        c1 = (k1 * dynamic_range) ** 2
        c2 = (k2 * dynamic_range) ** 2

        num = (2.0 * mean_1 * mean_2 + c1) * (2.0 * covariance + c2)
        den = (mean_1 * mean_1 + mean_2 * mean_2 + c1) * (variance_1 + variance_2 + c2)

        contrast_sensitivity = (2.0 * covariance + c2) / (variance_1 + variance_2 + c2)
        ssim = num / den

        return ssim, contrast_sensitivity
