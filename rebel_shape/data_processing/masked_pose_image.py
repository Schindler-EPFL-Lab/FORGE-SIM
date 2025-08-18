import drjit as dr
import mitsuba as mi

from rebel_shape.plugin_wrappers.perspective_camera import PerspectiveCamera


class MaskedPoseImage:
    """
    Contains an image (rgb/thermal or etc.), a camera and (optionally) a mask which
    correspond to the same view.
    """

    def __init__(
        self,
        image: "mi.TensorXf",
        camera: PerspectiveCamera,
        mask: "mi.TensorXf | None" = None,
    ) -> None:
        """
        Initializes a new instance of MaskedPoseImage with an `image` Tensor, its
        associated `camera` and an optional `mask` Tensor.

        If the `image.ndim` is 3 and `mask.ndim` is 2 or if the `image.ndim` is 2 and
        `mask.ndim` is 3 the class will automatically update mask ndim to match
        the `image` ndim.

        :raises ValueError if
            1. mask has less than 2 or more than 3 dimentions,
            2. mask resolution (height x width) is not equal to the
            image resolution,
            3. if mask values are not 0 or 1.
        """
        self._original_image = image
        self._camera = camera
        self._mask = mask
        self._resolution = image.shape[:2]
        self._ndim = image.ndim

        if self._mask is None:
            self._image = self._original_image
            return

        if self._mask.ndim < 2 or self._mask.ndim > 3:
            raise ValueError(
                f"Mask ndim must be either 2 or 3 but not {self._mask.ndim}"
            )

        # increate the mask dimention if needed
        if self._mask.ndim == 2 and self._original_image.ndim == 3:
            self._mask = self._mask[:, :, None]
        # decrease the mask dimention if needed
        elif self._mask.ndim == 3 and self._original_image.ndim == 2:
            self._mask = self._mask[:, :, 0]

        mask_resolution = self._mask.shape[:2]
        if mask_resolution != self._resolution:
            raise ValueError(
                f"Mask resolution {mask_resolution} does not match "
                f"image resolution {self._resolution}."
            )

        if dr.min(self._mask) < 0 or dr.max(self._mask) > 1:
            raise ValueError(
                "Mask values should be in range [0, 1]",
            )

        mask_rounded = dr.round(self._mask)
        if not dr.allclose(mask_rounded, self._mask):
            raise ValueError("Mask values should be either 0 or 1")

        self._image = self.apply_mask(self._original_image)

    @property
    def image(self) -> "mi.TensorXf":
        """
        Returns masked image.

        If a mask was presented during the class initialization then the
        method returns the masked image. Otherwise it returns the original image.
        """
        return self._image

    @property
    def sensor(self) -> "mi.Sensor":
        """
        Return the mitsuba sensor created from original camera.
        """
        return mi.load_dict(self._camera.mitsuba_dict)

    @property
    def camera(self) -> PerspectiveCamera:
        """
        Returns the original camera.
        """
        return self._camera

    @property
    def original_image(self) -> "mi.TensorXf":
        """
        Returns the original image without masking.
        """
        return self._original_image

    def apply_mask(self, tensor: "mi.TensorXf") -> "mi.TensorXf":
        """
        Applies a mask on a given image. Image values where mask is 0 become
        0 on and where mask is 1 are not changed.

        `tensor` is a mitsuba TensorXf representing the image

        :raises ValueError if number of dimentions of the input image is different or
        if the resolution of the input tensor is different from the original image
        """
        if tensor.ndim != self._ndim:
            raise ValueError(
                f"dim number of the input tensor is {tensor.ndim} while it should "
                f"match dim number {self._ndim} of the original image"
            )
        if tensor.shape[:2] != self._resolution:
            raise ValueError(
                f"Resolution of the input tensor {tensor.shape[:2]} "
                f"does not match the mask resolution {self._resolution}."
            )

        if self._mask is None:
            return tensor

        return tensor * self._mask

    @property
    def mask(self) -> "mi.TensorXf | None":
        """
        Returns the original mask. The mask dimention is updated to match the
        initial image dimention. If the original mask is None then the method returns
        None
        """
        return self._mask
