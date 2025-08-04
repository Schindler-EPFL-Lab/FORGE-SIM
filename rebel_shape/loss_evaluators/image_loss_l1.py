import drjit as dr
import mitsuba as mi

from rebel_shape.loss_evaluators.image_loss import ImageLoss


class ImageLossL1(ImageLoss):
    """
    Simple L1 loss function - an average over all absolute per-pixel differences
    """

    def evaluate(self, image: "mi.TensorXf", ref_image: "mi.TensorXf") -> "mi.TensorXf":
        return dr.mean(dr.abs(image - ref_image))
