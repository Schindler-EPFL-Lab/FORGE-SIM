import drjit as dr
import mitsuba as mi

from forge_sim.loss_evaluators.image_loss import ImageLoss


class ImageLossL2(ImageLoss):
    """
    Simple L2 loss function - an average over all squared per-pixel differences
    """

    def evaluate(self, image: "mi.TensorXf", ref_image: "mi.TensorXf") -> "mi.TensorXf":
        return dr.mean(dr.square(image - ref_image))
