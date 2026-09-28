import mitsuba as mi

from forge_sim.loss_evaluators.image_loss import ImageLoss


class CompositeImageLoss(ImageLoss):
    """
    A weighted combination of image loss functions.

    Allows combining different image losses (like `ImageLossL1`, etc.) with
    custom weights. The weights must form a partition of unity (sum to 1.0).
    """

    def __init__(self, losses_and_weights: list[tuple[ImageLoss, float]]) -> None:
        """
        Initialize the composite loss with a list of (loss function, weight) tuples.

        Wrights, the second component in `losses_and_weights`, must sum to 1.0.
        It is intended that the loss functions in the list are unique, but this
        is not strictly enforced.

        :raises ValueError: If weights don't sum to 1.0 (with a small tolerance),
            or if the `losses_and_weights` list is empty.
        """
        if len(losses_and_weights) == 0:
            raise ValueError("At least one loss function must be provided")

        weight_sum = sum(weight for _, weight in losses_and_weights)
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Loss weights must sum to 1.0, got {weight_sum}")

        self._losses_and_weights = losses_and_weights

    def evaluate(self, image: "mi.TensorXf", ref_image: "mi.TensorXf") -> "mi.TensorXf":
        composite_loss = mi.TensorXf(0.0)

        for loss_fn, weight in self._losses_and_weights:
            loss_value = loss_fn.evaluate(image, ref_image)
            composite_loss += weight * loss_value

        return composite_loss
