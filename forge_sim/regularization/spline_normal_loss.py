import mitsuba as mi

from forge_sim.meshing.tessellated_spline_surface import TessellatedSplineSurface
from forge_sim.regularization.normal_loss import NormalLoss
from forge_sim.regularization.regularization import Regularization


class SplineNormalLoss(Regularization):
    """
    A normal loss for tessellated spline surfaces.

    A wrapper around the `NormalLoss` that iterates over the spline patch meshes,
    computes the normal loss for each patch, and sums them up to provide a total loss
    for the entire spline surface.

    For more details about when to use this loss, see the documentation of
    `SplineNormalLoss`.
    """

    def __init__(
        self, tessellated_spline: TessellatedSplineSurface, weight: float = 1.0
    ) -> None:
        """
        Initializes the `SplineNormalLoss` with a tessellated spline surface and a
        `weight` that is divided by the number of patches in the spline surface.

        Internally, a list of `NormalLoss` instances is created for each patch in the
        tessellated spline surface. `weight` is passed to the constructor for each
        patch.

        :raises ValueError: If the weight is negative.
        """
        super().__init__(weight=weight / len(tessellated_spline))
        self._tessellated_spline = tessellated_spline

        self._per_patch_losses = [
            NormalLoss(mesh=patch, weight=weight) for patch in self._tessellated_spline
        ]

    def evaluate(self) -> "mi.TensorXf":
        """
        Evaluate the spline normal loss by averaging the individual patch normal losses.

        :returns: The total loss as a Tensor of shape (1,).
        """
        total_loss = mi.TensorXf(0.0)

        for patch_loss in self._per_patch_losses:
            total_loss += patch_loss.evaluate()

        return total_loss
