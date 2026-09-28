import mitsuba as mi

from forge_sim.meshing.tessellated_spline_surface import TessellatedSplineSurface
from forge_sim.regularization.position_loss import PositionLoss
from forge_sim.regularization.regularization import Regularization


class SplinePositionLoss(Regularization):
    """
    A position loss for tessellated spline surfaces.

    A wrapper around the `PositionLoss` that iterates over the spline patch meshes,
    computes the position loss for each patch, and sums them up to provide a total loss
    for the entire spline surface.

    For more details about when to use this loss, see the documentation of
    `SplinePositionLoss`.
    """

    def __init__(
        self, tessellated_spline: TessellatedSplineSurface, weight: float = 1.0
    ) -> None:
        """
        Initializes the `SplinePositionLoss` with a tessellated spline surface and a
        `weight` that is divided by the number of patches in the spline surface.

        Internally, a list of `PositionLoss` instances is created for each patch in the
        tessellated spline surface. `weight` is passed to the constructor for each
        patch.

        :raises ValueError: If the weight is negative.
        """
        super().__init__(weight=weight)
        self._tessellated_spline = tessellated_spline

        self._per_patch_losses = [
            PositionLoss(mesh=patch, weight=weight / len(self._tessellated_spline))
            for patch in self._tessellated_spline
        ]

    def evaluate(self) -> "mi.TensorXf":
        """
        Evaluate the spline patch loss by summing the individual patch position losses.

        :returns: The total loss as a Tensor of shape (1,).
        """
        total_loss = mi.TensorXf(0.0)

        for patch_loss in self._per_patch_losses:
            total_loss += patch_loss.evaluate()

        return total_loss
