import mitsuba as mi

from forge_sim.regularization.regularization import Regularization


class CompositeRegularization(Regularization):
    """
    Allows multiple regularization terms to be combined into a single loss function
    that preserves the `Regularization` interface.

    It receives a list of `Regularization` instances and a `composite_weight` that
    is applied to the loss computed from the combined regularization terms. Each
    regularization term has its own weight. The `composite_weight` can be viewed as
    a single scaling factor while individual regularization terms have weights that
    reflect their importance in the overall loss.
    """

    def __init__(
        self, regularization_terms: list[Regularization], composite_weight: float = 1.0
    ) -> None:
        """
        Initializes the `CompositeRegularization` with a list of `regularization_terms`
        and a `composite_weight` that scales the total loss.

        The `composite_weight` is applied to the sum of the individual regularization
        terms' losses, and should typically be set to 1.0.

        :raises ValueError: If the list of `regularization_terms` is empty or if
            `composite_weight` is negative.
        """
        if len(regularization_terms) == 0:
            raise ValueError("The list of `regularization_terms` cannot be empty.")
        if composite_weight < 0:
            raise ValueError("`composite_weight` must be non-negative.")

        super().__init__(weight=composite_weight)
        self._regularization_list = regularization_terms

    def evaluate(self) -> "mi.TensorXf":
        """
        Evaluates the composite regularization loss by summing the individual losses
        from each regularization term in the list. Finally, it scales the total loss
        by the `composite_weight`.

        :returns: A tensor with a single element representing the total loss.
        """
        loss = mi.TensorXf(0.0)

        for regularization in self._regularization_list:
            loss += regularization.evaluate()

        return loss * self.weight