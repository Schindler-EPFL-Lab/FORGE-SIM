import abc

import mitsuba as mi


class Regularization(abc.ABC):
    """
    Interface that defines a regularization loss for optimization problems.

    While the primary loss in reconstruction comes from how closely the reconstructed
    shape matches the target images, regularization losses are used to "guide" the
    optimization process towards more desirable solutions, such as smoother surfaces.

    The origin of the loss is not a priori clear, so the `evaluate` method takes no
    arguments. This means that the reference to the scene or objects that are being
    optimized should be stored in the class instance itself, providing a level
    of encapsulation and flexibility for different types of losses.
    """

    def __init__(self, weight: float) -> None:
        """
        Initializes the `Regularization` loss with a given weight.

        :raises ValueError: If `weight` is negative.
        """
        if weight < 0:
            raise ValueError("Weight must be non-negative.")

        self._weight = weight

    @property
    def weight(self) -> float:
        """
        Weight of the regularization term.

        :returns: The weight of the regularization term.
        """
        return self._weight

    @abc.abstractmethod
    def evaluate(self) -> "mi.TensorXf":
        """
        Returns a Tensor with a single element representing the regularization loss.

        :returns: Tensor of shape (1,) containing the regularization loss value.
        :raises NotImplementedError: if the concrete implementation has not
            implemented this property
        """
        raise NotImplementedError(
            "Classes that inherit from `Regularization` need to implement the "
            "`evaluate` method."
        )
