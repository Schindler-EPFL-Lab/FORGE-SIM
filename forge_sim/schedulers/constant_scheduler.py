from forge_sim.schedulers.optimization_scheduler import OptimizationScheduler


class ConstantScheduler(OptimizationScheduler):
    """Optimization scheduler that maintains a constant `learning_rate`.

    This is the simplest learning rate scheduler that keeps the learning rate
    constant throughout the entire optimization procedure. Typically, it is not
    the best as small step size means the optimization would progress too slowly, while
    a large learning rate later on introduces oscillations.
    """

    def __init__(self, learning_rate=1.0) -> None:
        """
        `learning_rate` is the fixed learning rate used for all optimization steps

        :raises ValueError: If `learning_rate` is not positive.
        """
        if learning_rate <= 0.0:
            raise ValueError("Learning rate must be a positive real number")

        self._learning_rate = learning_rate

    def learning_rate_for_step(self, step: int) -> float:
        """
        Returns the learning rate for the given step.

        Since this is a constant scheduler, this method always returns the same
        learning rate, regardless of the `step`.

        :returns: The fixed learning rate.
        """
        return self._learning_rate
