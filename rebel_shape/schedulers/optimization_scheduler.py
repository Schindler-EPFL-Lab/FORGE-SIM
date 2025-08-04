import abc


class OptimizationScheduler(abc.ABC):
    """
    Interface for optimization learning rate schedulers.

    Defines the interface for learning rate schedulers used during optimization.
    Different optimization approaches may require different learning rate schedules
    (constant, exponential decay, etc.) to achieve optimal results.

    Concrete implementations (like `ConstantScheduler`) should inherit from this class
    and implement the `learning_rate_for_step` method to define the learning rate
    for every `step` >= 0
    """

    @abc.abstractmethod
    def learning_rate_for_step(self, step: int) -> float:
        """
        Returns the learning rate for the given step.

        Calculates the appropriate learning rate based on the current optimization
        `step` and the scheduler's internal parameters. It is called during the
        optimization process to adjust the learning rate.

        :returns: The learning rate for the provided `step`
        :raises NotImplementedError: if the method was not implemented by a child class
        """
        raise NotImplementedError(
            "Classes that inherit from `OptimizationScheduler` need to implement "
            "`learning_rate_for_step`."
        )
