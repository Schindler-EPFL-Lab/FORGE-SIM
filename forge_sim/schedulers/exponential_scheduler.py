import math

from forge_sim.schedulers.optimization_scheduler import OptimizationScheduler


class ExponentialScheduler(OptimizationScheduler):
    """
    Optimization scheduler that changes the learning rate exponentially between
    a starting learning rate at a specified iteration and an ending learning rate at
    another specified iteration.

    The learning rate follows a curve of the form a * exp(b * x) + c. It is a good
    default choice as it typically suppresses oscillations that would usually occur
    if the learning rate is constant throughout optimization.
    """

    def __init__(
        self,
        start_learning_rate: float,
        end_learning_rate: float,
        start_iteration: int = 0,
        end_iteration: int = 100,
        steepness: float = 2.5,
    ) -> None:
        """
        Initialize the exponential scheduler.

        Before `start_iteration`, the learning rate is equal to `start_learning_rate`
        while after `end_iteration`, the learning rate is equal to `end_learning_rate`.
        Between these iterations, the learning rate follows a curve of the form:
        a * exp(b * x) + c, where:
            - x is the relative step size (subtracted start_iteration from step)
            - a is the magnitude of the change in learning rate
            - b is the steepness of the curve
                - positive if `end_learning_rate` > `start_learning_rate`
                - negative if end_learning_rate < start_learning_rate
            - c is the offset of the curve

        `steepness` controls the characteristic length defined as
        (`end_iteration` - `start_iteration`) / steepness.
        When `end_learning_rate` > `start_learning_rate`, `steepness` controls how late
        in the optimization procedure the larger steps kick in. On the other hand, for
        `end_learning_rate` < `start_learning_rate` it controls how quickly smaller
        steps become usual. Choosing values close to 0 reproduces a linear schedule.

        :raises ValueError: If `start_learning_rate`, `end_learning_rate`,
            `start_iteration`, `end_iteration`, or `steepness` are not positive,
            if `start_learning_rate` and `end_learning_rate` are the same,
            or if `end_iteration` is smaller than `start_iteration`.
        """
        if start_learning_rate < 0.0 or end_learning_rate < 0.0:
            raise ValueError("Learning rates must be non-negative real numbers")
        if start_learning_rate == end_learning_rate:
            raise ValueError(
                "Learning rates must be different, use `ConstantScheduler` instead"
            )

        if start_iteration < 0 or end_iteration < 0:
            raise ValueError("Iterations must be non-negative integers")
        if end_iteration <= start_iteration:
            raise ValueError("End iteration must be greater than start iteration")
        if steepness <= 0.0:
            raise ValueError("Steepness must be a positive real number")

        self._start_learning_rate = start_learning_rate
        self._end_learning_rate = end_learning_rate
        self._start_iteration = start_iteration
        self._end_iteration = end_iteration
        self._steepness = (
            steepness if end_learning_rate > start_learning_rate else -steepness
        )

        transition_length = end_iteration - start_iteration

        self._magnitude = (end_learning_rate - start_learning_rate) / (
            math.exp(self._steepness) - 1.0
        )
        self._offset = -self._magnitude + self._start_learning_rate
        self._exponent_factor = self._steepness / transition_length

    def learning_rate_for_step(self, step: int) -> float:
        """
        Returns the learning rate for the given step.

        - If step < `start_iteration`, returns `start_learning_rate`
        - If step > `end_iteration`, returns `end_learning_rate`
        - Otherwise, returns a value along the exponential curve
          (increasing or decreasing)

        :return: The learning rate for the given step.
        """
        if step < self._start_iteration:
            return self._start_learning_rate
        if step >= self._end_iteration:
            return self._end_learning_rate

        # Calculate normalized step (0 to transition_length)
        normalized_step = step - self._start_iteration

        # Calculate learning rate using the exponential formula
        exp_factor = math.exp(self._exponent_factor * normalized_step)
        return self._magnitude * exp_factor + self._offset
