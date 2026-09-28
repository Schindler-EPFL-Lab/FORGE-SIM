import drjit as dr
import mitsuba as mi


class SigmoidParametrization:
    """
    Provides methods to apply a sigmoid transformation and its inverse.

    This is useful for ensuring that certain parameters remain within a specific
    range, typically between 0 and 1. In particular, it is introduced for texture
    optimization.

    Naively optimizing the texture by simply updating the albedo runs the risk of
    exiting the interval of physical solutions, [0.0, 1.0]. Just clipping the values
    to [0.0, 1.0] would not be enough as the gradients would be undefined at the
    boundaries and this approach exhibited many artifacts when it was attempted

    Applying a sigmoid projection 1.0 / (1 + exp(-`_projection_steepness` * x)) maps
    the real line to the interval [0.0, 1.0] (additional clipping is applied to avoid
    numerical issues), enabling the use of standard optimization techniques and
    ensuring smooth gradients.

    Additionally, to prevent issues with underflow, the interval is clamped to
    [`SIGMOID_EPSILON`, 1.0 - `SIGMOID_EPSILON`] to make sure the computed mapping
    is always in range [0.0, 1.0] without getting stuck at the extremes.
    """

    SIGMOID_EPSILON: float = 1e-4

    def __init__(self, projection_steepness: float = 1.0) -> None:
        """
        Initializes the sigmoid parametrization with `projection_steepness`.

        The steepness is used to compute the approximate clipping for `from_parametric`
        method.

        :raises ValueError: If `projection_steepness` is a negative number.
        """
        if projection_steepness <= 0:
            raise ValueError(
                f"Projection steepness must be positive, got {projection_steepness}."
            )
        self._projection_steepness = projection_steepness

        self._approximate_clip = -dr.log(self.SIGMOID_EPSILON) / projection_steepness
        """
        The smallest and largest real numbers that are passed to avoid numerical issues
        when applying the sigmoid function. 
        """

    def from_parametric(self, params: "mi.TensorXf") -> "mi.TensorXf":
        """
        Projects the input `params` from the real line to the interval
        [`SIGMOID_EPSILON`, 1.0 - `SIGMOID_EPSILON`] using the sigmoid function.

        Input `params` are clipped to [-2 `_approximate_clip`, 2 `_approximate_clip`]
        to avoid numerical issues when applying the sigmoid function.
        """
        # Avoid floating point underflow (happens at -1e-38)
        clipped_params = dr.clip(
            params,
            -2.0 * self._approximate_clip,
            2.0 * self._approximate_clip,
        )

        return self.SIGMOID_EPSILON + (1.0 - 2.0 * self.SIGMOID_EPSILON) / (
            1.0 + dr.exp(-self._projection_steepness * clipped_params)
        )

    def to_parametric(self, params: "mi.TensorXf") -> "mi.TensorXf":
        """
        Projects the input `params` from the interval [0.0, 1.0] to the real line using
        the inverse sigmoid function.

        Input `params` are clipped to [2 `SIGMOID_EPSILON`, 1.0 - 2 `SIGMOID_EPSILON`]
        to avoid numerical issues when applying the inverse sigmoid function. Note
        that due to some peculiar numerical behavior, value 2 * `SIGMOID_EPSILON` is not
        mapped exactly to the negative of -2 * `SIGMOID_EPSILON` (about 3E-05
        difference).
        """
        clipped_params = dr.clip(
            params,
            2.0 * self.SIGMOID_EPSILON,
            1.0 - 2.0 * self.SIGMOID_EPSILON,
        )

        return (1.0 / self._projection_steepness) * dr.log(
            (clipped_params - self.SIGMOID_EPSILON)
            / (1.0 - self.SIGMOID_EPSILON - clipped_params)
        )
