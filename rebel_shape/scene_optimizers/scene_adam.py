import abc

import mitsuba as mi

from rebel_shape.scene_optimizers.scene_optimizer import (
    MitsubaOptimizationType,
    SceneOptimizer,
)


class SceneAdam(SceneOptimizer, abc.ABC):
    """
    Generalized Adam optimizer that wraps the `mi.ad.Adam` Mitsuba class.

    Scene properties can fundamentally be updated at different stages of the
    optimization step. Each step goes through all images (sequentially or randomly)
    and the details determine the strategy, e.g. one update (full), multiple updates
    (batched), update after each image (single)... This is controlled with `batch_size`.

    Furthermore, this class does not implement a concrete `run_optimization_step`
    method as when optimizing multiple parameters there are multiple update strategies.
    These strategies (sequential, interleaved, and simultaneous) are implemented in
    respective classes.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        batch_size: int = 5,
        uniform: bool = False,
        randomize_image_order: bool = True,
    ) -> None:
        """
        `learning_rate` determines the step size. It is overridden by the learning rate
        defined by `OptimizationScheduler` when calling `optimize`. Each parameter
        can also have its own learning rate set via `set_learning_rate`.

        `uniform` controls whether each element of the parameters vector has a separate
        variance estimate or the estimate is uniform (highest applied to all). This is
        useful when optimizing discontinuous objective functions (e.g. vertex
        positions).

        `randomize_image_order` controls whether the order of images is randomized
        during the optimization step. This can make the optimization process more robust
        by preventing the optimizer from getting stuck in local minima.

        :raises ValueError: If `learning_rate` is not positive or if `batch_size` is
            less than 1.
        """
        super().__init__()

        if learning_rate <= 0.0:
            raise ValueError("`learning_rate` has to be a positive real number")
        if batch_size < 1:
            raise ValueError("`batch_size` has to be a positive integer")

        self._batch_size = batch_size
        self._randomize_image_order = randomize_image_order

        self._adam = mi.ad.Adam(lr=learning_rate, uniform=uniform)

    def set_parameters(self, parameters: dict[str, MitsubaOptimizationType]) -> None:
        """
        Initializes the `mi.ad.Adam` optimizer with variables that will be optimized.

        This method needs to be called before starting the optimization process to
        configure the optimizer with the properties that will be optimized. An error
        is raised if this method is not called before `optimize`.

        `parameters` is a dictionary whose keys are user defined names for
        optimization variables and values are 1D Mitsuba float32 arrays (`mi.Float`) or
        tensors (`mi.TensorXf`) that represent the current state of the variable.

        No checks are performed on the keys of the `parameters` dictionary as this is
        a configuration step. It can also be called during optimization to change
        the shape of the parameters (e.g. when optimizing a mesh and the number of
        vertices changes).
        """
        for key, parameter in parameters.items():
            self._adam[key] = parameter

    def set_learning_rate(self, learning_rates: dict[str, float]) -> None:
        """
        Set the learning rate for each parameter of the underlying `mi.ad.Adam`
        optimizer.

        This method is used to set the learning rate for each parameter that is
        being optimized. It is called before each optimization step to adjust the
        learning rate based on the current iteration. The learning rate of just a
        subset of parameters can be set.

        :raises ValueError: if `learning_rates` dictionary does not contain
            a valid parameter key or if the learning rate is not positive.
        """
        for key, learning_rate in learning_rates.items():
            if key not in self._adam.keys():
                raise ValueError(f"Parameter '{key}' not found in optimizer variables.")
            if learning_rate <= 0.0:
                raise ValueError(
                    f"Learning rate for '{key}' must be a positive real number."
                )

        self._adam.set_learning_rate(learning_rates)

    @property
    def parameters(self) -> dict[str, MitsubaOptimizationType]:
        """
        Returns the internal state of the `mi.ad.Adam` optimizer.

        Mitsuba optimizers work with a set of parameters that are accessed by an
        appropriate key. This property returns a dictionary of the current state of all
        parameters that are being optimized.

        :returns: A dictionary where keys are parameter names and values are
            `mi.Float` or `mi.TensorXf` objects representing the current state of
            the parameters.
        """
        return dict(self._adam)
