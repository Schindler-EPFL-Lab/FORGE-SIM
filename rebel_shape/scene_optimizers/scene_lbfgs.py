import drjit as dr
import mitsuba as mi
import numpy as np

from rebel_shape import logger
from rebel_shape.data_processing.masked_pose_image import MaskedPoseImage
from rebel_shape.scene_optimizers.lbfgs import LBFGS
from rebel_shape.scene_optimizers.scene_optimizer import (
    MitsubaOptimizationType,
    SceneOptimizer,
)


class SceneLBFGS(SceneOptimizer):
    """
    L-BFGS scene optimizer that wraps the Mitsuba `LBFGS` optimizer.

    L-BFGS is a quasi-Newton optimization method that is typically used for
    deterministic optimization. It estimates the Hessian of the objective function
    based on gradients and parameters from a set of previous optimization step.
     This allows it to learn the scale of the problem and thus the choice of the
    learning rate becomes much less impactful.

    The `history_size` is an adjustable parameter, and it should be chosen large
    enough to stabilize the optimization, but not so large that it stalls it with
    outdated curvature data.

    Additionally, unlike stochastic optimization methods that typically take a fixed
    size step, regardless of whether this step reduces the objective function value,
    L-BFGS uses backtracking line search with Armijo and (optionally) Wolfe conditions.
    This ensures a sufficient decrease of the objective function - Armijo, and
    the directional derivative - Wolfe.

    It is typically not used for shape optimization as the objective is noisy,
    significantly reducing the fast Newton-like convergence near a minimum. While it
    is not a great choice of gradient preconditioned mesh optimization, it was found
    to work well for spline optimization, preserving their smoothness properties.

    Another particularity of the optimizer is that it typically requires the full
    gradient, i.e. a contribution from all optimization views. This can be very
    expensive and might overflow memory. Thus, an option to use it in batched mode is
    provided. This ensures that the objective for a particular batch does not increase,
    even though the overall objective might. Combining it with a larger history seems
    to produce qualitatively similar results as the full gradient.

    L-BFGS usage for shape optimization:

    By default, it expects to use full gradients, i.e. all images are considered when
    computing the gradient leading to a single update at each iteration (epoch).
    This is memory intensive: in practical experiments about 50 images can be optimized
    while fitting into a 16 GB memory buffer (GPU memory of an NVIDIA T4 Azure
    instance).

    Furthermore, if there are many input images it is likely that they will overlap
    significantly, meaning that most images do not contribute significantly to the
    gradient. Thus, it is recommended to use multiple large batches with randomization.
    Larger batches (>30 images) that are randomly selected ensure diverse views, while
    the randomization removes the risk of correlating with the past gradients.
    The batches should be as similar in size as possible.

    If 200 images are split into 5 batches of 40 without randomization, then every 5
    iterations the same images would be considered for gradient computation and the
    past gradients with the same set of images would be strongly correlated, biasing
    the curvature estimate and resulting in overly large steps.

    Batching also allows for faster convergence without sacrificing stability, as line
    search is conducted for each batch and each batch approximates the full gradient
    sufficiently well to avoid issues in all but the most pathological cases.
    It is encouraged to keep a larger history in batched mode to have a better estimate
    of curvature; each batch still lacks some information. 5 to 10 past epochs should be
    kept; that is 20 to 40 past states if there are 4 batches per epoch.
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        history_size: int = 10,
        use_wolfe_conditions: bool = False,
        line_search_iterations: int = 15,
        batch_size: int | None = None,
        randomize_image_order: bool = True,
    ) -> None:
        """
        Initialize the L-BFGS optimizer with the specified parameters.

        `learning_rate` rate is the default step size for all variables; it is the
        initial step length that is adjusted during line search.

        `history_size` is the number of previous states that are used to compute a
        quasi-Newton approximation of the inverse Hessian matrix that in turn is used
        to compute the new search direction. Values older than this size are discarded
        from the history to save memory. Typical values are between 5 and 100,
        depending on the problem complexity and memory constraints.

        `use_wolfe_curvature` determines whether to use the Wolfe condition on the
        gradient when conducting line search or to just use the Armijo condition.
        In most cases it is not recommended to use the Wolfe condition as it is both
        more expensive to evaluate and can lead to stalling of the optimization process.

        `line_search_iterations` is the number of iterations of backtracking line search
        to run. It determines how many times the step size is halved until the
        conditions are satisfied. Values higher than 20 are typically not needed -
        taking a very small step (`learning_rate` / 2^20) in a wrong direction is
        unlikely to significantly destabilize the optimization process.

        `batch_size` is the number of sensors to use in each optimization step. If
        `batch_size` is `None`, the full gradient is computed and used for the
        optimization step. If `batch_size` is specified, the optimization is performed
        in batches, where each batch contains `batch_size` sensors. This can help
        reduce AD graph memory usage and speed up the optimization process,
        especially for scenes with many sensors.

        `randomize_image_order` controls whether the order of images is randomized
        during the optimization step. This can make the optimization process more robust
        by preventing the optimizer from getting stuck in local minima.

        :raises ValueError: If `learning_rate` is not positive or if `batch_size` is
            specified and is less than 1.
        """
        super().__init__()

        if learning_rate <= 0.0:
            raise ValueError("`learning_rate` has to be a positive real number")

        if batch_size is not None and batch_size < 1:
            raise ValueError(
                "`batch_size` has to be a positive integer or `None` "
                "for full batch optimization."
            )

        self._batch_size = batch_size
        self._randomize_image_order = randomize_image_order

        self._lbfgs = LBFGS(
            learning_rate=learning_rate,
            history_size=history_size,
            use_wolfe_curvature=use_wolfe_conditions,
            line_search_iterations=line_search_iterations,
        )

    def run_optimization_step(
        self,
        scene: "mi.Scene",
        scene_params: "mi.SceneParams",
        param_optimization_configs: list[SceneOptimizer.ParamOptimizationConfig],
        ref_images: list[MaskedPoseImage],
        seed: int,
    ) -> tuple["mi.TensorXf", "mi.TensorXf"]:
        """
        Execute a single optimization step.

        Images are rendered based on `scene` and `scene_params`.
        All views specified by sensors of `ref_images` are considered and `image_loss`
        is computed for the rendered images and `ref_images` for every internal
        parameter of the scene. The regularization loss is computed after each
        update (at the end of the batch) and added to the total loss.

        How the scene parameters should be updated is specified and which loss is
        applied is defined by `param_optimization_configs`. The order in the list
        defines the order in which they are optimized.

        Rendering uses the provided `seed` which should be different for each
        iteration of the optimization procedure to avoid correlation.

        The update order for `param_optimization_configs` is sequential, only one
        parameter configuration is optimized at a time, with the same order as they
        appear in the list. This either results in a single optimization step
        (full batch) or multiple optimization steps (batched) for each parameter.

        Batch indices are not permuted, unlike the typical behavior of Adam optimizers.
        This was done to make the optimization process more 'deterministic' and
        hopefully more stable, as the L-BFGS optimizer is not designed to handle
        stochastic updates.

        :returns: A tuple of tensors of shape (1,) that contain the total loss and the
            regularization loss of the optimization step.
        """
        sensor_count = len(ref_images)

        total_loss = mi.TensorXf(0.0)
        total_reg_loss = mi.TensorXf(0.0)

        batch_size = self._batch_size if self._batch_size else sensor_count

        ordered_indices = (
            np.arange(sensor_count)
            if not self._randomize_image_order
            else np.random.permutation(sensor_count)
        )

        for param_config in param_optimization_configs:
            for start_idx in range(0, sensor_count, batch_size):
                end_idx = min(start_idx + batch_size, sensor_count)
                batch_indices = ordered_indices[start_idx:end_idx]

                batch_image_loss, batch_reg_loss = self._objective_function(
                    scene=scene,
                    scene_params=scene_params,
                    ref_images=[ref_images[i] for i in batch_indices],
                    param_config=param_config,
                    parameters=self.parameters[param_config.parameter_key],
                    seed=seed,
                )
                total_batch_loss = batch_image_loss + batch_reg_loss

                # Set the line search function for this batch
                self._lbfgs.line_search_fn = lambda parameters: sum(
                    self._objective_function(
                        scene=scene,
                        scene_params=scene_params,
                        ref_images=[ref_images[i] for i in batch_indices],
                        param_config=param_config,
                        parameters=parameters[param_config.parameter_key],
                        seed=seed,
                    )
                )

                dr.backward(total_batch_loss)
                self._lbfgs.step(total_batch_loss)

                weight = len(batch_indices) / sensor_count
                total_loss += weight * batch_image_loss
                total_loss += weight * batch_reg_loss
                total_reg_loss += weight * batch_reg_loss

        total_loss /= len(param_optimization_configs)
        total_reg_loss /= len(param_optimization_configs)

        return total_loss, total_reg_loss

    def set_parameters(self, parameters: dict[str, MitsubaOptimizationType]) -> None:
        """
        Initializes the `LBFGS` optimizer with variables that will be optimized.

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
            self._lbfgs[key] = parameter

    def set_learning_rate(self, learning_rates: dict[str, float]) -> None:
        """
        Set the learning rate for each parameter of the underlying `LBFGS` optimizer.

        This method is used to set the learning rate for each parameter that is
        being optimized. It is called before each optimization step to adjust the
        learning rate based on the current iteration. The learning rate of just a
        subset of parameters can be set.

        L-BFGS typically uses a single learning rate for all parameters as it estimates
        the curvature and thus the optimal step size during the optimization. Typical
        learning rates are in the range [0.2, 2.0]. Values larger than 1.0 can be useful
        for problems with very small gradients and line search will lead to smaller
        steps if the objective function does not decrease.

        :raises ValueError: if `learning_rates` dictionary does not contain
            a valid parameter key or if the learning rate is not positive.
        """
        for key, learning_rate in learning_rates.items():
            if key not in self._lbfgs.keys():
                raise ValueError(f"Parameter '{key}' not found in optimizer variables.")
            if learning_rate <= 0.0:
                raise ValueError(
                    f"Learning rate for '{key}' must be a positive real number."
                )

            if learning_rate < 0.1:
                logger.warning(
                    f"Learning rate for '{key}' is set to {learning_rate}, "
                    "which is below the typical lower bound of 0.2. "
                )

        self._lbfgs.set_learning_rate(learning_rates)

    @property
    def parameters(self) -> dict[str, MitsubaOptimizationType]:
        """
        Returns the internal state of the `LBFGS` optimizer.

        Mitsuba optimizers work with a set of parameters that are accessed by an
        appropriate key. This property returns a dictionary of the current state of all
        parameters that are being optimized.

        :returns: A dictionary where keys are parameter names and values are
            `mi.Float` or `mi.TensorXf` objects representing the current state of
            the parameters.
        """
        return self._lbfgs.variables
