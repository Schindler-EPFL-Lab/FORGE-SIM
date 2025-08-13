import drjit as dr
import mitsuba as mi
import numpy as np

from rebel_shape.data_processing.masked_pose_image import MaskedPoseImage
from rebel_shape.scene_optimizers.scene_adam import SceneAdam
from rebel_shape.scene_optimizers.scene_optimizer import SceneOptimizer


class SceneAdamSimultaneous(SceneAdam):
    """
    Concrete implementation of the Adam optimizer that simultaneously optimizes all
    parameters.

    On top of providing batching, this class implements a simultaneous update
    strategy, where all parameters are optimized at once. This is perhaps the most
    common optimization strategy.

    Since each parameter can have its own loss, losses for all parameters are computed
    and summed before backpropagation. This allows the optimizer to update all
    parameters based on the combined gradient. Note that this can lead to
    situations where parameters interfere with each other during optimization, which
    can make it difficult to optimize them effectively.
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
        super().__init__(
            learning_rate=learning_rate,
            batch_size=batch_size,
            uniform=uniform,
            randomize_image_order=randomize_image_order,
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
        is computed for the rendered images and `ref_images`.. The regularization
        loss is computed after each update (at the end of the batch) and added to the
        total loss.

        Rendering uses the provided `seed` which should be different for each
        iteration of the optimization procedure to avoid correlation.

        All parameters specified by `param_optimization_configs` are optimized
        simultaneously. The loss is accumulated for all parameters before
        backpropagation, allowing the optimizer to update all parameters based on
        the combined gradient.

        :returns: A tuple of tensors of shape (1,) that contain the total loss and the
            regularization loss of the optimization step.
        """
        sensor_count = len(ref_images)

        total_loss = mi.TensorXf(0.0)
        total_reg_loss = mi.TensorXf(0.0)

        ordered_indices = (
            np.arange(sensor_count)
            if not self._randomize_image_order
            else np.random.permutation(sensor_count)
        )
        for start_idx in range(0, sensor_count, self._batch_size):
            end_idx = min(start_idx + self._batch_size, sensor_count)
            batch_indices = ordered_indices[start_idx:end_idx]

            batch_image_loss = mi.TensorXf(0.0)
            batch_reg_loss = mi.TensorXf(0.0)

            for param_config in param_optimization_configs:
                image_loss, reg_loss = self._objective_function(
                    scene=scene,
                    scene_params=scene_params,
                    ref_images=[ref_images[i] for i in batch_indices],
                    param_config=param_config,
                    parameters=self.parameters[param_config.parameter_key],
                    seed=seed,
                )
                batch_image_loss += image_loss
                batch_reg_loss += reg_loss

            batch_image_loss /= len(param_optimization_configs)
            batch_reg_loss /= len(param_optimization_configs)

            total_batch_loss = batch_image_loss + batch_reg_loss

            dr.backward(total_batch_loss)
            self._adam.step()

            weight = len(batch_indices) / sensor_count

            total_loss += weight * batch_image_loss
            total_loss += weight * batch_reg_loss
            total_reg_loss += weight * batch_reg_loss

        return total_loss, total_reg_loss
