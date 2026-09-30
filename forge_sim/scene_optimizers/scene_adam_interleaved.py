import drjit as dr
import mitsuba as mi

from forge_sim.data_processing.masked_pose_image import MaskedPoseImage
from forge_sim.scene_optimizers.scene_adam import SceneAdam
from forge_sim.scene_optimizers.scene_optimizer import SceneOptimizer


class SceneAdamInterleaved(SceneAdam):
    """
    Concrete implementation of the Adam optimizer that sequentially optimizes the
    parameters.

    On top of providing batching, this class implements an interleaved update
    strategy, where batches of parameters are optimized in the order specified by
    `param_optimization_configs`.

    Each parameter is optimized in a round-robin fashion at the batch level, meaning
    that for each batch of images, the optimizer will update one parameter from each
    configuration before moving to the next batch. This allows for a more balanced
    optimization process, where each parameter gets updated in each batch, rather than
    optimizing all parameters at once (simultaneous) or one after another after all
    images (sequential).

    This pattern is sometimes referred to block coordinate descent, where a block of
    common parameters is optimized at a time.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        batch_size: int = 5,
        uniform: bool = False,
        randomize_image_order: bool = True,
        image_order_seed: int = 42,
    ) -> None:
        """
        `learning_rate` determines the step size. It is overridden by the learning rate
        defined by `OptimizationScheduler` when calling `optimize`. Each parameter
        can also have its own learning rate set via `set_learning_rate`.

        `uniform` controls whether each element of the parameters vector has a separate
        variance estimate or the estimate is uniform (highest applied to all). This is
        useful when optimizing discontinuous objective functions (e.g. vertex
        positions).

        `randomize_image_order` controls whether the order of images is shuffled
        before each optimization step. This can help improve convergence by reducing
        correlations between images. The shuffling is done using a fixed seed
        (`image_order_seed`) to ensure reproducibility. As it is possible for batches
        to have different sizes, the order of batches is also shuffled.

        :raises ValueError: If `learning_rate` is not positive or if `batch_size` is
            less than 1.
        """
        super().__init__(
            learning_rate=learning_rate,
            batch_size=batch_size,
            uniform=uniform,
            randomize_image_order=randomize_image_order,
            image_order_seed=image_order_seed,
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

        The order in which the parameters are optimized is defined by
        `param_optimization_configs`, mirroring the order from the list.
        The batches are interleaved in a round-robin fashion, based on the
        aforementioned order. The optimizer updates one parameter from each
        configuration before moving to the next batch. This allows for a more
        balanced optimization process as it allows each parameter to update the
        scene based on its gradient, not a cumulative gradient of all parameters.

        :returns: A tuple of tensors of shape (1,) that contain the total loss and the
            regularization loss of the optimization step.
        """
        sensor_count = len(ref_images)

        total_loss = mi.TensorXf(0.0)
        total_reg_loss = mi.TensorXf(0.0)

        for batch_indices in self._batch_maker.iteration_batch_indices(sensor_count):
            for param_config in param_optimization_configs:
                batch_image_loss, batch_reg_loss = self._objective_function(
                    scene=scene,
                    scene_params=scene_params,
                    ref_images=[ref_images[i] for i in batch_indices],
                    param_config=param_config,
                    parameters=self.parameters[param_config.parameter_key],
                    seed=seed,
                )
                total_batch_loss = batch_image_loss + batch_reg_loss

                dr.backward(total_batch_loss)
                self._adam.step()

                weight = len(batch_indices) / sensor_count

                total_loss += weight * batch_image_loss
                total_loss += weight * batch_reg_loss
                total_reg_loss += weight * batch_reg_loss

        total_loss /= len(param_optimization_configs)
        total_reg_loss /= len(param_optimization_configs)

        return total_loss, total_reg_loss
