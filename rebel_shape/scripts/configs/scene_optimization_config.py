from dataclasses import dataclass

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape.loss_evaluators.composite_loss import CompositeImageLoss
from rebel_shape.loss_evaluators.image_loss_l1 import ImageLossL1
from rebel_shape.loss_evaluators.image_loss_l2 import ImageLossL2
from rebel_shape.loss_evaluators.image_loss_msssim import ImageLossMSSSIM
from rebel_shape.scene_optimizers.scene_adam_sequential import SceneAdamSequential
from rebel_shape.scene_optimizers.scene_lbfgs import SceneLBFGS
from rebel_shape.scene_optimizers.scene_optimizer import SceneOptimizer
from rebel_shape.schedulers.constant_scheduler import ConstantScheduler
from rebel_shape.schedulers.exponential_scheduler import ExponentialScheduler
from rebel_shape.schedulers.optimization_scheduler import OptimizationScheduler


@dataclass(kw_only=True)
class SceneOptimizationConfig(ReverseCli):
    """
    Common configuration for scene optimization from sparse views.

    This configuration is applicable to all scripts that optimize a scene. These
    include the choice of loss functions, optimizers, and learning rate schedules,
    that are common across different optimization tasks such as shape, texture,
    and pose optimization.

    Common optimization settings are available as properties or methods based on the
    configuration parameters:

    1. Using the MS-SSIM loss in combination with L1 loss (weight 0.85 to L1 loss),
    2. The choice of optimizer depends on the `use_lbfgs` parameter, and if L-BFGS
    is used, typically the batch size is set to `None` (i.e., no batching).
    3. The learning rate scheduler is set to `ConstantScheduler` if the start and end
    learning rates are the same, or to `ExponentialScheduler` if they differ.
    """

    use_msssim_loss: bool = True
    """
    If `True`, the MS-SSIM loss is used as a secondary loss for optimization. 
    Otherwise, the L2 loss is used instead.
    """
    l1_loss_weight: float = 0.85
    """
    Weight of the L1 primary image loss in the composite loss. The other loss is 
    either SSIM or L2, depending on `use_ssim_loss`, and is weighted by 
    1.0 - `shape_l1_loss_weight`
    """

    learning_rate_start: float
    """Adam learning rate at the start of shape optimization"""
    learning_rate_end: float
    """Adam learning rate at the end of shape optimization"""
    learning_rate_steepness: float
    """How quickly the shape learning rate converges to the end value"""
    batch_size: int | None
    """Number of images used for gradient computation"""

    uniform_adam: bool
    """
    Whether to use uniform estimate for Adam optimizer when `use_lbfgs` is False.
    """
    use_lbfgs: bool
    """
    Whether to use LBFGS optimizer instead of Adam. If set to `True`, the
    `SceneLBFGS` optimizer is used instead of `SceneAdam`. L-BFGS typically requires
    larger learning rates to be effective (close to 1.0), while these are generally
    too large for Adam.
    """
    lbfgs_history_size: int = 15
    """
    Applied only if `use_lbfgs` is True. Sets how many past gradient information 
    is used to compute the new search direction. If there are multiple batches per each
    iteration, the history should be increased by the number of batches. 
    """

    def __post_init__(self) -> None:
        if self.l1_loss_weight < 0.0 or self.l1_loss_weight > 1.0:
            raise ValueError("`l1_loss_weight` must be a real number in [0.0, 1.0].")
        if self.lbfgs_history_size < 1:
            raise ValueError("`lbfgs_history_size` must be a positive integer.")

        if self.learning_rate_start < 0.0:
            raise ValueError("`learning_rate_start` must be non-negative.")
        if self.learning_rate_end < 0.0:
            raise ValueError("`learning_rate_end` must be non-negative.")
        if self.learning_rate_steepness <= 0.0:
            raise ValueError(
                "`learning_rate_steepness` must be a positive real number."
            )

        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("`batch_size` must be a positive integer")
        if not self.use_lbfgs and self.batch_size is None:
            raise ValueError(
                "`batch_size` must be set if `use_lbfgs` is False. "
                + "LBFGS optimizer does not use batch size."
            )

    @property
    def image_loss(self) -> CompositeImageLoss:
        """
        Returns a composite image loss that combines L1 loss and either
        MS-SSIM or L2 loss, depending on the `use_msssim_loss` flag.

        The L1 loss is weighted by `l1_loss_weight`, while the other loss is weighted
        by `1.0 - l1_loss_weight`. The secondary loss is set to MS-SSIM if
        `use_msssim_loss` is `True`, otherwise it is set to L2 loss.

        :returns: A composite image loss object that combines L1 and either MS-SSIM or
            L2 loss with correct weights.
        """
        return CompositeImageLoss(
            losses_and_weights=[
                (ImageLossL1(), self.l1_loss_weight),
                (
                    ImageLossMSSSIM() if self.use_msssim_loss else ImageLossL2(),
                    1.0 - self.l1_loss_weight,
                ),
            ]
        )

    @property
    def scene_optimizer(self) -> SceneOptimizer:
        """
        Returns an appropriate scene optimizer based on the `use_lbfgs` flag.

        If `use_lbfgs` is `True`, an instance of `SceneLBFGS` is returned with the
        specified `batch_size` and `lbfgs_history_size`. By default, the batch size
        should be set to `None` (all images are used for each iteration), but
        larger batch sizes can be used if memory is constrained.

        Otherwise, an instance of `SceneAdamSequential` is returned with the
        specified `batch_size` and `uniform` set depending on the `uniform_adam` flag.

        :returns: An instance of `SceneOptimizer` that is either a `SceneLBFGS` or
            `SceneAdamSequential`, depending on the `use_lbfgs` flag.
        """
        if self.use_lbfgs:
            return SceneLBFGS(
                batch_size=self.batch_size, history_size=self.lbfgs_history_size
            )

        assert self.batch_size is not None
        return SceneAdamSequential(
            batch_size=self.batch_size, uniform=self.uniform_adam
        )

    def scheduler(self, end_iteration: int) -> OptimizationScheduler:
        """
        Returns an appropriate learning rate scheduler based on the learning rate
        parameters.

        If the start and end learning rates are the same, a `ConstantScheduler`
        is returned. If they differ, an `ExponentialScheduler` is returned that
        smoothly transitions from the start learning rate to the end learning rate
        over the specified number of iterations.

        Since the `end_iteration` parameter is not available in the configuration,
        it is passed as an argument to this method. This allows the scheduler to
        know how many iterations it has to adjust the learning rate over.

        :returns: An instance of `OptimizationScheduler` that is either a
            `ConstantScheduler` or an `ExponentialScheduler`, depending on the
            learning rate parameters.
        """
        if self.learning_rate_start == self.learning_rate_end:
            return ConstantScheduler(learning_rate=self.learning_rate_start)

        return ExponentialScheduler(
            start_learning_rate=self.learning_rate_start,
            end_learning_rate=self.learning_rate_end,
            steepness=self.learning_rate_steepness,
            end_iteration=end_iteration,
        )
