from dataclasses import dataclass

from dataclasses_reverse_cli.reverse_cli import ReverseCli


@dataclass
class ShapeOptimization(ReverseCli):
    shape_smoothing_strength: float = 100.0
    """Laplacian smoothing parameter for shape optimization"""
    shape_l1_loss_weight: float = 0.9
    """
    Weight of the L1 image loss for shape in composite loss. The other loss is either
    SSIM or L2, depending on `use_ssim_loss`, and is weighted by 
    1.0 - `shape_l1_loss_weight`
    """
    shape_learning_rate_start: float = 0.6
    """Adam learning rate at the start of shape optimization"""
    shape_learning_rate_end: float = 0.15
    """Adam learning rate at the end of shape optimization"""
    shape_learning_rate_steepness: float = 1.5
    """How quickly the shape learning rate converges to the end value"""
    shape_batch_size: int = 12
    """Number of images used for shape gradient computation"""

    def __post_init__(self) -> None:
        if self.shape_smoothing_strength <= 0.0:
            raise ValueError(
                "`shape_projection_smoothing` must be a positive real number."
            )
        if self.shape_l1_loss_weight < 0.0 or self.shape_l1_loss_weight > 1.0:
            raise ValueError(
                "`shape_l1_loss_weight` must be a real number in [0.0, 1.0]."
            )

        if self.shape_learning_rate_start < 0.0:
            raise ValueError("`shape_learning_rate_start` must be non-negative.")
        if self.shape_learning_rate_end < 0.0:
            raise ValueError("`shape_learning_rate_end` must be non-negative.")
        if self.shape_learning_rate_end > self.shape_learning_rate_start:
            raise ValueError(
                "`shape_learning_rate_end` must be smaller than "
                "`shape_learning_rate_start`."
            )
        if self.shape_learning_rate_steepness <= 0.0:
            raise ValueError(
                "`shape_learning_rate_steepness` must be a positive real number."
            )
        if self.shape_batch_size < 1:
            raise ValueError("`batch_size` must be a positive integer")
