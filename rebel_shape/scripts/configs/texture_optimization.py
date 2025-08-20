from dataclasses import dataclass

from dataclasses_reverse_cli.reverse_cli import ReverseCli


@dataclass
class TextureOptimization(ReverseCli):
    texture_resolution: int = 1024
    """Resolution of the texture used for optimization"""
    texture_projection_steepness: float = 1.0
    """Steepness of the sigmoid projection for texture optimization"""
    texture_l1_loss_weight: float = 0.9
    """
    Weight of the L1 image loss for texture in composite loss. The other loss is either
    SSIM or L2, depending on `use_ssim_loss`, and is weighted by 
    1.0 - `texture_l1_loss_weight`
    """
    texture_learning_rate_start: float = 0.005
    """Adam learning rate at the start of shape optimization"""
    texture_learning_rate_end: float = 0.025
    """Adam learning rate at the end of shape optimization"""
    texture_learning_rate_steepness: float = 3.0
    """How late in the optimization procedure the larger steps kick in"""
    texture_batch_size: int = 1
    """Number of images used for texture gradient computation"""

    def __post_init__(self) -> None:
        if self.texture_resolution < 1:
            raise ValueError("`texture_resolution` must be a positive integer.")
        if self.texture_projection_steepness <= 0.0:
            raise ValueError(
                "`texture_projection_steepness` must be a positive real number."
            )
        if self.texture_l1_loss_weight < 0.0 or self.texture_l1_loss_weight > 1.0:
            raise ValueError(
                "`texture_l1_loss_weight` must be a real number in [0.0, 1.0]."
            )

        if self.texture_learning_rate_start < 0.0:
            raise ValueError("`texture_learning_rate_start` must be non-negative.")
        if self.texture_learning_rate_end < 0.0:
            raise ValueError("`texture_learning_rate_end` must be non-negative.")
        if self.texture_learning_rate_end < self.texture_learning_rate_start:
            raise ValueError(
                "`texture_learning_rate_start` must be smaller than "
                "`texture_learning_rate_end`."
            )
        if self.texture_learning_rate_steepness <= 0.0:
            raise ValueError(
                "`texture_learning_rate_steepness` must be a positive real number."
            )
        if self.texture_batch_size < 1:
            raise ValueError("`batch_size` must be a positive integer")
