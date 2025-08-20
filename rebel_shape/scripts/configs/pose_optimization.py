from dataclasses import dataclass

from dataclasses_reverse_cli.reverse_cli import ReverseCli


@dataclass
class PoseOptimization(ReverseCli):
    initial_sphere_radius: float = 2.0
    """Radius of the initial sphere used for optimization"""
    optimize_initial_pose: bool = True
    """Whether to execute pose optimization before shape and texture optimization"""
    pose_optimization_iterations: int = 10
    """How many times the """
    pose_learning_rate_start: float = 0.1
    """Adam learning rate at the start of initial pose optimization"""
    pose_learning_rate_end: float = 0.02
    """Adam learning rate at the end of initial pose optimization"""
    pose_learning_rate_steepness: float = 2.0
    """How quickly the pose learning rate converges to the end value"""
    pose_batch_size: int = 12
    """Number of images used for pose gradient computation"""

    def __post_init__(self) -> None:
        if self.pose_learning_rate_start < 0.0:
            raise ValueError("`pose_learning_rate_start` must be non-negative.")
        if self.pose_learning_rate_end < 0.0:
            raise ValueError("`pose_learning_rate_end` must be non-negative.")
        if self.pose_learning_rate_end > self.pose_learning_rate_start:
            raise ValueError(
                "`pose_learning_rate_end` must be smaller than "
                "`pose_learning_rate_start`."
            )
        if self.pose_learning_rate_steepness <= 0.0:
            raise ValueError(
                "`pose_learning_rate_steepness` must be a positive real number."
            )
        if self.pose_batch_size < 1:
            raise ValueError("`batch_size` must be a positive integer")
