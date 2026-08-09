from dataclasses import dataclass

from rebel_shape.scripts.configs.scene_optimization_config import (
    SceneOptimizationConfig,
)


@dataclass(kw_only=True)
class PoseOptimizationConfig(SceneOptimizationConfig):
    """
    Configuration for pose optimization.

    The initial guess is a sphere(oid) with a specified radius, which needs to be
    moved, rotated, and scaled to best fit the initial configuration of the scene.
    Thus, this is the first step if the pose is not known a priori.

    Optimization is performed for `optimization_iterations` iterations, where the
    pose is optimized using the specified loss functions and optimizers. Typically,
    a simple L1 loss is best suited for this task, as it is robust to outliers and
    provides a good initial guess for the pose.

    `jitter` adds a small symmetry-breaking perturbation to the initial scale and
    translation (proportional to the radius); `seed` controls reproducibility of
    that draw. They are honored by `PoseOptimizer` and ignored by callers that
    do not plumb them through.

    The radius of the initial sphere proxy is not configured here. The dedicated
    pose script computes it via a golden-section search; the combined
    mesh/spline scripts expose it as a top-level field of their own parameter
    dataclass.
    """

    optimization_iterations: int = 10
    """Number of iterations for pose optimization"""
    jitter: float = 0.01
    """
    Magnitude of the symmetry-breaking perturbation applied to the initial pose.
    Each of the 6 translation/scale components receives `jitter * N(0, 1)` (with
    translation scaled by the radius). Set to 0 to disable.
    """
    seed: int = 0
    """Random seed used to draw the jitter values."""

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.optimization_iterations < 1:
            raise ValueError("`optimization_iterations` must be a positive integer.")
        if self.jitter < 0.0:
            raise ValueError("`jitter` must be a non-negative real number.")
