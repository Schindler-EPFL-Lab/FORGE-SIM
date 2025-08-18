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
    """

    initial_sphere_radius: float
    """Radius of the initial sphere used for optimization"""
    optimization_iterations: int = 10
    """Number of iterations for pose optimization"""

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.initial_sphere_radius <= 0.0:
            raise ValueError("`initial_sphere_radius` must be a positive real number.")
        if self.optimization_iterations < 1:
            raise ValueError("`optimization_iterations` must be a positive integer.")
