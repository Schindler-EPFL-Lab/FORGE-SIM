from dataclasses import dataclass

from forge_sim.scripts.configs.scene_optimization_config import (
    SceneOptimizationConfig,
)


@dataclass(kw_only=True)
class ShapeOptimizationConfig(SceneOptimizationConfig):
    """
    Configuration for mesh-based shape optimization.

    Mesh-based shape optimization requires a single parameter, how strongly the
    gradients are diffused over the surface of the mesh (Laplacian parametrization).

    Smoothing can also be applied to spline-based shape optimization, but this
    typically results in over-smoothing and is not recommended.
    """

    smoothing_strength: float
    """
    Laplacian smoothing smoothing strength for shape optimization. 
    Higher values (20 - 200) are advised for finer meshes (10 - 100 thousand 
    vertices).
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.smoothing_strength < 0.0:
            raise ValueError("`smoothing_strength` must be a non-negative real number.")
