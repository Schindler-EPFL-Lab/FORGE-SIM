from dataclasses import dataclass

from rebel_shape.scripts.configs.scene_optimization_config import (
    SceneOptimizationConfig,
)


@dataclass(kw_only=True)
class TextureOptimizationConfig(SceneOptimizationConfig):
    """
    Configuration for texture optimization of a scene.

    The texture resolution dependents on whether a composite spline (typically 6 sides)
    or a mesh (typically a single mesh) is used for optimization. 1024 - 2048 is needed
    for a mesh, while 512 - 1024 is sufficient for a composite spline.

    Projection steepness is used to control the sigmoid projection of the texture
    optimization. The default value of 1.0 was found to work sufficiently well
    as increasing it would require reducing the learning rate, yielding the same
    effect.
    """

    texture_resolution: int
    """Resolution of the texture used for optimization."""
    projection_steepness: float = 1.0
    """Steepness of the sigmoid projection for texture optimization"""

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.texture_resolution < 1:
            raise ValueError("`texture_resolution` must be a positive integer.")
        if self.projection_steepness <= 0.0:
            raise ValueError(
                "`texture_projection_steepness` must be a positive real number."
            )
