from pathlib import Path
from typing import Any

import mitsuba as mi

from rebel_shape.loss_evaluators.image_loss import ImageLoss
from rebel_shape.loss_evaluators.image_loss_l2 import ImageLossL2
from rebel_shape.property_optimizers.bitmap_texture_optimizer import (
    BitmapTextureOptimizer,
)
from rebel_shape.property_optimizers.mesh_optimizer import MeshOptimizer
from rebel_shape.property_optimizers.property_optimizer import PropertyOptimizer
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.scene_optimizers.scene_optimizer import SceneOptimizer
from rebel_shape.schedulers.optimization_scheduler import OptimizationScheduler


class MeshTextureOptimizer(PropertyOptimizer):
    """
    A class that combines shape and texture optimization of a single triangle mesh.

    It is a thin wrapper around two classes: `MeshOptimizer` and
    `TextureOptimizer` and is used to optimize the vertices and texture of a Mitsuba
    scene. The optimization is performed simultaneously, updating first the shape and
    then the texture in each iteration.
    """

    def __init__(
        self,
        scene: "mi.Scene",
        scene_params: "mi.SceneParams",
        optimization_events: OptimizationEvents,
        shape_key: str,
        texture_key: str,
        shape_scheduler: OptimizationScheduler,
        texture_scheduler: OptimizationScheduler,
        shape_loss: ImageLoss = ImageLossL2(),
        texture_loss: ImageLoss = ImageLossL2(),
        shape_smoothing_strength: float = 25.0,
        texture_projection_steepness: float = 1.0,
        texture_save_interval: int = 5,
        flip_along_y: bool = True,
    ) -> None:
        """
        The `scene` is a loaded Mitsuba scene object, while `scene_params` are a
        dictionary of parameters that are to be optimized. `optimization_events` store
        all actions that happen at a specific iteration throughout the optimization.

        `shape_key` is a string that represents the shape whose vertex positions
         will be optimized. Once ".vertex_positions" is appended to it, it should
        produce a valid key in `scene_params` that contains the vertex positions of
        the mesh. Analogously, `texture_key` is a string that represents the texture
        that is to be optimized and must b be a valid key in `scene_params`

        `shape_smoothing_strength` is a free parameter, that modifies how strongly the
        gradients at a particular vertex are diffused. Larger values have higher
        stability when optimizing while smaller ones are better to recover higher
        detail.

        `texture_projection_steepness` is a free parameter that controls the steepness
        of the sigmoid projection and higher values can lead to faster convergence at
        the cost of less stability and more artifacts.

        `shape_scheduler` controls the learning rate throughout the optimization
        process, while `shape_loss` is the function used to compute the difference
        between the rendered image and the reference image. Analogously, `texture_loss`
        and  `texture_scheduler` determine the optimization of the Bitmap texture.
        Note that the loss functions for shape and texture optimization can be
        different, allowing for flexibility in the optimization process.

        Textures typically have a much larger memory footprint than meshes, so instead
        of saving them every iteration, they are saved every `texture_save_interval`
        iterations. Optionally, they can be flipped along the y-axis (height) for
        easier use in Blender and other tools that have the `u = 0`, `v = 0` in the
        bottom left corner. The default is `u = 0`, `v = 0` in the top left corner.

        :raises ValueError: If `shape_key` does not correspond to a valid shape in
            `scene_params`, if `shape_smoothing_strength` is a negative number,
            if `texture_key` is not found in `scene_params`, or if the texture key
            does not correspond to a property of type `mi.TensorXf`, if
            `texture_projection_steepness` is not a positive number, or if the
            `texture_save_interval` is not a positive integer.
        """
        super().__init__(
            scene=scene,
            scene_params=scene_params,
            optimization_events=optimization_events,
        )

        self._shape_optimizer = MeshOptimizer(
            scene=scene,
            scene_params=scene_params,
            optimization_events=optimization_events,
            shape_key=shape_key,
            shape_scheduler=shape_scheduler,
            smoothing_strength=shape_smoothing_strength,
            shape_loss=shape_loss,
        )
        self._texture_optimizer = BitmapTextureOptimizer(
            scene=scene,
            scene_params=scene_params,
            optimization_events=optimization_events,
            optimization_key=texture_key,
            texture_keys=[texture_key],
            texture_scheduler=texture_scheduler,
            projection_steepness=texture_projection_steepness,
            texture_loss=texture_loss,
            texture_save_interval=texture_save_interval,
            flip_along_y=flip_along_y,
        )

    @property
    def optimization_parameters(self) -> dict[str, Any]:
        """
        A dictionary which is a union of the optimization parameters
        from both the shape and texture optimizers.
        """
        shape_params = self._shape_optimizer.optimization_parameters
        texture_params = self._texture_optimizer.optimization_parameters

        return shape_params | texture_params

    @property
    def parameter_optimization_configs(
        self,
    ) -> list[SceneOptimizer.ParamOptimizationConfig]:
        """
        A list of two `ParamOptimizationConfig` objects, one for shape and one for
        texture optimization. Each config describes how to optimize the respective
        property.
        """
        shape_configs = self._shape_optimizer.parameter_optimization_configs
        texture_configs = self._texture_optimizer.parameter_optimization_configs

        return shape_configs + texture_configs

    def save_callback(
        self, iteration: int, output_directory: Path | None = None
    ) -> None:
        """
        Callback passed to `SceneOptimizer` to save the current mesh and texture.

        Passes the `iteration` number and an optional `output_directory` to both
        the shape and texture optimizers' `save_callback` methods.
        """
        self._shape_optimizer.save_callback(
            iteration=iteration, output_directory=output_directory
        )
        self._texture_optimizer.save_callback(
            iteration=iteration, output_directory=output_directory
        )
