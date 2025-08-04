from pathlib import Path
from typing import Any

import mitsuba as mi

from rebel_shape.loss_evaluators.image_loss import ImageLoss
from rebel_shape.loss_evaluators.image_loss_l2 import ImageLossL2
from rebel_shape.meshing.tessellated_spline_surface import TessellatedSplineSurface
from rebel_shape.property_optimizers.bitmap_texture_optimizer import (
    BitmapTextureOptimizer,
)
from rebel_shape.property_optimizers.property_optimizer import PropertyOptimizer
from rebel_shape.property_optimizers.spline_optimizer import SplineOptimizer
from rebel_shape.regularization.regularization import Regularization
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.scene_optimizers.scene_optimizer import SceneOptimizer
from rebel_shape.schedulers.optimization_scheduler import OptimizationScheduler


class SplineTextureOptimizer(PropertyOptimizer):
    """
    A class that combines spline and texture optimization of a tessellated composite
    spline surface.

    It is a thin wrapper around two other optimization classes: `SplineOptimizer` and
    `BitmapTextureOptimizer`. It is used to simultaneously optimize the global control
    points of a spline surface and the texture of each spline patch. First, the control
    points are optimized, followed by the texture in each iteration.
    """

    def __init__(
        self,
        scene: "mi.Scene",
        scene_params: "mi.SceneParams",
        optimization_events: OptimizationEvents,
        tessellated_spline: TessellatedSplineSurface,
        shape_scheduler: OptimizationScheduler,
        texture_scheduler: OptimizationScheduler,
        shape_loss: ImageLoss = ImageLossL2(),
        texture_loss: ImageLoss = ImageLossL2(),
        shape_regularization: Regularization | None = None,
        parametrization_smoothing: float = 4.0,
        texture_projection_steepness: float = 1.0,
        texture_save_interval: int = 5,
        flip_along_y: bool = True,
    ) -> None:
        """
        The `scene` is a loaded Mitsuba scene object, while `scene_params` are a
        dictionary of parameters that are to be optimized. `optimization_events` store
        all actions that happen at a specific iteration throughout the optimization.

        `tessellated_spline` is a spline surface whose global control points are to be
        optimized. All meshes that constitute the spline surface have to be
        present in `scene_params` with their vertex positions accessible via a key
        consisting of the mesh name followed by ".vertex_positions". The names of the
        meshes have to be unique.

        `shape_scheduler` controls the learning rate throughout the optimization
        process, while `shape_loss` is the function used to compute the difference
        between the rendered image and the reference image. Analogously, `texture_loss`
        and  `texture_scheduler` determine the optimization of the Bitmap texture.
        Note that the loss functions for shape and texture optimization can be
        different, allowing for flexibility in the optimization process.

        `smoothing_strength` is a free parameter that modifies how strongly the
        gradients at a particular vertex are diffused. Larger values have higher
        stability when optimizing while smaller ones are better to recover higher
        detail.

        `texture_projection_steepness` is a free parameter that controls the steepness
        of the sigmoid projection and higher values can lead to faster convergence at
        the cost of less stability and more artifacts.

        Textures typically have a much larger memory footprint than meshes, so instead
        of saving them every iteration, they are saved every `texture_save_interval`
        iterations. Optionally, they can be flipped along the y-axis (height) for
        easier use in Blender and other tools that have the `u = 0`, `v = 0` in the
        bottom left corner. The default is `u = 0`, `v = 0` in the top left corner.

        :raises ValueError: If all meshes in `tessellated_spline` are not present
            in `scene_params` or if `smoothing_strength` is a negative number.
        """
        super().__init__(
            scene=scene,
            scene_params=scene_params,
            optimization_events=optimization_events,
        )

        self._tessellated_spline = tessellated_spline

        self._shape_optimizer = SplineOptimizer(
            scene=self._scene,
            scene_params=self._scene_params,
            optimization_events=self._optimization_events,
            tessellated_spline=self._tessellated_spline,
            shape_scheduler=shape_scheduler,
            shape_loss=shape_loss,
            regularization=shape_regularization,
            smoothing_strength=parametrization_smoothing,
        )

        self._texture_optimizer = BitmapTextureOptimizer(
            scene=self._scene,
            scene_params=self._scene_params,
            optimization_events=self._optimization_events,
            optimization_key=tessellated_spline.mesh_name + "_texture",
            texture_keys=[
                f"{mesh.name}.bsdf.reflectance.data"
                for mesh in self._tessellated_spline
            ],
            texture_scheduler=texture_scheduler,
            texture_loss=texture_loss,
            projection_steepness=texture_projection_steepness,
            texture_save_interval=texture_save_interval,
            flip_along_y=flip_along_y,
        )

    def h_refinement(self, factor: int, scene_optimizer: SceneOptimizer):
        """
        Increase the number of knot spans in each direction of the spline by `factor`.

        This method invokes the `h_refinement` method of the underlying tessellated
        spline surface, which increases the total number of control points by
        dividing each knot span into `factor` knot_spans, along both surface
        directions.

        It also takes as input the `scene_optimizer` used in the main optimization
        loop to update the parameters in the optimizer to the new control points.

        :raises ValueError: if `factor` is not a positive integer.
        """
        if factor < 1:
            raise ValueError("`factor` must be a positive integer.")

        self._tessellated_spline.composite_spline.h_refinement(factor=factor)
        self._shape_optimizer._initialize_parametrization()

        scene_optimizer.set_parameters(self.optimization_parameters)

    @property
    def optimization_parameters(self) -> dict[str, Any]:
        """
        A list of two `ParamOptimizationConfig` objects, one for shape and one for
        texture optimization. Each config describes how to optimize the respective
        property.
        """
        shape_params = self._shape_optimizer.optimization_parameters
        texture_params = self._texture_optimizer.optimization_parameters

        return shape_params | texture_params

    @property
    def parameter_optimization_configs(
        self,
    ) -> list[SceneOptimizer.ParamOptimizationConfig]:
        """
        A list of two `ParamOptimizationConfig` objects, one for spline and one for
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
        Callback passed to `SceneOptimizer` to save the current spline mesh and texture.

        Passes the `iteration` number and an optional `output_directory` to both
        the shape and texture optimizers' `save_callback` methods.
        """
        self._shape_optimizer.save_callback(
            iteration=iteration, output_directory=output_directory
        )
        self._texture_optimizer.save_callback(
            iteration=iteration, output_directory=output_directory
        )
