from datetime import datetime
from pathlib import Path
from typing import Any

import mitsuba as mi

from rebel_shape.loss_evaluators.image_loss import ImageLoss
from rebel_shape.loss_evaluators.image_loss_l2 import ImageLossL2
from rebel_shape.property_optimizers.property_optimizer import PropertyOptimizer
from rebel_shape.property_optimizers.sigmoid_parametrization import (
    SigmoidParametrization,
)
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.scene_optimizers.scene_optimizer import SceneOptimizer
from rebel_shape.schedulers.optimization_scheduler import OptimizationScheduler


class AttributeTextureOptimizer(PropertyOptimizer):
    """
    Handles optimization of textures that are mesh properties, i.e. attached to
    vertices or faces.

    The optimization is carried out for a particular shape and its particular attached
    attribute. `SigmoidParametrization` is used to produce a mapping between the real
    line and the valid color range ([0.0, 1.0]). Can only optimize one attribute
    texture at a time.

    After each optimization iteration it saves the mesh as this way the attribute that
    is optimized is logged.
    """

    def __init__(
        self,
        scene: "mi.Scene",
        scene_params: "mi.SceneParams",
        optimization_events: OptimizationEvents,
        shape_key: str,
        attribute_key: str,
        texture_scheduler: OptimizationScheduler,
        projection_steepness: float = 1.0,
        texture_loss: ImageLoss = ImageLossL2(),
    ) -> None:
        """
        The `scene` is a loaded Mitsuba scene object, while `scene_params` are a
        dictionary of parameters that are to be optimized. `optimization_events` store
        all actions that happen at a specific iteration throughout the optimization.

        `shape_key` is the name of a mesh in the scene that is optimized. When the
        `shape_key` and `attribute_key` are joined with "." they have to produce a
        valid `scene_params` key. `attribute_key` must contain either "vertex_" or
        "face_", as prescribed by Mitsuba.

        `projection_steepness` is a free parameter that controls the steepness of the
        sigmoid projection and higher values can lead to faster convergence at the cost
        of less stability and artifacts.

        `texture_loss` and `texture_scheduler` determine the progress of optimization
        while `optimization_events` store all actions that happen at a specific
        iteration throughout the optimization.

        :raises ValueError: If the `texture_key` is not found in `scene_params` or if
            it is not of type `mi.TensorXf`, if the resolutions of textures in
            `texture_keys` are not the same, if `projection_steepness` is not a positive
            number, or if the `texture_save_interval` is not a positive integer.
        """
        super().__init__(
            scene=scene,
            scene_params=scene_params,
            optimization_events=optimization_events,
        )

        if not ("vertex_" in attribute_key or "face_" in attribute_key):
            raise ValueError(
                f"Attribute texture key '{attribute_key}' must contain 'vertex_' or "
                f"'face_' to indicate it is an attribute texture."
            )

        self._shape_key = shape_key
        self._texture_key = f"{shape_key}.{attribute_key}"

        if self._texture_key not in self._scene_params:
            raise ValueError(
                f"`texture_key` property {self._texture_key} not found in scene "
                f"parameters"
            )

        self._sigmoid_parametrization = SigmoidParametrization(projection_steepness)
        self._texture_scheduler = texture_scheduler
        self._texture_loss = texture_loss

    def _texture_update_callback(self, params: "mi.Float") -> None:
        """
        Callback that is passed to `SceneOptimizer` to update the texture property.

        `optimizer_params` are a vector of float32 values that have to be projected
        to [0.0, 1.0]
        """
        self._scene_params[self._texture_key] = (
            self._sigmoid_parametrization.from_parametric(params=params)
        )

    @property
    def optimization_parameters(self) -> dict[str, Any]:
        """
        A dictionary with a single key-value pair where the key is the
        `shape_key` and the value is a vector of float32 (`mi.Float`) values that
        represent flattened attribute texture data in the parametric space.
        """
        return {
            self._texture_key: self._sigmoid_parametrization.to_parametric(
                self._scene_params[self._texture_key]
            )
        }

    @property
    def parameter_optimization_configs(
        self,
    ) -> list[SceneOptimizer.ParamOptimizationConfig]:
        """
        A list containing a single `ParamOptimizationConfig` object that
        describes how to optimize the attribute texture of the mesh.
        """
        texture_param_config = SceneOptimizer.ParamOptimizationConfig(
            parameter_key=self._texture_key,
            scene_update_callback=self._texture_update_callback,
            image_loss=self._texture_loss,
            scheduler=self._texture_scheduler,
        )

        return [texture_param_config]

    def save_callback(
        self, iteration: int, output_directory: Path | None = None
    ) -> None:
        """
        Callback that is passed to `SceneOptimizer` to save the mesh property.

        Some properties are bound to the mesh (e.g. vertex temperature) and need to
        be saved as a mesh. The mesh key is not immediately available so it
        is extracted from the `texture_key` string. The mesh is saved in the
        `output_directory` with a name that includes the current timestamp and
        iteration number, if it is not None.

        :raises ValueError: If the mesh with the specified `shape_key` is not found
            in the scene.
        """
        if output_directory is None:
            return

        mesh = [msh for msh in self._scene.shapes() if msh.id() == self._shape_key][0]
        if mesh is None:
            raise ValueError(
                f"Mesh with ID '{self._shape_key}' not found in the scene."
            )

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        shape_name = "-".join(self._shape_key.split("."))
        mesh_name = f"{shape_name}_{timestamp}_iteration-{iteration:03}.ply"

        mesh_output_path_str = str(output_directory / mesh_name)
        mesh.write_ply(mesh_output_path_str)
