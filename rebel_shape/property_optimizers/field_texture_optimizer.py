from datetime import datetime
from pathlib import Path
from typing import Any

import drjit as dr
import jsonpickle
import mitsuba as mi

from rebel_shape.loss_evaluators.image_loss import ImageLoss
from rebel_shape.loss_evaluators.image_loss_l2 import ImageLossL2
from rebel_shape.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)
from rebel_shape.property_optimizers.bitmap_texture_optimizer import (
    BitmapTextureOptimizer,
)
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.schedulers.optimization_scheduler import OptimizationScheduler


class FieldTextureOptimizer(BitmapTextureOptimizer):
    """
    Handles optimization of textures that are attached to the surface of the tessellated
    IGA spline surface.

    Just like Mitsuba supports attaching values to faces and vertices of a mesh,
    the functionality used by `AttributeTextureOptimizer`, this class attaches values
    to the control points of the composite spline surface. Since Mitsuba cannot deal
    with spline surfaces directly, the spline is evaluated at evenly spaced parametric
    coordinates, forming a Bitmap texture which is then attached to the tessellated
    mesh associated to each spline patch.

    Note that the field values attached to the surface must produce values in range
    [0.0, 1.0] when evaluated as a texture, as they are used as colors in the
    Mitsuba scene. The `SigmoidParametrization` is used to ensure that the values
    are always in this range, allowing for optimization of the field values directly
    on the real line, while guaranteeing that the resulting texture is always valid.
    """

    def __init__(
        self,
        scene: "mi.Scene",
        scene_params: "mi.SceneParams",
        optimization_events: OptimizationEvents,
        tessellated_iga_spline: TessellatedIGASplineSurface,
        property_key: str,
        texture_scheduler: OptimizationScheduler,
        projection_steepness: float = 3.0,
        texture_loss: ImageLoss = ImageLossL2(),
        texture_save_interval: int = 5,
        flip_along_y: bool = True,
    ) -> None:
        """
        The `scene` is a loaded Mitsuba scene object, while `scene_params` are a
        dictionary of parameters that are to be optimized. `optimization_events` store
        all actions that happen at a specific iteration throughout the optimization.

        `projection_steepness` is a free parameter that controls the steepness of the
        sigmoid projection and higher values can lead to faster convergence at the cost
        of less stability and artifacts.

        `tessellated_iga_spline` is a composite spline surface that has associated
        field values of shape (num_control_points, dofs_per_control_point) that are
        to be optimized. The `property_key` is the name of the bitmap texture that
        represents the field values. It must be such that `{mesh_name}.{property_key}`
        is a valid key in `scene_params`, where `mesh_name` is the name of each mesh
        that represents a patch of the tessellated composite spline surface.

        Textures typically have a much larger memory footprint than meshes, so instead
        of saving them every iteration, they are saved every `texture_save_interval`
        iterations. Optionally, they can be flipped along the y-axis (height) for
        easier use in Blender and other tools that have the `u = 0`, `v = 0` in the
        bottom left corner. The default is `u = 0`, `v = 0` in the top left corner.

        `texture_loss` and `texture_scheduler` determine the progress of optimization
        while `optimization_events` store all actions that happen at a specific
        iteration throughout the optimization.

        :raises ValueError: If the `texture_key` is not found in `scene_params` or if
            it is not of type `mi.TensorXf`, if the resolutions of textures in
            `texture_keys` are not the same, if `projection_steepness` is not a positive
            number, or if the `texture_save_interval` is not a positive integer.
        """
        texture_keys = [
            f"{mesh.name}.{property_key}" for mesh in tessellated_iga_spline
        ]
        self._tessellated_iga_spline = tessellated_iga_spline
        self._property_key = property_key

        super().__init__(
            scene=scene,
            scene_params=scene_params,
            optimization_events=optimization_events,
            optimization_key=tessellated_iga_spline.spline_name + "_texture",
            texture_keys=texture_keys,
            texture_scheduler=texture_scheduler,
            projection_steepness=projection_steepness,
            texture_loss=texture_loss,
            texture_save_interval=texture_save_interval,
            flip_along_y=flip_along_y,
        )

    def _texture_update_callback(self, params: "mi.TensorXf") -> None:
        """
        Callback that is passed to `SceneOptimizer` to update the texture property.

        `params` are a vector of float32 values that have to be projected to [0.0, 1.0]
        and reshaped to match the global field values of the tessellated IGA spline,
        which have shape (num_control_points, dofs_per_control_point).

        The spline surface is then evaluated using the `evaluate_field_texture`
        method for each patch, and the resulting textures are updated in the
        `scene_params` dictionary under the corresponding texture keys.
        """
        new_field_values = dr.reshape(
            dtype=mi.TensorXf,
            value=self._sigmoid_parametrization.from_parametric(params),
            shape=self._tessellated_iga_spline.global_field_values.shape,
        )
        self._tessellated_iga_spline.global_field_values = new_field_values

        width, height, _ = self._resolution

        for texture_key, patch in zip(
            self._texture_keys, self._tessellated_iga_spline.composite_spline
        ):
            new_field_texture = patch.evaluate_field_texture(width=width, height=height)
            self._scene_params[texture_key] = new_field_texture

        self._scene_params.update()

    @property
    def optimization_parameters(self) -> dict[str, Any]:
        """
        A dictionary with a single key-value pair where the key is the
        `optimization_key` and the value is a flattened Tensor of shape
        (num_control_points, dofs_per_control_point) that represents the field
        values projected to the parametric space.
        """
        return {
            self._optimization_key: self._sigmoid_parametrization.to_parametric(
                dr.ravel(self._tessellated_iga_spline.global_field_values)
            )
        }

    def save_callback(
        self, iteration: int, output_directory: Path | None = None
    ) -> None:
        """
        Callback that is passed to `SceneOptimizer` to save the tessellated IGA spline.

        Encodes the current state of the IGA tessellated spline surface with jsonpickle
        and saves it to `output_directory` with a name that includes the current
        timestamp and `iteration` number.

        If `output_directory` is None, the method does nothing.
        """
        if output_directory is None:
            return

        super().save_callback(iteration=iteration, output_directory=output_directory)

        encoded_spline = jsonpickle.encode(self._tessellated_iga_spline)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_name = (
            f"{self._tessellated_iga_spline.spline_name}_{timestamp}"
            f"_iteration-{iteration:03}.json"
        )
        with open(output_directory / save_name, "w") as spline_file:
            spline_file.write(encoded_spline)
