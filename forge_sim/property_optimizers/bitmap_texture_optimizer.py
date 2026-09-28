from datetime import datetime
from pathlib import Path
from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np

from forge_sim.loss_evaluators.image_loss import ImageLoss
from forge_sim.loss_evaluators.image_loss_l2 import ImageLossL2
from forge_sim.property_optimizers.property_optimizer import PropertyOptimizer
from forge_sim.property_optimizers.sigmoid_parametrization import (
    SigmoidParametrization,
)
from forge_sim.scene_optimizers.optimization_events import OptimizationEvents
from forge_sim.scene_optimizers.scene_optimizer import SceneOptimizer
from forge_sim.schedulers.optimization_scheduler import OptimizationScheduler


class BitmapTextureOptimizer(PropertyOptimizer):
    """
    Handles optimization of textures that are represented as 2D images (bitmaps).
    Can handle optimization of multiple such textures so long as their resolution is
    the same.

    Intended to be used for optimizing either an image texture of a single mesh or
    image textures of multiple patches with the same texture resolution. The latter is
    the common case when carrying out optimization with meshes composed of patches
    or when optimizing spline surfaces.

    Since .exr files typically have a much larger memory footprint than mesh files
    they should not be saved at each iteration.
    """

    def __init__(
        self,
        scene: "mi.Scene",
        scene_params: "mi.SceneParams",
        optimization_events: OptimizationEvents,
        optimization_key: str,
        texture_keys: list[str],
        texture_scheduler: OptimizationScheduler,
        projection_steepness: float = 1.0,
        texture_loss: ImageLoss = ImageLossL2(),
        texture_save_interval: int = 5,
        flip_along_y: bool = True,
    ) -> None:
        """
        The `scene` is a loaded Mitsuba scene object, while `scene_params` are a
        dictionary of parameters that are to be optimized. `optimization_events` store
        all actions that happen at a specific iteration throughout the optimization.

        `optimization_key` is a unique identified used for the texture optimization
        variable that is passed to the `SceneOptimizer` instance. `texture_keys` are
        valid keys for `scene_parameters` that point to parameters of type
        `mi.TensorXf`, the way textures are represented in Mitsuba.

        `projection_steepness` is a free parameter that controls the steepness of the
        sigmoid projection and higher values can lead to faster convergence at the cost
        of less stability and artifacts.

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
        super().__init__(
            scene=scene,
            scene_params=scene_params,
            optimization_events=optimization_events,
        )

        for texture_key in texture_keys:
            if texture_key not in scene_params:
                raise ValueError(f"{texture_key} was not found in scene parameters")
            if not isinstance(scene_params[texture_key], mi.TensorXf):
                raise ValueError(
                    "The data referenced by `bitmap_texture_key` must be a "
                    f"`mi.TensorXf` object. Got {type(scene_params[texture_key])}."
                )

        resolution = scene_params[texture_keys[0]].shape
        for texture_key in texture_keys:
            if scene_params[texture_key].shape != resolution:
                raise ValueError(
                    f"Texture {texture_key} has a different resolution than the "
                    f"first texture."
                )

        if texture_save_interval < 1:
            raise ValueError(
                "`texture_save_interval` must be a positive integer!, "
                f"got {texture_save_interval}."
            )

        self._optimization_key = optimization_key
        self._texture_keys = texture_keys
        self._resolution = resolution

        self._texture_scheduler = texture_scheduler
        self._texture_loss = texture_loss
        self._sigmoid_parametrization = SigmoidParametrization(projection_steepness)

        self._texture_save_interval = texture_save_interval
        self._flip_along_y = flip_along_y

    def _texture_update_callback(self, params: "mi.TensorXf") -> None:
        """
        Callback that is passed to `SceneOptimizer` to update the texture property.

        `optimizer_params` are a vector of float32 values that have to be projected
        to [0.0, 1.0]
        """
        for i, texture_key in enumerate(self._texture_keys):
            self._scene_params[texture_key] = (
                self._sigmoid_parametrization.from_parametric(params[i, :, :, :])
            )
        self._scene_params.update()

    @property
    def optimization_parameters(self) -> dict[str, Any]:
        """
        A dictionary with a single key-value pair where the key is the
        `optimization_key` and the value is a `mi.TensorXf` of shape
        `(len(self._texture_keys), W, H, C)` where `W` and `H` are the width and height
        of the textures and `C` is the number of channels (typically 3 for RGB).
        """
        all_textures = dr.zeros(
            dtype=mi.TensorXf,
            shape=(len(self._texture_keys), *self._resolution),
        )
        for i, texture_key in enumerate(self._texture_keys):
            all_textures[i] = dr.ravel(self._scene_params[texture_key])

        return {
            self._optimization_key: self._sigmoid_parametrization.to_parametric(
                all_textures
            )
        }

    @property
    def parameter_optimization_configs(
        self,
    ) -> list[SceneOptimizer.ParamOptimizationConfig]:
        """
        A list containing a single `ParamOptimizationConfig` object that
        describes how to optimize bitmap textures of multiple meshes.
        """
        texture_param_config = SceneOptimizer.ParamOptimizationConfig(
            parameter_key=self._optimization_key,
            scene_update_callback=self._texture_update_callback,
            image_loss=self._texture_loss,
            scheduler=self._texture_scheduler,
        )

        return [texture_param_config]

    def save_callback(
        self, iteration: int, output_directory: Path | None = None
    ) -> None:
        """
        Callback that is passed to `SceneOptimizer` to save the textures.

        For each texture key that the optimizer is responsible for, extracts the
        texture data from `scene_params`, flips it along the y-axis if `_flip_along_y`
        is set to True, and saves it as a `.exr` file in the specified
        `output_directory`. The name contains the texture key, a timestamp, and the
        current iteration number.

        The method is called every `texture_save_interval` iterations to reduce the
        overall memory footprint of the optimization process.

        If `output_directory` is None, the method returns without saving.
        """
        if (iteration + 1) % self._texture_save_interval != 0:
            return

        if output_directory is None:
            return

        for texture_key in self._texture_keys:
            texture = mi.Bitmap(
                self._scene_params[texture_key][::-1, :, :]
                if self._flip_along_y
                else self._scene_params[texture_key]
            )

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            texture_name = "-".join(texture_key.split("."))
            output_name = f"{texture_name}_{timestamp}_iteration-{iteration:03}.exr"

            output_name_path_str = str(output_directory / output_name)
            texture.write(output_name_path_str)

    @staticmethod
    def generate_empty_texture(
        base_color: "mi.Color3f",
        texture_output_path: Path,
        height: int = 256,
        width: int = 256,
    ) -> None:
        """
        Generates a texture of size `width` x `height` with a specified `base_color`.

        The texture is saved to `texture_output_path` which includes the name and
        extension of the texture. It is recommended to use a `.exr` extension.

        The method is necessary because Mitsuba requires loading textures from disk,
        and a blank initial texture is needed to start the optimization process.

        :raises ValueError: If the directory to which the texture should be saved
            does not exist.
        """
        if not texture_output_path.parent.exists():
            raise ValueError(
                f"Output directory {texture_output_path.parent} does not exist."
            )

        data = np.full(
            (height, width, 3), base_color.numpy().squeeze(), dtype=np.float32
        )
        bitmap = mi.Bitmap(data)
        bitmap.write(str(texture_output_path))
