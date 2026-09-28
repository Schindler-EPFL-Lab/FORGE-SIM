from datetime import datetime
from pathlib import Path
from typing import Any

import drjit as dr
import mitsuba as mi

from forge_sim.loss_evaluators.image_loss import ImageLoss
from forge_sim.loss_evaluators.image_loss_l2 import ImageLossL2
from forge_sim.property_optimizers.laplacian_parametrization import (
    LaplacianParametrization,
)
from forge_sim.property_optimizers.property_optimizer import PropertyOptimizer
from forge_sim.scene_optimizers.optimization_events import OptimizationEvents
from forge_sim.scene_optimizers.scene_optimizer import SceneOptimizer
from forge_sim.schedulers.optimization_scheduler import OptimizationScheduler


class MeshOptimizer(PropertyOptimizer):
    """
    A class that performs shape optimization in a manner similar to the one shown in
    https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html

    This class implements a parametrization in terms of the topological Laplacian.
    It uses the `LaplacianParametrization` class to perform the optimization on
    parametrized mesh vertices. The space in which the optimization is performed
    smoothens out the gradients of the mesh vertices, allowing for larger steps
    during optimization while still maintaining a valid mesh structure.

    In general, mesh optimization is a challenging task due to sparse silhouette
    gradients that have large magnitudes. This can lead to large updates of the
    edges of the mesh, leading to mangling and state which cannot be recovered.
    Thus, the strength of the smoothing must be carefully chosen. If it is too high,
    the resulting mesh will be too smooth and will not recover fine details.
    """

    def __init__(
        self,
        scene: "mi.Scene",
        scene_params: "mi.SceneParams",
        optimization_events: OptimizationEvents,
        shape_key: str,
        shape_scheduler: OptimizationScheduler,
        smoothing_strength: float = 25.0,
        shape_loss: ImageLoss = ImageLossL2(),
    ) -> None:
        """
        The `scene` is a loaded Mitsuba scene object, while `scene_params` are a
        dictionary of parameters that are to be optimized. `optimization_events` store
        all actions that happen at a specific iteration throughout the optimization.

        `shape_key` is a string that represents the shape whose vertex positions
         will be optimized. Once ".vertex_positions" is appended to it, it should
        produce a valid key in `scene_params` that contains the vertex positions of
        the mesh.

        `shape_scheduler` controls the learning rate throughout the optimization
        process while `shape_loss` is the function used to compute the difference
        between the rendered image and the reference image.

        `smoothing_strength` is a free parameter, that modifies how strongly the
        gradients at a particular vertex are diffused. Larger values have higher
        stability when optimizing while smaller ones are better to recover higher
        detail.

        :raises ValueError: If `shape_key` does not correspond to a valid shape in
            `scene_params`, or if `smoothing_strength` is a negative number.
        """
        super().__init__(
            scene=scene,
            scene_params=scene_params,
            optimization_events=optimization_events,
        )

        self._shape_key = shape_key
        self._vertices_key = shape_key + ".vertex_positions"
        self._faces_key = shape_key + ".faces"

        if self._vertices_key not in self._scene_params:
            raise ValueError(
                f"`shape_key` property is not valid - cannot get vertices key "
                f"{self._vertices_key}"
            )

        if smoothing_strength < 0.0:
            raise ValueError("`smoothing_strength` must be a positive real number.")
        self._smoothing = smoothing_strength

        self._shape_loss = shape_loss
        self._shape_scheduler = shape_scheduler

        self._parametrization: LaplacianParametrization
        self._initialize_parametrization()

    def _initialize_parametrization(self) -> None:
        """
        Helper method that should be called whenever the connectivity of the mesh
        changes - vertex updates do not trigger it as the combinatorial Laplacian does
        not depend on the positions of the vertices.

        Updates the private instance of `LaplacianParametrization` with new vertices
        and faces. New vertices are a required parameter to perform deduplication of
        vertices that might arise due to seamless textures.
        """
        vertices = dr.reshape(
            dtype=mi.TensorXf,
            value=self._scene_params[self._vertices_key],
            shape=(-1, 3),
        )
        faces = dr.reshape(
            dtype=mi.TensorXu,
            value=self._scene_params[self._faces_key],
            shape=(-1, 3),
        )

        self._parametrization = LaplacianParametrization(
            nodes=vertices, faces=faces, smoothing_strength=self._smoothing
        )

    def update_smoothing_strength(
        self,
        new_smoothing_strength: float,
        scene_optimizer: SceneOptimizer,
    ) -> None:
        """
        Update the smoothing strength used in the Laplacian parametrization.

        As optimization progresses gradients become smaller and it is useful to reduce
        the strength of gradient smoothing. This method updates the smoothing
        strength to `new_smoothing_strength`.

        It also takes in the `scene_optimizer` used in the main optimization loop to
        update the parameters in the optimizer to new parametrization values derived
        from the current vertex positions.

        :raises ValueError: If `new_smoothing_strength` is a negative number.
        """
        if new_smoothing_strength < 0.0:
            raise ValueError("`new_smoothing_strength` must be a positive real number.")

        self._smoothing = new_smoothing_strength
        self._initialize_parametrization()

        scene_optimizer.set_parameters(self.optimization_parameters)

    def _shape_update_callback(self, params: "mi.Float") -> None:
        """
        Callback that is passed to `SceneOptimizer` to update the shape property.

        `params` are a vector of float32 values (`mi.Float`) that are in the parametric
        space. Transforming them back yields flattened vertex positions which are
        then assigned to the vertex positions of the mesh in the scene.
        """
        self._scene_params[self._vertices_key] = self._parametrization.from_parametric(
            params
        )
        self._scene_params.update()

    @property
    def optimization_parameters(self) -> dict[str, Any]:
        """
        A dictionary with a single key-value pair where the key is the
        `shape_key` and the value is a vector of float32 (`mi.Float`) values that
        represent flattened vertex positions in the parametric space.
        """
        return {
            self._shape_key: (
                self._parametrization.to_parametric(
                    self._scene_params[self._vertices_key]
                )
            )
        }

    @property
    def parameter_optimization_configs(
        self,
    ) -> list[SceneOptimizer.ParamOptimizationConfig]:
        """
        A list containing a single `ParamOptimizationConfig` object that
        describes how to optimize the vertices of the mesh.
        """
        shape_param_config = SceneOptimizer.ParamOptimizationConfig(
            parameter_key=self._shape_key,
            scene_update_callback=self._shape_update_callback,
            image_loss=self._shape_loss,
            scheduler=self._shape_scheduler,
        )

        return [shape_param_config]

    def save_callback(
        self, iteration: int, output_directory: Path | None = None
    ) -> None:
        """
        Callback that is passed to `SceneOptimizer` to save the current mesh.

        Takes the internal state and creates a new mesh that is then saved to
        `output_directory` with a name that includes the current timestamp and
        `iteration` number.

        If `output_directory` is None, the mesh is not saved and the method returns
        immediately.

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
