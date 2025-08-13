from datetime import datetime
from pathlib import Path
from typing import Any

import drjit as dr
import jsonpickle
import mitsuba as mi

from rebel_shape.loss_evaluators.image_loss import ImageLoss
from rebel_shape.loss_evaluators.image_loss_l2 import ImageLossL2
from rebel_shape.meshing.tessellated_spline_surface import TessellatedSplineSurface
from rebel_shape.property_optimizers.laplacian_parametrization import (
    LaplacianParametrization,
)
from rebel_shape.property_optimizers.property_optimizer import PropertyOptimizer
from rebel_shape.regularization.regularization import Regularization
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.scene_optimizers.scene_optimizer import SceneOptimizer
from rebel_shape.schedulers.optimization_scheduler import OptimizationScheduler


class SplineOptimizer(PropertyOptimizer):
    """
    A class that performs optimization of a tessellated composite spline surface.

    Mitsuba cannot optimize spline surfaces directly, so the `TessellatedSplineSurface`
    class maintains a `CompositeMesh` that represents the appearance of the spline
    surface. This is then used to optimize the global control points of the spline
    surface.

    This optimization procedure has very similar considerations as the shape
    optimization of a mesh, with the difference being that splines are intrinsically
    more stable than meshes and require smaller smoothing strengths to avoid
    catastrophic mangling. It is possible to optimize splines without any smoothing,
    but this requires a higher degree than typical (cubic is typical) and some care
    when choosing the learning rate.

    Control points form quads, and in this form it can be thought of as optimizing
    a mesh with quad faces. Thus, the `LaplaceParametrization` is again used in the
    same capacity as in the `MeshOptimizer` class, but now it is applied to the
    global control points and quads of the spline surface.s
    """

    def __init__(
        self,
        scene: "mi.Scene",
        scene_params: "mi.SceneParams",
        optimization_events: OptimizationEvents,
        tessellated_spline: TessellatedSplineSurface,
        shape_scheduler: OptimizationScheduler,
        shape_loss: ImageLoss = ImageLossL2(),
        regularization: Regularization | None = None,
        smoothing_strength: float = 4.0,
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
        process while `shape_loss` is the function used to compute the difference
        between the rendered image and the reference image.

        `smoothing_strength` is a free parameter, that modifies how strongly the
        gradients at a particular vertex are diffused. Larger values have higher
        stability when optimizing while smaller ones are better to recover higher
        detail.

        :raises ValueError: If all meshes in `tessellated_spline` are not present
            in `scene_params` or if `smoothing_strength` is a negative number.
        """
        super().__init__(
            scene=scene,
            scene_params=scene_params,
            optimization_events=optimization_events,
        )

        for mesh in tessellated_spline:
            if mesh.name + ".vertex_positions" not in scene_params:
                raise ValueError(
                    f"Mesh {mesh.name} is not present in the scene parameters."
                )

        self._tessellated_spline = tessellated_spline

        self._shape_loss = shape_loss
        self._regularization = regularization
        self._shape_scheduler = shape_scheduler

        self._parametrization: LaplacianParametrization

        if smoothing_strength < 0.0:
            raise ValueError("`smoothing_strength` must be a positive real number.")
        self._smoothing = smoothing_strength

        self._initialize_parametrization()

    def _initialize_parametrization(self) -> None:
        """
        Helper method that should be called whenever the vertices or faces
        of the mesh change.
        """
        self._parametrization = LaplacianParametrization(
            nodes=self._tessellated_spline.global_control_points,
            faces=self._tessellated_spline.global_quads,
            smoothing_strength=self._smoothing,
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
        self._initialize_parametrization()

        scene_optimizer.set_parameters(self.optimization_parameters)

    def _shape_update_callback(self, params: "mi.Float") -> None:
        """
        Callback that is passed to `SceneOptimizer` to update the shape property.

        `params` are a vector of float32 values that are in the Large steps projected
        parameter space. Transforming them back yields flattened vertex positions.
        """
        new_control_points = dr.reshape(
            dtype=mi.TensorXf,
            value=self._parametrization.from_parametric(params),
            shape=(-1, 3),
        )
        self._tessellated_spline.global_control_points = new_control_points

        for mesh in self._tessellated_spline:
            self._scene_params[mesh.name + ".vertex_positions"] = dr.ravel(
                mesh.vertices
            )
        self._scene_params.update()
        # Update normals after the vertices have changed.
        for mesh in self._tessellated_spline:
            self._scene_params[mesh.name + ".vertex_normals"] = dr.ravel(
                mesh.vertex_normals
            )

    @property
    def optimization_parameters(self) -> dict[str, Any]:
        return {
            self._tessellated_spline.mesh_name: self._parametrization.to_parametric(
                dr.ravel(self._tessellated_spline.global_control_points)
            ),
        }

    @property
    def parameter_optimization_configs(
        self,
    ) -> list[SceneOptimizer.ParamOptimizationConfig]:
        shape_param_config = SceneOptimizer.ParamOptimizationConfig(
            parameter_key=self._tessellated_spline.mesh_name,
            scene_update_callback=self._shape_update_callback,
            image_loss=self._shape_loss,
            scheduler=self._shape_scheduler,
            regularization=self._regularization,
        )
        return [shape_param_config]

    def save_callback(
        self, iteration: int, output_directory: Path | None = None
    ) -> None:
        """
        Callback that is passed to `SceneOptimizer` to save the current state of the
        tessellated spline surface and its meshes.

        Encodes the current state of the tessellated spline surface with jsonpickle
        and saves it to `output_directory` with a name that includes the current
        timestamp and `iteration` number. The meshes that represent the spline
        during optimization are also saved in the same directory.

        If `output_directory` is None, the spline and the meshes are not saved and the
        method returns immediately.
        """
        if output_directory is None:
            return

        encoded_spline = jsonpickle.encode(self._tessellated_spline)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_name = (
            f"{self._tessellated_spline.spline_name}_{timestamp}"
            f"_iteration-{iteration:03}.json"
        )
        with open(output_directory / save_name, "w") as spline_file:
            spline_file.write(encoded_spline)

        for mesh in self._tessellated_spline:
            mesh_name = f"{mesh.name}_{timestamp}_iteration-{iteration:03}"

            mesh.save_to_ply(output_dir=output_directory, save_name=mesh_name)
