import drjit as dr
import mitsuba as mi

from rebel_shape.loss_evaluators.image_loss import ImageLoss
from rebel_shape.loss_evaluators.image_loss_l2 import ImageLossL2
from rebel_shape.meshing.tessellated_spline_surface import TessellatedSplineSurface
from rebel_shape.property_optimizers.pose_optimizer import PoseOptimizer
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.schedulers.optimization_scheduler import OptimizationScheduler


class SplinePoseOptimizer(PoseOptimizer):
    """
    Concrete implementation of the `InitialPoseOptimizer` for spline surfaces.

    This optimizer is designed to adjust the initial control points of a
    tessellated spline surface by applying a transformation matrix to the
    global control points. It updates the vertex positions of the meshes accordingly.
    """

    def __init__(
        self,
        scene: "mi.Scene",
        scene_params: "mi.SceneParams",
        optimization_events: OptimizationEvents,
        tessellated_spline: TessellatedSplineSurface,
        scheduler: OptimizationScheduler,
        image_loss: ImageLoss = ImageLossL2(),
    ) -> None:
        """
        The `scene` is a loaded Mitsuba scene object, while `scene_params` are a
        dictionary of parameters that are to be optimized. `optimization_events` store
        all actions that happen at a specific iteration throughout the optimization.

        `tessellated_spline` is a spline surface whose initial control points are
        to be optimized. All meshes that constitute the spline surface have to be
        present in `scene_params` with their vertex positions accessible via the
        mesh name followed by ".vertex_positions".

        The `scheduler` controls the learning rate throughout the optimization
        process while `image_loss` is the function used to compute the difference
        between the rendered image and the reference image.

        :raises ValueError: If any of the meshes in `tessellated_spline` is not
            present in `scene_params`.
        """
        super().__init__(
            scene=scene,
            scene_params=scene_params,
            optimization_events=optimization_events,
            optimization_key=tessellated_spline.mesh_name,
            scheduler=scheduler,
            image_loss=image_loss,
        )

        self._tessellated_spline = tessellated_spline

        for mesh in tessellated_spline:
            if mesh.name + ".vertex_positions" not in scene_params:
                raise ValueError(
                    f"Mesh {mesh.name} is not present in the scene parameters."
                )

        self._initial_control_points = dr.unravel(
            dtype=mi.Point3f,
            array=tessellated_spline.global_control_points.array,
        )

    def _update_positions(self, tf_new_pose: "mi.Transform4f") -> None:
        """
        Updates the global control points of the spline surface based on the provided
        transformation matrix `tf_new_pose`.

        After applying the transformation, it updates the vertex positions of the
        meshes in the tessellated spline surface to reflect the impact of the new
        control points.
        """
        new_control_points = tf_new_pose @ self._initial_control_points

        new_control_points = dr.ravel(new_control_points)
        new_control_points = dr.reshape(
            dtype=mi.TensorXf,
            value=new_control_points,
            shape=self._tessellated_spline.global_control_points.shape,
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
