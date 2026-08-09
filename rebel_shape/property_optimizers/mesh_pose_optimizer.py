import drjit as dr
import mitsuba as mi

from rebel_shape.loss_evaluators.image_loss import ImageLoss
from rebel_shape.loss_evaluators.image_loss_l2 import ImageLossL2
from rebel_shape.property_optimizers.pose_optimizer import PoseOptimizer
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.schedulers.optimization_scheduler import OptimizationScheduler


class MeshPoseOptimizer(PoseOptimizer):
    """
    Concrete implementation of the `InitialPoseOptimizer` for mesh shapes.

    This optimizer is designed to adjust the initial pose of a mesh by updating
    the vertex positions based on a transformation matrix.
    """

    def __init__(
        self,
        scene: "mi.Scene",
        scene_params: "mi.SceneParams",
        optimization_events: OptimizationEvents,
        shape_key: str,
        scheduler: OptimizationScheduler,
        image_loss: ImageLoss = ImageLossL2(),
        jitter: float = 0.0,
        seed: int = 0,
        initial_scale: float = 1.0,
    ) -> None:
        """
        The `scene` is a loaded Mitsuba scene object, while `scene_params` are a
        dictionary of parameters that are to be optimized. `optimization_events` store
        all actions that happen at a specific iteration throughout the optimization.

        `shape_key` is a unique key that identifies the mesh shape in the scene. When
        the `shape_key` is joined with ".vertex_positions", it should produce
        a valid key in `scene_params` that contains the vertex positions of the mesh.

        The `scheduler` controls the learning rate throughout the optimization
        process while `image_loss` is the function used to compute the difference
        between the rendered image and the reference image.

        :raises ValueError: If the the vertices key derived from `shape_key` is not
            found in `scene_params`.
        """

        super().__init__(
            scene=scene,
            scene_params=scene_params,
            optimization_events=optimization_events,
            optimization_key=shape_key,
            scheduler=scheduler,
            image_loss=image_loss,
            jitter=jitter,
            seed=seed,
            initial_scale=initial_scale,
        )

        self._shape_key = shape_key
        self._vertices_key = shape_key + ".vertex_positions"

        if self._vertices_key not in self._scene_params:
            raise ValueError(
                f"`shape_key` property is not valid - cannot get vertices key "
                f"{self._vertices_key}"
            )

        self._initial_vertex_positions = dr.unravel(
            mi.Point3f, self._scene_params[self._vertices_key]
        )

    def _update_positions(self, tf_new_pose: "mi.Transform4f") -> None:
        """
        Updates the mesh vertices to the new positions based on the provided
        transformation matrix `tf_new_pose`.

        This method is used internally to apply the transformation to the initial
        vertex positions and update the scene parameters.
        """
        new_vertices = tf_new_pose @ self._initial_vertex_positions

        self._scene_params[self._vertices_key] = dr.ravel(new_vertices)
        self._scene_params.update()
