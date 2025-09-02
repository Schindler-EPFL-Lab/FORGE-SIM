import abc
from pathlib import Path
from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np

from rebel_shape.data_processing.masked_pose_image import MaskedPoseImage
from rebel_shape.loss_evaluators.image_loss import ImageLoss
from rebel_shape.loss_evaluators.image_loss_l2 import ImageLossL2
from rebel_shape.property_optimizers.property_optimizer import PropertyOptimizer
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.scene_optimizers.scene_optimizer import SceneOptimizer
from rebel_shape.schedulers.optimization_scheduler import OptimizationScheduler


class PoseOptimizer(PropertyOptimizer):
    """
    An abstract class that performs optimization of the pose (and scale) of some
    property in 3D space updated by the `_update_positions` method.

    It is assumed that the initial shape is centered at the origin. In that case, it
    is not at all uncommon that it is not visible from some viewpoints as the
    cameras are typically centered at the origin as well. However, good placement of the
    mesh goes a long way towards getting good optimization results as shape optimization
    is in general bad at moving vertices or control points around.

    The solution is to start with a reasonable starting position, called the
    convergence point, which is the position at which cameras look, on average. This
    makes the shape visible at first and provides a reasonable initial guess for the
    position. If all cameras are nearly parallel, worse results are expected.
    Then, the position, scale, and orientation of the mesh is optimized to match as
    best as possible the input images.

    Classes that inherit have to implement a single method that takes in the current
    transformation and has to update the shape in the scene.

    A Mitsuba tutorial that inspired the class:
    https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/object_pose_estimation.html
    """

    SCALE_EPSILON = 1e-3
    """
    Smallest scale that is allowed for the scaling part of the transform.
    The inverse is the largest scale that is allowed.
    """

    def __init__(
        self,
        scene: "mi.Scene",
        scene_params: "mi.SceneParams",
        optimization_events: OptimizationEvents,
        optimization_key: str,
        scheduler: OptimizationScheduler,
        image_loss: ImageLoss = ImageLossL2(),
    ) -> None:
        """
        The `scene` is a loaded Mitsuba scene object, while `scene_params` are a
        dictionary of parameters that are to be optimized. `optimization_events` store
        all actions that happen at a specific iteration throughout the optimization.

        `optimization_key` is a unique identifier used for the pose optimization
        variable that is passed to the `SceneOptimizer` instance.

        The `scheduler` controls the learning rate throughout the optimization
        process while `image_loss` is the function used to compute the difference
        between the rendered image and the reference image.
        """
        super().__init__(
            scene=scene,
            scene_params=scene_params,
            optimization_events=optimization_events,
        )

        self._optimization_key = optimization_key

        # Updated in `optimize` to the convergence point
        self._initial_position_guess = (0.0, 0.0, 0.0)

        self._loss = image_loss
        self._scheduler = scheduler

    def _shape_update_callback(self, params: "mi.Float") -> None:
        """
        Callback that is passed to `SceneOptimizer` to update the shape property.

        `params` are a vector of float32 values, assumed by convention to contain 9
        values: first 3 are translation, second 3 scaling, and last 3 rotation angles.

        The values are unpacked and clipped. Then, the current transformation matrix
        is created and passed to the `_update_positions` method that updates the
        appearance of the scene.

        For more details on the convention, see `optimization_parameters`.
        """

        params = dr.reshape(mi.TensorXf, params, (9, 1))

        translation = mi.Point3f(params[0:3])
        scaling = mi.Point3f(
            dr.clip(
                value=params[3:6],
                min=self.SCALE_EPSILON,
                max=1.0 / self.SCALE_EPSILON,
            )
        )
        rotation_angles = mi.Point3f(
            (180.0 / dr.pi) * dr.clip(params[6:9], -dr.pi, dr.pi)
        )

        tf_new_pose = (
            mi.Transform4f()
            .scale(scaling)
            .translate(translation)
            .rotate(axis=mi.Point3f(1, 0, 0), angle=rotation_angles.x)
            .rotate(axis=mi.Point3f(0, 1, 0), angle=rotation_angles.y)
            .rotate(axis=mi.Point3f(0, 0, 1), angle=rotation_angles.z)
        )

        self._update_positions(tf_new_pose=tf_new_pose)

    @abc.abstractmethod
    def _update_positions(self, tf_new_pose: "mi.Transform4f") -> None:
        """
        Updates the mesh vertices to the new positions based on the provided
        transformation matrix `tf_new_pose`.

        This method is used internally to apply the transformation to the initial
        positions (of e.g. vertices or control points) and update the scene parameters
        to reflect the new positions.

        :raises NotImplementedError: if the method was not implemented by a child class.
        """
        raise NotImplementedError(
            "Classes that inherit from InitialPoseOptimizer must implement "
            "the `_update_positions` method."
        )

    def optimize(
        self,
        ref_images: list[MaskedPoseImage],
        scene_optimizer: SceneOptimizer,
        num_iters: int = 10,
        views_to_render: list["mi.Sensor"] | None = None,
        save_directory_path: Path | None = None,
    ) -> None:
        """
        Executes pose optimization for `num_iters` steps.

        The user provides a list of MaskedPose reference images `ref_images`.
        `scene_optimizer` determines the underlying scene optimization
        algorithm (e.g. `SceneAdam`).

        Optionally, `views_to_render` can be used to render the current state of
        the scene for provided views and optionally save it to `save_directory_path`.

        Parameter checking occurs in `SceneOptimizer.optimize` method. At each
        iteration the mesh is saved to `save_directory_path` (if the path is not
        provided it is saved at the current directory level).

        """
        convergence_point = self.compute_camera_convergence_point(
            sensors=[image.sensor for image in ref_images]
        )
        self._initial_position_guess = convergence_point

        super().optimize(
            ref_images=ref_images,
            scene_optimizer=scene_optimizer,
            num_iters=num_iters,
            views_to_render=views_to_render,
            save_directory_path=save_directory_path,
        )

    @property
    def optimization_parameters(self) -> dict[str, Any]:
        """
        Sets the parameters for the `scene_optimizer` instance. Intended to be used
        before the optimization process starts.

        The convention for passing the parameters to the optimizer is the following:
        - 3 components for translation (center of the mesh): indices 0, 1, 2
        - 3 components for scaling (the initial orientation): indices 3, 4, 5
        - 3 components for rotation Euler angles (XYZ convention): indices 6, 7, 8
        A total of 9 components

        The initial translation is set to `initial_position_guess` and its role is to
        make sure the mesh is visible initially.
        """
        flattened_params = mi.Float(
            self._initial_position_guess[0],
            self._initial_position_guess[1],
            self._initial_position_guess[2],
            # Initial scale is 1.0
            1.0,
            1.0,
            1.0,
            # Initial angles are 0.0 (no rotation)
            0.0,
            0.0,
            0.0,
        )

        return {self._optimization_key: dr.ravel(flattened_params)}

    @property
    def parameter_optimization_configs(
        self,
    ) -> list[SceneOptimizer.ParamOptimizationConfig]:
        shape_param_config = SceneOptimizer.ParamOptimizationConfig(
            parameter_key=self._optimization_key,
            scene_update_callback=self._shape_update_callback,
            image_loss=self._loss,
            scheduler=self._scheduler,
        )

        return [shape_param_config]

    def save_callback(
        self, iteration: int, output_directory: Path | None = None
    ) -> None:
        """
        `InitialPoseOptimizer` does not save anything by default at each iteration.
        """
        return

    @staticmethod
    def compute_camera_convergence_point(
        sensors: list["mi.Sensor"],
    ) -> tuple[float, float, float]:
        """
        Computes a point in 3D space that all `sensors` are approximately looking at.
        Makes sure the mesh is visible in the beginning.

        This function works by finding the point that minimizes the sum of squared
        distances to all camera rays. Each camera ray is defined by the camera's
        position and forward direction (first 3 components of the second column,
        in Mitsuba camera convention the Z-axis looks forward).

        The optimum point is found using the least squares. For math, check out
        https://math.stackexchange.com/questions/61719/finding-the-intersection-point-of-many-lines-in-3d-point-closest-to-all-lines

        :returns: A tuple (x, y, z) representing the convergence point
        """
        # Matrices for the least squares problem
        A = np.zeros((3 * len(sensors), 3))
        b = np.zeros(3 * len(sensors))

        for i, sensor in enumerate(sensors):
            # Extract camera position and direction from transformation matrix
            pose_matrix = sensor.world_transform().matrix.numpy()

            # Forward direction is the positive z-axis in camera space
            direction = np.array(
                [pose_matrix[0, 2], pose_matrix[1, 2], pose_matrix[2, 2]]
            )
            direction = direction / np.linalg.norm(direction)

            id_mat = np.identity(3)
            position = np.array(
                [pose_matrix[0, 3], pose_matrix[1, 3], pose_matrix[2, 3]]
            )

            A[3 * i : 3 * (i + 1)] = id_mat - np.outer(direction, direction)
            b[3 * i : 3 * (i + 1)] = (
                (id_mat - np.outer(direction, direction)) @ position
            ).flatten()

        convergence_point, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        return (
            float(convergence_point[0]),
            float(convergence_point[1]),
            float(convergence_point[2]),
        )
