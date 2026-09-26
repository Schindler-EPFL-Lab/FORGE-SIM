import abc
import math
from pathlib import Path
from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np

from rebel_shape import logger
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
        jitter: float = 0.0,
        seed: int = 0,
        initial_scale: float = 1.0,
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

        `initial_scale` sets the initial scaling component of the TRS state, which
        is also the radius scale applied to the (unit-radius) initial proxy.
        Typically the caller computes this via `find_initial_scale_gss` before
        optimization starts.

        `jitter` (default 0.0) adds a small symmetry-breaking perturbation to the
        initial translation (proportional to `initial_scale`) and scale: each of
        the 6 components receives `jitter * N(0, 1)` drawn once and cached at
        construction. `seed` controls the random draw for reproducibility.
        """
        if jitter < 0.0:
            raise ValueError("`jitter` must be non-negative.")
        if initial_scale <= 0.0:
            raise ValueError("`initial_scale` must be a positive real number.")

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

        self._jitter = jitter
        self._seed = seed
        self._initial_scale = initial_scale

        # Draw the jitter once at construction so that repeated reads of
        # `optimization_parameters` do not re-randomise the initial values.
        rng = np.random.default_rng(seed)
        self._jitter_draws = rng.standard_normal(6)

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

        # Standard TRS (matrix = Tr @ R @ S). Mitsuba chains on the right, so the
        # first call is the leftmost factor. Scale-last keeps the translation
        # independent of scale/rotation, which is what downstream consumers
        # (e.g. `bbox_transforms` JSON) assume.
        tf_new_pose = (
            mi.Transform4f()
            .translate(translation)
            .rotate(axis=mi.Point3f(1, 0, 0), angle=rotation_angles.x)
            .rotate(axis=mi.Point3f(0, 1, 0), angle=rotation_angles.y)
            .rotate(axis=mi.Point3f(0, 0, 1), angle=rotation_angles.z)
            .scale(scaling)
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
        compute_initial_scale_gss: bool = False,
        gss_max_iters: int = 25,
        gss_samples_per_pixel: int = 4,
        gss_coarse_samples: int = 8,
    ) -> None:
        """
        Executes pose optimization for `num_iters` steps.

        The user provides a list of MaskedPose reference images `ref_images`.
        `scene_optimizer` determines the underlying scene optimization
        algorithm (e.g. `SceneAdam`).

        The convergence point of the camera rays is computed up front and
        becomes the initial translation guess. If `compute_initial_scale_gss`
        is `True`, an isotropic golden-section search is then run via
        `find_initial_scale_gss` with `gss_max_iters` iterations at
        `gss_samples_per_pixel` spp; the resulting radius replaces
        `self.initial_scale`. The post-GSS value is available on
        `self.initial_scale` for callers that need to forward it (e.g. to
        `pose_params_to_bbox_transforms`).

        Optionally, `views_to_render` can be used to render the current state of
        the scene for provided views and optionally save it to `save_directory_path`.

        Parameter checking occurs in `SceneOptimizer.optimize` method. At each
        iteration the mesh is saved to `save_directory_path` (if the path is not
        provided it is saved at the current directory level).

        Before any optimization step runs, the jittered initial pose is applied and
        `show_optimization_state` is called once for iteration 0, rendering and
        saving `views_to_render` (to `save_directory_path`, if provided) at the
        initial, pre-optimization state, so it is recorded in addition to whatever
        `PropertyOptimizer.optimize`'s own per-iteration cadence (governed by
        `display_state_interval`) saves during the `num_iters` optimization steps
        that follow. As a result, `PoseOptimizer.optimize` saves one more state than
        the base loop alone would over the same run.
        """
        convergence_point = self.compute_camera_convergence_point(
            sensors=[image.sensor for image in ref_images]
        )
        self._initial_position_guess = convergence_point

        if compute_initial_scale_gss:
            self._initial_scale = self.find_initial_scale_gss(
                ref_images=ref_images,
                max_iters=gss_max_iters,
                samples_per_pixel=gss_samples_per_pixel,
                coarse_samples=gss_coarse_samples,
            )

        # Apply the jittered initial pose and render a snapshot so callers can
        # eyeball where LBFGS is starting from before any gradient step runs.
        with dr.suspend_grad():
            self._shape_update_callback(
                self.optimization_parameters[self._optimization_key]
            )
        self.show_optimization_state(
            iteration=0,
            views_to_render=views_to_render,
            save_directory_path=save_directory_path,
        )

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
        make sure the mesh is visible initially. A seeded jitter is added to the
        translation (proportional to `initial_scale`) and to the per-axis scaling
        component (`initial_scale * (1 + jitter * z)`) to break perfect symmetry
        in the initial pose, which otherwise tends to trap LBFGS in saddles.
        """
        z = self._jitter_draws
        flattened_params = mi.Float(
            self._initial_position_guess[0] + self._initial_scale * self._jitter * z[0],
            self._initial_position_guess[1] + self._initial_scale * self._jitter * z[1],
            self._initial_position_guess[2] + self._initial_scale * self._jitter * z[2],
            self._initial_scale * (1.0 + self._jitter * z[3]),
            self._initial_scale * (1.0 + self._jitter * z[4]),
            self._initial_scale * (1.0 + self._jitter * z[5]),
            # Initial angles are 0.0 (no rotation)
            0.0,
            0.0,
            0.0,
        )

        return {self._optimization_key: dr.ravel(flattened_params)}

    @property
    def initial_scale(self) -> float:
        """
        Current initial scale used for the proxy. Updated by
        `find_initial_scale_gss` (directly or via `optimize` with
        `compute_initial_scale_gss=True`).
        """
        return self._initial_scale

    @property
    def initial_position_guess(self) -> tuple[float, float, float]:
        """
        Current initial position guess (the camera convergence point once
        `optimize` has been called).
        """
        return self._initial_position_guess

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

    def find_initial_scale_gss(
        self,
        ref_images: list[MaskedPoseImage],
        max_iters: int = 25,
        samples_per_pixel: int = 4,
        upper_bound_safety: float = 0.9,
        lower_bound_fraction: float = 0.01,
        coarse_samples: int = 8,
    ) -> float:
        """
        Two-phase search for the isotropic scale that minimizes the silhouette
        loss when the proxy is placed at `self._initial_position_guess`.

        Phase 1 sweeps `coarse_samples` evenly-spaced radii across the full
        bracket `[lower_bound_fraction * upper, upper_bound_safety *
        min_camera_distance]`, where `min_camera_distance` is the minimum
        Euclidean distance from the convergence point to any sensor origin in
        `ref_images`. The safety factor keeps the proxy from starting tangent
        to a camera (degenerate silhouette gradient). The coarse sweep guards
        against degenerate configurations (e.g. cameras in a ring) where the
        silhouette loss is flat for some radii and GSS alone would converge to
        a meaningless point inside that flat region.

        Phase 2 runs golden-section search inside a sub-bracket centered on
        the best coarse sample, refining to ~1% of the original `upper`.

        Each evaluation calls `self._update_positions` with
        `translate(convergence_point) @ scale((r, r, r))`, renders all views at
        `samples_per_pixel` spp, and sums `self._loss.evaluate` over views.

        :returns: The mid-point of the final bracket as a Python float.
        """
        if len(ref_images) == 0:
            raise ValueError("`ref_images` cannot be empty.")
        if max_iters < 1:
            raise ValueError("`max_iters` must be a positive integer.")
        if samples_per_pixel < 1:
            raise ValueError("`samples_per_pixel` must be a positive integer.")
        if coarse_samples < 3:
            raise ValueError("`coarse_samples` must be at least 3.")

        cp = np.asarray(self._initial_position_guess, dtype=np.float64)

        min_camera_distance = float("inf")
        for image in ref_images:
            sensor_matrix = np.asarray(
                image.sensor.world_transform().matrix.numpy()
            ).squeeze()
            origin = sensor_matrix[:3, 3]
            distance = float(np.linalg.norm(origin - cp))
            if distance < min_camera_distance:
                min_camera_distance = distance

        upper = upper_bound_safety * min_camera_distance
        lower = lower_bound_fraction * upper
        if upper <= lower:
            raise RuntimeError(
                "Camera convergence point is too close to a camera origin; "
                "cannot bracket the radius search."
            )
        logger.debug(
            f"GSS bracket [{lower:.4g}, {upper:.4g}] "
            f"(min camera distance = {min_camera_distance:.4g})"
        )

        # Phase 1: coarse linear sweep across the full bracket.
        coarse_scales = np.linspace(lower, upper, coarse_samples)
        coarse_losses = [
            self._evaluate_isotropic_scale_loss(
                scale=float(s),
                ref_images=ref_images,
                samples_per_pixel=samples_per_pixel,
            )
            for s in coarse_scales
        ]
        best_idx = int(np.argmin(coarse_losses))
        for s, fs in zip(coarse_scales, coarse_losses):
            logger.debug(f"GSS coarse: scale={s:.4g} loss={fs:.4g}")
        logger.debug(
            f"GSS coarse best: scale={coarse_scales[best_idx]:.4g} "
            f"loss={coarse_losses[best_idx]:.4g} (idx {best_idx}/{coarse_samples - 1})"
        )

        if np.allclose(coarse_losses, coarse_losses[0]):
            logger.warning(
                "GSS coarse sweep produced a flat loss: no scale separates the "
                "proxy from the background. Returning the midpoint of the bracket; "
                "the camera configuration is likely degenerate."
            )
            return 0.5 * (lower + upper)

        # Phase 2: golden-section refinement inside a 2-step neighborhood of
        # the best coarse sample.
        a = float(coarse_scales[max(best_idx - 1, 0)])
        b = float(coarse_scales[min(best_idx + 1, coarse_samples - 1)])
        logger.debug(f"GSS refine bracket [{a:.4g}, {b:.4g}]")

        phi_inv = (math.sqrt(5.0) - 1.0) / 2.0
        c = b - phi_inv * (b - a)
        d = a + phi_inv * (b - a)
        fc = self._evaluate_isotropic_scale_loss(
            scale=c,
            ref_images=ref_images,
            samples_per_pixel=samples_per_pixel,
        )
        fd = self._evaluate_isotropic_scale_loss(
            scale=d,
            ref_images=ref_images,
            samples_per_pixel=samples_per_pixel,
        )

        for i in range(max_iters):
            if (b - a) < 0.01 * upper:
                logger.debug(
                    f"GSS converged after {i} iterations (bracket < 1% of upper)"
                )
                break
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - phi_inv * (b - a)
                fc = self._evaluate_isotropic_scale_loss(
                    scale=c,
                    ref_images=ref_images,
                    samples_per_pixel=samples_per_pixel,
                )
            else:
                a, c, fc = c, d, fd
                d = a + phi_inv * (b - a)
                fd = self._evaluate_isotropic_scale_loss(
                    scale=d,
                    ref_images=ref_images,
                    samples_per_pixel=samples_per_pixel,
                )
            logger.debug(
                f"GSS iter {i + 1:2d}: bracket [{a:.4g}, {b:.4g}] "
                f"width={b - a:.4g} | f({c:.4g})={fc:.4g} f({d:.4g})={fd:.4g}"
            )

        return 0.5 * (a + b)

    def _evaluate_isotropic_scale_loss(
        self,
        scale: float,
        ref_images: list[MaskedPoseImage],
        samples_per_pixel: int,
    ) -> float:
        """
        Renders all `ref_images` with the proxy placed at
        `self._initial_position_guess` and scaled isotropically by `scale`,
        and returns the summed `self._loss` across views.

        Gradients are suspended; this is intended as a forward-only loss
        evaluator (e.g. for `find_initial_scale_gss`).

        If `_update_positions` rejects the transform (e.g. very small scales
        collapse the proxy enough that some vertex normals become zero-length
        and fail the normalization invariant), a warning is logged and
        `float("inf")` is returned so the search can reject this point.
        """
        with dr.suspend_grad():
            tf = (
                mi.Transform4f()
                .translate(mi.Point3f(*self._initial_position_guess))
                .scale(mi.Point3f(scale, scale, scale))
            )
            try:
                self._update_positions(tf_new_pose=tf)
            except ValueError as exc:
                logger.warning(
                    f"Rejecting scale={scale:.4g} during loss evaluation: {exc}"
                )
                return float("inf")
            total = mi.Float(0.0)
            for image in ref_images:
                rendered = mi.render(
                    scene=self._scene,
                    params=self._scene_params,
                    sensor=image.sensor,
                    spp=samples_per_pixel,
                )
                total = total + self._loss.evaluate(rendered, image.image)
            return float(total.numpy().item())

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
