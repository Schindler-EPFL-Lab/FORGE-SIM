import abc
import uuid
from dataclasses import dataclass
from typing import Callable, TypeAlias, Union

import mitsuba as mi
import mlflow
import numpy as np

from rebel_shape.data_processing.masked_pose_image import MaskedPoseImage
from rebel_shape.loss_evaluators.image_loss import ImageLoss
from rebel_shape.regularization.regularization import Regularization
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.schedulers.optimization_scheduler import OptimizationScheduler

MitsubaOptimizationType: TypeAlias = Union["mi.Float", "mi.TensorXf"]
"""
Type alias for the type of parameters that can be optimized. Most of the time it is a
`mi.Float`, e.g. for flattened vertex positions, but it can also be a `mi.TensorXf`, 
e.g. for bitmap textures.
"""


class SceneOptimizer(abc.ABC):
    """
    Interface representing the optimization method for differentiable rendering.

    It provides an abstraction around `drjit` calls to back-propagate gradients
    to the correct scene properties. The user provides a closure that updates the
    specified scene parameters in the `optimize` method, making the optimization
    method generic.

    Classes that optimize concrete properties (e.g. `MeshOptimizer` which optimizes
    vertex positions) are expected to call concrete implementations of
    `SceneOptimizer`.

    The `optimize` method is the main entry point for optimization, and it typically
    does not depend on the concrete optimizer; only the `run_optimization_step`
    is different. Methods for setting the initial state of the parameters and learning
    rate are also required.
    """

    @dataclass
    class ParamOptimizationConfig:
        """
        Settings for the optimization of a single parameter.

        Each parameter influences the scene in a different way and a different
        schedule and loss function are needed to optimize it. This class provides
        a structured way to define these settings.
        """

        parameter_key: str
        """
        The key by which the parameter is internally identified. Mitsuba is based on
        dictionaries and the internal optimization state for this parameter is 
        identified by this key.
        """
        image_loss: ImageLoss
        """
        Image loss used to compute the difference between rendered and reference images.
        """
        scheduler: OptimizationScheduler
        """Controls the learning rate based on the current iteration."""
        scene_update_callback: Callable[[MitsubaOptimizationType], None]
        """
        Defines how the scene parameters are updated as well as all other actions
        that need to happen afterwards. It is a closure that takes the current
        internal state of the optimizer as input. The type of this state is either
        a flattened array of floats (`mi.Float`, almost all cases) or a tensor of 
        floats (`mi.TensorXf`, used for bitmap textures).
        """
        regularization: Regularization | None = None
        """
        Regularization term that is applied to the parameter during optimization.
        It is applied after the image loss is compute for all images. If `None`,
        no regularization is applied.
        """

    def __init__(
        self,
        max_batch_size: int | None = None,
        randomize_image_order: bool = True,
        image_order_seed: int = 42,
    ) -> None:
        """
        Initializes the optimizer.

        This method should be overridden by child classes to set up any necessary state
        as it only initializes the optimizer with a unique identifier.

        `max_batch_size` is the maximum number of images that can be used for gradient
        update in a single shape update during an optimization step (effectively each
        step is an epoch over all images).

        At optimization time, if the number of images in `ref_images` is
        larger than `max_batch_size`, multiple shape updates are performed in
        `run_optimization_step`, each considering a subset of images. This limits
        memory consumption and is especially useful for optimizers like Adam.
        Setting it to `None` means that all images are used for each update.

        `randomize_image_order` controls whether the order of images is shuffled
        before each optimization step. This can help improve convergence by reducing
        correlations between images. The shuffling is done using a fixed seed
        (`image_order_seed`) to ensure reproducibility. As it is possible for batches
        to have different sizes, the order of batches is also shuffled.
        """
        self._optimizer_id = str(uuid.uuid4())
        """
        Unique identifier for the optimizer instance. This is useful when multiple
        optimizers are used in a composite optimizer to distinguish between them.
        """

        if max_batch_size is not None and max_batch_size < 1:
            raise ValueError("`max_batch_size` has to be a positive integer")

        self._max_batch_size = max_batch_size
        self._randomize_image_order = randomize_image_order
        self._image_order_seed = image_order_seed

        self._rng = np.random.default_rng(image_order_seed)

    @abc.abstractmethod
    def set_parameters(self, parameters: dict[str, MitsubaOptimizationType]) -> None:
        """
        Initializes the optimizer with variables that will be optimized.

        This method needs to be called before starting the optimization process to
        configure the optimizer with the properties that will be optimized. An error
        is raised if this method is not called before `optimize`.

        `parameters` is a dictionary whose keys are user defined names for
        optimization variables and values are 1D Mitsuba float32 arrays (`mi.Float`) or
        tensors (`mi.TensorXf`) that represent the current state of the variable.

        No checks are performed on the keys of the `parameters` dictionary as this is
        a configuration step. It can also be called during optimization to change
        the shape of the parameters (e.g. when optimizing a mesh and the number of
        vertices changes).

        :raises NotImplementedError: if the method was not implemented by a child class.
        """
        raise NotImplementedError(
            "Classes that inherit from `SceneOptimizer` need to implement the "
            "`set_parameters` method."
        )

    @abc.abstractmethod
    def set_learning_rate(self, learning_rates: dict[str, float]) -> None:
        """
        Set the learning rate for each parameter.

        This method is used to set the learning rate for each parameter that is
        being optimized. It is called before each optimization step to adjust the
        learning rate based on the current iteration. The learning rate of just a
        subset of parameters can be set.

        :raises NotImplementedError: if the method was not implemented by a child class.
        :raises ValueError: if `learning_rates` dictionary does not contain
            a valid parameter key or if the learning rate is not positive.
        """
        raise NotImplementedError(
            "Classes that inherit from `SceneOptimizer` need to implement the "
            "`set_learning_rate` method."
        )

    @staticmethod
    def _objective_function(
        scene: "mi.Scene",
        scene_params: "mi.SceneParams",
        ref_images: list[MaskedPoseImage],
        param_config: "SceneOptimizer.ParamOptimizationConfig",
        parameters: MitsubaOptimizationType,
        seed: int,
    ) -> tuple["mi.TensorXf", "mi.TensorXf"]:
        """
        Common objective function for optimization.

        This function computes the total loss and regularization loss for a given
        `scene` and its `scene_params`. It renders images for all sensors of
        `ref_images`, computes the image loss based on the `ref_images`, and applies
        regularization if specified in `param_config`.

        `parameters` are the current state of the variable that is being optimized, and
        it is passed to the `scene_update_callback` closure to update the scene
        parameters before rendering and ensure gradient backpropagation .

        `seed` is used to ensure that the rendered images are different for each
        iteration of the optimization procedure, avoiding correlation between
        iterations.

        :returns: A tuple of tensors of shape (1,) that contain the total image loss and
            the regularization loss for the provided parameters.
        """
        if len(ref_images) == 0:
            raise ValueError("At least one image is required `ref_images` list")

        param_config.scene_update_callback(parameters)

        batch_reg_loss = mi.TensorXf(0.0)
        batch_image_loss = mi.TensorXf(0.0)

        for ref_image in ref_images:
            image = mi.render(
                scene=scene,
                params=scene_params,
                sensor=ref_image.sensor,
                seed=seed,
            )
            image = ref_image.apply_mask(image)
            loss = param_config.image_loss.evaluate(
                image=image, ref_image=ref_image.image
            )
            batch_image_loss += loss

        if param_config.regularization is not None:
            reg_loss = param_config.regularization.evaluate()
            batch_reg_loss += reg_loss

        # Normalize the image loss by the number of sensors
        batch_image_loss /= len(ref_images)

        return batch_image_loss, batch_reg_loss

    def _batch_sizes(self, num_images: int) -> list[int]:
        """
        Compute batch sizes to split `num_images` into batches whose sizes differ by at
        most 1 and do not exceed `self._max_batch_size`.

        Behavior:
        - If `self._max_batch_size` >= `num_images`, return a single batch.
        - Otherwise choose the minimal number of batches that do not exceed
          `self._max_batch_size` and distribute `num_images` as evenly as possible
          across those batches. The first `remainder` batches will have size `base +
          1`, the rest `base`.

        Examples:
        - num_images=20, max_batch_size=50 -> num_batches=1 -> [20]
        - num_images=119, max_batch_size=50 -> num_batches=3 -> [40, 40, 39]
        - num_images=118, max_batch_size=50 -> num_batches=3 -> [39, 40, 39]

        :returns: List of batch sizes summing to `num_images`.
        :raises ValueError: if `num_images` is less than 1.
        """
        if num_images < 1:
            raise ValueError("`num_images` must be at least 1.")

        if self._max_batch_size is None or self._max_batch_size >= num_images:
            return [num_images]

        # Compute the minimal number of batches such that no batch exceeds
        # `self._max_batch_size`, i.e. num_batches * max_batch_size >= num_images
        num_batches = (num_images + self._max_batch_size - 1) // self._max_batch_size

        # It is always possible to split num_images into num_batches such that
        # the batch sizes differ by at most 1.
        base = num_images // num_batches
        remainder = num_images % num_batches

        sizes = [base + 1] * remainder + [base] * (num_batches - remainder)

        return sizes

    def iteration_batch_indices(self, num_images: int) -> list[list[int]]:
        """
        Compute the indices of images for each batch in an iteration.

        First, the number of batches and their sizes are determined using
        `self._batch_sizes`. All batches differ by at most 1 in size. Then, the indices
        of images are computed for each batch.

        If `self._randomize_image_order` is `True`, the order of images is shuffled
        before computing the batch indices. The shuffling is done using a fixed seed
        (`self._image_order_seed`) to ensure reproducibility.

        Furthermore, if `self._randomize_image_order` is `True`, the order of batches
        is also shuffled to avoid possible bias when batches have different sizes.

        :returns: A list of lists, where each inner list contains the indices of
            images for a batch.
        """
        batch_sizes = self._batch_sizes(num_images)
        indices = np.arange(num_images, dtype=np.uint32)

        # Randomize the order of images and batch sizes if enabled
        if self._randomize_image_order:
            self._rng.shuffle(indices)
            batch_sizes = self._rng.permutation(batch_sizes)

        batch_indices: list[list[int]] = []
        start = 0

        for size in batch_sizes:
            end = start + size
            batch_indices.append(indices[start:end].tolist())
            start = end

        return batch_indices

    def optimize(
        self,
        scene: "mi.Scene",
        scene_params: "mi.SceneParams",
        param_optimization_configs: list[ParamOptimizationConfig],
        ref_images: list[MaskedPoseImage],
        optimization_events: OptimizationEvents,
        num_iters: int,
    ) -> None:
        """
        Executes the optimization procedure on selected scene parameters.

        Images are rendered based on `scene` and `scene_params`.
        Optimization proceeds for `num_iters`; each iteration considers all
        sensors of `ref_images` and computes a loss based on the rendered images and
        `ref_images`.

        No assumptions are made about which parameter(s) of the scene are optimized.
        Each has unique properties, defined by the `param_optimization_configs` list.
        The order in the list typically defines the order in which they are optimized,
        but this is the responsibility of the `run_optimization_step` method.

        The most important properties are 'parameter_key', used to identify
        the parameter, `scene_update_callback`, a closure defining actions
        after the gradient has been applied, `scheduler`, which defines the learning
        rate for the parameter, `image_loss`, which defines how the loss is
        computed, and `regularization_term`, which penalizes bad configurations.
        For more details, see `ParamOptimizationConfig`.

        Additional events that should be triggered at the end of iteration, after
        the gradient has been applied and loss computed, can be added to
        `optimization_events` and will be triggered.

        `mlflow` records the learning rate, total loss, regularization loss, and
        reconstruction loss for each iteration. The learning rate is logged for each
        parameter, while the losses are logged as a single value for the entire
        optimization step.

        :raises RuntimeError: if `set_parameters` was not called before calling
            `optimize` - this is detected by checking if all parameters in
            `param_optimization_configs` are initialized in the optimizer.

        :raises ValueError: if `ref_images` is empty,
            or if `param_optimization_configs` is empty,
            or if `num_iters` is less than 1.
        """
        if len(ref_images) == 0:
            raise ValueError("At least one image is required in 'ref_images' list.")
        if num_iters < 1:
            raise ValueError("`num_iters` has to be at least 1.")
        if len(param_optimization_configs) == 0:
            raise ValueError("`param_optimization_configs` cannot be empty.")

        for param_config in param_optimization_configs:
            if param_config.parameter_key not in self.parameters.keys():
                raise RuntimeError(
                    f"Parameter {param_config.parameter_key} is not set."
                    "Call `set_parameters` with this key before starting optimization."
                )

        for iteration in range(num_iters):
            # Set the learning rate for each parameter based on the current iteration
            learning_rates: dict[str, float] = {
                config.parameter_key: config.scheduler.learning_rate_for_step(iteration)
                for config in param_optimization_configs
            }
            self.set_learning_rate(learning_rates)

            for key, learning_rate in learning_rates.items():
                mlflow.log_metric(
                    f"learning rate - {key}", learning_rate, step=iteration
                )

            # Run the optimization step and compute the loss
            loss, reg_loss = self.run_optimization_step(
                scene=scene,
                scene_params=scene_params,
                param_optimization_configs=param_optimization_configs,
                ref_images=ref_images,
                seed=iteration,
            )

            # Log the losses with mlflow
            loss_float = float(loss.numpy().astype(np.float64))
            reg_loss_float = float(reg_loss.numpy().astype(np.float64))

            mlflow.log_metric("total loss", loss_float, step=iteration)
            mlflow.log_metric("regularization loss", reg_loss_float, step=iteration)
            mlflow.log_metric(
                "reconstruction loss", loss_float - reg_loss_float, step=iteration
            )

            # Trigger events for the current iteration
            optimization_events.trigger_events(iteration=iteration)

    @abc.abstractmethod
    def run_optimization_step(
        self,
        scene: "mi.Scene",
        scene_params: "mi.SceneParams",
        param_optimization_configs: list["SceneOptimizer.ParamOptimizationConfig"],
        ref_images: list[MaskedPoseImage],
        seed: int,
    ) -> tuple["mi.TensorXf", "mi.TensorXf"]:
        """
        Execute a single optimization step.

        Images are rendered based on `scene` and `scene_params`.
        All views specified by sensors of `ref_images` are considered and
        `image_loss` is computed between the rendered images and `ref_images` for every
        internal parameter of the scene. The regularization loss is computed after each
        update (at the end of the batch) and added to the total loss.

        How the scene parameters should be updated is specified and which loss is
        applied is defined by `param_optimization_configs`. The order in the list
        defines the order in which they are optimized.

        Rendering uses the provided `seed` which should be different for each
        iteration of the optimization procedure to avoid correlation.

        :returns: A tuple of tensors of shape (1,) that contain the total loss and the
            regularization loss of the optimization step.
        :raises NotImplementedError: if the method was not implemented by a child class.
        """
        raise NotImplementedError(
            "Classes that inherit from `SceneOptimizer` need to implement the "
            + "`run_optimization_step` method."
        )

    @property
    @abc.abstractmethod
    def parameters(self) -> dict[str, MitsubaOptimizationType]:
        """
        Returns the internal state of the optimizer.

        Mitsuba optimizers work with a set of parameters that are accessed by an
        appropriate key. This property returns a dictionary of the current state of all
        parameters that are being optimized.

        :returns: A dictionary where keys are parameter names and values are
            `mi.Float` or `mi.TensorXf` objects representing the current state of
            the parameters.
        :raises NotImplementedError: if the method was not implemented by a child class.
        """
        raise NotImplementedError(
            "Classes that inherit from `SceneOptimizer` need to implement the "
            + "`variables` property."
        )

    @property
    def optimizer_id(self) -> str:
        """
        Returns the unique identifier of the optimizer.

        This ID is primarily used to be able to distinguish different optimizers when
        a composite optimizer is used, e.g. when multiple optimizers are used for
        different parameters of the scene.

        :returns: A string representing a unique identifier for the optimizer.
        """
        return self._optimizer_id
