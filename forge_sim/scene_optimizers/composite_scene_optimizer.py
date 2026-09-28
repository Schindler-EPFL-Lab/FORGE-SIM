import mitsuba as mi

from forge_sim.data_processing.masked_pose_image import MaskedPoseImage
from forge_sim.scene_optimizers.scene_optimizer import (
    MitsubaOptimizationType,
    SceneOptimizer,
)


class CompositeSceneOptimizer(SceneOptimizer):
    """
    Composite Scene optimizer that allows using different scene optimizers
    for different scene properties.

    This optimizer manages multiple `SceneOptimizer` instances, each responsible for
    updating a subset of scene parameter with its own settings (e.g. learning rate,
    batch size). This is useful when different scene properties require different
    optimization strategies.

    During optimization, each parameter is updated sequentially according to the order
    specified in `param_update_order`, each using its dedicated optimizer.
    """

    def __init__(
        self, optimizer_config: list[tuple[SceneOptimizer, list[str]]]
    ) -> None:
        """
        Initializes the composite optimizer with a list of optimizers and the parameter
        keys they are responsible for optimizing.

        `optimizer_config` is a list of tuples, where each tuple contains an instance
        of `SceneOptimizer` and a list of parameter keys that this optimizer will
        handle. Each optimizer in the list must have a unique `optimizer_id` and
        all parameter keys must be unique across all optimizers.

        :raises ValueError: If any optimizer in `optimizer_config` has a non-unique
            `optimizer_id`, if any parameter key is associated with more than one
            optimizer, or if any optimizer has no associated parameter keys.
        """
        # The defaults are satisfactory as they will not be used.
        super().__init__()

        optimizer_ids = [opt.optimizer_id for opt, _ in optimizer_config]
        if len(optimizer_ids) != len(set(optimizer_ids)):
            raise ValueError(
                "All optimizers in `optimizer_config` must have unique IDs."
            )

        self._scene_optimizers = {opt.optimizer_id: opt for opt, _ in optimizer_config}
        """
        Dictionary mapping optimizer IDs to their corresponding `SceneOptimizer`
        instances. This allows routing optimization parameters to the correct optimizer
        based on the parameter key.
        """

        self._parameter_to_optimizer_map: dict[str, str] = {}
        """
        Maps each parameter key to its corresponding `SceneAdam` optimizer.
        This is used to route optimization parameters to the correct optimizer.
        """

        for opt, param_keys in optimizer_config:
            if len(param_keys) == 0:
                raise ValueError(
                    "Each optimizer in `optimizer_config` must have at least one "
                    "associated parameter key."
                )
            for key in param_keys:
                if key in self._parameter_to_optimizer_map:
                    raise ValueError(
                        f"Parameter '{key}' is already associated with an optimizer."
                    )
                self._parameter_to_optimizer_map[key] = opt.optimizer_id

    def set_parameters(self, parameters: dict[str, MitsubaOptimizationType]) -> None:
        """
        Initializes the underlying optimizers with variables that will be optimized.

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

        The parameters are routed to the optimizer instance that is responsible for
        a given parameter. The `parameters` dictionary should contain keys that match
        the keys defined in the `optimizer_config` when initializing the
        `CompositeSceneOptimizer`.

        :raises RuntimeError: If a parameter does not have an associated `SceneAdam`
            optimizer.
        """
        per_optimizer_params: dict[str, dict[str, MitsubaOptimizationType]] = {}

        for param_key, param_value in parameters.items():
            if param_key not in self._parameter_to_optimizer_map:
                raise RuntimeError(f"There is no associated optimizer for {param_key}.")

            optimizer_key = self._parameter_to_optimizer_map[param_key]

            per_optimizer_params.setdefault(optimizer_key, {})[param_key] = param_value

        for optimizer_key, params_dict in per_optimizer_params.items():
            self._scene_optimizers[optimizer_key].set_parameters(parameters=params_dict)

    def set_learning_rate(self, learning_rates: dict[str, float]) -> None:
        """
        Set the learning rate for each parameter in the composite optimizer.

        This method is used to set the learning rate for each parameter that is
        being optimized. It is called before each optimization step to adjust the
        learning rate based on the current iteration. The learning rate of just a
        subset of parameters can be set.

        Since this is a composite optimizer, it routes the learning rates to the
        optimizer instance that is responsible for that parameter. The `learning_rates`
        dictionary should contain keys that match the keys defined in the
        `optimizer_config` when initializing the `CompositeSceneOptimizer`.

        :raises ValueError: if `learning_rates` dictionary does not contain
            a valid parameter key or if the learning rate is not positive.
        """
        per_optimizer_learning_rates: dict[str, dict[str, float]] = {}

        for param_key, learning_rate in learning_rates.items():
            if param_key not in self._parameter_to_optimizer_map:
                raise RuntimeError(f"There is no associated optimizer for {param_key}.")
            if learning_rate <= 0.0:
                raise ValueError(
                    f"Learning rate must be positive, got {param_key}:{learning_rate}."
                )

            optimizer_key = self._parameter_to_optimizer_map[param_key]

            per_optimizer_learning_rates.setdefault(optimizer_key, {})[param_key] = (
                learning_rate
            )

        for optimizer_key, lr_dict in per_optimizer_learning_rates.items():
            self._scene_optimizers[optimizer_key].set_learning_rate(
                learning_rates=lr_dict
            )

    def run_optimization_step(
        self,
        scene: "mi.Scene",
        scene_params: "mi.SceneParams",
        param_optimization_configs: list[SceneOptimizer.ParamOptimizationConfig],
        ref_images: list[MaskedPoseImage],
        seed: int = 0,
    ) -> tuple["mi.TensorXf", "mi.TensorXf"]:
        """
        Execute a single optimization step for all optimizers.

        Images are rendered based on `scene` and `scene_params`.
        All views specified by sensors of `ref_images` are considered and `image_loss`
        is computed between the rendered images and `ref_images` for every
        internal parameter of the scene. The regularization loss is computed after each
        update (at the end of the batch) and added to the total loss.

        How the scene parameters should be updated is specified and which loss is
        applied is defined by `param_optimization_configs`. The order in the list
        defines the order in which they are optimized.

        Rendering uses the provided `seed` which should be different for each
        iteration of the optimization procedure to avoid correlation.

        First it is determined which parameters are associated with which optimizer
        and then each optimizer runs its own optimization step with the parameters
        it is responsible for. The total loss and regularization loss are averaged
        across all optimizers and returned.

        :returns: A tuple of tensors of shape (1,) that contain the total loss and the
            regularization loss of the optimization step.
        """
        total_loss = mi.TensorXf(0.0)
        total_reg_loss = mi.TensorXf(0.0)

        per_optimizer_param_configs: dict[
            str, list[SceneOptimizer.ParamOptimizationConfig]
        ] = {}

        for param_config in param_optimization_configs:
            if param_config.parameter_key not in self._parameter_to_optimizer_map:
                raise RuntimeError(
                    f"No associated optimizer for {param_config.parameter_key}."
                )

            optimizer_key = self._parameter_to_optimizer_map[param_config.parameter_key]
            per_optimizer_param_configs.setdefault(optimizer_key, []).append(
                param_config
            )

        for optimizer_key, param_configs in per_optimizer_param_configs.items():
            loss, reg_loss = self._scene_optimizers[
                optimizer_key
            ].run_optimization_step(
                scene=scene,
                scene_params=scene_params,
                param_optimization_configs=param_configs,
                ref_images=ref_images,
                seed=seed,
            )

            total_loss += loss
            total_reg_loss += reg_loss

        total_loss /= len(param_optimization_configs)
        total_reg_loss /= len(param_optimization_configs)

        return total_loss, total_reg_loss

    @property
    def parameters(self) -> dict[str, MitsubaOptimizationType]:
        """
        Returns the internal state of all parameters from all optimizers.

        Mitsuba optimizers work with a set of parameters that are accessed by an
        appropriate key. Furthermore, this class provides a unified interface for
        accessing the parameters of all optimizers in a composite optimizer.
        All states of the parameters are gathered and returned as a single dictionary.

        :returns: A dictionary where keys are parameter names and values are
            `mi.Float` or `mi.TensorXf` objects representing the current state of
            the parameters.
        """
        variables: dict[str, MitsubaOptimizationType] = {}
        for optimizer in self._scene_optimizers.values():
            # Each optimizer already provides its own dictionary of parameters
            variables.update(optimizer.parameters)

        return variables
