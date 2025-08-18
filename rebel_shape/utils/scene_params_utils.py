import drjit as dr
import mitsuba as mi


class ChangeRevertSceneParameters:
    """
    Enables reassigning values of scene parameters and revert it back.
    This is helpful when we need to transform some scene parameters and them revert
    our changes back. For example transform object into a dark object and then
    revert it back to the original object.
    """

    def __init__(
        self,
        scene_params: "mi.SceneParameters",
    ):
        """
        Initializes the class with parameters of the scene `scene_params` that might
        be changed if `.change` is called
        """
        self._original_values = {}
        self._scene_params = scene_params

    def _update_and_save_copy(
        self, dict_to_change: dict[str, "mi.TensorXf | float"], save_copy: bool = False
    ):
        for key, value in dict_to_change.items():
            if save_copy:
                self._original_values[key] = dr.copy(self._scene_params[key])
            if isinstance(value, float):
                value = dr.full(
                    mi.TensorXf, value=value, shape=self._scene_params[key].shape
                )
            self._scene_params[key] = value
        self._scene_params.update()

    def change(self, values: dict[str, "mi.TensorXf | float"]) -> None:
        """
        Updates scene parameters with new values.
        For each (key, value) pair in the `values` dictionary:
            - If `value` is a `mi.TensorXf`, it is directly assigned to the
            corresponding scene parameter.
            - If `value` is a `float`, a new `mi.TensorXf` is created with the same
            shape as the existing parameter, filled with this float, and assigned to
            the parameter.

        :raises
            RuntimeError: If attempting to change a parameter that has already been
            changed and not yet reverted.
        """

        common_keys = set(values.keys()) & set(self._original_values.keys())
        if len(common_keys) > 0:
            raise RuntimeError(
                f"Cannot change values of {common_keys}."
                f"The parameters must be reverted before being updated the second time"
            )
        self._update_and_save_copy(values, save_copy=True)

    def revert(self) -> None:
        """
        Undoes changes of scene parameters which were done in the previous `self.change`
        call
        """
        self._update_and_save_copy(self._original_values, save_copy=False)
        self._original_values = {}
