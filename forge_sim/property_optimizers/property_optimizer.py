import abc
from datetime import datetime
from pathlib import Path
from typing import Any

import mitsuba as mi

from forge_sim.data_processing.masked_pose_image import MaskedPoseImage
from forge_sim.scene_optimizers.optimization_events import OptimizationEvents
from forge_sim.scene_optimizers.scene_optimizer import SceneOptimizer
from forge_sim.utils.image_utils import (
    show_comparison_with_reference,
    show_rendered_views,
)


class PropertyOptimizer(abc.ABC):
    """
    An interface that prescribes the behavior of an optimizer that updates
    a property of a Mitsuba scene. Specific implementations of this class are meant
    to be used in optimization scripts.

    It provides an abstraction around `SceneOptimizer` instances to automate and
    simplify optimization of specific scene properties. The main optimization loop is
    very similar for all scene properties that are of interest to optimize, so the
    interface prescribes 2 properties and 1 callback:

        1. `optimization_parameters` returns a `dict` whose values are the initial
           optimization variables and are passed to the `SceneOptimizer`
        2. `parameter_optimization_configs` returns parameter optimization configs
            that are passed to `SceneOptimizer.optimize` method. They define the loss,
            regularization, scheduling and how the scene is updated after each
            parameter update.
        3. `save_callback(iteration, output_directory)` is called once per iteration to
            save the state of the optimization to the specified directory (None
            signals that progress should not be saved)

    Additionally, methods to compare the current state of the optimization with the
    reference are provided, as well as a more extensive method meant to be used at
    the end of the optimization to visually compare the final result for all images.
    """

    def __init__(
        self,
        scene: "mi.Scene",
        scene_params: "mi.SceneParams",
        optimization_events: OptimizationEvents,
        display_state_interval: int = 2,
    ) -> None:
        """
        The `scene` is a loaded Mitsuba scene object, while `scene_params` are a
        dictionary of parameters that are to be optimized. `optimization_events` store
        all actions that happen at a specific iteration throughout the optimization.

        `display_state_interval` is the number of iterations after which the
        optimization state is displayed. When a small number of images is used for
        optimization rendering, the current state of the scene at each iteration
        can be very expensive so it is recommended to set this value to a larger
        number (5 - 10). Setting it to 1 is useful for debugging and visualization
        purposes.
        """
        self._scene = scene
        self._scene_params = scene_params
        self._optimization_events = optimization_events
        self._display_state_interval = display_state_interval

    def optimize(
        self,
        ref_images: list["MaskedPoseImage"],
        scene_optimizer: SceneOptimizer,
        num_iters: int = 100,
        views_to_render: list["mi.Sensor"] | None = None,
        save_directory_path: Path | None = None,
    ) -> None:
        """
        Fills optimization events with default per iteration behavior and invokes the
        main optimization loop.

        Images are rendered based on `scene` and `scene_params`.
        Optimization proceeds for `num_iters` and each iteration considers all
        views specified by sensors of `ref_images`. `image_loss` is computed by
        comparing the rendered images to corresponding `ref_images`.

        Optionally, `views_to_render` can be used to render the current state of
        the scene and save it to `save_directory_path`. If `save_directory_path`
        is not provided, the images are not saved. It also controls whether
        optimization progress is saved at each iteration.
        """
        # Configure the scene_optimizer
        scene_optimizer.set_parameters(self.optimization_parameters)

        for i in range(num_iters):
            # Show current state every `display_state_interval` iterations
            if (i + 1) % self._display_state_interval == 0:
                self.optimization_events.add_event(
                    iteration=i,
                    callback=lambda it=i: self.show_optimization_state(
                        iteration=it,
                        views_to_render=views_to_render,
                        save_directory_path=save_directory_path,
                    ),
                )
            # Save progress
            self.optimization_events.add_event(
                iteration=i,
                callback=lambda it=i: self.save_callback(
                    iteration=it, output_directory=save_directory_path
                ),
            )

        # Call the main optimization loop with parameter_optimization_configs
        scene_optimizer.optimize(
            scene=self.scene,
            scene_params=self.scene_params,
            param_optimization_configs=self.parameter_optimization_configs,
            ref_images=ref_images,
            num_iters=num_iters,
            optimization_events=self.optimization_events,
        )

    @property
    @abc.abstractmethod
    def optimization_parameters(self) -> dict[str, Any]:
        """
        A dictionary of parameters that are used to optimize the scene.

        The keys of the dictionary are unique parameter keys that are used to track
        them throughout the optimization process. Values are the initial parameter
        values that are passed to the `SceneOptimizer` for configuration at the start
        of the optimization or when the optimizer is reset.

        :returns: A dictionary where keys are unique parameter identifiers and values
                  are the initial parameter values to be optimized.
        :raises NotImplementedError: if the method was not implemented by a child class.
        """
        raise NotImplementedError(
            "Classes that inherit from `PropertyOptimizer` need to implement the "
            "`_optimization_parameters` method."
        )

    @property
    @abc.abstractmethod
    def parameter_optimization_configs(
        self,
    ) -> list[SceneOptimizer.ParamOptimizationConfig]:
        """
        Returns a list of `SceneOptimizer.ParamOptimizationConfig` objects that
        define the parameters to be optimized, their update callbacks, image loss,
        and scheduler.

        Each concrete implementation of `PropertyOptimizer` is expected to define
        its own set of parameters that are being optimized.

        :returns: A list of `SceneOptimizer.ParamOptimizationConfig` objects that
                    define how scene properties are optimized
        :raises NotImplementedError: if the method was not implemented by a child class.
        """
        raise NotImplementedError(
            "Classes that inherit from `PropertyOptimizer` need to implement the "
            "`_get_parameter_optimization_configs` method."
        )

    @abc.abstractmethod
    def save_callback(
        self, iteration: int, output_directory: Path | None = None
    ) -> None:
        """
        Saves the current state of the scene to a file.

        The `iteration` is used to generate unique filenames for all saves from that
        iteration. If `output_directory` is provided, the file is saved there. If
        not, it is not saved.

        If the save files are large, it is simple to save every N-th iteration by
        checking if `(iteration + 1) % N == 0` inside of this method.

        Write everything under `output_directory`. The entry-point script uploads that
        one directory to the experiment tracker when the run ends, so a file written
        anywhere else is saved to disk but never recorded as an artifact.

        :raises NotImplementedError: if the method was not implemented by a child class.
        """
        raise NotImplementedError(
            "Classes that inherit from `PropertyOptimizer` need to implement the "
            "`_save_callback` method."
        )

    @property
    def scene(self) -> "mi.Scene":
        """
        The Mitsuba scene object that is being optimized.

        A scene must be unique for each optimization task/script.
        """
        return self._scene

    @property
    def scene_params(self) -> "mi.SceneParams":
        """
        The Mitsuba scene parameters object that is being optimized.

        It is important to have a unique instance of scene parameters for all
        optimizers in each optimization task/script, as otherwise it is possible for
        changes in one optimizer to not correctly propagate to another optimizer that
        uses the same scene.
        """
        return self._scene_params

    @property
    def optimization_events(self) -> OptimizationEvents:
        """
        The `OptimizationEvents` object that is used to store per iteration callbacks
        to perform specific actions.

        This property enables adding events even after the `PropertyOptimizer` instance
        has been initialized.
        """
        return self._optimization_events

    def show_comparison_with_reference(
        self,
        ref_images: list[MaskedPoseImage],
        samples_per_pixel: int = 256,
        save_path: Path | None = None,
    ) -> None:
        """
        Show a comparison between current scene and reference.

        Current `scene_params`are used to render images with `samples_per_pixel`
        samples for all views in `sensors` of `ref_images` and compare them with
        `ref_images`.
        If a `save_path` is provided the figure is saved.

        :raises ValueError: If the `ref_images` or `sensor` lists are empty,
            or if the `ref_images` and `sensors` lists do not have the same length.
        """
        if len(ref_images) == 0:
            raise ValueError("At least one image is required in 'ref_images' list")

        rendered_views: list[mi.Bitmap] = []
        for image in ref_images:
            rendered_image = mi.render(
                scene=self.scene,
                params=self.scene_params,
                sensor=image.sensor,
                spp=samples_per_pixel,
            )
            rendered_views.append(mi.util.convert_to_bitmap(rendered_image))

        show_comparison_with_reference(
            rendered_views=rendered_views,
            reference_views=[
                mi.util.convert_to_bitmap(image.original_image) for image in ref_images
            ],
            save_path=save_path,
        )

    def show_optimization_state(
        self,
        iteration: int,
        views_to_render: list["mi.Sensor"] | None = None,
        save_directory_path: Path | None = None,
    ) -> None:
        """
        Renders the current state of the scene for `views_to_render` and
        saves it to `save_directory_path` if provided.

        `iteration` is the current iteration of the optimization
        and is used to generate a unique filename and seed for rendering.

        `views_to_render` is a list of sensors that may or may not be one of the
        `sensors` used for optimization. It is often the case that the validation
        views are used to render the current state of the scene. By default, this
        method is called in `SceneOptimizer.optimize` at each iteration.

        If `views_to_render` is `None`, no images are rendered. If the
        `save_directory_path` is not provided, the images are not saved.

        :raises ValueError: if `views_to_render` is an empty list, or if
            `iteration` is negative
        """
        if iteration < 0:
            raise ValueError("'iteration` must be a positive integer.")

        if views_to_render is not None and len(views_to_render) == 0:
            raise ValueError(
                "`views_to_render` list cannot be empty, pass `None` if "
                "random choice of optimization views is desired"
            )

        if views_to_render is None:
            return

        images: list["mi.TensorXf"] = []

        for sensor in views_to_render:
            images.append(
                mi.render(
                    scene=self.scene,
                    params=self.scene_params,
                    sensor=sensor,
                    seed=iteration,
                )
            )

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = (
            None
            if save_directory_path is None
            else save_directory_path.joinpath(
                f"views_{timestamp}_iteration-{iteration:03}.png"
            )
        )

        show_rendered_views(
            views=[mi.util.convert_to_bitmap(image) for image in images],
            save_path=save_path,
            titles=[f"View {i + 1}" for i in range(len(images))],
        )
