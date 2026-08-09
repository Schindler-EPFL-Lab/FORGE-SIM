from pathlib import Path
from typing import Any

import mitsuba as mi

from rebel_shape.loss_evaluators.image_loss import ImageLoss
from rebel_shape.loss_evaluators.image_loss_l2 import ImageLossL2
from rebel_shape.meshing.regular_composite_mesh import RegularCompositeMesh
from rebel_shape.property_optimizers.bitmap_texture_optimizer import (
    BitmapTextureOptimizer,
)
from rebel_shape.property_optimizers.property_optimizer import PropertyOptimizer
from rebel_shape.property_optimizers.regular_composite_mesh_optimizer import (
    RegularCompositeMeshOptimizer,
)
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.scene_optimizers.scene_optimizer import SceneOptimizer
from rebel_shape.schedulers.optimization_scheduler import OptimizationScheduler


class RegularCompositeMeshTextureOptimizer(PropertyOptimizer):
    """
    Joint shape + per-patch texture optimizer for a `RegularCompositeMesh`.

    Mesh-side analog of `SplineTextureOptimizer`
    (`rebel_shape/property_optimizers/spline_texture_optimizer.py`). Wraps a
    `RegularCompositeMeshOptimizer` for the shape and a `BitmapTextureOptimizer`
    bound to one texture key per patch — exactly the per-patch texture pattern
    used on the spline side at
    `rebel_shape/property_optimizers/spline_texture_optimizer.py:115-119` — so
    each cube face has its own independent texture map and no UV seams cross
    between patches.

    No regularization parameter is exposed (user request).
    """

    def __init__(
        self,
        scene: "mi.Scene",
        scene_params: "mi.SceneParams",
        optimization_events: OptimizationEvents,
        composite_mesh: RegularCompositeMesh,
        shape_scheduler: OptimizationScheduler,
        texture_scheduler: OptimizationScheduler,
        shape_loss: ImageLoss = ImageLossL2(),
        texture_loss: ImageLoss = ImageLossL2(),
        parametrization_smoothing: float = 4.0,
        texture_projection_steepness: float = 1.0,
        texture_save_interval: int = 5,
        flip_along_y: bool = True,
        mesh_save_interval: int | None = 10,
    ) -> None:
        """
        `composite_mesh` is the `RegularCompositeMesh` whose vertices and per-patch
        textures are optimized jointly. Each patch's
        ``"<name>.bsdf.reflectance.data"`` must already be a `mi.TensorXf` in
        `scene_params`, and the textures across patches must share the same
        resolution.

        `shape_scheduler`/`texture_scheduler` and `shape_loss`/`texture_loss`
        control the two sub-optimizations independently.
        `parametrization_smoothing` is the Laplacian smoothing applied to vertex
        gradients (see `RegularCompositeMeshOptimizer`).
        `texture_projection_steepness`, `texture_save_interval`, and
        `flip_along_y` are forwarded verbatim to `BitmapTextureOptimizer`.
        `mesh_save_interval` controls per-patch PLY save cadence on the shape
        side.

        :raises ValueError: Any of the validation conditions in the wrapped
            sub-optimizers (missing scene params, non-tensor textures, mismatched
            resolutions, non-positive intervals, etc.).
        """
        super().__init__(
            scene=scene,
            scene_params=scene_params,
            optimization_events=optimization_events,
        )

        self._composite_mesh = composite_mesh

        self._shape_optimizer = RegularCompositeMeshOptimizer(
            scene=self._scene,
            scene_params=self._scene_params,
            optimization_events=self._optimization_events,
            composite_mesh=composite_mesh,
            shape_scheduler=shape_scheduler,
            shape_loss=shape_loss,
            smoothing_strength=parametrization_smoothing,
            mesh_save_interval=mesh_save_interval,
        )

        self._texture_optimizer = BitmapTextureOptimizer(
            scene=self._scene,
            scene_params=self._scene_params,
            optimization_events=self._optimization_events,
            optimization_key=composite_mesh.name + "_texture",
            texture_keys=[
                f"{mesh.name}.bsdf.reflectance.data" for mesh in composite_mesh
            ],
            texture_scheduler=texture_scheduler,
            texture_loss=texture_loss,
            projection_steepness=texture_projection_steepness,
            texture_save_interval=texture_save_interval,
            flip_along_y=flip_along_y,
        )

    def h_refinement(self, factor: int, scene_optimizer: SceneOptimizer) -> None:
        """
        Delegates to the shape optimizer's `h_refinement`. Textures are
        per-patch and live in the local `[0, 1]^2` UV frame, so they remain
        valid after grid refinement — no texture update is required.

        :raises ValueError: If `factor` is not a positive integer.
        """
        self._shape_optimizer.h_refinement(
            factor=factor, scene_optimizer=scene_optimizer
        )

    @property
    def optimization_parameters(self) -> dict[str, Any]:
        """Union of the shape and texture sub-optimizer parameters."""
        shape_params = self._shape_optimizer.optimization_parameters
        texture_params = self._texture_optimizer.optimization_parameters
        return shape_params | texture_params

    @property
    def parameter_optimization_configs(
        self,
    ) -> list[SceneOptimizer.ParamOptimizationConfig]:
        """One config from the shape optimizer and one from the texture optimizer."""
        return (
            self._shape_optimizer.parameter_optimization_configs
            + self._texture_optimizer.parameter_optimization_configs
        )

    @property
    def composite_mesh(self) -> RegularCompositeMesh:
        """The composite mesh being optimized."""
        return self._composite_mesh

    def save_callback(
        self, iteration: int, output_directory: Path | None = None
    ) -> None:
        """Forwards to both sub-optimizers."""
        self._shape_optimizer.save_callback(
            iteration=iteration, output_directory=output_directory
        )
        self._texture_optimizer.save_callback(
            iteration=iteration, output_directory=output_directory
        )
