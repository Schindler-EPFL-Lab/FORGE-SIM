from datetime import datetime
from pathlib import Path
from typing import Any

import drjit as dr
import jsonpickle
import mitsuba as mi

from rebel_shape.loss_evaluators.image_loss import ImageLoss
from rebel_shape.loss_evaluators.image_loss_l2 import ImageLossL2
from rebel_shape.meshing.regular_composite_mesh import RegularCompositeMesh
from rebel_shape.property_optimizers.laplacian_parametrization import (
    LaplacianParametrization,
)
from rebel_shape.property_optimizers.property_optimizer import PropertyOptimizer
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.scene_optimizers.scene_optimizer import SceneOptimizer
from rebel_shape.schedulers.optimization_scheduler import OptimizationScheduler


class RegularCompositeMeshOptimizer(PropertyOptimizer):
    """
    Shape-only optimizer for a `RegularCompositeMesh`.

    Mirrors `SplineOptimizer` (`rebel_shape/property_optimizers/spline_optimizer.py`)
    but operates directly on the composite mesh's global vertex buffer instead of
    spline control points. The `LaplacianParametrization` therefore lives in vertex
    space — no spline indirection — and seam vertices are shared by construction
    because the `RegularCompositeMesh`'s `PatchManager` deduplicates them at
    patch boundaries. This eliminates the gradient-tearing failure mode that the
    single-mesh `MeshOptimizer` suffers from in the presence of UV seams.

    No regularization is applied: the user explicitly requested it be omitted,
    relying on the topology-level seam averaging in `update_global_representation`
    plus the Laplacian smoothing of gradients to keep the surface coherent.

    `h_refinement(factor, scene_optimizer)` mirrors the spline's refinement: the
    underlying composite mesh is refined in place, the parametrization is
    rebuilt, and the scene optimizer is reset to the new parameter shape.
    """

    def __init__(
        self,
        scene: "mi.Scene",
        scene_params: "mi.SceneParams",
        optimization_events: OptimizationEvents,
        composite_mesh: RegularCompositeMesh,
        shape_scheduler: OptimizationScheduler,
        shape_loss: ImageLoss = ImageLossL2(),
        smoothing_strength: float = 4.0,
        mesh_save_interval: int | None = 10,
    ) -> None:
        """
        `scene` is a loaded Mitsuba scene, `scene_params` is its traversal, and
        `optimization_events` collects per-iteration callbacks.

        `composite_mesh` is the `RegularCompositeMesh` to optimize. Every patch's
        ``"<name>.vertex_positions"`` must be present in `scene_params`, otherwise
        the optimizer cannot push updated vertices to the scene.

        `shape_scheduler` controls the learning rate over iterations, and
        `shape_loss` is the per-image loss compared against reference renders.

        `smoothing_strength` is the Laplacian smoothing strength applied to the
        gradients; larger values diffuse gradients more aggressively (helps
        stability but blurs detail).

        `mesh_save_interval` controls how often each patch is dumped to PLY
        during optimization. If `None`, patches are not saved.

        :raises ValueError: If any patch's ``"<name>.vertex_positions"`` is not in
            `scene_params`, `smoothing_strength` is negative, or
            `mesh_save_interval` is non-positive.
        """
        super().__init__(
            scene=scene,
            scene_params=scene_params,
            optimization_events=optimization_events,
        )

        for mesh in composite_mesh:
            if mesh.name + ".vertex_positions" not in scene_params:
                raise ValueError(
                    f"Patch {mesh.name!r} is not present in the scene parameters.",
                )

        if smoothing_strength < 0.0:
            raise ValueError("`smoothing_strength` must be a non-negative real number.")
        if mesh_save_interval is not None and mesh_save_interval < 1:
            raise ValueError("`mesh_save_interval` must be a positive integer.")

        self._composite_mesh = composite_mesh
        self._shape_loss = shape_loss
        self._shape_scheduler = shape_scheduler
        self._smoothing = smoothing_strength
        self._mesh_save_interval = mesh_save_interval

        self._parametrization: LaplacianParametrization
        self._initialize_parametrization()

    def _initialize_parametrization(self) -> None:
        """
        Rebuilds the `LaplacianParametrization` from the composite mesh's current
        global vertices and faces. Called once in `__init__`, and again after
        smoothing-strength changes or h-refinement, which both invalidate the
        previous Laplacian.
        """
        self._parametrization = LaplacianParametrization(
            nodes=self._composite_mesh.global_vertices,
            faces=self._composite_mesh.global_faces,
            smoothing_strength=self._smoothing,
        )

    def update_smoothing_strength(
        self,
        new_smoothing_strength: float,
        scene_optimizer: SceneOptimizer,
    ) -> None:
        """
        Updates the Laplacian smoothing strength and resets the scene optimizer.

        Useful when gradients shrink as optimization progresses and stronger detail
        recovery is desired (typically by reducing the smoothing strength).

        :raises ValueError: If `new_smoothing_strength` is negative.
        """
        if new_smoothing_strength < 0.0:
            raise ValueError(
                "`new_smoothing_strength` must be a non-negative real number.",
            )

        self._smoothing = new_smoothing_strength
        self._initialize_parametrization()
        scene_optimizer.set_parameters(self.optimization_parameters)

    def h_refinement(self, factor: int, scene_optimizer: SceneOptimizer) -> None:
        """
        Refines the underlying composite mesh by `factor` along each grid direction
        and resets the parametrization and scene optimizer state.

        Mirrors `SplineOptimizer.h_refinement` at
        `rebel_shape/property_optimizers/spline_optimizer.py:163`. After the
        composite mesh is refined every patch carries a freshly built Mitsuba mesh
        at the new resolution; their `vertex_positions`, `faces`,
        `vertex_texcoords`, and `vertex_normals` are pushed into `scene_params`
        and `scene_params.update()` is called so the renderer picks them up.

        :raises ValueError: If `factor` is not a positive integer.
        """
        if factor < 1:
            raise ValueError("`factor` must be a positive integer.")

        self._composite_mesh.h_refinement(factor=factor)

        for mesh in self._composite_mesh:
            self._scene_params[mesh.name + ".vertex_positions"] = dr.ravel(
                mesh.vertices,
            )
            self._scene_params[mesh.name + ".faces"] = dr.ravel(mesh.faces)
            self._scene_params[mesh.name + ".vertex_texcoords"] = dr.ravel(
                mesh.texcoords,
            )
        self._scene_params.update()
        # Do NOT push vertex_normals back: `scene_params.update()` recomputes
        # them from positions+faces, and writing our (possibly stale) values
        # afterwards would freeze them so they stop tracking subsequent
        # shape deformation. Empirically verified on Mitsuba 3.

        self._initialize_parametrization()
        scene_optimizer.set_parameters(self.optimization_parameters)

    def _shape_update_callback(self, params: "mi.Float") -> None:
        """
        Closure passed to `SceneOptimizer` to apply a parametric-space update.

        Converts `params` back to global Cartesian vertex positions through the
        Laplacian parametrization, propagates them to every patch via the
        composite mesh's `global_vertices` setter, and pushes the new positions
        through `scene_params`. `scene_params.update()` triggers Mitsuba's own
        vertex-normal recomputation from the new positions; we deliberately do
        NOT push vertex_normals afterwards because doing so would freeze them
        and stop them from tracking further shape deformation.
        """
        new_global_vertices = dr.reshape(
            dtype=mi.TensorXf,
            value=self._parametrization.from_parametric(params),
            shape=(-1, 3),
        )
        self._composite_mesh.global_vertices = new_global_vertices

        for mesh in self._composite_mesh:
            self._scene_params[mesh.name + ".vertex_positions"] = dr.ravel(
                mesh.vertices,
            )
        self._scene_params.update()

    @property
    def optimization_parameters(self) -> dict[str, Any]:
        """
        A single-entry dictionary mapping the composite mesh name to its current
        global vertex positions in the Laplacian-parametric space.
        """
        return {
            self._composite_mesh.name: self._parametrization.to_parametric(
                dr.ravel(self._composite_mesh.global_vertices),
            ),
        }

    @property
    def parameter_optimization_configs(
        self,
    ) -> list[SceneOptimizer.ParamOptimizationConfig]:
        """
        A single `ParamOptimizationConfig` describing how the composite mesh's
        vertices are optimized. No regularization term is attached, by user
        request.
        """
        shape_param_config = SceneOptimizer.ParamOptimizationConfig(
            parameter_key=self._composite_mesh.name,
            scene_update_callback=self._shape_update_callback,
            image_loss=self._shape_loss,
            scheduler=self._shape_scheduler,
            regularization=None,
        )
        return [shape_param_config]

    @property
    def composite_mesh(self) -> RegularCompositeMesh:
        """The composite mesh being optimized."""
        return self._composite_mesh

    def save_callback(
        self,
        iteration: int,
        output_directory: Path | None = None,
    ) -> None:
        """
        Saves the composite mesh every `mesh_save_interval` iterations.

        The full composite is written as a single jsonpickle ``.json`` (the
        mesh-side analog of the spline path's `TessellatedSplineSurface` JSON, so
        the topology can be reloaded with `RegularCompositeMesh.from_file` without
        re-stitching the per-patch PLYs), and each patch is additionally written to
        its own PLY.

        Does nothing if `output_directory` is `None` or `mesh_save_interval` is
        `None`.
        """
        if output_directory is None:
            return
        if self._mesh_save_interval is None:
            return
        if (iteration + 1) % self._mesh_save_interval != 0:
            return

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        composite_name = (
            f"{self._composite_mesh.name}_{timestamp}_iteration-{iteration:03}.json"
        )
        with open(output_directory / composite_name, "w") as composite_file:
            composite_file.write(jsonpickle.encode(self._composite_mesh))

        for mesh in self._composite_mesh:
            mesh_name = f"{mesh.name}_{timestamp}_iteration-{iteration:03}"
            mesh.save_to_ply(output_dir=output_directory, save_name=mesh_name)
