"""
Step 2b: Spline shape+texture reconstruction from the synthetic dataset.

Reconstructs a 6-patch cubed-sphere B-spline surface from the on-disk dataset
with fixed, known camera poses (no pose optimisation, poses are exact for
synthetic data). Mirrors the standard spline-reconstruction protocol
(`optimize_thermoscenes_spline` / `optimize_suzanne_spline_texture`): joint
shape (L-BFGS) + texture (Adam) optimisation, G1 + curvature regularisation, and
coarse-to-fine h-refinement.

Deliverable 2: the reconstructed surface JSON, the 6 per-patch texture EXRs, and
final-view renders.

Run:  python -m forge_sim.scripts.end_to_end.reconstruction.reconstruct_surface
"""

import time

import drjit as dr
import mitsuba as mi
import tyro

from forge_sim import logger
from forge_sim.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from forge_sim.plugin_wrappers.prb_integrator import GuidingOptions, PRBIntegrator
from forge_sim.property_optimizers.spline_texture_optimizer import (
    SplineTextureOptimizer,
)
from forge_sim.regularization.composite_regularization import CompositeRegularization
from forge_sim.regularization.spline_curvature_loss import SplineCurvatureLoss
from forge_sim.regularization.spline_g1_loss import SplineG1Loss
from forge_sim.scene_optimizers.composite_scene_optimizer import (
    CompositeSceneOptimizer,
)
from forge_sim.scene_optimizers.optimization_events import OptimizationEvents
from forge_sim.scripts.configs.spline_shape_optimization_config import (
    SplineShapeOptimizationConfig,
)
from forge_sim.scripts.configs.texture_optimization_config import (
    TextureOptimizationConfig,
)
from forge_sim.scripts.end_to_end.dataset.dataset_modality import DatasetModality
from forge_sim.scripts.end_to_end.dataset.dataset_runner import (
    DatasetGenerationRunner,
)
from forge_sim.scripts.end_to_end.end_to_end_config import EndToEndConfig
from forge_sim.scripts.end_to_end.reconstruction.reconstruct_surface_runner import (
    ReconstructSurfaceRunner,
)
from forge_sim.splines.cube_initialization import cubed_sphere


def _shape_config(config: EndToEndConfig) -> SplineShapeOptimizationConfig:
    """Shape-optimisation settings: L-BFGS with coarse-to-fine h-refinement."""
    rc = config.reconstruction
    return SplineShapeOptimizationConfig(
        smoothing_strength=0.0,
        learning_rate_start=1.0,
        learning_rate_end=1.0,
        learning_rate_steepness=1.0,
        use_lbfgs=True,
        uniform_adam=True,
        batch_size=None,
        lbfgs_history_size=10,
        h_refinement_iterations=rc.h_refinement_iterations,
        degree=rc.degree,
        knot_spans=rc.knot_spans,
        tessellation_subdivisions=rc.tessellation_subdivisions,
    )


def _texture_config(config: EndToEndConfig) -> TextureOptimizationConfig:
    """Texture-optimisation settings: Adam on small batches, ramping LR."""
    return TextureOptimizationConfig(
        texture_resolution=config.reconstruction.texture_resolution,
        learning_rate_start=0.03,
        learning_rate_end=0.12,
        learning_rate_steepness=2.0,
        use_lbfgs=False,
        uniform_adam=False,
        batch_size=2,
    )


def reconstruct(config: EndToEndConfig) -> None:
    """Optimise a cubed-sphere spline against the dataset and collect the output."""
    mi.set_variant("cuda_ad_rgb" if config.use_gpu else "llvm_ad_rgb")
    rc = config.reconstruction
    shape_params = _shape_config(config)
    texture_params = _texture_config(config)

    # Use the deterministic cameras that rendered the dataset (object at origin,
    # consistent with the cubed-sphere init) instead of the re-framing loaders.
    references = DatasetGenerationRunner.load_synthetic_references(
        config, DatasetModality.RGB
    )
    ref_images, eval_images = references.train_images, references.eval_images
    logger.info(f"[step 2b] Loaded {len(ref_images)} train / {len(eval_images)} eval")

    integrator = PRBIntegrator(
        max_bounces=4,
        bounces_without_russian_roulette=4,
        spp_continuous=rc.continuous_samples,
        spp_projective=rc.visibility_samples,
        spp_indirect=rc.visibility_samples,
        guiding=GuidingOptions.NONE,
    )

    cubedsphere = cubed_sphere(
        name=config.name,
        degree=rc.degree,
        spline_subdivisions=rc.knot_spans,
        tessellation_subdivisions=rc.tessellation_subdivisions,
        bsdf=DiffuseBSDF(
            DiffuseBSDF.TextureFromTensor(
                tensor=dr.full(
                    dtype=mi.TensorXf,
                    value=0.5,
                    shape=(rc.texture_resolution, rc.texture_resolution, 3),
                )
            )
        ),
    )

    scene_dict = {
        "type": "scene",
        "integrator": integrator.mitsuba_dict,
        "emitter": {"type": "envmap", "filename": str(config.dataset.envmap)},
    }
    for mesh in reversed(cubedsphere):
        scene_dict[mesh.name] = mesh.mitsuba_dict
    scene = mi.load_dict(scene_dict)

    shape_key = cubedsphere.mesh_name
    texture_key = cubedsphere.mesh_name + "_texture"

    shape_regularization = CompositeRegularization(
        regularization_terms=[
            SplineG1Loss(composite_spline=cubedsphere, weight=1.0),
            SplineCurvatureLoss(tessellated_spline=cubedsphere, weight=2e-3),
        ],
        composite_weight=1.0e-2,
    )

    optimizer = SplineTextureOptimizer(
        scene=scene,
        scene_params=mi.traverse(scene),
        optimization_events=OptimizationEvents(),
        tessellated_spline=cubedsphere,
        shape_scheduler=shape_params.scheduler(end_iteration=rc.iterations),
        texture_scheduler=texture_params.scheduler(end_iteration=rc.iterations),
        shape_loss=shape_params.image_loss,
        texture_loss=texture_params.image_loss,
        shape_regularization=shape_regularization,
        parametrization_smoothing=shape_params.smoothing_strength,
    )

    composite_scene_optimizer = CompositeSceneOptimizer(
        optimizer_config=[
            (shape_params.scene_optimizer, [shape_key]),
            (texture_params.scene_optimizer, [texture_key]),
        ],
    )

    for iteration in rc.h_refinement_iterations:
        optimizer.optimization_events.add_event(
            iteration=iteration,
            callback=lambda: optimizer.h_refinement(
                factor=2, scene_optimizer=composite_scene_optimizer
            ),
        )

    views = [eval_images[i].sensor for i in range(min(len(eval_images), 5))]
    start = time.time()
    optimizer.optimize(
        ref_images=ref_images,
        scene_optimizer=composite_scene_optimizer,
        num_iters=rc.iterations,
        views_to_render=views,
        save_directory_path=config.reconstruction_dir,
    )
    logger.info(f"[step 2b] Optimisation done in {time.time() - start:.1f}s")

    optimizer.show_comparison_with_reference(
        ref_images=eval_images,
        samples_per_pixel=rc.continuous_samples * 4,
        save_path=config.reconstruction_dir / "final_comparison.png",
    )
    ReconstructSurfaceRunner(config, started_at=start).collect_outputs()


if __name__ == "__main__":
    reconstruct(tyro.cli(EndToEndConfig))
