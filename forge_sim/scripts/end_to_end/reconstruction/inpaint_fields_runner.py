import tempfile
from pathlib import Path

import drjit as dr
import jsonpickle
import mitsuba as mi

from forge_sim import logger
from forge_sim.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)
from forge_sim.meshing.tessellated_spline_surface import TessellatedSplineSurface
from forge_sim.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from forge_sim.plugin_wrappers.prb_integrator import GuidingOptions, PRBIntegrator
from forge_sim.property_optimizers.field_texture_optimizer import (
    FieldTextureOptimizer,
)
from forge_sim.scene_optimizers.optimization_events import OptimizationEvents
from forge_sim.scripts.configs.texture_optimization_config import (
    TextureOptimizationConfig,
)
from forge_sim.scripts.end_to_end.dataset.dataset_modality import DatasetModality
from forge_sim.scripts.end_to_end.dataset.dataset_runner import (
    DatasetGenerationRunner,
)
from forge_sim.scripts.end_to_end.end_to_end_config import EndToEndConfig
from forge_sim.scripts.end_to_end.reconstruction.inpainted_field import InpaintedField


class FieldInpaintingRunner:
    """
    Inpaints the temperature and material fields onto the recovered spline, one
    field at a time, through the same image-space model as `inpaint_data_spline`.

    Each field is recovered by the production image-space inpainting model: the
    surface is rendered as an area emitter over a black background and per-control-
    point field values are optimised so the renders match the measured single-channel
    images. Temperature uses the thermal images; the binary material uses the
    windows-mask images (steel rendered as 1, wood as 0; the 0.5 initial value is the
    "gray background" of undefined regions).
    """

    def __init__(self, config: EndToEndConfig) -> None:
        """Stores the `config` every field this runner inpaints is driven by."""
        self._config = config

    def inpaint(self, kind: InpaintedField) -> None:
        """Inpaint one field onto the recovered spline, save it beside its renders."""
        is_temperature = kind is InpaintedField.TEMPERATURE
        modality = (
            DatasetModality.THERMAL if is_temperature else DatasetModality.MATERIAL
        )
        out_dir = (
            self._config.temperature_dir
            if is_temperature
            else self._config.material_dir
        )
        dest = (
            self._config.inpainted_temperature_spline_path
            if is_temperature
            else self._config.inpainted_material_spline_path
        )
        texture_params = self._texture_params()

        references = DatasetGenerationRunner.load_synthetic_references(
            self._config, modality
        )
        train_images = references.train_images
        eval_images = references.eval_images

        tessellated = TessellatedSplineSurface.from_file(self._config.spline_path)
        initial_field_values = [
            self._config.thermal.inpaint_initial_color
            * dr.ones(dtype=mi.TensorXf, shape=(spline.control_points.shape[0], 1))
            for spline in tessellated.composite_spline
        ]
        spline = TessellatedIGASplineSurface.from_tessellated_spline_surface(
            tessellated_spline_surface=tessellated,
            per_patch_field_values=initial_field_values,
        )
        spline.composite_spline.h_refinement(
            factor=self._config.thermal.inpaint_h_refinement
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            scene = self._build_emitter_scene(spline, Path(temp_dir))

            optimizer = FieldTextureOptimizer(
                scene=scene,
                scene_params=mi.traverse(scene),
                optimization_events=OptimizationEvents(),
                property_key="emitter.radiance.data",
                tessellated_iga_spline=spline,
                texture_scheduler=texture_params.scheduler(
                    end_iteration=self._config.thermal.inpaint_iterations
                ),
                texture_loss=texture_params.image_loss,
            )
            logger.info(
                f"[step 3/4] Inpainting {kind.value} ({len(train_images)} views) -> "
                f"{out_dir}"
            )
            optimizer.optimize(
                ref_images=train_images,
                scene_optimizer=texture_params.scene_optimizer,
                num_iters=self._config.thermal.inpaint_iterations,
                views_to_render=[img.sensor for img in eval_images],
                save_directory_path=out_dir,
            )
            optimizer.show_comparison_with_reference(
                ref_images=eval_images,
                samples_per_pixel=self._config.thermal.inpaint_continuous_samples * 4,
                save_path=out_dir / "Final_comparison.png",
            )

        dest.write_text(jsonpickle.encode(spline))
        logger.info(f"[step 3/4] {kind.value} field saved -> {dest.name}")

    def _texture_params(self) -> TextureOptimizationConfig:
        """The inpainting optimiser settings: constant LR, plain Adam, pure L1 loss."""
        lr = self._config.thermal.inpaint_learning_rate
        return TextureOptimizationConfig(
            texture_resolution=self._config.thermal.inpaint_field_texture_resolution,
            learning_rate_start=lr,
            learning_rate_end=lr,
            learning_rate_steepness=4.0,
            use_lbfgs=False,
            uniform_adam=False,
            batch_size=6,
            use_msssim_loss=False,
            l1_loss_weight=1.0,
        )

    def _build_emitter_scene(
        self, spline: TessellatedIGASplineSurface, temp_dir: Path
    ) -> "mi.Scene":
        """
        A scene where each patch is a black-BSDF area emitter reading its field
        texture (the inpainting "glow" model).
        """
        integrator = PRBIntegrator(
            max_bounces=2,
            bounces_without_russian_roulette=2,
            spp_continuous=self._config.thermal.inpaint_continuous_samples,
            spp_projective=1,
            spp_indirect=1,
            guiding=GuidingOptions.NONE,
        )
        scene_dict = {"type": "scene", "integrator": integrator.mitsuba_dict}
        initial_color = self._config.thermal.inpaint_initial_color
        for mesh in spline:
            texture_path = temp_dir / f"initial_{mesh.name}.exr"
            FieldTextureOptimizer.generate_empty_texture(
                base_color=mi.Color3f(initial_color, initial_color, initial_color),
                texture_output_path=texture_path,
                width=self._config.thermal.inpaint_field_texture_resolution,
                height=self._config.thermal.inpaint_field_texture_resolution,
            )
            mesh.save_to_ply(output_dir=temp_dir, save_name=mesh.name)
            scene_dict[mesh.name] = {
                "type": "ply",
                "filename": str(temp_dir / f"{mesh.name}.ply"),
                "bsdf": DiffuseBSDF(
                    reflectance=DiffuseBSDF.Albedo(albedo=0.0)
                ).mitsuba_dict,
                "emitter": {
                    "type": "area",
                    "radiance": {"type": "bitmap", "filename": str(texture_path)},
                },
            }
        return mi.load_dict(scene_dict)
