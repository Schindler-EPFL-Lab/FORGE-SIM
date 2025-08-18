import tempfile
import unittest
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np

from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from rebel_shape.plugin_wrappers.prb_integrator import GuidingOptions, PRBIntegrator
from rebel_shape.property_optimizers.bitmap_texture_optimizer import (
    BitmapTextureOptimizer,
)
from rebel_shape.utils.scene_params_utils import ChangeRevertSceneParameters


class TestChangeRevertSceneParameters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mi.set_variant("llvm_ad_rgb")

    def setUp(self):
        super().setUp()
        local_dir = Path(__file__).parent.resolve()
        test_envmap_path = local_dir / "data/textures/overcast_soil_1k.exr"
        shape_path = local_dir / "data/meshes/test_cube.ply"

        base_samples = 4
        integrator = PRBIntegrator(
            max_bounces=4,
            bounces_without_russian_roulette=4,
            spp_continuous=base_samples,
            spp_projective=base_samples // 4,
            spp_indirect=base_samples // 4,
            guiding=GuidingOptions.NONE,
        )

        self.image_texture_shape = (256, 256, 3)

        with tempfile.TemporaryDirectory() as temp_dir:
            initial_texture_path = Path(temp_dir) / "initial_texture.exr"
            BitmapTextureOptimizer.generate_empty_texture(
                base_color=mi.Color3f(0.1, 0.2, 0.3),
                texture_output_path=initial_texture_path,
                width=self.image_texture_shape[0],
                height=self.image_texture_shape[1],
            )

            scene_dict = {
                "type": "scene",
                "integrator": integrator.mitsuba_dict,
                "emitter": {
                    "type": "envmap",
                    "filename": str(test_envmap_path),
                },
                "shape": {
                    "type": "ply",
                    "filename": str(shape_path),
                    "bsdf": DiffuseBSDF(
                        reflectance=DiffuseBSDF.Texture(
                            texture_path=initial_texture_path
                        )
                    ).mitsuba_dict,
                },
            }

            self.scene_source = mi.load_dict(scene_dict)
            self.scene_params = mi.traverse(self.scene_source)

        self.original_image_texture = mi.TensorXf(
            np.full(
                shape=self.image_texture_shape,
                fill_value=np.array([0.1, 0.2, 0.3]),
            )
        )

        self.original_envmap_texture = dr.copy(self.scene_params["emitter.data"])

    def test_one_change(self):
        change_revert_scene_params = ChangeRevertSceneParameters(self.scene_params)

        change_revert_scene_params.change({"shape.bsdf.reflectance.data": 1.0})

        self.assertTrue(
            dr.allclose(
                self.scene_params["shape.bsdf.reflectance.data"],
                mi.TensorXf(np.ones(self.image_texture_shape, dtype=np.float32)),
            )
        )

        change_revert_scene_params.revert()

        self.assertTrue(
            dr.allclose(
                self.scene_params["shape.bsdf.reflectance.data"],
                self.original_image_texture,
            )
        )

    def test_one_change_does_not_change_others(self):
        change_revert_scene_params = ChangeRevertSceneParameters(self.scene_params)

        change_revert_scene_params.change({"shape.bsdf.reflectance.data": 1.0})

        self.assertTrue(
            dr.allclose(
                self.scene_params["emitter.data"],
                self.original_envmap_texture,
            )
        )

        change_revert_scene_params.revert()

        self.assertTrue(
            dr.allclose(
                self.scene_params["emitter.data"],
                self.original_envmap_texture,
            )
        )

    def test_invalid_multiple_changes(self):
        change_revert_scene_params = ChangeRevertSceneParameters(self.scene_params)

        change_revert_scene_params.change({"shape.bsdf.reflectance.data": 1.0})

        with self.assertRaises(RuntimeError):
            change_revert_scene_params.change({"shape.bsdf.reflectance.data": 1.0})

    def test_multiple_changes(self):
        change_revert_scene_params = ChangeRevertSceneParameters(self.scene_params)

        change_revert_scene_params.change({"shape.bsdf.reflectance.data": 0.7})

        self.assertTrue(
            dr.allclose(
                self.scene_params["shape.bsdf.reflectance.data"],
                mi.TensorXf(
                    np.full(self.image_texture_shape, fill_value=0.7, dtype=np.float32)
                ),
            )
        )

        change_revert_scene_params.change({"emitter.data": 0.5})

        self.assertTrue(
            dr.allclose(
                self.scene_params["emitter.data"],
                mi.TensorXf(
                    np.full(
                        self.original_envmap_texture.shape,
                        fill_value=0.5,
                        dtype=np.float32,
                    )
                ),
            )
        )

        change_revert_scene_params.revert()

        self.assertTrue(
            dr.allclose(
                self.scene_params["shape.bsdf.reflectance.data"],
                self.original_image_texture,
            )
        )

        self.assertTrue(
            dr.allclose(
                self.scene_params["shape.bsdf.reflectance.data"],
                self.original_image_texture,
            )
        )
