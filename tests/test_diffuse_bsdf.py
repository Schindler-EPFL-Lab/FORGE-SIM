import unittest
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np

from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF


class TestDiffuseBSDF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up a mi_variant for testing."""
        mi.set_variant("llvm_ad_rgb")

    def setUp(self):
        """Set up test resources."""
        local_dir = Path(__file__).parent.resolve()
        self.test_texture_path = local_dir / "data/textures/overcast_soil_1k.exr"

    def test_default_initialization(self):
        """Test creating DiffuseBSDF with default parameters."""
        bsdf = DiffuseBSDF()
        self.assertIsInstance(bsdf, DiffuseBSDF)

        expected_dict = {
            "type": "diffuse",
            "reflectance": {"type": "rgb", "value": [0.5, 0.5, 0.5]},
        }
        self.assertEqual(bsdf.mitsuba_dict, expected_dict)

    def test_albedo_reflectance(self):
        """Test DiffuseBSDF with custom Albedo."""
        bsdf = DiffuseBSDF(reflectance=DiffuseBSDF.Albedo(albedo=0.7))

        expected_dict = {
            "type": "diffuse",
            "reflectance": {"type": "rgb", "value": [0.7, 0.7, 0.7]},
        }
        self.assertEqual(bsdf.mitsuba_dict, expected_dict)

    def test_albedo_bounds(self):
        """Test that invalid albedo values raise ValueError."""
        with self.assertRaises(ValueError) as context:
            DiffuseBSDF.Albedo(albedo=-0.1)
        self.assertIn("[0.0, 1.0]", str(context.exception))

        with self.assertRaises(ValueError) as context:
            DiffuseBSDF.Albedo(albedo=1.1)
        self.assertIn("[0.0, 1.0]", str(context.exception))

    def test_solid_color_reflectance(self):
        """Test DiffuseBSDF with SolidColor reflectance."""
        color = DiffuseBSDF.SolidColor(red=0.8, green=0.3, blue=0.2)
        bsdf = DiffuseBSDF(reflectance=color)

        expected_dict = {
            "type": "diffuse",
            "reflectance": {"type": "rgb", "value": [0.8, 0.3, 0.2]},
        }
        self.assertEqual(bsdf.mitsuba_dict, expected_dict)

    def test_solid_color_bounds(self):
        """Test that invalid color values raise ValueError."""
        # Test invalid red
        with self.assertRaises(ValueError) as context:
            DiffuseBSDF.SolidColor(red=-0.1, green=0.5, blue=0.5)
        self.assertIn("[0.0, 1.0]", str(context.exception))

        # Test invalid green
        with self.assertRaises(ValueError) as context:
            DiffuseBSDF.SolidColor(red=0.5, green=1.1, blue=0.5)
        self.assertIn("[0.0, 1.0]", str(context.exception))

        # Test invalid blue
        with self.assertRaises(ValueError) as context:
            DiffuseBSDF.SolidColor(red=0.5, green=0.5, blue=1.1)
        self.assertIn("[0.0, 1.0]", str(context.exception))

    def test_texture_reflectance(self):
        """Test DiffuseBSDF with Texture reflectance."""
        texture = DiffuseBSDF.Texture(texture_path=self.test_texture_path)
        bsdf = DiffuseBSDF(reflectance=texture)

        expected_dict = {
            "type": "diffuse",
            "reflectance": {"type": "bitmap", "filename": str(self.test_texture_path)},
        }
        self.assertEqual(bsdf.mitsuba_dict, expected_dict)

    def test_texture_invalid_path(self):
        """Test that non-existent texture path raises ValueError."""
        invalid_path = Path("/nonexistent/texture.png")

        with self.assertRaises(ValueError) as context:
            DiffuseBSDF.Texture(texture_path=invalid_path)
        self.assertIn("does not exist", str(context.exception))

    def test_mesh_property_texture_reflectance(self):
        """Test DiffuseBSDF with Texture reflectance."""
        mesh_property_texture = DiffuseBSDF.MeshAttributeTexture(
            attribute_key="vertex_colors",
        )
        bsdf = DiffuseBSDF(reflectance=mesh_property_texture)

        expected_dict = {
            "type": "diffuse",
            "reflectance": {
                "type": "mesh_attribute",
                "name": "vertex_colors",
            },
        }
        self.assertEqual(bsdf.mitsuba_dict, expected_dict)

    def test_mesh_property_texture_invalid_key(self):
        """Test that invalid mesh property key raises ValueError."""
        with self.assertRaises(ValueError) as context:
            DiffuseBSDF.MeshAttributeTexture(attribute_key="invalid_key")
        self.assertIn("`vertex_` or `face_`", str(context.exception))

    def test_reflectance_classes_mitsuba_dict(self):
        """Test mitsuba_dict for all reflectance classes."""
        # Test Albedo class
        albedo = DiffuseBSDF.Albedo(albedo=0.3)
        self.assertEqual(albedo.mitsuba_dict, {"type": "rgb", "value": [0.3, 0.3, 0.3]})

        # Test SolidColor class
        color = DiffuseBSDF.SolidColor(red=0.1, green=0.2, blue=0.3)
        self.assertEqual(color.mitsuba_dict, {"type": "rgb", "value": [0.1, 0.2, 0.3]})

        # Test Texture class
        texture = DiffuseBSDF.Texture(texture_path=self.test_texture_path)
        self.assertEqual(
            texture.mitsuba_dict,
            {"type": "bitmap", "filename": str(self.test_texture_path)},
        )

        # Test MeshAttributeTexture class
        mesh_property_texture = DiffuseBSDF.MeshAttributeTexture(
            attribute_key="vertex_colors",
        )
        self.assertEqual(
            mesh_property_texture.mitsuba_dict,
            {"type": "mesh_attribute", "name": "vertex_colors"},
        )

    def test_texture_from_tensor_invalid_input(self):
        """Test TextureFromTensor raises ValueError for invalid input."""
        tensor_invalid_cases = [
            dr.full(dtype=mi.TensorXf, value=0.5, shape=(1, 128, 128, 3)),
            dr.full(dtype=mi.TensorXf, value=0.5, shape=(128, 128, 4)),
            dr.full(dtype=mi.TensorXf, value=-1.0, shape=(128, 128, 3)),
            dr.full(dtype=mi.TensorXf, value=2.0, shape=(128, 128, 3)),
        ]
        for i, tensor_invalid in enumerate(tensor_invalid_cases):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    _ = DiffuseBSDF.TextureFromTensor(tensor=tensor_invalid)

    def test_texture_from_tensor(self):
        """Test TextureFromTensor outputs correct Mitsuba dict for valid tensors."""
        np.random.seed(42)
        tensor_valid_cases = [
            mi.TensorXf(np.random.rand(103, 97, 3)) for _ in range(10)
        ]

        for i, tensor_valid in enumerate(tensor_valid_cases):
            with self.subTest(i=i):
                bsdf_mitsuba_dict = DiffuseBSDF.TextureFromTensor(
                    tensor=tensor_valid
                ).mitsuba_dict

                expected_keys = set(["type", "data"])
                self.assertEqual(set(bsdf_mitsuba_dict.keys()), expected_keys)
                self.assertEqual(bsdf_mitsuba_dict["type"], "bitmap")
                self.assertTrue(dr.allclose(bsdf_mitsuba_dict["data"], tensor_valid))
