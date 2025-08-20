import tempfile
import unittest
from json import JSONDecodeError
from pathlib import Path

import jsonpickle
import mitsuba as mi
import numpy as np
from PIL import Image

from rebel_shape.meshing.composite_spline_surface_factory import (
    CompositeSplineSurfaceFactory,
)
from rebel_shape.meshing.patch import Patch
from rebel_shape.meshing.tessellated_spline_surface import TessellatedSplineSurface
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from rebel_shape.splines.bspline_patch import BSplinePatch


class TestTessellatedSplineSurface(unittest.TestCase):
    """Unit tests for TessellatedSplineSurface class."""

    def setUp(self):
        """Set up a simple tessellated spline surface for testing."""
        mi.set_variant("llvm_ad_rgb")
        # Use BSplinePatch for testing, not mocks, too complex to mock properly
        # Unit square with 1 knot span in each direction = Bezier patch
        self.patch1 = BSplinePatch.unit_square(
            patch_name="patch1", degree=2, u_subdivisions=1, v_subdivisions=1
        )
        self.patch1.control_points = mi.TensorXf(
            self.patch1.control_points.numpy() + np.array([0.5, 0.5, 0.0])
        )

        self.patch2 = BSplinePatch.unit_square(
            patch_name="patch2", degree=2, u_subdivisions=1, v_subdivisions=1
        )
        self.patch2.control_points = mi.TensorXf(
            self.patch2.control_points.numpy() + np.array([1.5, 0.5, 0.0])
        )

        self.patch3 = BSplinePatch.unit_square(
            patch_name="patch3", degree=2, u_subdivisions=1, v_subdivisions=1
        )
        self.patch3.control_points = mi.TensorXf(
            self.patch3.control_points.numpy() + np.array([1.5, 1.5, 0.0])
        )

        self.composite_factory = CompositeSplineSurfaceFactory("test_composite")
        self.composite_factory.add_patch(self.patch1)
        self.composite_factory.add_patch(self.patch2)
        self.composite_factory.add_patch(self.patch3)

        self.composite_factory.link_sides(
            first_patch_name=self.patch1.name,
            first_patch_side=Patch.Side.RIGHT,
            second_patch_name=self.patch2.name,
            second_patch_side=Patch.Side.LEFT,
        )
        self.composite_factory.link_sides(
            first_patch_name=self.patch2.name,
            first_patch_side=Patch.Side.TOP,
            second_patch_name=self.patch3.name,
            second_patch_side=Patch.Side.BOTTOM,
        )

        self.patches = [self.patch1, self.patch2, self.patch3]
        self.connections = self.composite_factory._connection_list

        self.spline_surface = self.composite_factory.build()

        self.tessellated_surface = TessellatedSplineSurface(
            composite_spline_surface=self.spline_surface,
            tessellation_subdivisions_per_patch=2,
        )

    def test_init(self):
        """Test the initialization of TessellatedSplineSurface."""
        self.assertIsInstance(self.tessellated_surface, TessellatedSplineSurface)
        self.assertEqual(self.tessellated_surface.spline_name, "test_composite")
        self.assertEqual(
            self.tessellated_surface.mesh_name, "test_composite_tessellated"
        )

        self.assertEqual(self.tessellated_surface.global_vertices.shape[0], 21)
        self.assertEqual(self.tessellated_surface.global_faces.shape[0], 24)
        self.assertEqual(self.tessellated_surface.global_control_points.shape[0], 21)

    def test_getitem(self):
        """Test the __getitem__ method"""
        for idx, patch in enumerate(self.tessellated_surface._tessellated_mesh):
            self.assertEqual(self.tessellated_surface[idx], patch)

        with self.assertRaises(IndexError):
            _ = self.tessellated_surface[3]

    def test_len(self):
        """Test the __len__ method"""
        self.assertEqual(len(self.tessellated_surface), 3)

    def test_iter(self):
        """Test the __iter__ method"""
        patches = list(iter(self.tessellated_surface))
        self.assertEqual(patches, list(self.tessellated_surface._tessellated_mesh))

    def test_control_point_setter(self):
        """Test setting control points."""
        new_control_points = (
            self.tessellated_surface.global_control_points.numpy()
            + np.array([0.0, 0.0, 0.5])
        )

        # Modify one control point to observe normal update behavior
        new_control_points[8, 2] += 1.0
        self.tessellated_surface.global_control_points = mi.TensorXf(new_control_points)

        # Verify that the control points were updated
        np.testing.assert_array_equal(
            self.tessellated_surface.global_control_points.numpy(),
            new_control_points,
        )

        # Check that the vertices have correct positions after updating control points
        # Bezier basis makes it "easy enough" to compute expected vertices
        expected_vertices = np.array(
            [
                [0.0, 0.0, 0.5],
                [0.5, 0.0, 0.5],
                [1.0, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [0.5, 0.5, 0.5 + 0.25 * 0.25],
                [1.0, 0.5, 0.5 + 0.25],
                [0.0, 1.0, 0.5],
                [0.5, 1.0, 0.5 + 0.25],
                [1.0, 1.0, 0.5 + 1.0],  # Modified control point = vertex pos here!
                [1.5, 0.0, 0.5],
                [2.0, 0.0, 0.5],
                [1.5, 0.5, 0.5 + 0.25 * 0.25],
                [2.0, 0.5, 0.5],
                [1.5, 1.0, 0.5 + 0.25],
                [2.0, 1.0, 0.5],
                [1.0, 1.5, 0.5 + 0.25],
                [1.5, 1.5, 0.5 + 0.25 * 0.25],
                [2.0, 1.5, 0.5],
                [1.0, 2.0, 0.5],
                [1.5, 2.0, 0.5],
                [2.0, 2.0, 0.5],
            ],
            dtype=np.float32,
        )

        # Verify that the vertices were updated correctly
        np.testing.assert_array_almost_equal(
            self.tessellated_surface._tessellated_mesh.global_vertices.numpy(),
            expected_vertices,
        )

        # Verify that the control points in each patch were updated correctly
        for patch_idx, patch in enumerate(
            self.tessellated_surface._composite_spline_surface
        ):
            np.testing.assert_array_equal(
                patch.control_points.numpy(),
                new_control_points[
                    self.tessellated_surface._composite_spline_surface._patch_to_global_map[
                        patch_idx
                    ]
                ],
            )

        # Verify tessellated mesh vertices and normals are consistent
        for patch_idx, patch in enumerate(self.tessellated_surface):
            np.testing.assert_array_equal(
                patch.vertices.numpy(),
                self.tessellated_surface._tessellated_mesh.global_vertices.numpy()[
                    self.tessellated_surface._composite_spline_surface._patch_to_global_map[
                        patch_idx
                    ]
                ],
            )
            np.testing.assert_array_equal(
                patch.vertex_normals.numpy(),
                self.tessellated_surface._tessellated_mesh.global_vertex_normals.numpy()[
                    self.tessellated_surface._composite_spline_surface._patch_to_global_map[
                        patch_idx
                    ]
                ],
            )

    def test_serialization(self):
        """
        Test serialization and deserialization of CompositeSplineSurface.

        Since the tessellated mesh is the only object that is not serialized in
        a standard way, all critical components are checked for consistency.
        """
        encoded = jsonpickle.encode(self.tessellated_surface)
        decoded = jsonpickle.decode(encoded)

        np.testing.assert_array_equal(
            decoded.global_vertices.numpy(),
            self.tessellated_surface.global_vertices.numpy(),
        )
        np.testing.assert_array_equal(
            decoded.global_faces.numpy(),
            self.tessellated_surface.global_faces.numpy(),
        )
        np.testing.assert_array_equal(
            decoded.global_vertex_normals.numpy(),
            self.tessellated_surface.global_vertex_normals.numpy(),
        )

        for mesh, decoded_mesh in zip(self.tessellated_surface, decoded):
            np.testing.assert_array_almost_equal(
                mesh.vertices.numpy(), decoded_mesh.vertices.numpy()
            )
            np.testing.assert_array_almost_equal(
                mesh.vertex_normals.numpy(), decoded_mesh.vertex_normals.numpy()
            )
            np.testing.assert_array_almost_equal(
                mesh.faces.numpy(), decoded_mesh.faces.numpy()
            )
            np.testing.assert_array_almost_equal(
                mesh.texcoords.numpy(), decoded_mesh.texcoords.numpy()
            )

    def test_from_file_with_textures_success(self):
        """Test successful loading of tessellated spline surface with textures."""
        # First save the current tessellated surface to a file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save tessellated surface
            surface_file = temp_path / "test_surface.json"
            with open(surface_file, "w") as f:
                _ = f.write(jsonpickle.encode(self.tessellated_surface))

            # Create texture files matching patch names
            texture1 = temp_path / "diffuse_patch1_texture.png"
            texture2 = temp_path / "material_patch2_albedo.jpg"
            texture3 = temp_path / "patch3_color.bmp"

            # Create dummy texture files
            for texture_path in (texture1, texture2, texture3):
                img = Image.fromarray(
                    np.array(
                        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]],
                        dtype=np.uint8,
                    )
                )
                img.save(texture_path)

            texture_paths = (texture3, texture1, texture2)

            # Load tessellated surface with textures
            loaded_surface = TessellatedSplineSurface.from_file_with_textures(
                composite_spline_surface_path=surface_file, texture_paths=texture_paths
            )

            # Verify the surface was loaded correctly
            self.assertEqual(loaded_surface.spline_name, "test_composite")
            self.assertEqual(len(loaded_surface), 3)

            # Verify textures were applied in correct order (patch1, patch2, patch3)
            expected_texture_order = [texture1, texture2, texture3]
            for patch, texture_path in zip(loaded_surface, expected_texture_order):
                self.assertIsInstance(patch.bsdf, DiffuseBSDF)

                expected_mitsuba_dict = {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "bitmap",
                        "filename": str(texture_path),
                    },
                }
                self.assertEqual(patch.bsdf.mitsuba_dict, expected_mitsuba_dict)

    def test_from_file_with_textures_file_errors(self):
        """Test error handling for file-related issues."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test nonexistent surface file
            nonexistent_file = temp_path / "nonexistent.json"
            texture_file = temp_path / "patch1.png"
            texture_file.touch()

            with self.assertRaises(FileNotFoundError):
                TessellatedSplineSurface.from_file_with_textures(
                    composite_spline_surface_path=nonexistent_file,
                    texture_paths=[texture_file, texture_file, texture_file],
                )

            # Test invalid JSON file
            invalid_file = temp_path / "invalid.json"
            with open(invalid_file, "w") as f:
                f.write("invalid json content")

            with self.assertRaises(JSONDecodeError):
                TessellatedSplineSurface.from_file_with_textures(
                    composite_spline_surface_path=invalid_file,
                    texture_paths=[texture_file, texture_file, texture_file],
                )
