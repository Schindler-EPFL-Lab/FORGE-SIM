import tempfile
import unittest
from pathlib import Path

import mitsuba as mi
import numpy as np

from rebel_shape.meshing.mesh import Mesh
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF


class TestMesh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the Mitsuba variant and test data."""
        mi.set_variant("scalar_rgb")

        cls.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        cls.faces = np.array([[0, 1, 2]], dtype=np.uint32)
        cls.texcoords = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
        cls.normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float64)

    def test_init_mesh(self):
        """Test creating a mesh with basic parameters."""
        mesh = Mesh(
            mesh_name="test_mesh",
            vertices=self.vertices,
            faces=self.faces,
            bsdf=DiffuseBSDF(DiffuseBSDF.Albedo(0.8)),
        )

        self.assertEqual(mesh._mesh.vertex_count(), 3)
        self.assertEqual(mesh._mesh.face_count(), 1)
        self.assertTrue(mesh._mesh.has_vertex_normals())
        self.assertFalse(mesh._mesh.has_vertex_texcoords())
        self.assertEqual(mesh.name, "test_mesh")
        np.testing.assert_allclose(mesh.vertices.numpy(), self.vertices)
        np.testing.assert_allclose(mesh.faces.numpy(), self.faces)

        # No texture coordinates by default -> raises error
        with self.assertRaises(ValueError):
            _ = mesh.texcoords

    def test_init_with_normals_and_texcoords(self):
        """Test creating a mesh with normals and texture coordinates."""
        mesh = Mesh(
            mesh_name="test_mesh",
            vertices=self.vertices,
            faces=self.faces,
            vertex_normals=self.normals,
            texture_coordinates=self.texcoords,
        )

        self.assertTrue(mesh._mesh.has_vertex_normals())
        self.assertTrue(mesh._mesh.has_vertex_texcoords())
        np.testing.assert_allclose(mesh.vertex_normals.numpy(), self.normals)
        np.testing.assert_allclose(mesh.texcoords.numpy(), self.texcoords)

    def test_init_with_texture_coordinates(self):
        """Test creating a mesh with texture coordinates."""
        mesh = Mesh(
            mesh_name="test_mesh",
            vertices=self.vertices,
            faces=self.faces,
            texture_coordinates=self.texcoords,
        )

        self.assertTrue(mesh._mesh.has_vertex_texcoords())
        np.testing.assert_allclose(mesh.texcoords.numpy(), self.texcoords)

    def test_init_with_normals(self):
        """Test creating a mesh with custom normals."""
        mesh = Mesh(
            mesh_name="test_mesh",
            vertices=self.vertices,
            faces=self.faces,
            vertex_normals=self.normals,
        )

        self.assertTrue(mesh._mesh.has_vertex_normals())
        np.testing.assert_allclose(mesh.vertex_normals.numpy(), self.normals)

    def test_invalid_input(self):
        """Test invalid inputs for mesh creation."""
        # Invalid vertices
        with self.assertRaises(ValueError):
            Mesh(mesh_name="test_mesh", vertices=np.array([]), faces=self.faces)
        with self.assertRaises(ValueError):
            Mesh(
                mesh_name="test_mesh",
                vertices=np.array([[0, 0], [1, 0]]),
                faces=self.faces,
            )

        # Invalid faces
        with self.assertRaises(ValueError):
            Mesh(mesh_name="test_mesh", vertices=self.vertices, faces=np.array([]))
        with self.assertRaises(ValueError):
            Mesh(
                mesh_name="test_mesh",
                vertices=self.vertices,
                faces=np.array([[0, 1], [1, 2]]),
            )

        # Mismatched texcoords
        with self.assertRaises(ValueError):
            Mesh(
                mesh_name="test_mesh",
                vertices=self.vertices,
                faces=self.faces,
                texture_coordinates=np.array([[0, 0], [1, 0]], dtype=np.float64),
            )
        # Texcoords not in [0, 1]
        with self.assertRaises(ValueError):
            Mesh(
                mesh_name="test_mesh",
                vertices=self.vertices,
                faces=self.faces,
                texture_coordinates=np.array([[0, 0], [1.1, 0]], dtype=np.float64),
            )

        # Mismatched normals
        with self.assertRaises(ValueError):
            Mesh(
                "test_mesh",
                vertices=self.vertices,
                faces=self.faces,
                vertex_normals=np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float64),
            )
        # Normals not unit length
        with self.assertRaises(ValueError):
            Mesh(
                "test_mesh",
                vertices=self.vertices,
                faces=self.faces,
                vertex_normals=np.array([[0, 0, 1.1], [0, 0, 1]], dtype=np.float64),
            )

    def test_from_ply_file(self):
        local_dir = Path(__file__).parent.resolve()
        dataset_path = local_dir / "data/meshes/test_cube.ply"

        mesh = Mesh.from_ply_file(dataset_path)
        self.assertEqual(mesh.vertices.shape, (14, 3))
        self.assertEqual(mesh.faces.shape, (12, 3))
        self.assertEqual(mesh.texcoords.shape, (14, 2))
        self.assertEqual(mesh.vertex_normals.shape, (14, 3))

    def test_setters(self):
        """Test setting vertex normals and texture coordinates."""
        mesh = Mesh(
            mesh_name="test_mesh",
            vertices=self.vertices,
            faces=self.faces,
            vertex_normals=self.normals,
            texture_coordinates=self.texcoords,
        )

        # Set new vertex coordinates
        new_vertices = np.array([[0, 0, 0], [0, 2, 0], [0, 2, 2]], dtype=np.float64)
        mesh.vertices = mi.TensorXf(new_vertices)
        np.testing.assert_allclose(mesh.vertices.numpy(), new_vertices)

        # Set new vertex normals
        new_normals = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        mesh.vertex_normals = mi.TensorXf(new_normals)
        np.testing.assert_allclose(mesh.vertex_normals.numpy(), new_normals)

    def test_setters_invalid(self):
        """Test invalid setters."""
        mesh = Mesh(
            mesh_name="test_mesh",
            vertices=self.vertices,
            faces=self.faces,
            vertex_normals=self.normals,
            texture_coordinates=self.texcoords,
        )

        # Invalid vertex coordinates
        with self.assertRaisesRegex(ValueError, "2D tensor"):
            mesh.vertices = mi.TensorXf(np.array([0, 0]))
        with self.assertRaises(ValueError):
            mesh.vertices = mi.TensorXf(np.array([[0, 0], [1, 0], [0, 1]]))
        with self.assertRaises(ValueError):
            mesh.vertices = mi.TensorXf(np.array([[0, 0, 0], [1, 0, 0]]))

        # Invalid vertex normals
        with self.assertRaises(ValueError):
            mesh.vertex_normals = mi.TensorXf(np.array([[0, 0, 1], [0, 0, 1]]))
        with self.assertRaises(ValueError):
            mesh.vertex_normals = mi.TensorXf(np.array([[0, 1], [0, 1], [0, 1]]))
        with self.assertRaises(ValueError):
            mesh.vertex_normals = mi.TensorXf(
                np.array([[0, 0, 1.1], [0, 0, 1], [0, 0, 1]])
            )

    def test_set_vertices_at_index(self):
        """Test setting vertices at a specific index."""
        mesh = Mesh(
            mesh_name="test_mesh",
            vertices=self.vertices,
            faces=self.faces,
            vertex_normals=self.normals,
            texture_coordinates=self.texcoords,
        )

        # Set new vertex at index 1
        new_vertices = mi.TensorXf(np.array([[0, 1, 1]], dtype=np.float64))
        mesh.set_vertices_at(mi.ArrayXu(1), new_vertices)

        # Check if the vertex was updated correctly
        np.testing.assert_array_equal(mesh.vertices[1].numpy(), new_vertices[0].numpy())

        # Test invalid index
        with self.assertRaises(IndexError):
            mesh.set_vertices_at(mi.ArrayXu(3), new_vertices)

        with self.assertRaises(ValueError):
            mesh.set_vertices_at(mi.ArrayXu([0, 1]), new_vertices)

        with self.assertRaises(ValueError):
            mesh.set_vertices_at(
                indices=mi.ArrayXu(1),
                vertices=mi.TensorXf(np.array([[0, 1]], dtype=np.float64)),
            )

    def test_set_vertex_normals_at_indices(self):
        """Test setting vertex normals at specific indices."""
        mesh = Mesh(
            mesh_name="test_mesh",
            vertices=self.vertices,
            faces=self.faces,
            vertex_normals=self.normals,
            texture_coordinates=self.texcoords,
        )

        # Set new vertex normals at index 1
        new_normals = mi.TensorXf(np.array([[0, 1, 0]], dtype=np.float64))
        mesh.set_vertex_normals_at(indices=mi.ArrayXu(1), normals=new_normals)

        # Check if the vertex normal was updated correctly
        np.testing.assert_array_equal(
            mesh.vertex_normals[1].numpy(), new_normals[0].numpy()
        )

        # Test invalid index
        with self.assertRaises(IndexError):
            mesh.set_vertex_normals_at(
                indices=mi.ArrayXu(3),
                normals=new_normals,
            )

        with self.assertRaises(ValueError):
            mesh.set_vertex_normals_at(
                indices=mi.ArrayXu([0, 1]), normals=new_normals
            )

        with self.assertRaises(ValueError):
            mesh.set_vertex_normals_at(
                indices=mi.ArrayXu(1),
                normals=mi.TensorXf(np.array([[0, 1]], dtype=np.float64)),
            )

    def test_save_and_load(self):
        """Test saving the mesh."""
        mesh = Mesh(
            mesh_name="test_save_mesh", vertices=self.vertices, faces=self.faces
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save the mesh
            output_dir = Path(tmpdir)
            mesh.save_to_ply(output_dir)

            self.assertTrue((output_dir / "test_save_mesh.ply").exists())

    def test_mitsuba_dict(self):
        """Test the Mitsuba dictionary representation."""
        mesh = Mesh(
            mesh_name="test_mesh",
            vertices=self.vertices,
            faces=self.faces,
            texture_coordinates=self.texcoords,
            vertex_normals=self.normals,
        )

        # Check that it indeed is a Mitsuba mesh
        self.assertTrue(isinstance(mesh.mitsuba_dict, mi.Mesh))
