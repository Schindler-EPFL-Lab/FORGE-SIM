import unittest

import mitsuba as mi
import numpy as np

from rebel_shape.utils.mesh_utils import (
    _create_cube,
    _cube_map_parametrization,
    create_cubedsphere,
    create_mitsuba_mesh,
    subdivide_mesh,
)


class TestMeshUtils(unittest.TestCase):
    def setUp(self):
        """Initialize Mitsuba variant for testing."""
        mi.set_variant("scalar_rgb")

    def test_create_cube(self):
        """Test generation of cube vertices and faces."""
        radius = 1.5
        vertices, faces = _create_cube(radius=radius)

        self.assertEqual(len(vertices), 8, "Should have 8 vertices")
        self.assertEqual(len(faces), 12, "Should have 12 triangular faces")

        vertex_distances = np.sqrt(np.sum(vertices**2, axis=1))
        np.testing.assert_allclose(vertex_distances, radius)

        self.assertEqual(faces.shape[1], 3, "Faces should be triangles")

    def test_subdivide_mesh(self):
        """Test subdivision of a triangular mesh."""
        # Create a simple tetrahedron
        vertices = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]], dtype=np.float32
        )
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int32)

        radius = 1.0
        norms = np.sqrt(np.sum(vertices**2, axis=1))
        vertices = vertices / norms[:, np.newaxis]

        new_vertices, new_faces, _ = subdivide_mesh(
            vertices=vertices, faces=faces, radius=radius
        )

        self.assertEqual(len(new_faces), len(faces) * 4)

        # Check that all new vertices are on the sphere
        vertex_distances = np.sqrt(np.sum(new_vertices**2, axis=1))
        np.testing.assert_allclose(vertex_distances, radius)

    def test_cube_map_parametrization(self):
        """Test cube map parametrization of vertices."""
        # Test with points at the center of each cube face
        vertices = np.array(
            [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ],
            dtype=np.float32,
        )

        texcoords = _cube_map_parametrization(vertices)

        expected_uvs = np.array(
            [
                [0.625, 0.375],
                [0.125, 0.375],
                [0.375, 0.625],
                [0.375, 0.125],
                [0.375, 0.375],
                [0.875, 0.375],
            ]
        )
        np.testing.assert_allclose(texcoords, expected_uvs, rtol=1e-5)

    def test_create_mitsuba_mesh(self):
        """Test creation of a Mitsuba mesh from vertices and faces."""
        # Create a simple triangle
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        uv = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)

        mesh = create_mitsuba_mesh("test_mesh", faces, vertices, uv)
        self.assertIsInstance(mesh, mi.Mesh)

        self.assertEqual(mesh.vertex_count(), 3)
        self.assertEqual(mesh.face_count(), 1)
        self.assertTrue(mesh.has_vertex_texcoords())

        mesh_no_uv = create_mitsuba_mesh("test_mesh_no_uv", faces, vertices, None)
        self.assertFalse(mesh_no_uv.has_vertex_texcoords())

    def test_create_cubedsphere(self):
        """Test creation of a cubed sphere mesh."""
        radius = 1.5
        subdivisions = 1

        cubedsphere = create_cubedsphere(radius, subdivisions)
        self.assertIsInstance(cubedsphere, mi.Mesh)

        # Cube after one subdivision: ~24 vertices, 48 faces
        self.assertGreaterEqual(cubedsphere.vertex_count(), 24)
        self.assertEqual(cubedsphere.face_count(), 48)
        self.assertTrue(cubedsphere.has_vertex_texcoords())
