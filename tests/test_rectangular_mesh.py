import unittest

import mitsuba as mi
import numpy as np

from forge_sim.meshing.patch import Patch
from forge_sim.meshing.rectangular_mesh import RectangularMesh


class TestRectangularMesh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mi.set_variant("llvm_ad_rgb")
        # Create a simple test mesh
        cls.vertices = np.array(
            [[-0.5, -0.5, 0], [0.5, -0.5, 0], [-0.5, 0.5, 0], [0.5, 0.5, 0]],
            dtype=np.float64,
        )

        cls.faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)

        cls.texture_coordinates = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64
        )

    def setUp(self):
        """
        Build the mesh per test rather than caching it on the class.

        `RectangularMesh` wraps an `mi.Mesh`, and a Mitsuba object held on the class
        outlives the test that built it; that has been observed to abort unrelated
        tests inside Dr.Jit's thread pool. Only the plain numpy inputs are shared.
        """
        self.mesh = RectangularMesh(
            mesh_name="test_mesh",
            vertices=self.vertices,
            faces=self.faces,
            texture_coordinates=self.texture_coordinates,
        )

    def test_initialization(self):
        """Test whether the mesh is initialized correctly."""
        self.assertEqual(self.mesh.name, "test_mesh")
        self.assertEqual(self.mesh.vertices.shape, (4, 3))
        self.assertEqual(self.mesh.faces.shape, (2, 3))
        self.assertEqual(self.mesh.vertex_normals.shape, (4, 3))
        self.assertEqual(self.mesh.texcoords.shape, (4, 2))

    def test_get_side(self):
        """
        Test whether the correct vertices are returned for each side of the
        rectangular mesh.
        """
        left = self.mesh.get_node_indices_for(Patch.Side.LEFT)
        right = self.mesh.get_node_indices_for(Patch.Side.RIGHT)
        top = self.mesh.get_node_indices_for(Patch.Side.TOP)
        bottom = self.mesh.get_node_indices_for(Patch.Side.BOTTOM)

        # Check correct vertices are returned for each side
        np.testing.assert_array_almost_equal(
            left.numpy(), np.array([0, 2], dtype=np.uint32)
        )
        np.testing.assert_array_almost_equal(
            right.numpy(), np.array([1, 3], dtype=np.uint32)
        )
        np.testing.assert_array_almost_equal(
            top.numpy(), np.array([2, 3], dtype=np.uint32)
        )
        np.testing.assert_array_almost_equal(
            bottom.numpy(), np.array([0, 1], dtype=np.uint32)
        )

    def test_unit_square(self):
        """Test whether the mesh is created correctly from a transformed unit square."""
        mesh = RectangularMesh.unit_square(
            mesh_name="test_from_corners",
            height_subdivisions=2,
            width_subdivisions=2,
        )

        # Check mesh has correct dimensions
        self.assertEqual(mesh.vertices.shape, (9, 3))  # 3x3 grid
        self.assertEqual(mesh.faces.shape, (8, 3))  # 2x2 grid * 2 triangles
        self.assertEqual(mesh.texcoords.shape, (9, 2))

        expected_vertices = np.array(
            [
                [-0.5, -0.5, 0.0],
                [0.0, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [-0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [-0.5, 0.5, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.5, 0.0],
            ]
        )
        np.testing.assert_array_almost_equal(mesh.vertices.numpy(), expected_vertices)

        expected_faces = np.array(
            [
                [0, 1, 4],
                [0, 4, 3],
                [1, 2, 5],
                [1, 5, 4],
                [3, 4, 7],
                [3, 7, 6],
                [4, 5, 8],
                [4, 8, 7],
            ]
        )
        np.testing.assert_array_equal(mesh.faces.numpy(), expected_faces)

    def test_invalid_params_to_transformed_unit_square(self):
        """Test whether passing invalid parameters raises a ValueError."""
        with self.assertRaises(ValueError):
            RectangularMesh.unit_square(
                mesh_name="invalid_mesh",
                height_subdivisions=0,
                width_subdivisions=1,
            )

        with self.assertRaises(ValueError):
            RectangularMesh.unit_square(
                mesh_name="invalid_mesh",
                height_subdivisions=1,
                width_subdivisions=-1,
            )
