import unittest

import drjit as dr
import mitsuba as mi
import numpy as np

from forge_sim.meshing.composite_mesh_factory import CompositeMeshFactory
from forge_sim.meshing.mesh import Mesh
from forge_sim.meshing.patch import Patch
from forge_sim.meshing.rectangular_mesh import RectangularMesh


class TestCompositeMesh(unittest.TestCase):
    def setUp(self):
        mi.set_variant("llvm_ad_rgb")

        self.num_subdivisions = 2
        # Create simple rectangular meshes for testing
        self.mesh1 = RectangularMesh.unit_square(
            mesh_name="mesh1",
            width_subdivisions=self.num_subdivisions,
            height_subdivisions=self.num_subdivisions,
        )
        self.mesh1.vertices = mi.TensorXf(
            self.mesh1.vertices.numpy() + np.array([-0.5, 0.0, 0.0])
        )
        self.mesh2 = RectangularMesh.unit_square(
            mesh_name="mesh2",
            width_subdivisions=self.num_subdivisions,
            height_subdivisions=self.num_subdivisions,
        )
        self.mesh2.vertices = mi.TensorXf(
            self.mesh2.vertices.numpy() + np.array([0.5, 0.0, 0.0])
        )
        self.mesh3 = RectangularMesh.unit_square(
            mesh_name="mesh3",
            width_subdivisions=self.num_subdivisions,
            height_subdivisions=self.num_subdivisions,
        )
        self.mesh3.vertices = mi.TensorXf(
            self.mesh3.vertices.numpy() + np.array([0.5, -1.0, 0.0])
        )

        # Create a composite mesh with the first rectangular mesh
        self.composite_factory = CompositeMeshFactory("test_composite")
        self.composite_factory.add_patch(self.mesh1)
        self.composite_factory.add_patch(self.mesh2)
        self.composite_factory.link_sides(
            first_patch_name=self.mesh1.name,
            first_patch_side=Patch.Side.RIGHT,
            second_patch_name=self.mesh2.name,
            second_patch_side=Patch.Side.LEFT,
        )
        self.composite_mesh = self.composite_factory.build()

    def test_global_mesh_exposes_welded_global_mesh(self):
        """`global_mesh` returns the welded `Mesh` for the global vertices/faces."""
        global_mesh = self.composite_mesh.global_mesh

        self.assertIsInstance(global_mesh, Mesh)
        np.testing.assert_allclose(
            global_mesh.vertices.numpy(),
            self.composite_mesh.global_vertices.numpy(),
        )
        np.testing.assert_array_equal(
            global_mesh.faces.numpy(),
            self.composite_mesh.global_faces.numpy(),
        )

    def test_init(self):
        """Test whether the composite mesh initializes correctly."""
        self.composite_mesh = self.composite_factory.build()

        self.assertEqual(len(self.composite_mesh._patch_list), 2)
        self.assertEqual(len(self.composite_mesh._connection_list), 2)
        self.assertEqual(len(self.composite_mesh._patch_to_global_map), 2)

        self.assertEqual(
            len(self.composite_mesh._patch_to_global_map[0]),
            self.mesh1.vertices.shape[0],
        )
        self.assertEqual(
            len(self.composite_mesh._patch_to_global_map[1]),
            self.mesh2.vertices.shape[0],
        )
        self.assertEqual(
            len(self.composite_mesh._global_to_patches_map),
            self.mesh1.vertices.shape[0]
            + self.mesh2.vertices.shape[0]
            - len(self.mesh1.get_node_indices_for(Patch.Side.RIGHT)),
        )

    def test_update_global_mesh_indices(self):
        # Add meshes and link them
        self.composite_factory.add_patch(self.mesh3)
        self.composite_factory.link_sides(
            first_patch_name=self.mesh2.name,
            first_patch_side=Patch.Side.BOTTOM,
            second_patch_name=self.mesh3.name,
            second_patch_side=Patch.Side.TOP,
        )

        composite = self.composite_factory.build()

        # Check that the number of vertices is less than the sum of all mesh vertices
        # due to shared vertices at seams
        expected_vertices_count = (
            self.mesh1.vertices.shape[0]
            + self.mesh2.vertices.shape[0]
            + self.mesh3.vertices.shape[0]
            - len(self.mesh1.get_node_indices_for(Patch.Side.RIGHT))
            - len(self.mesh1.get_node_indices_for(Patch.Side.TOP))
        )

        self.assertEqual(composite.global_vertices.shape[0], expected_vertices_count)

    def test_getitem(self):
        """Test the __getitem__ method"""
        for idx, patch in enumerate([self.mesh1, self.mesh2]):
            self.assertEqual(self.composite_mesh[idx], patch)

        with self.assertRaises(IndexError):
            _ = self.composite_mesh[2]

    def test_len(self):
        """Test the __len__ method"""
        self.assertEqual(len(self.composite_mesh), 2)

    def test_iter(self):
        """Test the __iter__ method"""
        patches = list(iter(self.composite_mesh))
        self.assertEqual(patches, [self.mesh1, self.mesh2])

    def test_vertices_setter(self):
        """Test setting vertices of the composite mesh."""
        # Set new vertices
        current_vertices = self.composite_mesh.global_vertices.numpy()
        new_vertices = current_vertices + np.array([1.0, 0.0, 0.0])
        self.composite_mesh.global_vertices = mi.TensorXf(new_vertices)

        # Check that the vertices were updated correctly
        np.testing.assert_array_almost_equal(
            self.composite_mesh.global_vertices.numpy(), new_vertices
        )

        # Check member vertices
        for i, mesh in enumerate(self.composite_mesh._patch_list):
            member_vertices = self.composite_mesh[i].vertices
            np.testing.assert_array_almost_equal(
                member_vertices.numpy(),
                new_vertices[self.composite_mesh._patch_to_global_map[i]],
            )

    def test_update_global_mesh_vertices(self):
        """Test updating global vertices."""
        # Update global vertices
        new_vertices = self.mesh2.vertices.numpy() + np.array([0.0, 0.0, 1.0])
        self.composite_mesh[1].vertices = mi.TensorXf(new_vertices)

        self.composite_mesh.update_global_representation()

        # Check that global vertices match the updated member vertices
        np.testing.assert_array_almost_equal(
            self.composite_mesh.global_vertices[
                dr.ravel(
                    mi.TensorXu(
                        self.composite_mesh._patch_to_global_map[1][
                            np.arange(self.composite_mesh[1].vertices.shape[0])
                        ]
                    )
                )
            ].numpy(),
            self.composite_mesh[1].vertices.numpy(),
        )

        # Check that the edge of the patch that interacts with the other patch
        # has been updated correctly
        np.testing.assert_array_almost_equal(
            self.composite_mesh[0]
            .vertices[self.composite_mesh[0].get_node_indices_for(Patch.Side.RIGHT)]
            .numpy(),
            self.composite_mesh[1]
            .vertices[self.composite_mesh[1].get_node_indices_for(Patch.Side.LEFT)]
            .numpy(),
        )

    def test_update_global_mesh_vertices_flipped(self):
        """Test updating global vertices."""
        factory = CompositeMeshFactory("test_composite")
        factory.add_patch(self.mesh1)
        flipped_mesh2 = RectangularMesh.unit_square(
            mesh_name="mesh2_flipped",
            width_subdivisions=self.num_subdivisions,
            height_subdivisions=self.num_subdivisions,
        )
        flipped_mesh2.vertices = mi.TensorXf(
            -1.0 * flipped_mesh2.vertices.numpy() + np.array([0.5, 0.0, 1.0])
        )

        factory.add_patch(flipped_mesh2)
        factory.link_sides(
            first_patch_name=self.mesh1.name,
            first_patch_side=Patch.Side.RIGHT,
            second_patch_name=flipped_mesh2.name,
            second_patch_side=Patch.Side.RIGHT,
        )
        composite = factory.build()

        # Check that global vertices match the updated member vertices
        np.testing.assert_array_almost_equal(
            composite.global_vertices[
                dr.ravel(
                    mi.TensorXu(
                        composite._patch_to_global_map[1][
                            np.arange(composite[1].vertices.shape[0])
                        ]
                    )
                )
            ].numpy(),
            composite[1].vertices.numpy(),
        )

    def test_update_global_mesh_normals(self):
        """Test updating global vertex normals."""
        # Update global vertex normals
        new_normals = self.mesh1.vertex_normals.numpy() @ np.eye(3)[[2, 0, 1], :]
        self.composite_mesh[1].vertex_normals = mi.TensorXf(new_normals)

        self.composite_mesh.update_global_representation()

        # Check that the normals were updated correctly
        np.testing.assert_array_almost_equal(
            self.composite_mesh.global_vertex_normals[
                dr.ravel(
                    mi.TensorXu(
                        self.composite_mesh._patch_to_global_map[1][
                            np.arange(self.composite_mesh[1].vertices.shape[0])
                        ]
                    )
                )
            ].numpy(),
            self.composite_mesh[1].vertex_normals.numpy(),
        )

        # Check that the edge of the patch that interacts with the other patch
        # has been updated correctly
        np.testing.assert_array_almost_equal(
            self.composite_mesh[0]
            .vertex_normals[
                self.composite_mesh[0].get_node_indices_for(Patch.Side.RIGHT)
            ]
            .numpy(),
            self.composite_mesh[1]
            .vertex_normals[
                self.composite_mesh[1].get_node_indices_for(Patch.Side.LEFT)
            ]
            .numpy(),
        )
