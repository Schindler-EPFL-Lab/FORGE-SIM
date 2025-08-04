import unittest

import mitsuba as mi
import numpy as np
import jsonpickle

from rebel_shape.meshing.composite_spline_surface_factory import (
    CompositeSplineSurfaceFactory,
)
from rebel_shape.meshing.patch import Patch
from rebel_shape.meshing.tessellated_spline_surface import TessellatedSplineSurface
from rebel_shape.splines.bspline_patch import BSplinePatch


class TestTessellatedSplineSurface(unittest.TestCase):
    """Unit tests for TessellatedSplineSurface class."""

    def setUp(self):
        """Set up a simple tessellated spline surface for testing."""
        mi.set_variant("scalar_rgb")
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
