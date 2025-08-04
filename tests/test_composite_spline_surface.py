import unittest

import jsonpickle
import mitsuba as mi
import numpy as np

from rebel_shape.meshing.composite_mesh import CompositeMesh
from rebel_shape.meshing.composite_spline_surface import CompositeSplineSurface
from rebel_shape.meshing.composite_spline_surface_factory import (
    CompositeSplineSurfaceFactory,
)
from rebel_shape.meshing.patch import Patch
from rebel_shape.meshing.rectangular_mesh import RectangularMesh
from rebel_shape.splines.bspline_patch import BSplinePatch


class TestCompositeSplineSurface(unittest.TestCase):
    def setUp(self):
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

        self.surface = self.composite_factory.build()

    def test_init(self):
        """Test the initialization of CompositeSplineSurface"""
        self.assertIsInstance(self.surface, CompositeSplineSurface)
        self.assertEqual(self.surface.name, "test_composite")
        self.assertEqual(self.surface.global_control_points.shape[0], 21)

    def test_build_global_representation(self):
        """Test building of global control points"""
        global_cps = self.surface._global_control_points.numpy()
        expected_cps = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.5, 0.0],
                [1.0, 0.5, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.5, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.5, 0.5, 0.0],
                [2.0, 0.5, 0.0],
                [1.5, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [1.0, 1.5, 0.0],
                [1.5, 1.5, 0.0],
                [2.0, 1.5, 0.0],
                [1.0, 2.0, 0.0],
                [1.5, 2.0, 0.0],
                [2.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_array_almost_equal(global_cps, expected_cps)

        expected_quads = np.array(
            [
                [0, 1, 4, 3],
                [1, 2, 5, 4],
                [3, 4, 7, 6],
                [4, 5, 8, 7],
                [2, 9, 11, 5],
                [9, 10, 12, 11],
                [5, 11, 13, 8],
                [11, 12, 14, 13],
                [8, 13, 16, 15],
                [13, 14, 17, 16],
                [15, 16, 19, 18],
                [16, 17, 20, 19],
            ]
        )

        global_quads = self.surface._global_quads.numpy()
        np.testing.assert_array_equal(global_quads, expected_quads)

    def test_update_global_control_points(self):
        """Test updating global control points"""
        new_patch1_cps = mi.TensorXf(
            self.patch1.control_points.numpy() + np.array([0.0, 0.0, 1.0])
        )
        self.surface[0].control_points = new_patch1_cps
        self.surface.update_global_representation()

        updated_global_cps = self.surface._global_control_points.numpy()

        expected_edge_cps = np.array(
            [
                [1.0, 0.0, 1.0 / 2.0],
                [1.0, 0.5, 1.0 / 2.0],
                [1.0, 1.0, 1.0 / 3.0],
            ],
            dtype=np.float32,
        )

        # Check global averaged control points for the right edge of patch 0
        np.testing.assert_array_almost_equal(
            updated_global_cps[
                self.surface._patch_to_global_map[0][
                    self.surface[0].get_node_indices_for(Patch.Side.RIGHT)
                ],
            ],
            expected_edge_cps,
        )

        # Changes have to propagate to all patches
        np.testing.assert_array_almost_equal(
            self.surface[0].control_points[
                self.surface[0].get_node_indices_for(Patch.Side.RIGHT)
            ],
            expected_edge_cps,
        )
        np.testing.assert_array_almost_equal(
            self.surface[1].control_points[
                self.surface[1].get_node_indices_for(Patch.Side.LEFT)
            ],
            expected_edge_cps,
        )

    def test_getitem(self):
        """Test the __getitem__ method"""
        for idx, patch in enumerate(self.patches):
            self.assertEqual(self.surface[idx], patch)

        with self.assertRaises(IndexError):
            _ = self.surface[3]

    def test_len(self):
        """Test the __len__ method"""
        self.assertEqual(len(self.surface), 3)

    def test_iter(self):
        """Test the __iter__ method"""
        patches = list(iter(self.surface))
        self.assertEqual(patches, self.patches)

    def test_control_points_setter(self):
        """Test updating global control points"""
        new_cps = mi.TensorXf(
            self.surface.global_control_points.numpy() + np.array([0.0, 0.0, 0.5])
        )
        self.surface.global_control_points = new_cps

        for patch_idx, patch in enumerate(self.surface):
            np.testing.assert_array_almost_equal(
                patch.control_points.numpy(),
                new_cps.numpy()[self.surface._patch_to_global_map[patch_idx]],
            )

    def test_serialization(self):
        """
        Tests serialization and deserialization of CompositeSplineSurface.

        Since only global control points, global quads, and averaging weights require
        special handling, only these attributes are tested.
        """
        encoded = jsonpickle.encode(self.surface)
        decoded = jsonpickle.decode(encoded)

        self.assertEqual(decoded.name, self.surface.name)
        self.assertEqual(len(decoded), len(self.surface))

        # Check global representation
        np.testing.assert_array_almost_equal(
            decoded.global_control_points.numpy(),
            self.surface.global_control_points.numpy(),
        )
        np.testing.assert_array_equal(
            decoded.global_quads.numpy(), self.surface.global_quads.numpy()
        )

        np.testing.assert_array_equal(
            decoded._averaging_weights.numpy(),
            self.surface._averaging_weights.numpy(),
        )

    def test_composite_mesh_tesselation(self):
        """Test the composite_mesh_tesselation method"""
        tesselation = self.surface.composite_mesh_tesselation(subdivisions_per_patch=2)

        self.assertIsInstance(tesselation, CompositeMesh)
        self.assertEqual(tesselation.name, "test_composite_tessellated")
        self.assertEqual(len(tesselation), 3)

        for patch in tesselation:
            self.assertIsInstance(patch, RectangularMesh)
            self.assertEqual(patch.vertices.shape[0], 9)
            self.assertEqual(patch.faces.shape[0], 8)
            self.assertEqual(patch.texcoords.shape[0], 9)

        self.assertEqual(len(tesselation.global_vertices), 21)
        self.assertEqual(len(tesselation.global_faces), 24)

        # In this special case the control points are the same as the vertices
        np.testing.assert_array_almost_equal(
            tesselation.global_vertices.numpy(),
            self.surface.global_control_points.numpy(),
        )

    def test_h_refinement(self):
        self.surface.h_refinement(factor=3)

        self.assertEqual(self.surface.global_control_points.shape[0], 25 * 3 - 2 * 5)
        self.assertEqual(self.surface.global_quads.shape[0], 4 * 4 * 3)

        # Each patch should have 25 control points after refinement
        for mapping in self.surface._patch_to_global_map:
            self.assertEqual(mapping.shape[0], 25)

        self.assertEqual(
            self.surface._averaging_weights.shape[0],
            self.surface.global_control_points.shape[0],
        )
