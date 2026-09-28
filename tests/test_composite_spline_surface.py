import unittest

import jsonpickle
import mitsuba as mi
import numpy as np
from OCP.BRep import BRep_Tool
from OCP.GeomAdaptor import GeomAdaptor_Surface
from OCP.OCP.GeomAbs import GeomAbs_BSplineSurface
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape

from forge_sim.meshing.composite_mesh import CompositeMesh
from forge_sim.meshing.composite_spline_surface import CompositeSplineSurface
from forge_sim.meshing.composite_spline_surface_factory import (
    CompositeSplineSurfaceFactory,
)
from forge_sim.meshing.patch import Patch
from forge_sim.meshing.rectangular_mesh import RectangularMesh
from forge_sim.splines.bspline_patch import BSplinePatch


class TestCompositeSplineSurface(unittest.TestCase):
    def setUp(self):
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

    def test_to_brep(self):
        """Test conversion to BRep format using OCP TopoDS_Shape"""
        brep_shape = self.surface.to_brep(tolerance=1e-6)

        # Check that the result is a valid TopoDS_Shape
        self.assertIsInstance(brep_shape, TopoDS_Shape)
        self.assertFalse(brep_shape.IsNull())

        # Explore the faces in the B-Rep shape
        face_explorer = TopExp_Explorer(brep_shape, TopAbs_FACE)
        face_count = 0
        faces = []

        while face_explorer.More():
            face = TopoDS.Face_s(face_explorer.Current())
            faces.append(face)
            face_count += 1
            face_explorer.Next()

        # The composite surface has 3 patches, so we should have 3 faces
        self.assertEqual(
            face_count,
            len(self.surface),
            "Number of faces should match number of patches",
        )

        # Verify that each face corresponds to a valid B-Spline surface
        for i, face in enumerate(faces):
            # Get the underlying surface from the face
            surface_handle = BRep_Tool.Surface_s(face)
            self.assertIsNotNone(
                surface_handle, f"Face {i} should have a valid surface"
            )

            # Create a GeomAdaptor to check surface properties
            adaptor = GeomAdaptor_Surface(surface_handle)

            self.assertEqual(
                adaptor.GetType(),
                GeomAbs_BSplineSurface,
                f"Face {i} should be a B-Spline surface",
            )

        # Verify that the B-Rep can evaluate points correctly by comparing
        # with direct patch evaluation
        for patch_idx in range(len(self.surface)):
            patch = self.surface[patch_idx]
            face = faces[patch_idx]

            surface_handle = BRep_Tool.Surface_s(face)

            n_test_points = 7
            u_samples = np.linspace(0.0, 1.0, n_test_points)
            v_samples = np.linspace(0.0, 1.0, n_test_points)
            uv = np.meshgrid(u_samples, v_samples)

            uv_points = np.vstack([uv[0].ravel(), uv[1].ravel()]).T

            for u, v in uv_points:
                # Evaluate using the patch
                patch_point = patch.evaluate_coordinates(
                    parametric_coords=mi.TensorXf([[u, v]])
                ).numpy()[0]

                # Evaluate using the OCP surface
                ocp_point = surface_handle.Value(u, v)
                brep_point = np.array([ocp_point.X(), ocp_point.Y(), ocp_point.Z()])

                # Compare the results
                np.testing.assert_array_almost_equal(
                    patch_point,
                    brep_point,
                    decimal=5,
                    err_msg=f"Patch {patch_idx} point mismatch at (u={u}, v={v})",
                )
