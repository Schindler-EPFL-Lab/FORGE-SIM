import unittest

import mitsuba as mi
import numpy as np

from forge_sim.meshing.composite_spline_surface_factory import (
    CompositeSplineSurfaceFactory,
)
from forge_sim.meshing.mesh import Mesh
from forge_sim.meshing.patch import Patch
from forge_sim.meshing.tessellated_spline_surface import TessellatedSplineSurface
from forge_sim.regularization.normal_loss import NormalLoss
from forge_sim.regularization.position_loss import PositionLoss
from forge_sim.regularization.sparse_laplacian_matrix import SparseLaplacianMatrix
from forge_sim.regularization.spline_g1_loss import SplineG1Loss
from forge_sim.regularization.spline_normal_loss import SplineNormalLoss
from forge_sim.splines.bspline_patch import BSplinePatch


class TestRegularization(unittest.TestCase):
    """Unit tests for the C1 loss function in spline surfaces."""

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

    def test_combinatorial_laplacian_triangles(self):
        """Tests the combinatorial Laplacian for a triangle mesh."""
        triangle_faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
        indices, values = SparseLaplacianMatrix.combinatorial_laplacian(triangle_faces)

        # Checking the matrix is more intuitive than the indices and values directly
        matrix = np.zeros((4, 4), dtype=np.float32)
        for i, j, value in zip(indices[0], indices[1], values):
            matrix[i, j] += value

        self.assertEqual(indices.shape[0], 2)
        # 3 edges = 2 indices per edge, plus 3 diagonal entries, each vertex
        # connected to two others
        self.assertEqual(indices.shape[1], 20)

        np.testing.assert_array_almost_equal(
            matrix,
            np.array(
                [[2, -1, -1, 0], [-1, 3, -1, -1], [-1, -1, 3, -1], [0, -1, -1, 2]],
                dtype=np.float32,
            ),
        )

    def test_combinatorial_laplacian_quads(self):
        """Tests the combinatorial Laplacian for a quadrilateral mesh."""
        quad_faces = np.array([[0, 1, 3, 2]], dtype=np.uint32)
        indices, values = SparseLaplacianMatrix.combinatorial_laplacian(quad_faces)

        matrix = np.zeros((4, 4), dtype=np.float32)
        for i, j, value in zip(indices[0], indices[1], values):
            matrix[i, j] += value

        self.assertEqual(indices.shape[0], 2)
        # 4 edges = 2 indices per edge, plus 3 diagonal entries, each vertex
        # connected to two others
        self.assertEqual(indices.shape[1], 4 * 2 + 4 * 2)

        np.testing.assert_array_almost_equal(
            matrix,
            np.array(
                [[2, -1, -1, 0], [-1, 2, 0, -1], [-1, 0, 2, -1], [0, -1, -1, 2]],
                dtype=np.float32,
            ),
        )

    def test_position_loss_simple(self):
        """
        Test whether `PositionLoss` behaves correctly on a small mesh that was
        computed by hand.
        """
        faces = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3]], dtype=np.uint32)
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.5, 0.5, 0.0],
            ],
            dtype=np.float32,
        )
        mesh = Mesh(mesh_name="position_loss_test", vertices=vertices, faces=faces)
        regularization = PositionLoss(mesh=mesh)

        # The current mesh has no curvature, so the loss should be 0
        value = regularization.evaluate()

        matrix = np.array(
            [
                [2, -1, -1, 0, 0],
                [-1, 4, -1, -1, -1],
                [-1, -1, 3, -1, 0],
                [0, -1, -1, 3, -1],
                [0, -1, 0, -1, 2],
            ],
            dtype=np.float32,
        )
        distance_sums = np.array(
            [2.0, 4.121320344, 3.414213562, 2.707106781, 1.414213562], dtype=np.float32
        )

        laplacian_vector_prod = (
            matrix @ mesh.vertices.numpy() / distance_sums[:, np.newaxis]
        )
        dot_product = np.sum(laplacian_vector_prod * laplacian_vector_prod, axis=1)
        expected_curvature = np.mean(dot_product)

        self.assertAlmostEqual(
            value.numpy().astype(np.float64), expected_curvature, places=6
        )

    def test_normal_loss_simple(self):
        """
        Test whether `NormalLoss` behaves correctly on a small mesh that was
        computed by hand.
        """
        faces = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3]], dtype=np.uint32)
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.5],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.5, 0.5, 0.0],
            ],
            dtype=np.float32,
        )
        mesh = Mesh(mesh_name="normal_loss_test", vertices=vertices, faces=faces)
        regularization = NormalLoss(mesh=mesh)

        # The current mesh has no curvature, so the loss should be 0
        value = regularization.evaluate()

        matrix = np.array(
            [
                [2, -1, -1, 0, 0],
                [-1, 4, -1, -1, -1],
                [-1, -1, 3, -1, 0],
                [0, -1, -1, 3, -1],
                [0, -1, 0, -1, 2],
            ],
            dtype=np.float32,
        )
        distance_sums = np.array(
            [2.11803, 4.60209, 3.5, 2.82514, 1.57313], dtype=np.float32
        )

        laplacian_vector_prod = (
            matrix @ mesh.vertex_normals.numpy() / distance_sums[:, np.newaxis]
        )
        dot_product = np.sum(laplacian_vector_prod * laplacian_vector_prod, axis=1)
        expected_curvature = np.mean(dot_product)

        self.assertAlmostEqual(
            value.numpy().astype(np.float64), expected_curvature, places=6
        )

    def test_g1_loss(self):
        """
        Sanity check for the `SplineG1Loss` regularization function.

        Unfortunately, a test by hand could not be constructed easily, so the
        test is more of a sanity check that the loss function behaves as expected
        when the spline surface is modified.

        Proper verification has to be done by visually inspecting an optimization
        run with this loss function.
        """
        regularization = SplineG1Loss(composite_spline=self.tessellated_surface)

        # The current spline has no C1 discontinuities, so the loss should be 0
        value = regularization.evaluate()
        self.assertEqual(value.numpy().astype(np.float64), 0.0)

        # Modify control points to create a C1 discontinuity
        old_cps = self.tessellated_surface.global_control_points.numpy().copy()
        old_cps[8, 2] += 0.2
        old_cps[4, 2] += 0.1
        self.tessellated_surface.global_control_points = mi.TensorXf(old_cps)

        # Now the spline should have a C1 discontinuity
        new_value = regularization.evaluate()
        self.assertGreater(new_value.numpy().astype(np.float64), 0.0)

    def test_spline_normal_loss(self):
        """
        Tests whether the `SplineNormalLoss` is a sum of the per-patch
        regularization losses.
        """
        regularization = SplineNormalLoss(tessellated_spline=self.tessellated_surface)

        value = regularization.evaluate()
        self.assertEqual(value.numpy().astype(np.float64), 0.0)

        # Modify control points to create a normal discontinuity
        old_cps = self.tessellated_surface.global_control_points.numpy().copy()
        old_cps[8, 2] += 0.2
        old_cps[4, 2] += 0.1
        self.tessellated_surface.global_control_points = mi.TensorXf(old_cps)

        expected_value = mi.TensorXf(0.0)
        for patch_reg in regularization._per_patch_losses:
            expected_value += patch_reg.evaluate()

        expected_value /= len(regularization._per_patch_losses)

        self.assertAlmostEqual(
            expected_value.numpy().astype(np.float64), expected_value, places=6
        )

    def test_spline_position_loss(self):
        """
        Tests whether the `SplinePositionLoss` is a sum of the per-patch
        regularization losses.
        """
        regularization = SplineNormalLoss(tessellated_spline=self.tessellated_surface)

        expected_value = mi.TensorXf(0.0)
        for patch_reg in regularization._per_patch_losses:
            expected_value += patch_reg.evaluate()

        expected_value /= len(regularization._per_patch_losses)

        self.assertAlmostEqual(
            expected_value.numpy().astype(np.float64), expected_value, places=6
        )
