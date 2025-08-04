import unittest

import mitsuba as mi
import numpy as np

from rebel_shape.isogeometric_analysis.iga_manifold_problem import IGAManifoldProblem
from rebel_shape.meshing.composite_iga_spline_surface_factory import (
    CompositeIGASplineSurfaceFactory,
)
from rebel_shape.meshing.patch import Patch
from rebel_shape.splines.bspline_patch import BSplinePatch
from rebel_shape.splines.iga_spline_patch import IGASplinePatch


class TestIGAManifoldProblem(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mi.set_variant("scalar_rgb")

        # Use BSplinePatch for testing, not mocks, too complex to mock properly
        # Degree 1 patches are used for simplicity and have different knot span
        # configurations to test concatenation logic
        patch_1 = BSplinePatch.unit_square(
            patch_name="patch1", degree=1, u_subdivisions=1, v_subdivisions=2
        )
        patch_1.control_points = mi.TensorXf(
            patch_1.control_points.numpy() + np.array([0.5, 0.5, 0.0])
        )
        cls.patch_1 = IGASplinePatch(
            spline_patch=patch_1, field_values=patch_1.control_points
        )

        patch_2 = BSplinePatch.unit_square(
            patch_name="patch2", degree=1, u_subdivisions=2, v_subdivisions=2
        )
        patch_2.control_points = mi.TensorXf(
            patch_2.control_points.numpy() + np.array([1.5, 0.5, 0.0])
        )
        cls.patch_2 = IGASplinePatch(
            spline_patch=patch_2, field_values=patch_2.control_points
        )

        cls.composite_factory = CompositeIGASplineSurfaceFactory("test_composite")
        cls.composite_factory.add_patch(cls.patch_1)
        cls.composite_factory.add_patch(cls.patch_2)

        cls.composite_factory.link_sides(
            first_patch_name=cls.patch_1.name,
            first_patch_side=Patch.Side.RIGHT,
            second_patch_name=cls.patch_2.name,
            second_patch_side=Patch.Side.LEFT,
        )

        cls.patches = [cls.patch_1, cls.patch_2]
        cls.connections = cls.composite_factory._connection_list

        cls.iga_surface = cls.composite_factory.build()

    def test_concatenate_patch_local_matrices(self):
        """Test concatenation of local matrices from multiple patches."""
        # Create sample local matrices and indices
        local_matrices_1 = mi.TensorXf(np.random.rand(2, 4, 4))
        local_matrices_2 = mi.TensorXf(np.random.rand(4, 4, 4))
        local_matrices_list = [local_matrices_1, local_matrices_2]

        local_indices_1 = mi.TensorXu([[0, 1, 2, 3], [2, 3, 4, 5]])
        local_indices_2 = mi.TensorXu(
            [[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7], [4, 5, 7, 8]]
        )
        local_indices_list = [local_indices_1, local_indices_2]

        all_indices, all_matrices = (
            IGAManifoldProblem._concatenate_patch_local_matrices(
                composite_iga_spline=self.iga_surface,
                local_indices_list=local_indices_list,
                local_matrices_list=local_matrices_list,
            )
        )

        self.assertEqual(all_matrices.shape, (6, 4, 4))
        self.assertEqual(all_indices.shape, (6, 4))

        # Check that matrices are properly concatenated
        np.testing.assert_array_equal(
            all_matrices[0:2].numpy(), local_matrices_1.numpy()
        )
        np.testing.assert_array_equal(
            all_matrices[2:6].numpy(), local_matrices_2.numpy()
        )

        # Check that indices have received the global numbering
        expected_indices = np.array(
            [
                [0, 1, 2, 3],
                [2, 3, 4, 5],
                [1, 6, 3, 8],
                [6, 7, 8, 9],
                [3, 8, 5, 10],
                [8, 9, 10, 11],
            ]
        )
        np.testing.assert_array_equal(all_indices.numpy(), expected_indices)

    def test_concatenate_patch_local_matrices_invalid(self):
        """Test that exceptions are raised for invalid inputs."""
        # Raises error for empty input lists
        with self.assertRaises(ValueError):
            IGAManifoldProblem._concatenate_patch_local_matrices(
                composite_iga_spline=self.iga_surface,
                local_indices_list=[],
                local_matrices_list=[],
            )

        # Raises error for mismatched list lengths
        local_matrices_list = [mi.TensorXf(np.random.rand(2, 4, 4))]
        local_indices_list = [mi.TensorXu([[0, 1, 2, 3]]), mi.TensorXu([[4, 5, 6, 7]])]

        with self.assertRaises(ValueError):
            IGAManifoldProblem._concatenate_patch_local_matrices(
                composite_iga_spline=self.iga_surface,
                local_indices_list=local_indices_list,
                local_matrices_list=local_matrices_list,
            )

        # Different DOF count in local matrices
        local_matrices_1 = mi.TensorXf(np.random.rand(2, 4, 4))
        local_matrices_2 = mi.TensorXf(np.random.rand(3, 3, 3))
        local_matrices_list = [local_matrices_1, local_matrices_2]

        local_indices_1 = mi.TensorXu([[0, 1, 3, 4], [1, 2, 3, 4]])
        local_indices_2 = mi.TensorXu([[4, 5, 6], [5, 6, 7], [6, 7, 0]])
        local_indices_list = [local_indices_1, local_indices_2]

        with self.assertRaises(ValueError):
            IGAManifoldProblem._concatenate_patch_local_matrices(
                composite_iga_spline=self.iga_surface,
                local_indices_list=local_indices_list,
                local_matrices_list=local_matrices_list,
            )

    def test_concatenate_patch_local_vectors(self):
        """Test concatenation of local vectors from multiple patches."""
        # First patch has 2 knot spans and the second has 4 knot spans
        local_vectors_1 = mi.TensorXf(np.random.rand(2, 4))
        local_vectors_2 = mi.TensorXf(np.random.rand(4, 4))
        local_vectors_list = [local_vectors_1, local_vectors_2]

        local_indices_1 = mi.TensorXu([[0, 1, 2, 3], [2, 3, 4, 5]])
        local_indices_2 = mi.TensorXu(
            [[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7], [4, 5, 7, 8]]
        )
        local_indices_list = [local_indices_1, local_indices_2]

        all_indices, all_vectors = IGAManifoldProblem._concatenate_patch_local_vectors(
            composite_iga_spline=self.iga_surface,
            local_indices_list=local_indices_list,
            local_vectors_list=local_vectors_list,
        )

        self.assertEqual(all_vectors.shape, (6, 4))
        self.assertEqual(all_indices.shape, (6, 4))

        # Check that vectors are properly concatenated
        np.testing.assert_array_equal(all_vectors[0:2].numpy(), local_vectors_1.numpy())
        np.testing.assert_array_equal(all_vectors[2:6].numpy(), local_vectors_2.numpy())

        # Check that indices have received the global numbering
        expected_indices = np.array(
            [
                [0, 1, 2, 3],
                [2, 3, 4, 5],
                [1, 6, 3, 8],
                [6, 7, 8, 9],
                [3, 8, 5, 10],
                [8, 9, 10, 11],
            ]
        )
        np.testing.assert_array_equal(all_indices.numpy(), expected_indices)

    def test_concatenate_patch_local_vectors_invalid(self):
        """Test error handling for empty input lists."""
        with self.assertRaises(ValueError):
            IGAManifoldProblem._concatenate_patch_local_vectors(
                composite_iga_spline=self.iga_surface,
                local_indices_list=[],
                local_vectors_list=[],
            )

        # Test error handling for mismatched list lengths
        local_vectors_list = [mi.TensorXf(np.random.rand(2, 4))]
        local_indices_list = [mi.TensorXu([[0, 1, 2, 3]]), mi.TensorXu([[4, 5, 6, 7]])]
        with self.assertRaises(ValueError):
            IGAManifoldProblem._concatenate_patch_local_vectors(
                composite_iga_spline=self.iga_surface,
                local_indices_list=local_indices_list,
                local_vectors_list=local_vectors_list,
            )

        # Mismatched DOF count in local vectors
        local_vectors_1 = mi.TensorXf(np.random.rand(2, 4))
        local_vectors_2 = mi.TensorXf(np.random.rand(4, 3))  # Different DOF count
        local_vectors_list = [local_vectors_1, local_vectors_2]

        local_indices_1 = mi.TensorXu([[0, 1, 2, 3], [1, 2, 3, 4]])
        local_indices_2 = mi.TensorXu([[4, 5, 6], [5, 6, 7], [6, 7, 8]])
        local_indices_list = [local_indices_1, local_indices_2]

        with self.assertRaises(ValueError):
            IGAManifoldProblem._concatenate_patch_local_vectors(
                composite_iga_spline=self.iga_surface,
                local_indices_list=local_indices_list,
                local_vectors_list=local_vectors_list,
            )

    def test_knot_span_handler_properties(self):
        """Test knot span setup for numerical integration."""
        knot_span_handler_1 = IGAManifoldProblem.KnotSpanHandler(
            patch=self.patch_1, integration_order=2
        )
        knot_span_handler_2 = IGAManifoldProblem.KnotSpanHandler(patch=self.patch_2)

        self.assertEqual(knot_span_handler_1.num_knot_spans, 2)
        self.assertEqual(knot_span_handler_2.num_knot_spans, 4)

        self.assertEqual(knot_span_handler_1.min_span_size, 0.5)
        self.assertEqual(knot_span_handler_1.max_span_size, 1.0)
        self.assertEqual(knot_span_handler_2.min_span_size, 0.5)
        self.assertEqual(knot_span_handler_2.max_span_size, 0.5)

        np.testing.assert_array_equal(
            knot_span_handler_1.midpoint_parametric_coordinates.numpy(),
            np.array([[0.5, 0.25], [0.5, 0.75]]),
        )
        np.testing.assert_array_equal(
            knot_span_handler_2.midpoint_parametric_coordinates.numpy(),
            np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]),
        )

    def test_knot_span_handler_gl_interval_change(self):
        """Test Gauss-Legendre interval change for knot spans."""
        knot_span_handler = IGAManifoldProblem.KnotSpanHandler(
            patch=self.patch_1, integration_order=2
        )

        gl_positions = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))

        for u_i, v_i in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            with self.subTest(u_i=u_i, v_i=v_i):
                changed_interval = knot_span_handler._gl_interval_change(
                    sample_u=knot_span_handler._gl_positions[u_i],
                    sample_v=knot_span_handler._gl_positions[v_i],
                )
                expected_interval = mi.TensorXf(
                    [
                        [
                            0.5 + 0.5 * gl_positions[u_i],
                            0.25 + 0.25 * gl_positions[v_i],
                        ],
                        [
                            0.5 + 0.5 * gl_positions[u_i],
                            0.75 + 0.25 * gl_positions[v_i],
                        ],
                    ]
                )
                np.testing.assert_array_almost_equal(
                    changed_interval.numpy(), expected_interval.numpy()
                )

    def test_knot_span_handler_gl_weight_scale(self):
        """Test Gauss-Legendre weight scaling for knot spans."""
        knot_span_handler = IGAManifoldProblem.KnotSpanHandler(
            patch=self.patch_1, integration_order=2
        )

        weights = knot_span_handler._gl_weight_scaling(
            weight_u=knot_span_handler._gl_weights[0],
            weight_v=knot_span_handler._gl_weights[0],
        )
        expected_weights = mi.TensorXf([0.5 * 0.25, 0.5 * 0.25])
        np.testing.assert_array_almost_equal(weights.numpy(), expected_weights.numpy())

        knot_span_handler = IGAManifoldProblem.KnotSpanHandler(
            patch=self.patch_2, integration_order=2
        )
        weights = knot_span_handler._gl_weight_scaling(
            weight_u=knot_span_handler._gl_weights[0],
            weight_v=knot_span_handler._gl_weights[0],
        )
        expected_weights = mi.TensorXf([0.25 * 0.25 for _ in range(4)])

        np.testing.assert_array_almost_equal(weights.numpy(), expected_weights.numpy())

    def test_gl_coords_and_weights(self):
        """Test Gauss-Legendre coordinates and weights list generation."""
        knot_span_handler = IGAManifoldProblem.KnotSpanHandler(
            patch=self.patch_1, integration_order=2
        )
        gl_positions = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))

        coords_and_weights = knot_span_handler.gl_coords_and_weights()
        expected_weights = mi.TensorXf([0.5 * 0.25, 0.5 * 0.25])

        for i in range(2):
            for j in range(2):
                coords = coords_and_weights[i * 2 + j][0]

                expected_interval = mi.TensorXf(
                    [
                        [0.5 + 0.5 * gl_positions[j], 0.25 + 0.25 * gl_positions[i]],
                        [0.5 + 0.5 * gl_positions[j], 0.75 + 0.25 * gl_positions[i]],
                    ]
                )
                np.testing.assert_array_almost_equal(
                    coords.numpy(), expected_interval.numpy()
                )

                weights = coords_and_weights[i * 2 + j][1]
                np.testing.assert_array_almost_equal(
                    weights.numpy(), expected_weights.numpy()
                )

    def test_transform_basis_derivative_identity_metric(self):
        """Test basis function derivative transformation."""
        num_knot_spans = 3
        num_dofs = 4

        # Create mock basis derivatives
        basis_derivative = mi.TensorXf(np.random.rand(num_knot_spans, num_dofs, 2))

        # Create mock inverse metric tensor (identity for simplicity)
        inv_metric = mi.TensorXf(np.tile(np.eye(2), (num_knot_spans, 1, 1)))

        transformed = IGAManifoldProblem._transform_basis_derivative(
            basis_derivative=basis_derivative, inv_metric=inv_metric
        )

        # Check shape
        self.assertEqual(transformed.shape, (num_knot_spans, num_dofs, 2))

        # With identity metric, transformation should be identity
        np.testing.assert_array_almost_equal(
            transformed.numpy(), basis_derivative.numpy()
        )

    def test_transform_basis_derivative_non_identity_metric(self):
        """Test basis function derivative transformation with non-identity metric."""
        # Create basis derivatives
        basis_derivative = mi.TensorXf(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[2.0, 0.0], [0.0, 2.0], [1.0, 2.0]]]
        )

        # Create non-identity inverse metric
        inv_metric = mi.TensorXf([[[2.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 2.0]]])

        transformed = IGAManifoldProblem._transform_basis_derivative(
            basis_derivative=basis_derivative,
            inv_metric=inv_metric,
        )

        # Result should be [2*1 + 0*0, 0*1 + 1*0] = [2, 0]
        np.testing.assert_array_almost_equal(transformed[0, 0].numpy(), [2.0, 0.0])

        # Result should be [1*1 + 1*2, 0*1 + 2*2] = [3, 4]
        np.testing.assert_array_almost_equal(transformed[1, 2].numpy(), [3.0, 4.0])

    def test_local_matrix_scatter_indices(self):
        """Test generation of scatter indices for local matrix assembly."""
        num_knot_spans = 3
        num_dofs = 4
        offset = 1

        indices = IGAManifoldProblem._local_matrix_scatter_indices(
            num_knot_spans=num_knot_spans, num_dofs=num_dofs, offset=offset
        )

        # Check shape
        expected_length = num_knot_spans * num_dofs
        self.assertEqual(indices.shape, (expected_length,))

        # Check that indices are computed correctly
        expected_indices = []
        for span in range(num_knot_spans):
            base_idx = span * num_dofs * num_dofs + offset * num_dofs
            for dof in range(num_dofs):
                expected_indices.append(base_idx + dof)

        np.testing.assert_array_equal(indices.numpy(), np.array(expected_indices))

        # Manual tests for two specific cases
        indices = IGAManifoldProblem._local_matrix_scatter_indices(
            num_knot_spans=2, num_dofs=3, offset=0
        )
        np.testing.assert_array_equal(indices[0:3].numpy(), [0, 1, 2])
        np.testing.assert_array_equal(indices[3:6].numpy(), [9, 10, 11])

        indices = IGAManifoldProblem._local_matrix_scatter_indices(
            num_knot_spans=2, num_dofs=3, offset=2
        )
        np.testing.assert_array_equal(indices[0:3].numpy(), [6, 7, 8])
        np.testing.assert_array_equal(indices[3:6].numpy(), [15, 16, 17])

    def test_local_matrix_scatter_indices_invalid(self):
        """Test error handling for invalid parameters in scatter indices generation."""
        # Test non-positive num_knot_spans
        with self.assertRaises(ValueError):
            IGAManifoldProblem._local_matrix_scatter_indices(
                num_knot_spans=0, num_dofs=4, offset=1
            )

        # Test non-positive num_dofs
        with self.assertRaises(ValueError):
            IGAManifoldProblem._local_matrix_scatter_indices(
                num_knot_spans=3, num_dofs=0, offset=1
            )

        # Test negative offset
        with self.assertRaises(ValueError):
            IGAManifoldProblem._local_matrix_scatter_indices(
                num_knot_spans=3, num_dofs=4, offset=-1
            )

        # Test offset too large
        with self.assertRaises(ValueError):
            IGAManifoldProblem._local_matrix_scatter_indices(
                num_knot_spans=3, num_dofs=4, offset=4
            )
