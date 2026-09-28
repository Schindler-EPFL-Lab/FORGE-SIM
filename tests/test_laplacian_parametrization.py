import unittest

import drjit as dr
import mitsuba as mi
import numpy as np

from forge_sim.property_optimizers.laplacian_parametrization import (
    LaplacianParametrization,
)


class TestLaplacianParametrization(unittest.TestCase):
    def setUp(self):
        mi.set_variant("llvm_ad_rgb")

    def test_initialization(self):
        """
        Tests whether the Laplacian parametrization is initialized correctly.
        """
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
        )
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)

        parametrization = LaplacianParametrization(
            nodes=mi.TensorXf(vertices),
            faces=mi.TensorXu(faces),
            smoothing_strength=2.0,
        )

        self.assertEqual(isinstance(parametrization, LaplacianParametrization), True)
        self.assertEqual(parametrization._unique_faces.shape, (2, 3))
        self.assertEqual(parametrization._node_inverse_mapping.shape, (4,))
        self.assertEqual(parametrization._node_index.shape, (4,))

        self.assertEqual(parametrization._ls_cols.shape, (14,))
        self.assertEqual(parametrization._ls_rows.shape, (14,))
        self.assertEqual(parametrization._ls_values.shape, (14, 1))

    def test_large_steps_laplacian(self):
        """
        Tests the Large Steps Laplacian parametrization
        """
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
        )
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)

        indices, values = LaplacianParametrization.large_steps_laplacian(
            num_nodes=len(vertices), faces=faces, smoothing_strength=2.0
        )

        matrix = np.zeros((4, 4), dtype=np.float32)
        for i, j, value in zip(indices[0], indices[1], values):
            matrix[i, j] += value

        self.assertEqual(indices.shape[0], 2)
        # 3 edges = 2 indices per edge, plus 3 diagonal entries, each vertex
        # connected to two others + 4 diagonal entries
        self.assertEqual(indices.shape[1], 24)

        np.testing.assert_array_almost_equal(
            matrix,
            np.array(
                [[5, -2, -2, 0], [-2, 7, -2, -2], [-2, -2, 7, -2], [0, -2, -2, 5]],
                dtype=np.float32,
            ),
        )

    def test_to_parametric(self):
        """
        Tests the conversion from Laplacian parametrization to parametric space.

        Unfortunately, the conversion from the Laplacian parametrization to the
        vertex positions cannot be tested in `scalar_rgb` mode, because it results in
        a segfault.
        """
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
        )
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)

        parametrization = LaplacianParametrization(
            nodes=mi.TensorXf(vertices),
            faces=mi.TensorXu(faces),
            smoothing_strength=2.0,
        )

        reference_matrix = np.array(
            [[5, -2, -2, 0], [-2, 7, -2, -2], [-2, -2, 7, -2], [0, -2, -2, 5]],
            dtype=np.float32,
        )

        parametric = parametrization.to_parametric(dr.ravel(mi.TensorXf(vertices)))

        # Do not forget to apply the inverse mapping to match the node indexing
        reference_parametric = (
            reference_matrix @ vertices[parametrization._node_inverse_mapping]
        ).flatten()

        np.testing.assert_array_almost_equal(parametric.numpy(), reference_parametric)
