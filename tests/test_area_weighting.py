import unittest

import numpy as np

from rebel_shape.scripts.end_to_end.modal.area_weighting import (
    mode_column,
    vertex_areas,
)


class TestAreaWeighting(unittest.TestCase):
    def test_two_triangles_forming_unit_square_sum_to_one(self) -> None:
        """The lumped vertex areas of a unit square (two triangles) sum to 1.0."""
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        self.assertAlmostEqual(
            float(vertex_areas(vertices, faces).sum()), 1.0, places=12
        )

    def test_single_triangle_splits_its_area_evenly_across_its_vertices(self) -> None:
        """
        A 3-4-5 right triangle has area 6, so each of its three vertices,
        touched by exactly one face, must receive exactly 6 / 3 = 2.
        """
        vertices = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        np.testing.assert_allclose(
            vertex_areas(vertices, faces), [2.0, 2.0, 2.0], rtol=0.0, atol=1e-12
        )

    def test_degenerate_triangle_contributes_no_area(self) -> None:
        """Three collinear points span zero area, so every vertex gets zero."""
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        np.testing.assert_array_equal(vertex_areas(vertices, faces), np.zeros(3))

    def test_self_column_has_unit_norm(self) -> None:
        """`mode_column` normalises under the area-weighted inner product."""
        mode = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
        weights = np.array([1.0, 1.0])
        self.assertAlmostEqual(
            float(np.linalg.norm(mode_column(mode, weights))), 1.0, places=12
        )

    def test_column_applies_the_per_vertex_weight(self) -> None:
        """
        Two vertices sharing the same displacement value must not end up with
        the same weighted contribution: with weights `[1, 3]`, the second
        vertex's share of the normalised column is three times the first's,
        against a hand-computed `[1, 0, 0, 3, 0, 0] / sqrt(10)`.
        """
        mode = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        weights = np.array([1.0, 3.0])
        expected = np.array([1.0, 0.0, 0.0, 3.0, 0.0, 0.0]) / np.sqrt(10.0)
        np.testing.assert_allclose(
            mode_column(mode, weights), expected, rtol=0.0, atol=1e-12
        )

    def test_column_is_invariant_to_scaling_the_mode(self) -> None:
        """
        A normalised column is unchanged by a positive rescaling of the mode, and
        flips sign under a negative one, the property MAC relies on, since an
        eigenvector is only ever defined up to an arbitrary scale.
        """
        mode = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
        weights = np.array([1.0, 1.0])
        column = mode_column(mode, weights)
        np.testing.assert_allclose(
            mode_column(2.5 * mode, weights), column, rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            mode_column(-mode, weights), -column, rtol=0.0, atol=1e-12
        )

    def test_zero_mode_normalises_to_zero_not_nan(self) -> None:
        """An all-zero mode has no direction; the norm floor keeps it finite."""
        mode = np.zeros((2, 3))
        weights = np.array([1.0, 1.0])
        np.testing.assert_array_equal(mode_column(mode, weights), np.zeros(6))

    def test_sample_gathers_before_normalising(self) -> None:
        """
        Passing `sample` must equal gathering the mode at those indices first and
        then normalising: the identity the cross-mesh comparison relies on.
        """
        mode = np.arange(15.0).reshape(5, 3)
        sample = np.array([3, 1, 4])
        weights = np.array([1.0, 2.0, 0.5])
        np.testing.assert_allclose(
            mode_column(mode, weights, sample),
            mode_column(mode[sample], weights, None),
            rtol=0.0,
            atol=1e-12,
        )
