try:
    import open3d  # noqa: F401

    OPEN3D_IMPORTED = True
except ImportError:
    OPEN3D_IMPORTED = False

import unittest

import numpy as np

from rebel_shape.scripts.end_to_end.heat.heat_field_metrics_runner import (
    HeatFieldMetricsRunner,
)


class TestHeatFieldMetricsRunner(unittest.TestCase):
    def test_rel_l2_matches_hand_computed_value(self) -> None:
        """
        Area-weighted relative L2 of `diff = [1, 0, 1]` against
        `ref = [1, 2, 0]` under areas `[1, 1, 2]`:
        `sqrt(1*1+1*0+2*1) / sqrt(1*1+1*4+2*0) = sqrt(3) / sqrt(5)`.
        """
        diff = np.array([1.0, 0.0, 1.0])
        ref = np.array([1.0, 2.0, 0.0])
        areas = np.array([1.0, 1.0, 2.0])
        self.assertAlmostEqual(
            HeatFieldMetricsRunner._rel_l2(diff, ref, areas),
            np.sqrt(3.0) / np.sqrt(5.0),
            places=12,
        )

    def test_rel_l2_is_zero_when_the_reference_norm_vanishes(self) -> None:
        """
        A zero reference field makes the relative error undefined; the
        implementation resolves this as 0 rather than raising or returning NaN.
        """
        diff = np.array([1.0, 0.0, 1.0])
        areas = np.array([1.0, 1.0, 2.0])
        self.assertEqual(
            HeatFieldMetricsRunner._rel_l2(diff, np.zeros(3), areas), 0.0
        )

    @unittest.skipUnless(OPEN3D_IMPORTED, "open3d is not installed.")
    def test_build_transfer_reproduces_an_affine_field_at_its_own_vertices(
        self,
    ) -> None:
        """
        Barycentric interpolation reproduces an affine function exactly, so
        transferring `f(x, y) = 2x - 3y + 1` from a unit-square mesh onto a
        permutation of its own vertices, an edge midpoint and a shared-edge
        midpoint must equal `f` evaluated directly at those query points:
        an oracle independent of the transfer's own interpolation code.
        """
        source_vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        )
        source_faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)

        def affine_field(points: np.ndarray) -> np.ndarray:
            return 2.0 * points[:, 0] - 3.0 * points[:, 1] + 1.0

        field = affine_field(source_vertices)
        query = np.array(
            [
                [1.0, 1.0, 0.0],  # vertices, permuted
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 0.0],  # midpoint of the shared diagonal edge
                [0.5, 0.0, 0.0],  # midpoint of a boundary edge
            ]
        )

        transfer, distance = HeatFieldMetricsRunner._build_transfer(
            source_vertices, source_faces, query
        )

        np.testing.assert_allclose(
            transfer(field), affine_field(query), rtol=0.0, atol=1e-5
        )
        # Every query point lies exactly on the source surface, so the
        # projection distance the transfer reports must vanish.
        np.testing.assert_allclose(distance, 0.0, rtol=0.0, atol=1e-5)
