try:
    import open3d  # noqa: F401

    OPEN3D_IMPORTED = True
except ImportError:
    OPEN3D_IMPORTED = False

import json
import math
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from forge_sim.scripts.end_to_end.metrics.recompute_metrics_runner import (
    RecomputeMetricsRunner,
)
from forge_sim.scripts.end_to_end.modal.modal_frequencies import RIGID_BODY_FREQ_HZ


class TestRecomputeMetricsRunner(unittest.TestCase):
    def test_tied_ranks_are_averaged(self) -> None:
        """
        A tied value in `x` must not read as perfect correlation.

        `x`'s two middle entries are equal, so their averaged rank is 2.5 each;
        `y` carries no ties, so its ranks are its own values. The Pearson
        correlation of `[1, 2.5, 2.5, 4]` against `[1, 2, 3, 4]` is
        `4.5 / sqrt(4.5 * 5.0) = 0.94868...`, not the 1.0 a dense (untied)
        ranking of `x` as `[1, 2, 3, 4]` would report.
        """
        rho = RecomputeMetricsRunner.spearman(
            x=[1.0, 2.0, 2.0, 3.0], y=[1.0, 2.0, 3.0, 4.0]
        )
        self.assertAlmostEqual(rho, 0.9486832980505138, places=12)
        self.assertNotAlmostEqual(rho, 1.0, places=2)

    def test_tie_free_series_matches_dense_ranking(self) -> None:
        """
        With no ties, tie-averaging and dense ranking agree: a monotone series
        against itself is perfectly correlated, and against its reverse is
        perfectly anti-correlated.
        """
        x = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(
            RecomputeMetricsRunner.spearman(x=x, y=x), 1.0, places=12
        )
        self.assertAlmostEqual(
            RecomputeMetricsRunner.spearman(x=x, y=list(reversed(x))),
            -1.0,
            places=12,
        )

    def test_constant_series_is_undefined(self) -> None:
        """A constant series has no rank variance, so the correlation is NaN."""
        with self.assertWarns(Warning):
            rho = RecomputeMetricsRunner.spearman(
                x=[1.0, 1.0, 1.0], y=[1.0, 2.0, 3.0]
            )
        self.assertTrue(math.isnan(rho))

    @staticmethod
    def _write_eigenmodes_h5(
        path: Path,
        vertices: np.ndarray,
        faces: np.ndarray,
        displacement: np.ndarray,
    ) -> None:
        """
        Writes the `Eigenmodes.from_h5` layout with a zero rotation field, which
        `modal_mac` never reads.
        """
        with h5py.File(path, "w") as h5_file:
            h5_file.create_dataset("/Mesh/vertices", data=vertices.astype(np.float64))
            h5_file.create_dataset("/Mesh/topology", data=faces.astype(np.int64))
            for mode_index in range(displacement.shape[0]):
                mode = displacement[mode_index]
                h5_file.create_dataset(
                    f"/Modes/mode_{mode_index}/displacement", data=mode
                )
                h5_file.create_dataset(
                    f"/Modes/mode_{mode_index}/rotation", data=np.zeros_like(mode)
                )

    @staticmethod
    def _write_surface_h5(
        path: Path, vertices: np.ndarray, faces: np.ndarray
    ) -> None:
        """Writes the `/Mesh` vertex/topology layout `geometry_error` reads."""
        with h5py.File(path, "w") as h5_file:
            h5_file.create_dataset("/Mesh/vertices", data=vertices.astype(np.float32))
            h5_file.create_dataset("/Mesh/topology", data=faces.astype(np.int32))

    def test_modal_mac_of_identical_modes_is_one_and_orthogonal_modes_is_zero(
        self,
    ) -> None:
        """
        On a single triangle every vertex carries the same lumped area, so
        `mode_column` reduces to plain L2 normalisation and two axis-aligned unit
        displacement modes are orthonormal by construction: MAC(i, i) = 1,
        MAC(i, j) = 0 for i != j, independent of the area-weighting machinery
        `modal_mac` layers on top.
        """
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        mode_x = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        mode_y = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        displacement = np.stack([mode_x, mode_y])
        freq = np.array([10.0, 20.0])
        self.assertGreater(freq.min(), RIGID_BODY_FREQ_HZ)

        with tempfile.TemporaryDirectory() as directory:
            fem_h5 = Path(directory) / "fem.h5"
            iga_h5 = Path(directory) / "iga.h5"
            self._write_eigenmodes_h5(fem_h5, vertices, faces, displacement)
            self._write_eigenmodes_h5(iga_h5, vertices, faces, displacement)
            result = RecomputeMetricsRunner.modal_mac(
                fem_h5, iga_h5, fem_freq=freq, iga_freq=freq, cluster_gap=0.01
            )

        np.testing.assert_allclose(result.mac_matrix, np.eye(2), rtol=0.0, atol=1e-10)
        self.assertEqual(result.n_reordered, 0)

    def test_modal_mac_reports_reordering_when_shapes_swap(self) -> None:
        """
        Swapping the two IGA modes relative to their FEM counterparts must
        collapse the diagonal MAC to 0 while `best_mac` stays 1: the shape is
        still recovered, only under the other mode index.
        """
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        mode_x = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        mode_y = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        freq = np.array([10.0, 20.0])

        with tempfile.TemporaryDirectory() as directory:
            fem_h5 = Path(directory) / "fem.h5"
            iga_h5 = Path(directory) / "iga.h5"
            self._write_eigenmodes_h5(
                fem_h5, vertices, faces, np.stack([mode_x, mode_y])
            )
            self._write_eigenmodes_h5(
                iga_h5, vertices, faces, np.stack([mode_y, mode_x])
            )
            result = RecomputeMetricsRunner.modal_mac(
                fem_h5, iga_h5, fem_freq=freq, iga_freq=freq, cluster_gap=0.01
            )

        np.testing.assert_allclose(result.diagonal_mac, [0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(result.best_mac, [1.0, 1.0], atol=1e-10)
        self.assertEqual(result.best_partner_iga_mode, [2, 1])
        self.assertEqual(result.n_reordered, 2)

    def test_modal_mac_is_invariant_to_a_sign_flip(self) -> None:
        """
        An eigenvector's sign is arbitrary, so a FEM mode and its exact negation
        on the IGA side must still report MAC = 1, not -1: the correlation is
        squared, not signed.
        """
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        mode_x = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        freq = np.array([10.0, 20.0])

        with tempfile.TemporaryDirectory() as directory:
            fem_h5 = Path(directory) / "fem.h5"
            iga_h5 = Path(directory) / "iga.h5"
            self._write_eigenmodes_h5(
                fem_h5, vertices, faces, np.stack([mode_x, mode_x])
            )
            self._write_eigenmodes_h5(
                iga_h5, vertices, faces, np.stack([-mode_x, mode_x])
            )
            result = RecomputeMetricsRunner.modal_mac(
                fem_h5, iga_h5, fem_freq=freq, iga_freq=freq, cluster_gap=0.01
            )

        np.testing.assert_allclose(result.diagonal_mac, [1.0, 1.0], atol=1e-10)

    def test_modal_parity_fit_matches_hand_derived_affine_line(self) -> None:
        """
        `iga_spline_hz = 2 * fem_gt_hz + 5` is an exact affine relation, so the
        least-squares fit must recover slope 2, intercept 5 and R^2 = 1; the
        through-origin scale `(f.g)/(f.f)` is a different, independently
        computable quantity (11/3) that a fit sharing code with `slope` could
        not accidentally satisfy by construction.
        """
        fem_hz = [1.0, 2.0, 3.0, 4.0]
        iga_hz = [2.0 * f + 5.0 for f in fem_hz]
        modes = [
            {
                "elastic_mode": i + 1,
                "fem_gt_hz": f,
                "iga_spline_hz": g,
                "rel_error": abs(g - f) / f,
            }
            for i, (f, g) in enumerate(zip(fem_hz, iga_hz))
        ]
        report = {
            "num_rb_modes_fem": 6,
            "num_rb_modes_iga": 6,
            "num_compared_elastic_modes": len(modes),
            "mean_rel_error": 0.0,
            "median_rel_error": 0.0,
            "max_rel_error": 0.0,
            "modes": modes,
        }
        with tempfile.TemporaryDirectory() as directory:
            modal_json = Path(directory) / "modal_comparison.json"
            modal_json.write_text(json.dumps(report))
            result = RecomputeMetricsRunner.modal_parity(modal_json)

        self.assertAlmostEqual(result.slope, 2.0, places=9)
        self.assertAlmostEqual(result.intercept_hz, 5.0, places=9)
        self.assertAlmostEqual(result.r_squared, 1.0, places=9)
        self.assertAlmostEqual(result.scale_through_origin, 110.0 / 30.0, places=9)
        self.assertEqual(result.n_modes, 4)

    @unittest.skipUnless(OPEN3D_IMPORTED, "open3d is not installed.")
    def test_geometry_error_matches_known_offset_between_parallel_planes(
        self,
    ) -> None:
        """
        Two coincident unit-square meshes offset by `dz = 0.1` along their
        shared normal have a uniform surface-to-surface distance of exactly
        `dz` everywhere, an oracle from parallel-plane geometry, not from
        Open3D's own metric code. Chamfer sums both directions (`2 * dz`);
        Hausdorff and the directed mean/RMS distances report one (`dz`); the
        bounding-box diagonal is the GT square's own diagonal, `sqrt(2)`.
        """
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        offset = vertices.copy()
        offset[:, 2] += 0.1

        with tempfile.TemporaryDirectory() as directory:
            fem_h5 = Path(directory) / "fem.h5"
            iga_h5 = Path(directory) / "iga.h5"
            self._write_surface_h5(fem_h5, vertices, faces)
            self._write_surface_h5(iga_h5, offset, faces)
            result = RecomputeMetricsRunner.geometry_error(
                fem_h5, iga_h5, sample_points=2000, sample_seed=0
            )

        diag = math.sqrt(2.0)
        # Every `places=` below leaves at least a 4x margin over the largest
        # float32/quantisation noise observed for this fixture (Open3D's
        # metrics are computed in float32); the check exists to catch a wrong
        # scale factor or a swapped direction, not to chase the noise floor.
        self.assertAlmostEqual(result.bbox_diag, diag, places=6)
        self.assertAlmostEqual(result.chamfer, 0.2, places=5)
        self.assertAlmostEqual(result.hausdorff, 0.1, places=6)
        self.assertAlmostEqual(result.chamfer_pct_diag, 100 * 0.2 / diag, places=3)
        self.assertAlmostEqual(result.hausdorff_pct_diag, 100 * 0.1 / diag, places=5)
        self.assertAlmostEqual(
            result.mean_surf_dist_pct_diag, 100 * 0.1 / diag, places=5
        )
        self.assertAlmostEqual(
            result.rms_surf_dist_pct_diag, 100 * 0.1 / diag, places=5
        )
        self.assertAlmostEqual(result.area_ratio_rec_over_gt, 1.0, places=6)

    def test_heat_lambda1_matches_hand_computed_reciprocal(self) -> None:
        """`lambda_1 = 1 / T_relax` on each side, and their relative error."""
        with tempfile.TemporaryDirectory() as directory:
            iga_log = Path(directory) / "iga_heat_log.json"
            fem_log = Path(directory) / "fem_heat_log.json"
            iga_log.write_text(json.dumps({"relaxation_time": 2.0}))
            fem_log.write_text(json.dumps({"relaxation_time": 4.0}))
            result = RecomputeMetricsRunner.heat_lambda1(iga_log, fem_log)

        self.assertAlmostEqual(result.T_relax_iga_spline_s, 2.0, places=12)
        self.assertAlmostEqual(result.T_relax_fem_gt_s, 4.0, places=12)
        self.assertAlmostEqual(result.lambda1_iga_spline, 0.5, places=12)
        self.assertAlmostEqual(result.lambda1_fem_gt, 0.25, places=12)
        self.assertAlmostEqual(result.rel_error, 1.0, places=12)
