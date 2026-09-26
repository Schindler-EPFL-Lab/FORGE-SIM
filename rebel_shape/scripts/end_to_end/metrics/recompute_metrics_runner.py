import json
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from rebel_shape.scripts.end_to_end.metrics.mesh_metrics_report import MeshMetricsReport
from rebel_shape.scripts.end_to_end.modal.area_weighting import (
    mode_column,
    vertex_areas,
)
from rebel_shape.scripts.end_to_end.modal.degenerate_mode_runs import (
    degenerate_mode_runs,
)
from rebel_shape.scripts.end_to_end.modal.eigenmodes import Eigenmodes
from rebel_shape.scripts.end_to_end.modal.modal_comparison_report import (
    ModalComparisonReport,
)
from rebel_shape.scripts.end_to_end.modal.modal_frequencies import RIGID_BODY_FREQ_HZ


class RecomputeMetricsRunner:
    """
    The numerics behind every metric `recompute_metrics` reports: `modal_mac`,
    `modal_parity`, `heat_lambda1`, `geometry_error` and `spearman`. Each is
    independent: `_metrics_for_mesh` calls the first four per mesh, and
    `write_reports` calls `spearman` afterward to correlate the results.
    """

    @staticmethod
    def modal_mac(
        fem_h5: Path,
        iga_h5: Path,
        fem_freq: npt.NDArray[np.float64],
        iga_freq: npt.NDArray[np.float64],
        cluster_gap: float,
    ) -> MeshMetricsReport.ModalMac:
        """
        M1: area-weighted MAC between FEM-GT and IGA-spline elastic mode shapes.

        MAC(i,j) = |phi_i^T M psi_j|^2 / [(phi_i^T M phi_i)(psi_j^T M psi_j)] with M the
        lumped (diagonal vertex-area) mass, exactly the area-weighted normalized
        displacement correlation `align_modal_xdmf` already builds, squared, and from
        the same `area_weighting.mode_column` so the two cannot drift apart. IGA modes
        are resampled to the FEM vertices by nearest neighbour (surfaces coincide to
        <1%). Only the displacement DOFs enter (translational MAC, the shell
        convention).

        Reported:
          * `diagonal_mac`: index-paired (mode i vs i) shape agreement.
          * `best_mac`: each FEM mode's best MAC over all IGA modes, i.e.
            whether the shape is recovered at all, regardless of which IGA mode
            carries it. High best-MAC with low diagonal is a reordering, not lost
            shape.
          * per near-degenerate cluster (consecutive modes within `cluster_gap` of
            each other), `subspace_overlap` = mean cos^2 of the principal angles
            between the FEM and IGA cluster subspaces (M4): 1 = subspaces coincide,
            even when the individual within-cluster vectors are arbitrarily mixed.

        NB a full-spectrum Hungarian re-pairing is deliberately not used: on weakly
        correlated clusters it shuffles modes across the spectrum (see
        align_modal_xdmf).
        """
        fem = Eigenmodes.from_h5(fem_h5)
        iga = Eigenmodes.from_h5(iga_h5)
        n = min(fem.num_modes, iga.num_modes)
        fem_v = fem.vertices
        nn = cKDTree(iga.vertices).query(fem_v, k=1)[1]
        w = np.sqrt(np.clip(vertex_areas(fem_v, fem.faces), 0, None))

        ff = np.stack([mode_column(fem.displacement[m], w) for m in range(n)], axis=1)
        ii = np.stack(
            [mode_column(iga.displacement[m], w, nn) for m in range(n)], axis=1
        )
        mac_full = (ff.T @ ii) ** 2  # (n,n): rows FEM, cols IGA

        fem_el = np.where(fem_freq[:n] > RIGID_BODY_FREQ_HZ)[0]
        iga_el = np.where(iga_freq[:n] > RIGID_BODY_FREQ_HZ)[0]
        m = min(len(fem_el), len(iga_el))
        fem_el, iga_el = fem_el[:m], iga_el[:m]
        mac = mac_full[np.ix_(fem_el, iga_el)]  # (m,m) elastic MAC
        fe = fem_freq[fem_el]

        diag = np.diag(mac)
        best_j = mac.argmax(axis=1)
        best = mac[np.arange(m), best_j]

        clusters = []
        for run in degenerate_mode_runs(cluster_gap, fe):
            sub = mac[np.ix_(run, run)]
            clusters.append(
                MeshMetricsReport.Cluster(
                    fem_modes=[int(r + 1) for r in run],
                    size=len(run),
                    fem_hz=[float(fe[r]) for r in run],
                    subspace_overlap=float(sub.sum() / len(run)),
                    diag_mac=[float(mac[r, r]) for r in run],
                )
            )
        multi = [c for c in clusters if c.size > 1]

        return MeshMetricsReport.ModalMac(
            n_elastic=int(m),
            mac_matrix=mac.tolist(),
            diagonal_mac=diag.tolist(),
            mean_diagonal_mac=float(diag.mean()),
            best_mac=best.tolist(),
            best_partner_iga_mode=[int(j + 1) for j in best_j],
            mean_best_mac=float(best.mean()),
            n_best_mac_ge_0p5=int((best >= 0.5).sum()),
            n_reordered=int((best_j != np.arange(m)).sum()),
            n_clusters=len(clusters),
            n_multi_mode_clusters=len(multi),
            clusters=clusters,
            mean_cluster_subspace_overlap=float(
                np.mean([c.subspace_overlap for c in clusters])
            ),
        )

    @staticmethod
    def modal_parity(modal_json: Path) -> MeshMetricsReport.ModalParity:
        """M2: parity fit f_IGA vs f_FEM: slope+R^2, and the through-origin scale."""
        modes = ModalComparisonReport.from_json(modal_json).modes
        f = np.array([mode.fem_gt_hz for mode in modes])
        g = np.array([mode.iga_spline_hz for mode in modes])
        a, b = np.polyfit(f, g, 1)
        ss_res = float(np.sum((g - (a * f + b)) ** 2))
        ss_tot = float(np.sum((g - g.mean()) ** 2))
        return MeshMetricsReport.ModalParity(
            slope=float(a),
            intercept_hz=float(b),
            r_squared=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            scale_through_origin=float((f @ g) / (f @ f)),  # slope through the origin
            n_modes=len(modes),
        )

    @staticmethod
    def heat_lambda1(iga_log: Path, fem_log: Path) -> MeshMetricsReport.HeatEigenvalue:
        """H1(k=1): first heat eigenvalue lambda_1 = 1/T_relax on both geometries."""
        t_iga = float(json.loads(iga_log.read_text())["relaxation_time"])
        t_fem = float(json.loads(fem_log.read_text())["relaxation_time"])
        l_iga, l_fem = 1.0 / t_iga, 1.0 / t_fem
        return MeshMetricsReport.HeatEigenvalue(
            T_relax_fem_gt_s=t_fem,
            T_relax_iga_spline_s=t_iga,
            lambda1_fem_gt=l_fem,
            lambda1_iga_spline=l_iga,
            rel_error=abs(l_iga - l_fem) / l_fem,
        )

    @staticmethod
    def geometry_error(
        fem_h5: Path, iga_h5: Path, sample_points: int, sample_seed: int
    ) -> MeshMetricsReport.Geometry:
        """
        Sec.4: surface distance between the co-registered GT (FEM) and recovered
        (IGA) analysis meshes, normalised by the GT bbox diagonal (-> % of object
        size).

        Both surfaces are drawn at `sample_points` uniform samples, and Open3D's
        global RNG is seeded with `sample_seed` first, so the sampled metrics are
        reproducible rather than re-drawn on every run.
        """
        # Deferred: open3d is a heavy optional dependency, absent from the plain venv
        # install and only needed by this metric.
        import open3d as o3d

        def load(h5: Path) -> "o3d.t.geometry.TriangleMesh":
            """Read an exported analysis surface into an Open3D tensor mesh."""
            with h5py.File(h5, "r") as f:
                v = np.asarray(f["/Mesh/vertices"], dtype=np.float32)
                t = np.asarray(f["/Mesh/topology"], dtype=np.int32)
            return o3d.t.geometry.TriangleMesh(o3d.core.Tensor(v), o3d.core.Tensor(t))

        o3d.utility.random.seed(sample_seed)
        gt = load(fem_h5)  # FEM ground-truth
        rec = load(iga_h5)  # IGA recovered
        gt_v = gt.vertex.positions.numpy()
        diag = float(np.linalg.norm(gt_v.max(0) - gt_v.min(0)))

        mp = o3d.t.geometry.MetricParameters(n_sampled_points=sample_points)
        ch, hd = gt.compute_metrics(
            rec,
            (o3d.t.geometry.Metric.ChamferDistance,
             o3d.t.geometry.Metric.HausdorffDistance),
            mp,
        ).numpy()

        # Directed recovered->GT surface distance (mean/RMS) via raycasting.
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(gt)
        rec_pts = rec.sample_points_uniformly(sample_points).point.positions
        d = scene.compute_distance(rec_pts).numpy()

        def pct(x: float) -> float:
            """A distance as a percentage of the ground-truth bounding-box diagonal."""
            return 100.0 * float(x) / diag

        return MeshMetricsReport.Geometry(
            bbox_diag=diag,
            chamfer=float(ch),
            hausdorff=float(hd),
            chamfer_pct_diag=pct(ch),
            hausdorff_pct_diag=pct(hd),
            mean_surf_dist_pct_diag=pct(d.mean()),
            rms_surf_dist_pct_diag=pct(np.sqrt((d ** 2).mean())),
            area_ratio_rec_over_gt=float(
                rec.to_legacy().get_surface_area() / gt.to_legacy().get_surface_area()
            ),
        )

    @staticmethod
    def spearman(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
        """
        Spearman rank correlation, with ties averaged per SciPy's own definition.

        :returns: The correlation coefficient, or NaN where either input is constant
            (SciPy's own NaN in that case).
        """
        statistic = spearmanr(x, y).statistic
        return float(statistic)
