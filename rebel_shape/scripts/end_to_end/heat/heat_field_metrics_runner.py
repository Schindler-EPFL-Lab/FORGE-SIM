import json
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

from rebel_shape.scripts.end_to_end.heat.heat_field_report import HeatFieldReport
from rebel_shape.scripts.end_to_end.heat.iga_heat_report import IGAHeatReport
from rebel_shape.scripts.end_to_end.modal.area_weighting import vertex_areas


class HeatFieldMetricsRunner:
    """
    H2 numerics: the cross-surface field discrepancy, its transfer floor, and the
    floor-corrected excess between the two.

    `heat_field_l2` is the entry point `_metrics_for_mesh` reaches for; the rest
    of the class is its supporting numerics: the raw XDMF/HDF5 read, the
    closest-point transfer, and the area-weighted relative L2.
    """

    _HOT_THRESHOLD: float = 0.5
    """
    Temperature above which a vertex counts as "hot" for the pattern-coverage fraction.

    The checkerboard field runs between `temp_cold=0` and `temp_hot=1`, so its
    midpoint separates the two plateaus wherever the transition is monotone. That
    pair is asserted rather than checked: this metric reads finished artifacts and
    the logs beside them record neither bound, so a run that moved them would be
    measured against a cut that is no longer their midpoint.
    """

    @staticmethod
    def _require_gt_fem_dt_source(iga_log: Path) -> None:
        """
        Rejects an IGA run that did not reuse the FEM-derived time step.

        `heat_field_l2` pairs FEM snapshot k with IGA snapshot k as the same
        physical time and differentiates that pairing with `np.gradient` for the
        space-time excess, valid only when both solvers marched the same dt over
        the same saved steps, which the IGA run guarantees by reusing the dt the
        FEM ground-truth run derived from its own assembled K, M
        (`IGAHeatReport.TimeStepSource.GT_FEM`). A self-normalised run instead
        resolves its own dt from its own relaxation time and lands its saved steps
        at different physical instants, so pairing by index would difference
        snapshots that are not the same time: a structural mismatch, not a
        magnitude a tolerance could absorb.

        :raises ValueError: If `iga_log` does not carry `dt_source == "gt_fem"`.
        """
        raw = json.loads(iga_log.read_text())
        dt_source = raw.get("dt_source")
        expected = IGAHeatReport.TimeStepSource.GT_FEM.value
        if dt_source != expected:
            raise ValueError(
                f"{iga_log} carries dt_source={dt_source!r}, not {expected!r}; the "
                f"space-time field comparison pairs FEM and IGA snapshots by index "
                f"on the assumption that both solvers share one physical clock, "
                f"which only holds when the IGA run reused the FEM-derived dt. "
                f"Rerun the IGA heat step with the FEM ground-truth run already in "
                f"place."
            )

    @staticmethod
    def _load_heat(
        h5_path: Path, xdmf_path: Path
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        list[float],
        list[npt.NDArray[np.float64]],
    ]:
        """Vertices, faces, times, and the per-step temperature snapshots."""
        with h5py.File(h5_path, "r") as f:
            v = np.asarray(f["Mesh/Vertices"], dtype=np.float64)
            faces = np.asarray(f["Mesh/Topology"], dtype=np.int64)
            steps = sorted(
                (k for k in f["Function"] if k.startswith("Step_")),
                key=lambda s: int(s.split("_")[1]),
            )
            snaps = [
                np.asarray(f[f"Function/{s}"], dtype=np.float64).ravel() for s in steps
            ]
        times = [
            float(t.attrib["Value"]) for t in ET.parse(xdmf_path).getroot().iter("Time")
        ]
        return v, faces, times, snaps

    @staticmethod
    def _build_transfer(
        src_v: npt.NDArray[np.float64],
        src_f: npt.NDArray[np.int64],
        query: npt.NDArray[np.float64],
    ) -> tuple[
        Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        npt.NDArray[np.float64],
    ]:
        """
        Closest-point barycentric transfer operator from a source mesh onto `query`
        points, plus the projection distance. Geometry is fixed across timesteps, so it
        is built once and reused for every field.
        """
        # Deferred: open3d is a heavy optional dependency, absent from the plain venv
        # install and only needed by this metric.
        import open3d as o3d

        legacy = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.ascontiguousarray(src_v, np.float64)),
            o3d.utility.Vector3iVector(np.ascontiguousarray(src_f, np.int32)),
        )
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
        q = o3d.core.Tensor(query.astype(np.float32))
        ans = scene.compute_closest_points(q)
        prim = ans["primitive_ids"].numpy().astype(np.int64)
        bary = ans["primitive_uvs"].numpy().astype(np.float64)
        tri = src_f[prim]
        w1, w2 = bary[:, 0], bary[:, 1]
        w0 = 1.0 - w1 - w2
        closest = ans["points"].numpy().astype(np.float64)
        dist = np.linalg.norm(query - closest, axis=1)

        def transfer(field: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            """Interpolate a per-vertex source field at each query point's footpoint."""
            f = np.asarray(field, dtype=np.float64)
            return w0 * f[tri[:, 0]] + w1 * f[tri[:, 1]] + w2 * f[tri[:, 2]]

        return transfer, dist

    @staticmethod
    def _rel_l2(
        diff: npt.NDArray[np.float64],
        ref: npt.NDArray[np.float64],
        areas: npt.NDArray[np.float64],
    ) -> float:
        """Area-weighted (lumped-mass) relative L2 over a surface."""
        num = float(np.sqrt(np.sum(areas * diff * diff)))
        den = float(np.sqrt(np.sum(areas * ref * ref)))
        return num / den if den > 0 else 0.0

    @classmethod
    def heat_field_l2(
        cls,
        *,
        mesh: str,
        fem_h5: Path,
        fem_xdmf: Path,
        iga_h5: Path,
        iga_xdmf: Path,
        iga_log: Path,
    ) -> HeatFieldReport:
        """
        Cross-surface field discrepancy, its transfer floor, and the excess between
        them.

        :param mesh: Name the report is tagged with; the arithmetic does not use it.
        :param iga_log: The IGA run's `iga_heat_log.json`, checked for
            `dt_source == "gt_fem"` before any snapshot is paired by index; see
            `_require_gt_fem_dt_source`.
        :returns: The per-step cross/floor/excess curves plus the headline scalars
            (t0 cross, final and space-time excess, hot-area fractions, projection
            distances as a percentage of the bounding-box diagonal).
        :raises ValueError: If `iga_log` does not carry `dt_source == "gt_fem"`.
        """
        cls._require_gt_fem_dt_source(iga_log)
        fv, ff, ftimes, fsnaps = cls._load_heat(fem_h5, fem_xdmf)
        iv, iff, itimes, isnaps = cls._load_heat(iga_h5, iga_xdmf)
        n = min(len(fsnaps), len(isnaps))
        times = itimes[:n]
        diag = float(np.linalg.norm(iv.max(0) - iv.min(0)))

        fem_to_iga, d_fi = cls._build_transfer(fv, ff, iv)
        iga_to_fem, d_if = cls._build_transfer(iv, iff, fv)
        iga_areas = np.clip(vertex_areas(iv, iff), 0, None)
        fem_areas = np.clip(vertex_areas(fv, ff), 0, None)

        # Hot-area fraction at t0 (pattern-coverage recovery: an inpainting-quality
        # proxy immune to edge misregistration, unlike pointwise L2).
        hot_fem = float(
            (fem_areas * (fsnaps[0] > cls._HOT_THRESHOLD)).sum() / fem_areas.sum()
        )
        hot_iga = float(
            (iga_areas * (isnaps[0] > cls._HOT_THRESHOLD)).sum() / iga_areas.sum()
        )

        cross, floor, excess = [], [], []
        for k in range(n):
            f_on_i = fem_to_iga(fsnaps[k])
            i_on_f = iga_to_fem(isnaps[k])
            # symmetric cross-field discrepancy
            c = 0.5 * (
                cls._rel_l2(isnaps[k] - f_on_i, isnaps[k], iga_areas)
                + cls._rel_l2(fsnaps[k] - i_on_f, fsnaps[k], fem_areas)
            )
            # Symmetric transfer floor: round-trip each field through the other mesh and
            # back (no recovery) -> the artifact any field of this smoothness incurs.
            fl = 0.5 * (
                cls._rel_l2(fem_to_iga(i_on_f) - isnaps[k], isnaps[k], iga_areas)
                + cls._rel_l2(iga_to_fem(f_on_i) - fsnaps[k], fsnaps[k], fem_areas)
            )
            cross.append(c)
            floor.append(fl)
            excess.append(float(np.sqrt(max(c * c - fl * fl, 0.0))))

        cross = np.array(cross)
        floor = np.array(floor)
        exc = np.array(excess)
        t = np.array(times)
        dt = np.gradient(t) if len(t) > 1 else np.array([1.0])
        st_excess = (
            float(np.sqrt(np.sum(dt * exc**2) / np.sum(dt))) if np.sum(dt) else 0.0
        )

        return HeatFieldReport(
            mesh=mesh,
            n_steps=n,
            bbox_diag=diag,
            d_proj_pct_diag=HeatFieldReport.ProjectionDistance(
                fem_to_iga_mean=100 * float(d_fi.mean()) / diag,
                fem_to_iga_max=100 * float(d_fi.max()) / diag,
                iga_to_fem_mean=100 * float(d_if.mean()) / diag,
            ),
            hot_fraction_fem=hot_fem,  # t0 pattern coverage (inpainting quality)
            hot_fraction_iga=hot_iga,
            hot_fraction_abs_error=abs(hot_fem - hot_iga),
            t0_cross_relL2=float(cross[0]),  # transfer-limited, not inpainting error
            final_excess_relL2=float(exc[-1]),  # cleanest genuine (diffused) number
            transient_mean_excess=float(exc[1:].mean()),
            spacetime_excess=st_excess,
            max_cross_relL2=float(cross.max()),
            cross_relL2_curve=cross.tolist(),
            floor_relL2_curve=floor.tolist(),
            excess_relL2_curve=exc.tolist(),
            times=times,
        )
