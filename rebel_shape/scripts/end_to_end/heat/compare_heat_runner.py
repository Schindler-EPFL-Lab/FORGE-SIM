from pathlib import Path

import numpy as np
import numpy.typing as npt


class CompareHeatRunner:
    """Diagnostic support for `compare_heat`'s peak-gradient plot panel."""

    _DEGENERATE_AREA: float = 1e-30
    """
    Floor on a facet's doubled area, below which its normal is left unnormalised so a
    degenerate triangle cannot divide by zero. Far below any real facet's area on these
    meshes, whose target edge lengths are ~1e-2 in model units.
    """

    @classmethod
    def peak_gradient_series(
        cls, h5_path: Path
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]] | None:
        """
        Peak surface |grad T| per snapshot from an IGA-format .h5 (best effort).

        :returns: `(snapshot_index, peak_gradient)`, or None when the file or the HDF5
            reader is unavailable; the caller then omits that plot panel.
        """
        # Deferred: h5py is optional for this panel on purpose: it is a diagnostic
        # extra, so its absence must degrade the plot rather than fail the comparison.
        try:
            import h5py
        except ImportError:
            return None
        if not h5_path.exists():
            return None
        with h5py.File(h5_path, "r") as f:
            verts = np.asarray(f["Mesh"]["Vertices"], dtype=float)
            topo = np.asarray(f["Mesh"]["Topology"], dtype=np.int64)
            steps = sorted(
                (k for k in f["Function"].keys() if k.startswith("Step_")),
                key=lambda s: int(s.split("_")[1]),
            )
            temps = [
                np.asarray(f["Function"][s], dtype=float).reshape(-1) for s in steps
            ]

        p0, p1, p2 = verts[topo[:, 0]], verts[topo[:, 1]], verts[topo[:, 2]]
        e1, e2 = p1 - p0, p2 - p0
        normal = np.cross(e1, e2)
        area2 = np.linalg.norm(normal, axis=1, keepdims=True)
        normal = normal / np.maximum(area2, cls._DEGENERATE_AREA)

        peaks = []
        for t in temps:
            t0, t1, t2 = t[topo[:, 0]], t[topo[:, 1]], t[topo[:, 2]]
            # Solve grad g: g.e1=t1-t0, g.e2=t2-t0, g.n=0  (per triangle)
            a = np.stack([e1, e2, normal], axis=1)  # (m,3,3)
            rhs = np.stack([t1 - t0, t2 - t0, np.zeros_like(t0)], axis=1)  # (m,3)
            try:
                g = np.linalg.solve(a, rhs[:, :, None])[:, :, 0]
                peaks.append(float(np.max(np.linalg.norm(g, axis=1))))
            except np.linalg.LinAlgError:
                peaks.append(float("nan"))
        return np.arange(len(peaks)), np.array(peaks)
