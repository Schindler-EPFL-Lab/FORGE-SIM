import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from fem.end_to_end.mesh_quality.mesh_quality_report import MeshQualityReport
from fem.end_to_end.mesh_quality.triangle_quality import TriangleQuality
from fem.meshprep.mesh_repair import clean_mesh
from rebel_shape import logger
from rebel_shape.scripts.end_to_end.end_to_end_config import EndToEndConfig


class MeshQualityRunner:
    """
    Assesses triangle quality on the source mesh, remeshes it isotropically for
    quality, and reassesses: the before/after pair `mesh_quality_report.json`
    is built from, proving the remesh improves the triangulation for downstream
    finite-element analysis.

    Holds the configuration `_remesh` and the report both read. `run` is the
    entry point a caller reaches for.
    """

    @dataclass(frozen=True, kw_only=True)
    class RemeshedMesh:
        """The dedupped, isotropically-remeshed mesh `_remesh` produces from
        the source triangulation."""

        vertices: npt.NDArray[np.float64]
        """Vertex positions, shape (n, 3)."""
        faces: npt.NDArray[np.int64]
        """Triangle vertex indices, shape (m, 3)."""

    _REMESH_ITERATIONS: int = 20
    """
    Botsch-Kobbelt smoothing passes over the source triangulation.

    The remesher and the report it produces must be told the same number, or the
    report describes a mesh that was never built, so both read it from here.
    """

    def __init__(self, config: EndToEndConfig) -> None:
        """Stores the configuration the assessment reads from; does no I/O."""
        self._config = config

    def run(self) -> MeshQualityReport:
        """Assesses quality, remeshes, reassesses; writes report + histograms + PLY."""
        import gpytoolbox as gpy  # heavy; deferred to the two methods that call it

        logger.info(f"[step 1] Loading source mesh: {self._config.mesh_path}")
        vertices, faces = gpy.read_mesh(str(self._config.mesh_path))

        before_quality = TriangleQuality.from_mesh(vertices, faces)
        before = self._log_triangulation(
            before_quality, vertices, faces, label="before"
        )

        remeshed = self._remesh(vertices, faces)
        after_quality = TriangleQuality.from_mesh(remeshed.vertices, remeshed.faces)
        after = self._log_triangulation(
            after_quality, remeshed.vertices, remeshed.faces, label="after"
        )

        gpy.write_mesh(
            str(self._config.remeshed_mesh_path), remeshed.vertices, remeshed.faces
        )
        self._histograms(
            before_quality,
            after_quality,
            self._config.mesh_quality_dir / "quality_histograms.png",
        )
        return self._write_report(before, after)

    def _remesh(
        self, vertices: npt.NDArray[np.float64], faces: npt.NDArray[np.int64]
    ) -> RemeshedMesh:
        """
        Dedups the source (manifold-preserving) and isotropically remeshes it to
        `_config.remesh_target_edge` via Botsch-Kobbelt.
        """
        import gpytoolbox as gpy  # heavy; deferred to the two methods that call it

        logger.info(
            f"[step 1] Remeshing (Botsch-Kobbelt, target edge "
            f"{self._config.remesh_target_edge})"
        )
        # Dedup the source first (manifold-preserving): removes the duplicate/degenerate
        # elements that otherwise make Botsch emit slivers. merge_eps=0 -> no welding.
        cleaned = clean_mesh(vertices, faces, merge_eps=0.0)
        rv, rf = gpy.remesh_botsch(
            cleaned.vertices.astype(np.float64),
            cleaned.faces.astype(np.int32),
            i=self._REMESH_ITERATIONS,
            h=float(self._config.remesh_target_edge),
            project=True,
        )
        remeshed = clean_mesh(rv, rf, merge_eps=0.0)
        return MeshQualityRunner.RemeshedMesh(
            vertices=remeshed.vertices, faces=remeshed.faces.astype(np.int64)
        )

    @staticmethod
    def _log_triangulation(
        quality: TriangleQuality,
        vertices: npt.NDArray[np.float64],
        faces: npt.NDArray[np.int64],
        *,
        label: str,
    ) -> MeshQualityReport.Triangulation:
        """Reduces `quality` to a `Triangulation` summary and logs it under `label`."""
        topology = MeshQualityReport.Topology.from_mesh(vertices, faces)
        triangulation = MeshQualityReport.Triangulation.from_quality(quality, topology)
        skew = triangulation.skewness
        aspect = triangulation.aspect_ratio
        edge = triangulation.edge_length
        logger.info(
            f"[step 1] {label}: {triangulation.num_triangles} tris | "
            f"skew {skew.mean:.3f}/{skew.max:.3f} | "
            f"aspect {aspect.mean:.2f}/{aspect.max:.2f} | "
            f"edge [{edge.min:.4g}, {edge.max:.4g}]"
        )
        return triangulation

    def _write_report(
        self,
        before: MeshQualityReport.Triangulation,
        after: MeshQualityReport.Triangulation,
    ) -> MeshQualityReport:
        """Builds the before/after report and writes it beside the remeshed mesh."""
        report = MeshQualityReport(
            mesh_path=self._config.mesh_path,
            remeshed_path=self._config.remeshed_mesh_path,
            remesh=MeshQualityReport.Remesh(
                method=MeshQualityReport.Remesh.Method.BOTSCH,
                target_edge=self._config.remesh_target_edge,
                iterations=self._REMESH_ITERATIONS,
            ),
            before=before,
            after=after,
        )
        # `default=str` is the one remaining conversion to text `to_json_dict`
        # leaves for `json.dumps`: `mesh_path` and `remeshed_path` are `Path`,
        # and JSON has no path type.
        (self._config.mesh_quality_dir / "mesh_quality_report.json").write_text(
            json.dumps(report.to_json_dict(), indent=2, default=str)
        )
        logger.info(
            f"[step 1] Wrote report + remeshed mesh to {self._config.mesh_quality_dir}"
        )
        return report

    @staticmethod
    def _histograms(
        before: TriangleQuality, after: TriangleQuality, out_path: Path
    ) -> None:
        """Overlaid before/after histograms for skewness and aspect ratio."""
        import matplotlib  # heavy; deferred to the one call site that needs a plot

        matplotlib.use("Agg")  # headless backend; the runner has no display
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, before_values, after_values, xlabel, title, xmax in (
            (
                axes[0],
                before.skewness,
                after.skewness,
                "skewness",
                "Equiangle skewness (0 = ideal)",
                1.0,
            ),
            (
                axes[1],
                before.aspect_ratio,
                after.aspect_ratio,
                "aspect_ratio",
                "Aspect ratio R/2r (1 = ideal)",
                8.0,
            ),
        ):
            ax.hist(
                np.clip(before_values, 0, xmax),
                bins=60,
                alpha=0.6,
                label="before",
                color="tab:red",
                density=True,
            )
            ax.hist(
                np.clip(after_values, 0, xmax),
                bins=60,
                alpha=0.6,
                label="after",
                color="tab:green",
                density=True,
            )
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("density")
            ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
