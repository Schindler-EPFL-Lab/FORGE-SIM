from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from fem.end_to_end.mesh_quality.triangle_quality import TriangleQuality


@dataclass(frozen=True, kw_only=True)
class MeshQualityReport:
    """
    Before/after triangle-quality and topology record for one remesh.

    Written as `mesh_quality_report.json` by `analyze_mesh_quality.py`;
    `rebel_shape/scripts/end_to_end/metrics/summarize.py` reads it back.
    """

    @dataclass(frozen=True, kw_only=True)
    class Stats:
        """One distribution of a per-face indicator, reduced to order statistics."""

        min: float
        """Smallest value over all faces."""
        max: float
        """Largest value over all faces."""
        mean: float
        """Arithmetic mean over all faces."""
        median: float
        """50th percentile over all faces."""
        p95: float
        """95th percentile over all faces."""
        p99: float
        """99th percentile over all faces."""

        @classmethod
        def from_values(
            cls, values: npt.NDArray[np.float64]
        ) -> "MeshQualityReport.Stats":
            """Reduces a per-face array to the standard order statistics."""
            return cls(
                min=float(np.min(values)),
                max=float(np.max(values)),
                mean=float(np.mean(values)),
                median=float(np.median(values)),
                p95=float(np.percentile(values, 95)),
                p99=float(np.percentile(values, 99)),
            )

    @dataclass(frozen=True, kw_only=True)
    class Extremes:
        """The range of a length- or area-valued quantity over the whole mesh."""

        min: float
        """Smallest value over the mesh."""
        max: float
        """Largest value over the mesh."""

    @dataclass(frozen=True, kw_only=True)
    class Topology:
        """Open3D's topology verdicts on one triangulation."""

        num_vertices: int
        """Vertex count."""
        num_triangles: int
        """Triangle count."""
        is_edge_manifold: bool
        """Whether every edge is shared by at most two triangles."""
        is_vertex_manifold: bool
        """Whether every vertex has a single connected triangle fan."""
        is_watertight: bool
        """Whether the surface encloses a volume with no boundary."""
        num_non_manifold_edges: int
        """Count of edges shared by more than two triangles."""

        @classmethod
        def from_mesh(
            cls, vertices: npt.NDArray[np.float64], faces: npt.NDArray[np.int64]
        ) -> "MeshQualityReport.Topology":
            """
            Checks topology via Open3D (manifoldness, watertightness, degeneracies).
            """
            # Deferred: open3d is heavy, and only this one classmethod needs it.
            # Reading a recorded report back means importing this module for its
            # record types alone, which should not pay for it.
            import open3d as o3d

            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(vertices),
                o3d.utility.Vector3iVector(faces),
            )
            return cls(
                num_vertices=int(np.asarray(mesh.vertices).shape[0]),
                num_triangles=int(np.asarray(mesh.triangles).shape[0]),
                is_edge_manifold=bool(mesh.is_edge_manifold()),
                is_vertex_manifold=bool(mesh.is_vertex_manifold()),
                is_watertight=bool(mesh.is_watertight()),
                num_non_manifold_edges=int(len(mesh.get_non_manifold_edges())),
            )

    @dataclass(frozen=True, kw_only=True)
    class Triangulation:
        """
        One triangulation's quality indicators and topology.

        This is the `before` / `after` half of the report.
        """

        num_triangles: int
        """Triangle count."""
        skewness: "MeshQualityReport.Stats"
        """Distribution of equiangle skewness; 0 is ideal, 1 degenerate."""
        aspect_ratio: "MeshQualityReport.Stats"
        """Distribution of circumradius / (2 * inradius); 1 is ideal."""
        slenderness: "MeshQualityReport.Stats"
        """Distribution of reciprocal shape regularity; 1 is ideal, large degenerate."""
        min_angle_deg: "MeshQualityReport.Stats"
        """Distribution of the smallest interior angle per face [degrees]."""
        max_angle_deg: "MeshQualityReport.Stats"
        """Distribution of the largest interior angle per face [degrees]."""
        edge_length: "MeshQualityReport.Extremes"
        """Min/max edge length over the mesh."""
        triangle_area: "MeshQualityReport.Extremes"
        """Min/max triangle area over the mesh."""
        fraction_skewness_gt_0p5: float
        """Fraction of triangles with equiangle skewness above 0.5."""
        fraction_aspect_gt_4: float
        """Fraction of triangles with aspect ratio above 4."""
        fraction_min_angle_lt_20deg: float
        """Fraction of triangles with minimum interior angle below 20 degrees."""
        topology: "MeshQualityReport.Topology"
        """Manifoldness/watertightness verdict for this triangulation."""

        @classmethod
        def from_quality(
            cls,
            quality: TriangleQuality,
            topology: "MeshQualityReport.Topology",
        ) -> "MeshQualityReport.Triangulation":
            """
            Reduces per-face `quality` to a JSON-friendly summary of standard
            indicators.
            """
            all_edges = np.concatenate([quality.edge_min, quality.edge_max])
            return cls(
                num_triangles=int(len(quality.skewness)),
                skewness=MeshQualityReport.Stats.from_values(quality.skewness),
                aspect_ratio=MeshQualityReport.Stats.from_values(quality.aspect_ratio),
                slenderness=MeshQualityReport.Stats.from_values(quality.slenderness),
                min_angle_deg=MeshQualityReport.Stats.from_values(quality.min_angle),
                max_angle_deg=MeshQualityReport.Stats.from_values(quality.max_angle),
                edge_length=MeshQualityReport.Extremes(
                    min=float(all_edges.min()), max=float(all_edges.max())
                ),
                triangle_area=MeshQualityReport.Extremes(
                    min=float(quality.area.min()), max=float(quality.area.max())
                ),
                fraction_skewness_gt_0p5=float(np.mean(quality.skewness > 0.5)),
                fraction_aspect_gt_4=float(np.mean(quality.aspect_ratio > 4.0)),
                fraction_min_angle_lt_20deg=float(np.mean(quality.min_angle < 20.0)),
                topology=topology,
            )

    @dataclass(frozen=True, kw_only=True)
    class Remesh:
        """The isotropic remesh the `after` half was measured on."""

        class Method(Enum):
            """Remeshing algorithm; a closed set of one member today."""

            BOTSCH = "botsch"
            """Botsch-Kobbelt isotropic remeshing, via gpytoolbox."""

        method: "MeshQualityReport.Remesh.Method"
        """Remeshing algorithm."""
        target_edge: float
        """Target edge length passed to the remesher."""
        iterations: int
        """Number of remeshing iterations run."""

    mesh_path: Path
    """Path to the source mesh."""
    remeshed_path: Path
    """Path the remeshed surface was written to."""
    remesh: Remesh
    """The remesh that produced `after` from `before`."""
    before: Triangulation
    """Quality and topology of the source mesh."""
    after: Triangulation
    """Quality and topology of the remeshed mesh."""

    def to_json_dict(self) -> dict[str, Any]:
        """
        The report as the JSON object `mesh_quality_report.json` records.

        `remesh.method` crosses the file boundary as its `.value` string, the
        form the report carried before it was an enum; the two `Path` fields
        still cross through `json.dumps`'s own `default=str` at the write site.
        """
        report = asdict(self)
        report["remesh"]["method"] = self.remesh.method.value
        return report
