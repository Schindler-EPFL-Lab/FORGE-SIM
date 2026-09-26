import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from fem.end_to_end.ground_truth_bake.ground_truth_bake_report import (
    GroundTruthBakeReport,
)
from fem.end_to_end.mesh_quality.triangle_quality import TriangleQuality
from fem.fem_mesh_data import FEMMeshData
from fem.meshprep.mesh_repair import clean_mesh
from fem.meshprep.p2_geometry import curved_edge_nodes
from fem.meshprep.reference_surface import transfer_vertex_field_to_points
from fem.meshprep.repaired_mesh import RepairedMesh
from rebel_shape import logger
from rebel_shape.scripts.end_to_end.dataset.mesh_arrays import MeshArrays
from rebel_shape.scripts.end_to_end.dataset.synthetic_fields import (
    material_labels,
    synthetic_temperature,
)
from rebel_shape.scripts.end_to_end.end_to_end_config import EndToEndConfig
from rebel_shape.scripts.end_to_end.heat.thermal_sim_config import ThermalSimConfig


class GroundTruthBakeRunner:
    """
    Bakes the two ground-truth FEM analysis meshes (fine heat, coarser modal)
    from one source mesh, and the report describing what was written.

    Both meshes are handed to the FEM drivers as `.npz` files rather than in
    memory, because the drivers run as separate processes. Holds the config and
    the repaired base mesh that `_build_heat_mesh` and `_build_modal_mesh` both
    read, so neither has to re-derive it. `bake` is the entry point a caller
    reaches for.

    `bake` reads the source PLY's UVs through Mitsuba, so the caller must have
    already called `mi.set_variant`; this class never does, since the variant
    is a process-global that the launching script sets once, not library code.
    """

    MIN_AREA_KEPT: float = 0.90
    """
    Surface-area fraction below which the repair is reported as suspicious.

    Dropping the detached bodies is the repair's purpose, so some loss is expected:
    Suzanne's two eyeballs are 4.0 percent of its area, measured, and Spot loses
    nothing. The bound sits well clear of that and still catches the case the
    warning is about, a real limb or panel going with them, which costs far more.
    It is a warning threshold rather than a tolerance; nothing is rejected on it.
    """

    @dataclass(frozen=True, kw_only=True)
    class CurvedModalMesh:
        """
        What `_curve_modal_mesh` produces: the curved-geometry report entry,
        plus the mid-edge node positions and their basix edge index for the
        caller to fold into `modal_gt.npz`.
        """

        curved: "GroundTruthBakeReport.CurvedGeometry"
        """The curved-geometry report entry."""
        edge_nodes: npt.NDArray[np.float64]
        """Mid-edge node positions, one per unique mesh edge."""
        edge_index: npt.NDArray[np.int64]
        """Per-face edge indices into `edge_nodes`, in `basix_edge_table` column
        order."""

    _REPORT_NAME: str = "ground_truth_bake_report.json"
    """Name of the bake's own record, written at the output root."""

    def __init__(self, config: EndToEndConfig) -> None:
        """Stores the configuration the bake reads from; does no I/O."""
        self._config = config

    def bake(self) -> GroundTruthBakeReport:
        """Produce heat_gt.npz and modal_gt.npz from the source mesh + config."""
        source = MeshArrays.from_ply(self._config.mesh_path)
        base = self._prepare_base(source)
        heat_mesh = self._build_heat_mesh(source, base)
        modal_mesh = self._build_modal_mesh(base)

        report = GroundTruthBakeReport(heat=heat_mesh, modal=modal_mesh)
        self.write_bake_report(report, self._config.output_root / self._REPORT_NAME)
        return report

    @staticmethod
    def write_bake_report(report: GroundTruthBakeReport, path: Path) -> None:
        """
        Writes the bake's record beside its meshes, as step 1 writes
        `mesh_quality_report.json`.

        `asdict` leaves the two `Path` fields as objects and everything else as
        JSON's own scalar types, so `default=str` is reached for those two and
        nothing else.
        """
        path.write_text(json.dumps(asdict(report), indent=2, default=str) + "\n")
        logger.info(f"[gt] bake report written to {path}")

    def _prepare_base(self, source: MeshArrays) -> RepairedMesh:
        """
        Welds and repairs the source mesh into a single watertight body, or takes
        it exactly as provided when `repair_source_mesh` is False.
        """
        if self._config.repair_source_mesh:
            return self._repair_base(source)
        return self._use_source_as_is(source)

    def _repair_base(self, source: MeshArrays) -> RepairedMesh:
        """
        Welds UV-seam duplicates and drops detached interior sub-bodies, sealing
        the holes they leave.

        UVs are optional: the thermal field falls back to a world-space
        checkerboard when the source has none (bunny, armadillo). See
        `_heat_temperature_field`. Welding UV-seam duplicates and dropping the
        detached eyeball bodies closes the model into a single watertight
        manifold: those bodies lie inside the head hull, are invisible to the
        image reconstruction, and leave the shell self-intersecting and modally
        ill-posed. Only the eyeball bulge goes (the eyelid and brow detail is
        on the head), and the sockets they leave are sealed, then smoothed
        after remeshing into a natural lid rather than a flat patch. The closed
        surface that results matches the closed reconstructed B-rep in area
        closely, which an open or eyeball-laden shell would not.
        """
        base = RepairedMesh.from_raw(source.vertices, source.faces)
        logger.info(
            f"[gt] source {source.vertices.shape[0]} verts -> welded + repaired "
            f"base {base.vertices.shape[0]} verts / {base.faces.shape[0]} faces, "
            f"{len(base.seal_centres)} eye seal(s) r={base.seal_radius:.3g} "
            f"(native scale)"
        )
        # The repair drops detached sub-bodies, by design (Suzanne's eyeballs).
        # On a mesh whose detached parts are real geometry (a building's
        # fences or railings) that would silently amputate them, so surface
        # it loudly. Vertex counts cannot detect this (welding removes seam
        # duplicates); area can.
        area_kept = self._surface_area(base.vertices, base.faces) / self._surface_area(
            source.vertices, source.faces
        )
        if area_kept < self.MIN_AREA_KEPT:
            logger.warning(
                f"[gt] repair dropped {100 * (1 - area_kept):.1f}% of the source "
                f"surface area. Intended for detached interior bodies (Suzanne's "
                f"eyeballs); if this mesh's detached parts are REAL geometry, they "
                f"have just been removed. Consider --no-repair-source-mesh."
            )
        return base

    @staticmethod
    def _use_source_as_is(source: MeshArrays) -> RepairedMesh:
        """
        Takes the source mesh exactly as provided: no weld, no dropped
        bodies, no capping, no reorientation.

        For a mesh that is already watertight, manifold and single-component
        the repair has nothing to fix, and its 1e-5 vertex weld can collapse
        legitimately short edges into a non-manifold result. Nothing was cut,
        so there is nothing to seal.
        """
        base = RepairedMesh(
            vertices=source.vertices,
            faces=source.faces,
            seal_centres=np.empty((0, 3), dtype=np.float64),
            seal_radius=0.0,
        )
        logger.info(
            f"[gt] source mesh used AS-IS (repair disabled): "
            f"{base.vertices.shape[0]} verts / {base.faces.shape[0]} faces "
            f"(native scale)"
        )
        return base

    def _build_heat_mesh(
        self, source: MeshArrays, base: RepairedMesh
    ) -> GroundTruthBakeReport.HeatMesh:
        """
        Builds the fine heat-analysis mesh, writes `heat_gt.npz` and its PLY, and
        returns the report entry for it.
        """
        heat = self._remesh(
            base.vertices,
            base.faces,
            self._config.thermal.fem_remesh_edge,
            base.seal_centres,
            base.seal_radius,
        )
        temperature = self._heat_temperature_field(source, heat.vertices)
        material = self._heat_material_field(heat.vertices)

        heat_path = self._config.fem_heat_dir / "heat_gt.npz"
        np.savez(
            heat_path,
            vertices=heat.vertices,
            faces=heat.faces.astype(np.int64),
            temperature=temperature,
            material=material,
            scale=1.0,  # native scale; kept for the FEM driver's npz contract
        )
        self._write_ply(heat_path.with_suffix(".ply"), heat.vertices, heat.faces)
        quality = TriangleQuality.from_mesh(heat.vertices, heat.faces)
        logger.info(
            f"[gt] heat mesh: {heat.faces.shape[0]} tris, longest axis "
            f"{(heat.vertices.max(0) - heat.vertices.min(0)).max():.4g} m (native), "
            f"steel frac {material.mean():.3f}, hot frac "
            f"{(temperature > 0.5).mean():.3f}, skew max {quality.skewness.max():.3f} "
            f"-> {heat_path}"
        )
        return GroundTruthBakeReport.HeatMesh(
            path=heat_path,
            num_triangles=int(heat.faces.shape[0]),
            num_vertices=int(heat.vertices.shape[0]),
            steel_fraction=float(material.mean()),
            hot_fraction=float((temperature > 0.5).mean()),
        )

    def _heat_temperature_field(
        self, source: MeshArrays, heat_vertices: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Temperature at the heat mesh's vertices, transferred from the measured
        field when `gt_temperature_mesh_path` is set, otherwise from the source
        mesh's synthetic checkerboard.

        The FEM thermal field is the source geometry's per-vertex temperature,
        the same function `dataset.py` renders into the thermal images, evaluated
        on the same vertices, so the two cannot drift, linearly transferred onto
        the analysis mesh rather than re-sampled from the texture at the fine
        remesh vertices. Re-sampling would hand FEM an artificially sharp field the
        measurement never had; the remesh is the FEM pipeline's tax (Suzanne's
        eyes, the bunny's holes), not a field upgrade. Real-image targets have no
        known GT texture, so their ground-truth field is measured instead,
        inpainted onto the GT surface from the thermal images by
        `inpaint_data_mesh`, and travels the same barycentric route for the same
        reason: the measurement's own resolution is the honest one.
        """
        if self._config.gt_temperature_mesh_path is not None:
            measured, orig_temperature = MeshArrays.from_inpainted_ply(
                self._config.gt_temperature_mesh_path
            )
            source_v, source_f = measured.vertices, measured.faces
            logger.info(
                f"[gt] temperature from measured field "
                f"{self._config.gt_temperature_mesh_path.name} "
                f"({orig_temperature.min():.3f}..{orig_temperature.max():.3f})"
            )
        else:
            source_v, source_f = source.vertices, source.faces
            orig_temperature = synthetic_temperature(
                source.vertices, source.uvs, self._config.thermal
            )
        return transfer_vertex_field_to_points(
            heat_vertices, source_v, source_f, orig_temperature
        )

    def _heat_material_field(
        self, heat_vertices: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Material fraction at the heat mesh's vertices, transferred from the
        measured windows mask when `gt_material_mesh_path` is set, otherwise the
        geometric axis split.

        Left continuous; every consumer thresholds at 0.5
        (`fem/end_to_end/heat_gt/fem_heat_gt_mesh.py`,
        `rebel_shape/scripts/end_to_end/heat/iga_heat_simulation.py`).
        """
        if self._config.gt_material_mesh_path is not None:
            measured_material, orig_material = MeshArrays.from_inpainted_ply(
                self._config.gt_material_mesh_path
            )
            material = transfer_vertex_field_to_points(
                heat_vertices, measured_material.vertices, measured_material.faces,
                orig_material,
            )
            logger.info(
                f"[gt] material from measured mask "
                f"{self._config.gt_material_mesh_path.name} "
                f"({orig_material.min():.3f}..{orig_material.max():.3f}, "
                f"active frac "
                f"{(material > ThermalSimConfig.ACTIVE_MATERIAL_THRESHOLD).mean():.3f})"
            )
            return material
        return material_labels(
            heat_vertices, self._config.material_axis, self._config.material_threshold
        )

    def _build_modal_mesh(self, base: RepairedMesh) -> GroundTruthBakeReport.ModalMesh:
        """
        Builds the coarser, single-material modal-analysis mesh, writes
        `modal_gt.npz` and its PLY, and returns the report entry for it.
        """
        modal = self._remesh(
            base.vertices,
            base.faces,
            self._config.modal.fem_remesh_edge,
            base.seal_centres,
            base.seal_radius,
        )
        modal_path = self._config.fem_modal_dir / "modal_gt.npz"
        modal_arrays: dict[str, npt.NDArray | float] = {
            "vertices": modal.vertices,
            "faces": modal.faces.astype(np.int64),
            "scale": 1.0,
        }
        curved: GroundTruthBakeReport.CurvedGeometry | None = None
        if self._config.modal_geometry_order == 2:
            curved_modal = self._curve_modal_mesh(modal, base)
            curved = curved_modal.curved
            modal_arrays["edge_nodes"] = curved_modal.edge_nodes
            modal_arrays["edge_index"] = curved_modal.edge_index
        np.savez(modal_path, **modal_arrays)
        self._write_ply(modal_path.with_suffix(".ply"), modal.vertices, modal.faces)
        quality = TriangleQuality.from_mesh(modal.vertices, modal.faces)
        logger.info(
            f"[gt] modal mesh: {modal.faces.shape[0]} tris, longest axis "
            f"{(modal.vertices.max(0) - modal.vertices.min(0)).max():.4g} m "
            f"(native), skew max {quality.skewness.max():.3f} -> {modal_path}"
        )
        return GroundTruthBakeReport.ModalMesh(
            path=modal_path,
            num_triangles=int(modal.faces.shape[0]),
            num_vertices=int(modal.vertices.shape[0]),
            curved=curved,
        )

    def _curve_modal_mesh(
        self, modal: FEMMeshData, base: RepairedMesh
    ) -> CurvedModalMesh:
        """
        Curves the coarse modal mesh back onto the source surface (base.vertices/
        base.faces, not the remesh): the flat remesh chords across the geometry
        and loses area, i.e. shell mass. Returns the curved-geometry report entry
        together with the mid-edge node positions and their basix edge index, for
        the caller to fold into `modal_gt.npz`.
        """
        faces = modal.faces.astype(np.int64)
        edge_nodes, edge_index, n_straight = curved_edge_nodes(
            modal.vertices, faces, base.vertices, base.faces
        )
        curved_area = self._p2_surface_area(
            modal.vertices, faces, edge_nodes, edge_index
        )
        flat_area = self._surface_area(modal.vertices, faces)
        source_area = self._surface_area(base.vertices, base.faces)
        curved = GroundTruthBakeReport.CurvedGeometry(
            geometry_order=2,
            num_edge_nodes=int(edge_nodes.shape[0]),
            num_left_straight=n_straight,
            flat_area_fraction=flat_area / source_area,
            curved_area_fraction=curved_area / source_area,
        )
        logger.info(
            f"[gt] curved (order 2) modal mesh: {edge_nodes.shape[0]} mid-edge nodes, "
            f"{n_straight} left straight; surface area vs source "
            f"{100 * flat_area / source_area:.1f}% (flat) -> "
            f"{100 * curved_area / source_area:.1f}% (curved)"
        )
        return GroundTruthBakeRunner.CurvedModalMesh(
            curved=curved, edge_nodes=edge_nodes, edge_index=edge_index
        )

    @staticmethod
    def _remesh(
        vertices: npt.NDArray[np.float64],
        faces: npt.NDArray[np.int64],
        edge: float,
        seal_centres: npt.NDArray[np.float64] | None = None,
        seal_radius: float = 0.0,
    ) -> FEMMeshData:
        """
        Dedup + isotropic Botsch remesh + sliver-collapse (manifold-preserving),
        then smooth the socket-seal regions into a natural lid.

        The post-remesh clean collapses vertices closer than `0.05 * edge`, well
        below a good triangle's shortest edge (~0.5 * edge), so it only kills the
        occasional Botsch sliver without disturbing the bulk mesh. When
        `seal_centres` is given, the flat socket caps (left after the eyeballs are
        dropped) are Taubin-smoothed within `seal_radius` into a sealed-lid dome;
        the rest of the surface is pinned.
        """
        import gpytoolbox as gpy  # heavy; deferred to the two methods that call it

        cleaned = clean_mesh(vertices, faces, merge_eps=0.0)
        # i=40 (not 20): the extra Botsch iterations flip away the occasional obtuse
        # "cap" triangle that a shorter run leaves near the seam.
        rv, rf = gpy.remesh_botsch(
            cleaned.vertices.astype(np.float64),
            cleaned.faces.astype(np.int32),
            i=40,
            h=float(edge),
            project=True,
        )
        recleaned = clean_mesh(rv, rf, merge_eps=0.05 * float(edge))
        repaired = RepairedMesh(
            vertices=recleaned.vertices,
            faces=recleaned.faces.astype(np.int64),
            seal_centres=(
                seal_centres
                if seal_centres is not None
                else np.empty((0, 3), dtype=np.float64)
            ),
            seal_radius=seal_radius,
        ).smoothed()
        return FEMMeshData(
            vertices=repaired.vertices, faces=repaired.faces.astype(np.uint32)
        )

    @staticmethod
    def _surface_area(
        vertices: npt.NDArray[np.float64], faces: npt.NDArray[np.int64]
    ) -> float:
        """
        Total triangle area. Welding duplicates preserves it; dropping a body does
        not, which is what makes it the right check on the repair step.
        """
        tri = vertices[faces]
        return float(
            0.5
            * np.linalg.norm(
                np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
            ).sum()
        )

    @staticmethod
    def _p2_surface_area(
        vertices: npt.NDArray[np.float64],
        faces: npt.NDArray[np.int64],
        edge_nodes: npt.NDArray[np.float64],
        edge_index: npt.NDArray[np.int64],
    ) -> float:
        """
        Area of a curved (order-2) mesh, via the 4-way subdivision each P2
        triangle induces on its own mid-edge nodes. Still a lower bound (the
        sub-triangles are flat), but it captures the curvature the straight-sided
        area misses.

        The `edge_index` columns are read in the order
        `fem.meshprep.p2_geometry.basix_edge_table` defines, `[e12, e02, e01]`.
        """
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        e12 = edge_nodes[edge_index[:, 0]]
        e02 = edge_nodes[edge_index[:, 1]]
        e01 = edge_nodes[edge_index[:, 2]]
        total = 0.0
        for a, b, c in (
            (v0, e01, e02), (v1, e12, e01), (v2, e02, e12), (e01, e12, e02),
        ):
            total += 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
        return float(total)

    @staticmethod
    def _write_ply(
        path: Path, vertices: npt.NDArray[np.float64], faces: npt.NDArray[np.uint32]
    ) -> None:
        """Write the watertight analysis mesh next to its .npz for inspection."""
        import gpytoolbox as gpy  # heavy; deferred to the two methods that call it

        gpy.write_mesh(str(path), vertices.astype(np.float64), faces.astype(np.int32))
