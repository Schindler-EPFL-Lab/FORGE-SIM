from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class GroundTruthBakeReport:
    """
    What the ground-truth bake produced, for a caller that did not watch it.

    The two analysis meshes themselves go to disk as `.npz` (plus a `.ply`
    alongside for inspection), because the FEM drivers that consume them are
    separate processes; this record only describes what was written.
    """

    @dataclass(frozen=True, kw_only=True)
    class HeatMesh:
        """
        The fine heat-analysis mesh and the fields transferred onto it.

        It is refined enough to resolve the sharp checkerboard initial condition
        and the material interface, which is why it is not the modal mesh.
        """

        path: Path
        """Where the `heat_gt.npz` bundle was written."""
        num_triangles: int
        """Triangles after the isotropic remesh."""
        num_vertices: int
        """Vertices after the isotropic remesh."""
        steel_fraction: float
        """Mean of the material field; the active material's share of the surface."""
        hot_fraction: float
        """Vertex fraction whose transferred temperature exceeds the 0.5 midpoint."""

    @dataclass(frozen=True, kw_only=True)
    class CurvedGeometry:
        """
        The order-2 curving applied to the modal mesh, when one was requested.

        A flat remesh chords across the geometry and loses area, which is shell
        mass; the two area fractions are what make that loss, and its recovery,
        visible.
        """

        geometry_order: int
        """Geometric order of the modal mesh; 2 whenever this record is present."""
        num_edge_nodes: int
        """Mid-edge nodes projected back onto the source surface."""
        num_left_straight: int
        """Mid-edge nodes the projection could not place, left on the chord."""
        flat_area_fraction: float
        """Straight-sided modal mesh area, as a fraction of the source area."""
        curved_area_fraction: float
        """Curved (order-2) modal mesh area, as a fraction of the source area."""

    @dataclass(frozen=True, kw_only=True)
    class ModalMesh:
        """
        The coarser modal-analysis mesh, to which the FEM driver applies one
        material uniformly.
        """

        path: Path
        """Where the `modal_gt.npz` bundle was written."""
        num_triangles: int
        """Triangles after the isotropic remesh."""
        num_vertices: int
        """Vertices after the isotropic remesh."""
        curved: "GroundTruthBakeReport.CurvedGeometry | None"
        """The order-2 curving, or None when the mesh was left straight-sided."""

    heat: HeatMesh
    """The fine heat-analysis mesh."""
    modal: ModalMesh
    """The coarser modal-analysis mesh."""
