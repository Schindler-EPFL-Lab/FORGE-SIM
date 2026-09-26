from dataclasses import dataclass

from rebel_shape.scripts.configs.structural_material_config import (
    StructuralMaterialConfig,
)


@dataclass(frozen=True, kw_only=True)
class FEMModalGTReport:
    """
    What the FEM shell modal run on the ground-truth mesh reports about itself.

    Serialised as `fem_modal_gt_log.json` beside the mode-shape archive. The
    eigenvalues and the mode shapes go to `modal_analysis_results.npz`; this record
    is everything around them: the material and element the spectrum belongs to, what
    the mesh cleaning did, and how far the rigid-body modes sit below the first elastic
    one. `end_to_end.modal.align_modal_xdmf` reads `frequencies_hz` back from it rather
    than re-solving.
    """

    @dataclass(frozen=True, kw_only=True)
    class Element:
        """The shell discretisation the run assembled."""

        field_degree: int
        """Lagrange degree of the displacement and rotation fields."""
        use_bbar: bool
        """Whether the B-bar assumed-strain locking cure was applied."""
        use_psri: bool
        """Whether partial selective reduced integration was applied instead."""
        project_rotary_inertia: bool
        """Whether the drilling rotation's inertia was projected out."""

    mesh_npz: str
    """Path of the ground-truth mesh bundle the run read."""
    material: StructuralMaterialConfig
    """The uniform material the spectrum belongs to."""
    element: Element
    """The shell discretisation."""
    ndofs: int
    """Degrees of freedom the eigenproblem was solved on."""
    num_vertices_input: int
    """Vertices in the mesh as it arrived, before cleaning."""
    num_triangles_input: int
    """Triangles in the mesh as it arrived, before cleaning."""
    num_degenerate_triangles_dropped: int
    """Near-zero-area triangles the cleaning pass removed."""
    num_triangles: int
    """Triangles actually solved on."""
    max_equiangle_skew_input: float
    """Worst equiangle skew of the mesh as it arrived; 0 is equilateral."""
    num_rigid_body_modes_expected: int
    """Rigid-body modes the geometry should have (6 for a free-free surface)."""
    num_rigid_body_modes_detected: int
    """Rigid-body modes actually found below the spectral gap."""
    rb_to_first_elastic_gap_ratio: float
    """
    First elastic eigenvalue over the last rigid-body one.

    Large means the two families are cleanly separated; of order 1 means they are
    not, and the mode numbering below cannot be trusted.
    """
    solve_seconds: float
    """Wall-clock time of the eigensolve alone."""
    eigenvalues: list[float]
    """Eigenvalues as solved, in the solver's own order."""
    frequencies_hz: list[float]
    """Natural frequencies [Hz], paired positionally with `eigenvalues`."""
