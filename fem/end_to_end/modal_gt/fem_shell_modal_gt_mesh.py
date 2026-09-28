"""
Reissner-Mindlin shell modal analysis on a ground-truth triangle mesh, in DOLFINx.

This is the sibling of `shell_modal_analysis_spline.py`: instead of building the
mesh from a forge-sim spline mid-surface, it runs the identical geometrically-exact,
locking-free shell problem (`geometrically_exact_shell_problem.py`) on a bare
triangle mesh supplied as a NumPy `.npz`. The mesh is the ground-truth surface
itself, a faceted reconstruction with no underlying smooth surface, so the
geometry is order 1 (flat facets): there is no mid-surface to curve the mid-edge
nodes onto. The solver's geometrically-exact strain measures are rigid-body
invariant regardless of geometry order, so a flat order-1 mesh still yields six
exactly zero-energy rigid-body modes; only the absolute spectral accuracy is O(h^2)
in the geometry, as for any faceted shell.

Material is a single uniform steel (the "car case"): the spectrum is the six
near-zero free-free rigid-body modes followed by the elastic panel-bending modes.

The eigensolve uses SLEPc shift-invert with a MUMPS factorization, mass-shifted to
lift the 6-dimensional rigid-body nullspace, identical to the spline driver and
the IGA `ModalAnalysisSolver`. The output format (`modal_analysis_results.npz`
+ a mode-shape XDMF) mirrors `shell_modal_analysis_spline.py` exactly so the two
runs are directly comparable.

Input `.npz` contract (e.g. `outputs/suzanne_e2e/07_fem_modal/modal_gt.npz`):
    vertices  (n, 3) float64: already at physical size (longest axis = 4 m)
    faces     (m, 3) int:     triangle vertex indices
    scale     scalar:         informational only (the rescale factor already applied)

Run from the repo root in the `dolfinx-forge-sim` conda environment::

    python fem/end_to_end/modal_gt/fem_shell_modal_gt_mesh.py \
        --mesh-npz outputs/suzanne_e2e/07_fem_modal/modal_gt.npz

Performance note: the default B-bar locking cure (`--use-bbar`) assembles its
membrane/shear stiffness blocks in scipy via a sparse `spsolve` against the
per-cell reduced-space mass; at this DOF count (~3e5) that assembly alone takes
many minutes. For a fast run on a large mesh, prefer the PSRI cure, which is pure
UFL forms and assembles in seconds with the same rigid-body-clean spectrum::

    python fem/end_to_end/modal_gt/fem_shell_modal_gt_mesh.py \
        --mesh-npz outputs/suzanne_e2e/07_fem_modal/modal_gt.npz \
        --no-use-bbar --use-psri
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import tyro

# Repository root: makes `forge_sim` and this script's own `fem` package
# importable when it is run by path rather than as `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from fem.end_to_end.modal_gt.fem_shell_modal_gt_runner import FEMShellModalGTRunner
from fem.petsc_factor_solver import FactorSolver
from fem.shells.modal_result import ModalResult
from fem.shells.shell_element import ShellElement
from forge_sim.scripts.configs.structural_material_config import (
    StructuralMaterialConfig,
)


@dataclass(kw_only=True)
class FEMShellModalGTMeshParameters(ReverseCli):
    """Configuration for FEM Reissner-Mindlin modal analysis on a GT triangle mesh."""

    mesh_npz: Path
    """Path to the ground-truth mesh `.npz` (keys: vertices, faces, scale)."""
    output_folder: Path = Path("./outputs/suzanne_e2e/07_fem_modal/")
    """Directory for results (created if missing)."""

    num_modes: int = 20
    """Number of modes to compute."""

    material_config: StructuralMaterialConfig = field(
        default_factory=lambda: StructuralMaterialConfig(
            youngs_modulus=210.0e9,
            poisson_ratio=0.3,
            density=7850.0,
            thickness=0.0015,
            shear_correction_factor=5.0 / 6.0,
        )
    )
    """
    Uniform structural material (E, nu, rho, thickness, shear correction).

    Uniform steel (the car case), rather than the concrete-leaning defaults of
    `StructuralMaterialConfig` itself, so the script is runnable as-is. A pipeline run
    overrides all five values from the shared `ModalSimConfig`, so that this run and
    the IGA run it is compared against use one material.
    """

    field_degree: int = 2
    """
    Lagrange degree for displacement and rotation (2 or 3).

    Discretization: geometry is order 1 regardless, the GT mesh is the reference
    surface, with no mid-surface to curve the mid-edge nodes onto.
    """
    use_bbar: bool = True
    """B-bar element-local L2 strain projection (parameter-free locking cure).
    Mutually exclusive with `use_psri`."""
    use_psri: bool = False
    """Partial selective reduced integration. Mutually exclusive with `use_bbar`;
    with both off the shell is fully integrated."""
    drilling_stiffness_factor: float = 1.0
    """Scale factor on the drilling stabilization stiffness."""
    project_rotary_inertia: bool = True
    """Zero inertia on the drilling rotation (sends drilling modes to +inf)."""

    clean_degenerate: bool = True
    """
    Mesh hygiene: drop zero-/near-zero-area triangles (and any vertices they orphan)
    before the solve. A single sliver can pollute the low spectrum / rigid-body gap.
    The pass is manifold-preserving (it only removes faces, never adds or moves
    vertices).
    """
    min_triangle_area_ratio: float = 1.0e-8
    """A triangle is dropped when its area is below this fraction of the mean
    triangle area (only when `clean_degenerate` is True)."""

    solver_tolerance: float = 1.0e-9
    """Eigensolver: SLEPc convergence tolerance."""
    spectral_shift: float | None = None
    """Mass-proportional shift to lift the rigid-body nullspace. None -> heuristic."""
    factor_solver: FactorSolver = FactorSolver.MUMPS
    """PETSc direct factorization backend for the shift-invert eigensolve."""
    use_scipy_solver: bool = False
    """Use the scipy diagnostic solver (per-mode energy split) instead of SLEPc.
    Only for small meshes; assembles five scipy matrices."""

    export_xdmf: bool = True
    """
    Output: export mode-shape displacement fields to an XDMF time series for
    ParaView.
    """
    export_xdmf_indices: list[int] = field(default_factory=lambda: list(range(20)))
    """Mode indices to export."""

    num_rb_modes: int = 6
    """Expected rigid-body mode count (6 for a free-free surface)."""

    def __post_init__(self) -> None:
        if not self.mesh_npz.exists():
            raise ValueError(f"Mesh npz {self.mesh_npz} does not exist.")
        if self.num_modes < 1:
            raise ValueError("`num_modes` must be a positive integer.")
        if self.field_degree not in (2, 3):
            raise ValueError("`field_degree` must be 2 or 3.")
        # The material's own rules (positivity, the Poisson range, and the
        # unit-conversion magnitude warnings) are `StructuralMaterialConfig`'s, and
        # have already run by the time a sub-config can be assigned here.
        # Evaluated for the rejection it performs: the pair naming two cures is
        # the one combination the flags can express and the solver cannot.
        ShellElement.locking_cure_from_flags(self.use_bbar, self.use_psri)

    @property
    def locking_cure(self) -> ShellElement.LockingCure:
        """
        The locking cure the two flags select.

        :returns: The cure, as
            :meth:`ShellElement.locking_cure_from_flags` reads the flags.
        """
        return ShellElement.locking_cure_from_flags(self.use_bbar, self.use_psri)


def run_modal_analysis(params: FEMShellModalGTMeshParameters) -> ModalResult:
    """Runs the FEM shell modal analysis on a GT mesh and persists the results."""
    element = ShellElement(
        degree=params.field_degree,
        locking_cure=params.locking_cure,
        drilling=True,
        drilling_stiffness_factor=params.drilling_stiffness_factor,
        project_rotary_inertia=params.project_rotary_inertia,
        normal=ShellElement.NormalKind.GEOMETRIC,
    )
    return FEMShellModalGTRunner(
        mesh_npz=params.mesh_npz,
        output_folder=params.output_folder,
        material_config=params.material_config,
        element=element,
        clean_degenerate=params.clean_degenerate,
        min_triangle_area_ratio=params.min_triangle_area_ratio,
        num_modes=params.num_modes,
        spectral_shift=params.spectral_shift,
        solver_tolerance=params.solver_tolerance,
        factor_solver=params.factor_solver,
        use_scipy_solver=params.use_scipy_solver,
        export_xdmf=params.export_xdmf,
        export_xdmf_indices=params.export_xdmf_indices,
        num_rb_modes=params.num_rb_modes,
    ).run()


if __name__ == "__main__":
    parameters = tyro.cli(FEMShellModalGTMeshParameters)
    run_modal_analysis(params=parameters)
