"""
Reissner-Mindlin shell modal analysis on a forge-sim spline surface, in DOLFINx.

This is the FEM counterpart to the IGA
`forge_sim/scripts/reissner_mindlin_modal_analysis/reissner_mindlin_modal_analysis.py`.
It loads the same serialized spline `.json`, applies the same geometry rescaling
(`target_length_scale`), builds a curved (isoparametric) DOLFINx mesh on the
spline mid-surface, and solves the generalized
eigenvalue problem `K phi = omega^2 M phi` with the geometrically-exact,
locking-free shell formulation (see `geometrically_exact_shell_problem.py`).

Because the two solvers operate on identically-scaled geometry with the same uniform
material, their spectra should match: first the six rigid-body modes at (near) zero
with a large gap to the first elastic mode, then the elastic spectrum.

The eigensolve uses SLEPc shift-invert with a MUMPS factorization (memory-scalable),
mass-shifted to lift the 6-dimensional rigid-body nullspace, the same strategy as
the IGA `ModalAnalysisSolver`. Run from the repo root in the `dolfinx-forge-sim`
conda environment::

    python fem/shells/shell_modal_analysis_spline.py --spline-path <surface.json>

Memory note: peak memory is dominated by the MUMPS factorization, which stays modest
on a 2-manifold (nested-dissection ordering): ~7 GB at `subdivisions_per_patch=64`
(≈0.6 M DOFs) and ~21 GB at 128 (≈2.4 M DOFs) for a 6-patch surface, both fit on a
32 GB machine. S=64 already matches the IGA spectrum to < 0.12 %.
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mitsuba as mi
import numpy as np
import tyro

# Repository root: makes `forge_sim` and this script's own `fem` package
# importable when it is run by path rather than as `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from fem.petsc_factor_solver import FactorSolver
from fem.shells.eigenmode_xdmf import export_mode_shapes_xdmf
from fem.shells.geometrically_exact_shell_modal_solver import (
    GeometricallyExactShellModalSolver,
)
from fem.shells.geometrically_exact_shell_problem import GeometricallyExactShellProblem
from fem.shells.modal_result import ModalResult
from fem.shells.rigid_body_modes import report_rigid_body_modes
from fem.shells.shell_element import ShellElement
from fem.shells.spline_to_shell_mesh import build_shell_mesh, load_and_rescale_spline
from forge_sim import logger
from forge_sim.scripts.configs.structural_material_config import (
    StructuralMaterialConfig,
)


@dataclass(kw_only=True)
class ShellModalAnalysisParameters(ReverseCli):
    """Configuration for FEM Reissner-Mindlin modal analysis on a spline surface."""

    spline_path: Path
    """Path to the reconstructed spline surface (TessellatedSplineSurface JSON)."""
    output_folder: Path = Path("./outputs/shell_modal_fem/")
    """Directory for results (created if missing)."""

    num_modes: int = 20
    """Number of modes to compute."""

    material_config: StructuralMaterialConfig = field(
        default_factory=StructuralMaterialConfig
    )
    """Uniform structural material (E, nu, rho, thickness, shear correction)."""

    target_length_scale: float | None = 3.0
    """
    If set, scale so the longest control-point bbox extent equals this (meters).

    Must match the IGA run for the spectra to compare.
    """

    subdivisions_per_patch: int = 64
    """
    Discretization: quad subdivisions per patch per direction (2*S^2
    triangles/patch).
    """
    geometry_order: int = 2
    """Mesh geometry order: 1 (flat facets) or 2 (curved, nodes on the spline)."""
    field_degree: int = 2
    """Lagrange degree for displacement and rotation (2 or 3)."""
    use_bbar: bool = True
    """
    B-bar element-local L2 strain projection (locking cure). Recommended True:
    parameter-free, stronger than PSRI on thin bending-dominated geometry, and
    leaves the membrane-dominated sphere spectrum unchanged. Mutually exclusive
    with `use_psri`.
    """
    use_psri: bool = False
    """
    Partial selective reduced integration (older locking cure). Mutually
    exclusive with `use_bbar`; with both off the shell is fully integrated.
    """
    drilling_stiffness_factor: float = 1.0
    """Scale factor on the drilling stabilization stiffness."""

    solver_tolerance: float = 1.0e-9
    """Eigensolver: SLEPc convergence tolerance."""
    spectral_shift: float | None = None
    """
    Mass-proportional shift to lift the rigid-body nullspace, honoured by both
    solve paths. None -> heuristic.
    """
    factor_solver: FactorSolver = FactorSolver.MUMPS
    """PETSc direct factorization backend for the shift-invert eigensolve."""
    use_scipy_solver: bool = False
    """
    Use the scipy diagnostic solver instead of SLEPc. Only for small meshes: it
    assembles five scipy matrices. In exchange it fills the `mode_energies` entry of
    the saved `.npz` with the per-mode membrane/bending/shear/drilling split, which
    the SLEPc path leaves zero.
    """

    export_xdmf: bool = True
    """
    Output: export mode-shape displacement fields to an XDMF time series for
    ParaView.
    """
    export_xdmf_indices: list[int] = field(default_factory=lambda: list(range(20)))
    """Mode indices to export."""

    num_rb_modes: int = 6
    """Expected rigid-body mode count (6 for a free closed/free-free surface)."""

    def __post_init__(self) -> None:
        if not self.spline_path.exists():
            raise ValueError(f"Spline path {self.spline_path} does not exist.")
        if self.num_modes < 1:
            raise ValueError(
                f"`num_modes` must be a positive integer, got {self.num_modes}."
            )
        if self.geometry_order not in (1, 2):
            raise ValueError(
                f"`geometry_order` must be 1 or 2, got {self.geometry_order}."
            )
        if self.field_degree not in (2, 3):
            raise ValueError(
                f"`field_degree` must be 2 or 3, got {self.field_degree}."
            )
        if self.subdivisions_per_patch < 1:
            raise ValueError(
                "`subdivisions_per_patch` must be a positive integer, got "
                f"{self.subdivisions_per_patch}."
            )
        if self.target_length_scale is not None and self.target_length_scale <= 0.0:
            raise ValueError(
                "`target_length_scale` must be positive if provided, got "
                f"{self.target_length_scale}."
            )
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


def run_modal_analysis(params: ShellModalAnalysisParameters) -> ModalResult:
    """
    Runs the FEM shell modal analysis and persists the results.

    Loads and rescales the spline, builds the shell mesh at
    `params.geometry_order` (curved for the default 2, flat facets for 1),
    assembles and eigensolves the shell problem, checks the rigid-body-mode
    band, saves the results to `params.output_folder`, and optionally exports
    the mode shapes to XDMF.

    A Mitsuba variant must already be selected: loading the spline is a Dr.Jit
    operation, and `mi.set_variant` is a process-global side effect that belongs
    to the launching script, which sets it as its first act.

    `params` is the modal analysis configuration.

    :returns: The modal solve result (eigenvalues, eigenvectors, frequencies, and
        `mode_energies`, the latter only populated when
        `params.use_scipy_solver` is set). Also written to `params.output_folder`.
    """
    params.output_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading spline from {params.spline_path}")
    tessellated = load_and_rescale_spline(
        spline_path=params.spline_path,
        target_length_scale=params.target_length_scale,
    )

    logger.info(
        f"Building curved (order {params.geometry_order}) mesh: "
        f"{params.subdivisions_per_patch} subdivisions/patch..."
    )
    t0 = time.time()
    msh = build_shell_mesh(
        tessellated_spline=tessellated,
        subdivisions_per_patch=params.subdivisions_per_patch,
        order=params.geometry_order,
    )
    n_cells = msh.topology.index_map(msh.topology.dim).size_global
    logger.info(
        f"Mesh: {msh.geometry.x.shape[0]} geometry nodes, {n_cells} cells "
        f"(built in {time.time() - t0:.1f}s)."
    )

    mc = params.material_config
    element = ShellElement(
        degree=params.field_degree,
        locking_cure=params.locking_cure,
        drilling_stiffness_factor=params.drilling_stiffness_factor,
        normal=ShellElement.NormalKind.GEOMETRIC,
    )
    problem = GeometricallyExactShellProblem(
        msh,
        youngs_modulus=mc.youngs_modulus,
        poisson_ratio=mc.poisson_ratio,
        density=mc.density,
        thickness=mc.thickness,
        element=element,
        shear_correction=mc.shear_correction_factor,
    )
    solver = GeometricallyExactShellModalSolver(problem)
    logger.info(
        f"Shell space: {element.label()}, {problem.ndofs} DOFs. "
        f"Material: E={mc.youngs_modulus:.3e} nu={mc.poisson_ratio} "
        f"rho={mc.density} h={mc.thickness}."
    )

    t0 = time.time()
    if params.use_scipy_solver:
        result = solver.solve_modes(
            num_modes=params.num_modes, spectral_shift=params.spectral_shift
        )
    else:
        result = solver.solve_modes_slepc(
            num_modes=params.num_modes,
            spectral_shift=params.spectral_shift,
            tol=params.solver_tolerance,
            factor_solver=params.factor_solver,
        )
    logger.info(f"Eigensolve completed in {time.time() - t0:.1f}s.")

    report_rigid_body_modes(
        result.eigenvalues, result.frequencies, params.num_rb_modes
    )
    logger.info(
        "Natural frequencies (Hz):\n" + np.array2string(result.frequencies, precision=4)
    )

    output_path = params.output_folder / "modal_analysis_results.npz"
    # `mode_energies` is the whole reason to choose the scipy path; the SLEPc path
    # leaves it zero, which is its documented "not computed" value. Saved either
    # way so the archive layout does not depend on which solver produced it.
    np.savez(
        output_path,
        eigenvalues=result.eigenvalues,
        eigenvectors=result.eigenvectors,
        frequencies=result.frequencies,
        mode_energies=result.mode_energies,
    )
    logger.info(f"Saved modal results to {output_path}.")

    if params.export_xdmf:
        export_mode_shapes_xdmf(
            problem.V, result, params.export_xdmf_indices, params.output_folder
        )

    return result


if __name__ == "__main__":
    # Mitsuba variant must be set before any spline loading / Dr.Jit operation,
    # and it is a process-global side effect, so the launching script owns it.
    mi.set_variant("llvm_ad_mono")
    run_modal_analysis(params=tyro.cli(ShellModalAnalysisParameters))
