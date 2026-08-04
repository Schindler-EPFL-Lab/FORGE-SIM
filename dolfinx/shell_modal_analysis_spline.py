"""
Reissner-Mindlin shell modal analysis on a rebel-shape spline surface, in DOLFINx.

This is the FEM counterpart to the IGA
``rebel_shape/scripts/reissner_mindlin_modal_analysis/reissner_mindlin_modal_analysis.py``.
It loads the same serialized spline ``.json``, applies the *same* geometry rescaling
(``target_length_scale``), builds a curved (isoparametric) DOLFINx mesh on the
spline mid-surface, and solves the generalized
eigenvalue problem ``K phi = omega^2 M phi`` with the geometrically-exact,
locking-free shell formulation (see ``geometrically_exact_shell_solver.py``).

Because the two solvers operate on identically-scaled geometry with the same uniform
material, their spectra should match: first the six rigid-body modes at (near) zero
with a large gap to the first elastic mode, then the elastic spectrum.

The eigensolve uses SLEPc shift-invert with a MUMPS factorization (memory-scalable),
mass-shifted to lift the 6-dimensional rigid-body nullspace -- the same strategy as
the IGA ``ModalAnalysisSolver``. Run from the repo root in the ``dolfinx-rebel-shape``
conda environment::

    python dolfinx/shell_modal_analysis_spline.py --spline-path <surface.json>

Memory note: peak memory is dominated by the MUMPS factorization, which stays modest
on a 2-manifold (nested-dissection ordering): ~7 GB at ``subdivisions_per_patch=64``
(≈0.6 M DOFs) and ~21 GB at 128 (≈2.4 M DOFs) for a 6-patch surface -- both fit on a
32 GB machine. S=64 already matches the IGA spectrum to < 0.12 %.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import mitsuba as mi
import tyro

# Make the rebel_shape package importable when run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape import logger
from rebel_shape.scripts.configs.structural_material_config import (
    StructuralMaterialConfig,
)

from geometrically_exact_shell_solver import (
    GeometricallyExactShellSolver,
    ModalResult,
    ShellElement,
)
from spline_to_shell_mesh import build_shell_mesh, load_and_rescale_spline


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

    # geometry scaling -- MUST match the IGA run for the spectra to compare.
    target_length_scale: float | None = 3.0
    """If set, scale so the longest control-point bbox extent equals this (meters)."""

    # discretization
    subdivisions_per_patch: int = 64
    """Quad subdivisions per patch per direction (2*S^2 triangles/patch)."""
    geometry_order: int = 2
    """Mesh geometry order: 1 (flat facets) or 2 (curved, nodes on the spline)."""
    field_degree: int = 2
    """Lagrange degree for displacement and rotation (2 or 3)."""
    use_bbar: bool = True
    """B-bar element-local L2 strain projection (locking cure). Recommended True:
    parameter-free, stronger than PSRI on thin bending-dominated geometry, and
    leaves the membrane-dominated sphere spectrum unchanged. Takes precedence over
    ``use_psri``."""
    use_psri: bool = False
    """Partial selective reduced integration (older locking cure). Used only when
    ``use_bbar`` is False."""
    drilling_stiffness_factor: float = 1.0
    """Scale factor on the drilling stabilization stiffness."""

    # eigensolver
    solver_tolerance: float = 1.0e-9
    """SLEPc convergence tolerance."""
    spectral_shift: float | None = None
    """Mass-proportional shift to lift the rigid-body nullspace, honoured by both
    solve paths. None -> heuristic."""
    factor_solver: str = "mumps"
    """PETSc direct factorization backend for shift-invert (mumps/superlu_dist/...)."""
    use_scipy_solver: bool = False
    """Use the scipy diagnostic solver instead of SLEPc. Only for small meshes: it
    assembles five scipy matrices. In exchange it fills the ``mode_energies`` entry of
    the saved ``.npz`` with the per-mode membrane/bending/shear/drilling split, which
    the SLEPc path leaves zero."""

    # output
    export_xdmf: bool = True
    """Export mode-shape displacement fields to an XDMF time series for ParaView."""
    export_xdmf_indices: list[int] = field(default_factory=lambda: list(range(20)))
    """Mode indices to export."""

    n_rigid_body_modes: int = 6
    """Expected rigid-body mode count (6 for a free closed/free-free surface)."""

    def __post_init__(self) -> None:
        if not self.spline_path.exists():
            raise ValueError(f"Spline path {self.spline_path} does not exist.")
        if self.num_modes < 1:
            raise ValueError("`num_modes` must be a positive integer.")
        if self.geometry_order not in (1, 2):
            raise ValueError("`geometry_order` must be 1 or 2.")
        if self.field_degree not in (2, 3):
            raise ValueError("`field_degree` must be 2 or 3.")
        if self.subdivisions_per_patch < 1:
            raise ValueError("`subdivisions_per_patch` must be a positive integer.")
        if self.target_length_scale is not None and self.target_length_scale <= 0.0:
            raise ValueError("`target_length_scale` must be positive if provided.")


def _report_rigid_body_modes(
    eigenvalues: npt.NDArray[np.float64],
    frequencies: npt.NDArray[np.float64],
    n_rigid: int,
) -> None:
    """
    Logs the rigid-body-mode test.

    The spectrum must show ``n_rigid`` near-zero modes separated from the elastic
    modes by a large gap. The rigid-body count is detected robustly as the location
    of the largest relative gap in the low spectrum (not a fixed threshold).

    :param eigenvalues: Eigenvalues of the generalized eigenvalue problem.
    :param frequencies: Natural frequencies (Hz) corresponding to ``eigenvalues``.
    :param n_rigid: Expected number of rigid-body modes.
    """
    # Scan the rigid-body band plus a comparable stretch of elastic modes, so the
    # jump out of the band is inside the window and later, larger gaps higher in
    # the spectrum cannot outrank it. The `+ 4` keeps the window meaningful when
    # `n_rigid` is small. Detection needs at least two eigenvalues to form a ratio.
    n_scan = min(len(eigenvalues), 2 * n_rigid + 4)
    if n_scan < 2:
        logger.warning(
            f"Rigid-body-mode test skipped: only {n_scan} eigenvalue(s) computed, "
            f"at least 2 are needed to locate the spectral gap.")
        return
    abs_ev = np.abs(eigenvalues[:n_scan])
    # Floor the near-zero (rigid-body) eigenvalues at a small fraction of the spectral
    # scale: their post-shift residuals are tiny and noisy, and unfloored ratios
    # *within* that band can spuriously exceed the true RB->elastic jump. Flooring
    # collapses intra-band ratios to ~1 so only the real gap stands out. The factor
    # only has to sit between the rigid-body residuals (1e-13 of the scale or below,
    # being round-off about zero) and the elastic spectrum (where the smallest
    # eigenvalue is within a decade or two of the largest scanned one), so any value
    # in a very wide range works; 1e-6 is near the middle of it in log terms.
    capped = np.maximum(abs_ev, 1.0e-6 * float(abs_ev.max()))
    ratios = capped[1:] / capped[:-1]
    detected = int(np.argmax(ratios)) + 1
    gap = float(ratios[detected - 1])
    logger.info(
        f"Rigid-body-mode test: largest spectral gap after mode {detected} "
        f"(expected {n_rigid}), gap ratio = {gap:.3e}.")
    logger.info(
        f"  RB eigenvalues: {np.array2string(eigenvalues[:n_rigid], precision=3)}")
    if len(eigenvalues) > n_rigid:
        logger.info(
            f"  first elastic mode: lambda = {eigenvalues[n_rigid]:.4e}, "
            f"f = {frequencies[n_rigid]:.3f} Hz")
    if detected != n_rigid:
        logger.warning(
            f"Detected {detected} rigid-body modes but expected {n_rigid}; "
            "check geometry closure or eigensolver settings.")


def _export_mode_shapes_xdmf(
    solver: GeometricallyExactShellSolver,
    result: ModalResult,
    mode_indices: list[int],
    output_folder: Path,
    scale_factor: float = 0.1,
) -> None:
    """
    Writes eigenmodes in the SAME temporal ``eigenmodes.xdmf``/``.h5`` layout as the
    IGA runner (``reissner_mindlin_modal_runner.export_eigenmodes_to_xdmf``), so a FEM
    (GT-mesh) run and an IGA (spline) run open identically in ParaView for side-by-side
    comparison.

    The format: one temporal collection, one Uniform grid per mode (Time = export
    index), each referencing its OWN deformed geometry
    ``vertices - (unit_displacement * scale_factor * smallest_bbox_extent)`` plus the
    per-vertex ``displacement`` / ``rotation`` vectors and magnitudes. The values are
    taken at the mesh geometry vertices (the displacement and rotation sub-fields are
    interpolated to a P1 vector space and gathered into geometry-vertex order).

    :param solver: Shell solver holding the mesh (``solver.msh``) and mixed function
        space (``solver.V``) the eigenvectors are defined on.
    :param result: Modal solve result, with eigenvalues/frequencies/eigenvectors.
    :param mode_indices: Mode indices to export.
    :param output_folder: Directory the ``eigenmodes.xdmf``/``.h5`` pair is written to
        (created if missing).
    :param scale_factor: Deformation scale, as a fraction of the SMALLEST positive
        bounding-box extent (not an absolute length).
    """
    import basix.ufl
    import h5py
    from dolfinx import fem
    from scipy.spatial import cKDTree

    valid = [i for i in mode_indices if 0 <= i < result.eigenvectors.shape[1]]
    if len(valid) != len(mode_indices):
        # Requesting more indices than modes is the normal case (the default list
        # is a fixed range), so this is not an error -- but it must not be silent,
        # or a truncated eigensolve would quietly produce a short export.
        logger.warning(
            f"Skipping {len(mode_indices) - len(valid)} requested mode index/indices "
            f"outside the {result.eigenvectors.shape[1]} computed modes.")
    if not valid:
        return

    msh = solver.msh
    V = solver.V
    vertices = np.ascontiguousarray(msh.geometry.x[:, :3], dtype=np.float64)
    geom_dofmap = np.asarray(msh.geometry.dofmap)
    faces = np.ascontiguousarray(
        geom_dofmap.reshape(geom_dofmap.shape[0], -1)[:, :3], dtype=np.int32
    )
    n_vertices = vertices.shape[0]

    # Map each geometry vertex to its P1 dof index (so per-vertex fields land in the
    # same order as `vertices`/`faces`); a vector P1 dof array reshaped (n, 3) shares
    # this vertex blocking.
    vp1_scalar = fem.functionspace(msh, ("Lagrange", 1))
    p1_coords = np.asarray(vp1_scalar.tabulate_dof_coordinates()[:, :3])
    vtx_to_dof = cKDTree(p1_coords).query(vertices, k=1)[1]
    vp1_vec = fem.functionspace(
        msh, basix.ufl.element("Lagrange", msh.basix_cell(), 1, shape=(3,))
    )

    def _at_vertices(
        sub_index: int, eigvec: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Collapses sub-field ``sub_index`` of an eigenvector, interpolates it to the
        P1 vector space, and reorders it into geometry-vertex order.

        :param sub_index: Sub-space index into the mixed function space ``V``
            (0 for displacement, 1 for rotation).
        :param eigvec: Full eigenvector array, indexed by the mixed space's dofs.
        :returns: Per-vertex vector field, shape ``(n_vertices, 3)``.
        """
        sub_space, sub_map = V.sub(sub_index).collapse()
        f_sub = fem.Function(sub_space)
        f_sub.x.array[:] = eigvec[sub_map]
        f_p1 = fem.Function(vp1_vec)
        f_p1.interpolate(f_sub)
        return f_p1.x.array.reshape(-1, 3)[vtx_to_dof].astype(np.float64)

    extents = vertices.max(axis=0) - vertices.min(axis=0)
    positive = extents[extents > 1.0e-12]
    deformation_scale = float(scale_factor) * (
        float(positive.min()) if positive.size else 1.0
    )

    payloads: list[dict] = []
    for i in valid:
        eigvec = result.eigenvectors[:, i]
        disp = _at_vertices(0, eigvec)
        rot = _at_vertices(1, eigvec)
        # Canonical sign: an eigenvector is defined only up to sign, so FEM and IGA can
        # disagree by a global minus. Flip the whole mode so its largest-magnitude
        # displacement component is positive -- the same rule the IGA exporter applies,
        # so simple (non-degenerate) modes land on a matching sign for side-by-side
        # visualization. (Degenerate/repeated eigenvalues need subspace alignment, not
        # just a sign -- see note in the comparison step.)
        flat = disp.reshape(-1)
        if flat.size and flat[int(np.argmax(np.abs(flat)))] < 0.0:
            disp = -disp
            rot = -rot
        disp_max = float(np.max(np.linalg.norm(disp, axis=1)))
        disp_unit = disp / disp_max if disp_max > 1.0e-12 else disp
        payloads.append(
            {
                "mode_index": int(i),
                "eigenvalue": float(result.eigenvalues[i]),
                "frequency": float(result.frequencies[i]),
                "geometry": vertices - disp_unit * deformation_scale,
                "displacement": disp,
                "rotation": rot,
                "displacement_magnitude": np.linalg.norm(disp, axis=1).reshape(-1, 1),
                "rotation_magnitude": np.linalg.norm(rot, axis=1).reshape(-1, 1),
            }
        )

    output_folder.mkdir(parents=True, exist_ok=True)
    stem = output_folder / "eigenmodes"
    h5_path = stem.with_suffix(".h5")
    xdmf_path = stem.with_suffix(".xdmf")
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset("/Mesh/topology", data=faces.astype(np.int32))
        # Undeformed vertices, for the post-hoc MAC alignment step (align_modal_xdmf).
        h5_file.create_dataset("/Mesh/vertices", data=vertices.astype(np.float64))
        for k, mode in enumerate(payloads):
            for key in (
                "geometry",
                "displacement",
                "rotation",
                "displacement_magnitude",
                "rotation_magnitude",
            ):
                h5_file.create_dataset(
                    f"/Modes/mode_{k}/{key}",
                    data=np.asarray(mode[key], dtype=np.float64),
                )
    _write_eigenmodes_xdmf(
        xdmf_path, h5_path.name, n_vertices, faces.shape[0], payloads
    )
    logger.info(
        f"Exported {len(payloads)} mode shapes to {xdmf_path} "
        f"(IGA-compatible eigenmodes format)."
    )


def _write_eigenmodes_xdmf(
    xdmf_path: Path,
    h5_name: str,
    num_vertices: int,
    num_triangles: int,
    mode_payloads: list[dict],
) -> None:
    """
    Writes the temporal-collection XDMF descriptor.

    Identical structure to the IGA runner's ``_write_temporal_xdmf`` so the two
    outputs are interchangeable.

    :param xdmf_path: Output path for the ``.xdmf`` descriptor.
    :param h5_name: Name of the companion ``.h5`` file, referenced by relative path
        inside the XDMF ``DataItem`` entries.
    :param num_vertices: Number of geometry vertices, shared by every mode grid.
    :param num_triangles: Number of triangle cells, shared by every mode grid.
    :param mode_payloads: Per-mode data (as built in ``_export_mode_shapes_xdmf``),
        one dict per exported mode, in export order. Keyed by ``"mode_index"``
        (int), ``"eigenvalue"`` / ``"frequency"`` (float) and the five array
        datasets ``"geometry"``, ``"displacement"``, ``"rotation"``,
        ``"displacement_magnitude"``, ``"rotation_magnitude"``. Left as a bare
        ``dict`` rather than ``dict[str, object]`` because the value type is
        genuinely heterogeneous, and pinning it to ``object`` only makes the
        ``int(...)`` / ``float(...)`` reads below unresolvable to a type checker.
    """

    def _attr(name: str, attr_type: str, dataset: str, dims: str) -> list[str]:
        """Emits the XDMF markup for one node-centred attribute."""
        return [
            f'        <Attribute Name="{name}" '
            f'AttributeType="{attr_type}" Center="Node">',
            f'          <DataItem Dimensions="{dims}" '
            f'Format="HDF">{h5_name}:{dataset}</DataItem>',
            "        </Attribute>",
        ]

    lines = [
        '<?xml version="1.0"?>',
        '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>',
        '<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">',
        "  <Domain>",
        '    <Grid Name="eigenmodes" GridType="Collection" '
        'CollectionType="Temporal">',
    ]
    for k, mode in enumerate(mode_payloads):
        original = int(mode["mode_index"])
        lines.extend(
            [
                f'      <Grid Name="mode_{original}" GridType="Uniform">',
                '        <Topology TopologyType="Triangle" '
                f'NumberOfElements="{num_triangles}" NodesPerElement="3">',
                f'          <DataItem Dimensions="{num_triangles} 3" '
                f'NumberType="Int" Format="HDF">{h5_name}:/Mesh/topology</DataItem>',
                "        </Topology>",
                '        <Geometry GeometryType="XYZ">',
                f'          <DataItem Dimensions="{num_vertices} 3" '
                f'Format="HDF">{h5_name}:/Modes/mode_{k}/geometry</DataItem>',
                "        </Geometry>",
                f'        <Time Value="{k}" />',
                f"        <!-- mode_index={original} "
                f'lambda={float(mode["eigenvalue"]):.6e} '
                f'f={float(mode["frequency"]):.6f} Hz -->',
            ]
        )
        lines.extend(_attr("displacement", "Vector",
                           f"/Modes/mode_{k}/displacement", f"{num_vertices} 3"))
        lines.extend(_attr("rotation", "Vector",
                           f"/Modes/mode_{k}/rotation", f"{num_vertices} 3"))
        lines.extend(_attr("displacement_magnitude", "Scalar",
                           f"/Modes/mode_{k}/displacement_magnitude",
                           f"{num_vertices} 1"))
        lines.extend(_attr("rotation_magnitude", "Scalar",
                           f"/Modes/mode_{k}/rotation_magnitude",
                           f"{num_vertices} 1"))
        lines.append("      </Grid>")
    lines.extend(["    </Grid>", "  </Domain>", "</Xdmf>"])
    with open(xdmf_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def run_modal_analysis(params: ShellModalAnalysisParameters) -> ModalResult:
    """
    Runs the FEM shell modal analysis and persists the results.

    Sets the Mitsuba variant to ``llvm_ad_mono`` (a global side effect), then loads
    and rescales the spline, builds the shell mesh at ``params.geometry_order``
    (curved for the default 2, flat facets for 1), assembles and eigensolves the
    shell problem, checks the rigid-body-mode band, saves the results to
    ``params.output_folder``, and optionally exports the mode shapes to XDMF.

    :param params: Modal analysis configuration.
    :returns: The modal solve result (eigenvalues, eigenvectors, frequencies, and
        ``mode_energies`` -- the latter only populated when
        ``params.use_scipy_solver`` is set). Also written to ``params.output_folder``.
    """
    mi.set_variant("llvm_ad_mono")
    params.output_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading spline from {params.spline_path}")
    tessellated = load_and_rescale_spline(
        spline_path=params.spline_path,
        target_length_scale=params.target_length_scale,
    )

    logger.info(
        f"Building curved (order {params.geometry_order}) mesh: "
        f"{params.subdivisions_per_patch} subdivisions/patch...")
    t0 = time.time()
    msh = build_shell_mesh(
        tessellated_spline=tessellated,
        subdivisions_per_patch=params.subdivisions_per_patch,
        order=params.geometry_order,
    )
    n_cells = msh.topology.index_map(msh.topology.dim).size_global
    logger.info(f"Mesh: {msh.geometry.x.shape[0]} geometry nodes, {n_cells} cells "
                f"(built in {time.time() - t0:.1f}s).")

    mc = params.material_config
    element = ShellElement(
        degree=params.field_degree,
        bbar=params.use_bbar,
        psri=params.use_psri,
        drilling_stiffness_factor=params.drilling_stiffness_factor,
        normal="geometric",
    )
    solver = GeometricallyExactShellSolver(
        msh,
        youngs_modulus=mc.youngs_modulus,
        poisson_ratio=mc.poisson_ratio,
        density=mc.density,
        thickness=mc.thickness,
        element=element,
        shear_correction=mc.shear_correction_factor,
    )
    logger.info(f"Shell space: {element.label()}, {solver.ndofs} DOFs. "
                f"Material: E={mc.youngs_modulus:.3e} nu={mc.poisson_ratio} "
                f"rho={mc.density} h={mc.thickness}.")

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

    _report_rigid_body_modes(
        result.eigenvalues, result.frequencies, params.n_rigid_body_modes)
    logger.info("Natural frequencies (Hz):\n"
                + np.array2string(result.frequencies, precision=4))

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
        _export_mode_shapes_xdmf(
            solver, result, params.export_xdmf_indices, params.output_folder)

    return result


if __name__ == "__main__":
    parameters = tyro.cli(ShellModalAnalysisParameters)
    run_modal_analysis(params=parameters)
