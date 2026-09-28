import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from dolfinx import mesh
from mpi4py import MPI

from fem.dolfinx_mesh import create_manifold_mesh
from fem.end_to_end.mesh_quality.triangle_quality import TriangleQuality
from fem.end_to_end.modal_gt.fem_modal_gt_report import FEMModalGTReport
from fem.meshprep.cleaned_mesh import CleanedMesh
from fem.meshprep.mesh_repair import drop_degenerate_triangles
from fem.petsc_factor_solver import FactorSolver
from fem.shells.eigenmode_xdmf import export_mode_shapes_xdmf
from fem.shells.geometrically_exact_shell_modal_solver import (
    GeometricallyExactShellModalSolver,
)
from fem.shells.geometrically_exact_shell_problem import GeometricallyExactShellProblem
from fem.shells.modal_result import ModalResult
from fem.shells.rigid_body_modes import detect_rigid_body_modes, report_rigid_body_modes
from fem.shells.shell_element import ShellElement
from forge_sim import logger
from forge_sim.scripts.configs.structural_material_config import (
    StructuralMaterialConfig,
)


class FEMShellModalGTRunner:
    """
    Runs the geometrically-exact shell modal solve on a baked ground-truth
    triangle mesh, and persists the results.

    Holds the resolved run configuration every step of `run` reads (never
    the script's own `FEMShellModalGTMeshParameters`), so that this module
    never imports from the entry script that imports it. `run` is the entry
    point a caller reaches for. `build_gt_shell_mesh` is public because the
    driver's own tests exercise the mesh handover directly, without a solve.
    """

    @dataclass(frozen=True, kw_only=True)
    class CurvedGeometryArrays:
        """The baked order-2 geometry `_curved_geometry_arrays` reads out of a
        ground-truth `.npz`, or both fields `None` for a flat mesh."""

        edge_nodes: npt.NDArray[np.float64] | None
        """Mid-edge node positions, one per unique mesh edge."""
        edge_index: npt.NDArray[np.int64] | None
        """Per-face edge indices into `edge_nodes`, in `basix_edge_table` column
        order."""

    @dataclass(frozen=True, kw_only=True)
    class ProblemAndSolver:
        """The assembled shell problem `_build_problem` built, paired with the
        modal solver constructed over it."""

        problem: GeometricallyExactShellProblem
        """The assembled shell problem."""
        solver: GeometricallyExactShellModalSolver
        """The modal solver constructed over `problem`."""

    @dataclass(frozen=True, kw_only=True)
    class SolveResult:
        """The eigenpairs `_solve` produced, timed."""

        result: ModalResult
        """The eigenpairs, natural frequencies and per-mode energy split."""
        solve_seconds: float
        """Wall-clock time the eigensolve took."""

    def __init__(
        self,
        *,
        mesh_npz: Path,
        output_folder: Path,
        material_config: StructuralMaterialConfig,
        element: ShellElement,
        clean_degenerate: bool,
        min_triangle_area_ratio: float,
        num_modes: int,
        spectral_shift: float | None,
        solver_tolerance: float,
        factor_solver: FactorSolver,
        use_scipy_solver: bool,
        export_xdmf: bool,
        export_xdmf_indices: list[int],
        num_rb_modes: int,
    ) -> None:
        """Stores the resolved run configuration; does no I/O."""
        self._mesh_npz = mesh_npz
        self._output_folder = output_folder
        self._material_config = material_config
        self._element = element
        self._clean_degenerate = clean_degenerate
        self._min_triangle_area_ratio = min_triangle_area_ratio
        self._num_modes = num_modes
        self._spectral_shift = spectral_shift
        self._solver_tolerance = solver_tolerance
        self._factor_solver = factor_solver
        self._use_scipy_solver = use_scipy_solver
        self._export_xdmf = export_xdmf
        self._export_xdmf_indices = export_xdmf_indices
        self._num_rb_modes = num_rb_modes

    def run(self) -> ModalResult:
        """Runs the FEM shell modal analysis on a GT mesh and persists the results."""
        self._output_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading ground-truth mesh from {self._mesh_npz}")
        data = np.load(self._mesh_npz)
        vertices = np.asarray(data["vertices"], dtype=np.float64)
        faces = np.asarray(data["faces"], dtype=np.int64)
        scale = float(data["scale"]) if "scale" in data.files else float("nan")
        n_vertices_in, n_faces_in = vertices.shape[0], faces.shape[0]
        skew_before = self._log_loaded_mesh(vertices, faces, scale)

        cleaned = self._clean_mesh(vertices, faces)
        curved_geometry = self._curved_geometry_arrays(data, cleaned.num_dropped)

        msh = self.build_gt_shell_mesh(
            cleaned.vertices,
            cleaned.faces,
            edge_nodes=curved_geometry.edge_nodes,
            edge_index=curved_geometry.edge_index,
        )
        self._log_dolfinx_mesh(msh)

        problem_and_solver = self._build_problem(msh)
        solved = self._solve(problem_and_solver.solver)
        result = solved.result

        report_rigid_body_modes(
            result.eigenvalues, result.frequencies, self._num_rb_modes
        )
        logger.info(
            "Natural frequencies (Hz):\n"
            + np.array2string(result.frequencies, precision=4)
        )

        self._save_results(result)
        if self._export_xdmf:
            export_mode_shapes_xdmf(
                problem_and_solver.problem.V,
                result,
                self._export_xdmf_indices,
                self._output_folder,
            )

        self._write_log_json(
            result,
            n_vertices_in=n_vertices_in,
            n_faces_in=n_faces_in,
            n_dropped=cleaned.num_dropped,
            n_triangles=cleaned.faces.shape[0],
            max_skew_before=skew_before,
            solve_seconds=solved.solve_seconds,
        )
        return result

    @staticmethod
    def _log_loaded_mesh(
        vertices: npt.NDArray[np.float64], faces: npt.NDArray[np.int64], scale: float
    ) -> float:
        """Logs the loaded mesh and its quality; returns the max equiangle skew."""
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        logger.info(
            f"GT mesh: {vertices.shape[0]} vertices, {faces.shape[0]} triangles, "
            f"bbox extent {np.array2string(extent, precision=4)} "
            f"(longest {extent.max():.4f} m), baked scale {scale:.6g}."
        )
        quality = TriangleQuality.from_mesh(vertices, faces)
        skew_before = float(quality.skewness.max())
        areas = quality.area
        logger.info(
            f"Mesh quality before cleaning: max equiangle skew {skew_before:.4f}, "
            f"min/mean/max triangle area {areas.min():.3e}/{areas.mean():.3e}/"
            f"{areas.max():.3e}."
        )
        return skew_before

    def _clean_mesh(
        self, vertices: npt.NDArray[np.float64], faces: npt.NDArray[np.int64]
    ) -> CleanedMesh:
        """
        Drops zero-/near-zero-area triangles when `clean_degenerate` is set, and
        logs the result. A no-op, reporting zero dropped, otherwise.
        """
        if not self._clean_degenerate:
            return CleanedMesh(vertices=vertices, faces=faces, num_dropped=0)

        cleaned = drop_degenerate_triangles(
            vertices, faces, self._min_triangle_area_ratio
        )
        if cleaned.num_dropped:
            quality = TriangleQuality.from_mesh(cleaned.vertices, cleaned.faces)
            skew_after = float(quality.skewness.max())
            logger.info(
                f"Dropped {cleaned.num_dropped} degenerate triangle(s) "
                f"(area < {self._min_triangle_area_ratio:g} * mean); "
                f"max equiangle skew now {skew_after:.4f}, "
                f"{cleaned.vertices.shape[0]} vertices / "
                f"{cleaned.faces.shape[0]} triangles remain."
            )
        else:
            logger.info("No degenerate triangles found; mesh unchanged.")
        return cleaned

    @staticmethod
    def _curved_geometry_arrays(
        data: np.lib.npyio.NpzFile, n_dropped: int
    ) -> CurvedGeometryArrays:
        """
        Reads the baked mid-edge nodes and their per-face edge index, or both
        fields `None` for a flat mesh.

        Dropping degenerate triangles renumbers faces, which invalidates the
        baked per-face edge indices, so this falls back to flat rather than
        mis-wire it, since the two arrays travel through the `.npz` from a separate
        process with no shared object to disagree with.
        """
        edge_nodes = (
            np.asarray(data["edge_nodes"]) if "edge_nodes" in data.files else None
        )
        edge_index = (
            np.asarray(data["edge_index"]) if "edge_index" in data.files else None
        )
        if edge_nodes is not None and n_dropped:
            logger.warning(
                f"Curved geometry data discarded: {n_dropped} degenerate triangle(s) "
                "were dropped, so the baked mid-edge indices no longer match the "
                "mesh. Falling back to order 1 (re-bake to keep the curved geometry)."
            )
            return FEMShellModalGTRunner.CurvedGeometryArrays(
                edge_nodes=None, edge_index=None
            )
        if edge_nodes is not None:
            logger.info(
                f"Curved geometry: {edge_nodes.shape[0]} mid-edge nodes on the "
                "original surface (order 2)."
            )
        return FEMShellModalGTRunner.CurvedGeometryArrays(
            edge_nodes=edge_nodes, edge_index=edge_index
        )

    @staticmethod
    def build_gt_shell_mesh(
        vertices: npt.NDArray[np.float64],
        faces: npt.NDArray[np.int64],
        comm: MPI.Comm = MPI.COMM_SELF,
        edge_nodes: npt.NDArray[np.float64] | None = None,
        edge_index: npt.NDArray[np.int64] | None = None,
    ) -> mesh.Mesh:
        """
        Builds a DOLFINx triangle surface mesh (tdim=2, gdim=3) from raw vertex/face
        arrays: order 1 (flat facets), or order 2 (curved) when `edge_nodes` and
        `edge_index` are supplied.

        The analysis mesh is a remeshed, coarsened stand-in for the ground-truth
        surface, not the surface itself, so its flat facets chord across the
        geometry and lose area. Shell mass is `rho * h * area`, making that a
        physical mass deficit which biases frequencies high. Order 2 recovers the
        chord part of it by placing each mid-edge node on the source surface, at no
        cost in field DOFs. The projection is done once by the baking stage and
        arrives here as the two arrays, so the two coarse meshes cannot disagree
        about where the surface is. The geometrically-exact shell measures remain
        rigid-body invariant at either order, so the six-zero-mode check holds
        regardless.

        `vertices` is the vertex positions, shape `(n, 3)`, and `faces` is the
        triangle vertex indices, shape `(m, 3)`. `comm` is the MPI communicator
        the mesh is distributed over. `edge_nodes` is the mid-edge node positions,
        one per unique mesh edge, or `None` for a flat mesh; `edge_index` is the
        per-face edge indices into `edge_nodes`, in the `basix_edge_table`
        column order, or `None` for a flat mesh.

        :returns: The DOLFINx mesh, of geometry order 1 or 2 accordingly.
        :raises ValueError: If `edge_index` has a different number of rows than
            `faces`, which means the mesh was modified after the nodes were baked.
        """
        p1_vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        p1_faces = np.ascontiguousarray(faces, dtype=np.int64)

        if edge_nodes is None or edge_index is None:
            return create_manifold_mesh(comm, p1_faces, p1_vertices, degree=1)

        # Order 2: mid-edge nodes already projected onto the original (un-remeshed)
        # source surface by the baking stage. The assumed cell layout
        # [v0, v1, v2, e12, e02, e01] is defined by
        # fem.meshprep.p2_geometry.basix_edge_table, which is what ordered
        # edge_index's columns when the baking stage (a separate process,
        # reaching us only through the npz) wrote them.
        if edge_index.shape[0] != p1_faces.shape[0]:
            raise ValueError(
                f"curved mesh data is stale: {edge_index.shape[0]} edge-index rows "
                f"for {p1_faces.shape[0]} faces (the mesh was modified after baking)"
            )
        points = np.vstack([p1_vertices, np.ascontiguousarray(edge_nodes, np.float64)])
        p2_cells = np.column_stack(
            [
                p1_faces,
                p1_vertices.shape[0] + np.ascontiguousarray(edge_index, np.int64),
            ]
        )
        return create_manifold_mesh(comm, p2_cells, points, degree=2)

    @staticmethod
    def _log_dolfinx_mesh(msh: mesh.Mesh) -> None:
        """Logs the DOLFINx mesh handed back by `build_gt_shell_mesh`."""
        n_cells = msh.topology.index_map(msh.topology.dim).size_global
        n_nodes = msh.geometry.x.shape[0]
        logger.info(
            f"DOLFINx mesh: {n_nodes} geometry nodes, {n_cells} cells "
            f"(geometry order {msh.geometry.cmap.degree})."
        )

    def _build_problem(self, msh: mesh.Mesh) -> ProblemAndSolver:
        """Assembles the shell problem and its modal solver, and logs the space."""
        mc = self._material_config
        problem = GeometricallyExactShellProblem(
            msh,
            youngs_modulus=mc.youngs_modulus,
            poisson_ratio=mc.poisson_ratio,
            density=mc.density,
            thickness=mc.thickness,
            element=self._element,
            shear_correction=mc.shear_correction_factor,
        )
        solver = GeometricallyExactShellModalSolver(problem)
        logger.info(
            f"Shell space: {self._element.label()}, {problem.ndofs} DOFs. "
            f"Material: E={mc.youngs_modulus:.3e} nu={mc.poisson_ratio} "
            f"rho={mc.density} h={mc.thickness}."
        )
        return FEMShellModalGTRunner.ProblemAndSolver(problem=problem, solver=solver)

    def _solve(self, solver: GeometricallyExactShellModalSolver) -> SolveResult:
        """Runs the eigensolve (scipy or SLEPc, per `use_scipy_solver`) and times it."""
        t0 = time.time()
        if self._use_scipy_solver:
            result = solver.solve_modes(num_modes=self._num_modes)
        else:
            result = solver.solve_modes_slepc(
                num_modes=self._num_modes,
                spectral_shift=self._spectral_shift,
                tol=self._solver_tolerance,
                factor_solver=self._factor_solver,
            )
        solve_seconds = time.time() - t0
        logger.info(f"Eigensolve completed in {solve_seconds:.1f}s.")
        return FEMShellModalGTRunner.SolveResult(
            result=result, solve_seconds=solve_seconds
        )

    def _save_results(self, result: ModalResult) -> None:
        """
        Saves the eigenpairs to `modal_analysis_results.npz`.

        `mode_energies` is the whole reason to choose the scipy solve path; the
        SLEPc path leaves it zero, which is its documented "not computed" value.
        Saved either way so the archive layout does not depend on which solver
        produced it.
        """
        output_path = self._output_folder / "modal_analysis_results.npz"
        np.savez(
            output_path,
            eigenvalues=result.eigenvalues,
            eigenvectors=result.eigenvectors,
            frequencies=result.frequencies,
            mode_energies=result.mode_energies,
        )
        logger.info(f"Saved modal results to {output_path}.")

    def _write_log_json(
        self,
        result: ModalResult,
        *,
        n_vertices_in: int,
        n_faces_in: int,
        n_dropped: int,
        n_triangles: int,
        max_skew_before: float,
        solve_seconds: float,
    ) -> None:
        """Writes the run's record beside the mode-shape archive it describes."""
        num_rb = self._num_rb_modes
        eig = result.eigenvalues
        rb_to_elastic_gap = (
            float(abs(eig[num_rb]) / max(abs(eig[num_rb - 1]), 1e-300))
            if len(eig) > num_rb
            else float("nan")
        )
        detected_rb = detect_rigid_body_modes(eig, num_rb).count

        summary = FEMModalGTReport(
            mesh_npz=str(self._mesh_npz),
            material=self._material_config,
            element=FEMModalGTReport.Element(
                field_degree=self._element.degree,
                use_bbar=self._element.locking_cure is ShellElement.LockingCure.BBAR,
                use_psri=self._element.locking_cure is ShellElement.LockingCure.PSRI,
                project_rotary_inertia=self._element.project_rotary_inertia,
            ),
            ndofs=int(result.ndofs),
            num_vertices_input=int(n_vertices_in),
            num_triangles_input=int(n_faces_in),
            num_degenerate_triangles_dropped=int(n_dropped),
            num_triangles=int(n_triangles),
            max_equiangle_skew_input=float(max_skew_before),
            num_rigid_body_modes_expected=int(num_rb),
            num_rigid_body_modes_detected=int(detected_rb),
            rb_to_first_elastic_gap_ratio=rb_to_elastic_gap,
            solve_seconds=float(solve_seconds),
            eigenvalues=[float(v) for v in result.eigenvalues],
            frequencies_hz=[float(f) for f in result.frequencies],
        )
        log_path = self._output_folder / "fem_modal_gt_log.json"
        log_path.write_text(json.dumps(asdict(summary), indent=2))
        logger.info(f"Wrote run summary to {log_path}.")
