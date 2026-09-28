from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np

from forge_sim import logger
from forge_sim.isogeometric_analysis.eigenmode_xdmf_exporter import (
    EigenmodeXDMFExporter,
)
from forge_sim.isogeometric_analysis.modal_analysis_solver import ModalAnalysisSolver
from forge_sim.isogeometric_analysis.reissner_mindlin_modal_problem import (
    ReissnerMindlinModalProblem,
)
from forge_sim.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)
from forge_sim.meshing.tessellated_spline_surface import TessellatedSplineSurface
from forge_sim.scripts.configs.structural_material_config import (
    StructuralMaterialConfig,
)


def apply_h_refinement(
    tessellated_spline: TessellatedSplineSurface,
    h_refinement_factor: int = 1,
    target_spans: int | None = None,
) -> None:
    """
    Uniformly h-refines `tessellated_spline` in place.

    The factor is `h_refinement_factor`, unless `target_spans` is given: that
    OVERRIDES it and is resolved against the loaded spline's own base span count
    (`factor = round(target_spans / base_spans)`, at least 1), so a fixed
    effective resolution can be pinned across splines with different base spans.
    A resolved factor of 1 leaves the spline untouched.

    :param tessellated_spline: Surface to refine in place.
    :param h_refinement_factor: Uniform refinement factor per parametric direction.
    :param target_spans: Desired knot spans per direction; overrides
        `h_refinement_factor` when set.
    """
    factor = h_refinement_factor
    if target_spans is not None:
        first_patch = next(iter(tessellated_spline.composite_spline))
        base_spans = first_patch.interpolating_spline.u_spline.knot_spans.shape[0]
        factor = max(1, round(target_spans / base_spans))
        effective_spans = base_spans * factor
        if effective_spans != target_spans:
            logger.warning(
                f"target_spans={target_spans} not an integer multiple of the "
                f"{base_spans}-span base; using factor {factor} -> "
                f"{effective_spans} spans."
            )
        logger.info(
            f"target_spans={target_spans}: base {base_spans} spans/dir -> "
            f"h-refinement factor {factor} (-> {effective_spans} spans/dir)"
        )
    if factor > 1:
        logger.info(
            f"Applying h-refinement with factor {factor} to the input spline surface."
        )
        tessellated_spline.composite_spline.h_refinement(factor)


def prepare_modal_geometry(
    spline_path: Path,
    h_refinement_factor: int = 1,
    target_spans: int | None = None,
    target_length_scale: float | None = None,
) -> TessellatedIGASplineSurface:
    """
    Loads a saved spline surface and brings it into the state the modal analysis
    expects: h-refined, carrying the six shell DOFs per control point, and
    optionally rescaled.

    Shared by every modal entry point so that the geometry handed to
    `ReissnerMindlinModalRunner` is prepared identically, otherwise the
    resulting eigenfrequencies are not comparable between them.

    The steps, in order, are:

    1. load the surface and h-refine it (see `apply_h_refinement`),
    2. attach zeroed shell field values (the modal solution workspace),
    3. if `target_length_scale` is set, rescale so the longest axis of the
       control-point bounding box matches it.

    :param spline_path: Path to the saved (tessellated) spline surface.
    :param h_refinement_factor: Uniform refinement factor per parametric direction.
    :param target_spans: Desired knot spans per direction; overrides
        `h_refinement_factor` when set.
    :param target_length_scale: Target length for the longest axis of the
        CONTROL-POINT bounding box. `None` leaves the geometry at the scale it
        was loaded.

    :returns: The prepared IGA surface.

    :raises ValueError: If `target_length_scale` is set but the control-point
        bounding box is degenerate.
    """
    tessellated_spline = TessellatedSplineSurface.from_file(
        composite_spline_surface_path=spline_path
    )

    apply_h_refinement(
        tessellated_spline=tessellated_spline,
        h_refinement_factor=h_refinement_factor,
        target_spans=target_spans,
    )

    per_patch_field_values = [
        dr.zeros(
            dtype=mi.TensorXf,
            shape=(
                patch.control_points.shape[0],
                ReissnerMindlinModalProblem.DOFS_PER_CONTROL_POINT,
            ),
        )
        for patch in tessellated_spline.composite_spline
    ]

    tessellated_iga_spline = (
        TessellatedIGASplineSurface.from_tessellated_spline_surface(
            tessellated_spline_surface=tessellated_spline,
            per_patch_field_values=per_patch_field_values,
        )
    )


    if target_length_scale is not None:
        control_points_np = tessellated_iga_spline.global_control_points.numpy()
        bbox_min = np.min(control_points_np, axis=0)
        bbox_max = np.max(control_points_np, axis=0)
        bbox_extents = bbox_max - bbox_min
        longest_extent = float(np.max(bbox_extents))

        if longest_extent <= 1.0e-12:
            raise ValueError(
                "Cannot apply target-length scaling: longest bbox extent is near zero."
            )

        target_scale = target_length_scale / longest_extent
        tessellated_iga_spline.global_control_points = (
            tessellated_iga_spline.global_control_points * target_scale
        )

        logger.info(
            f"Applied target-length scaling with factor {target_scale:.6f} to match "
            + f"target length scale {target_length_scale}. "
            + f"Original longest extent was {longest_extent:.6f}."
        )

    return tessellated_iga_spline


class ReissnerMindlinModalRunner:
    """
    Runner for Reissner-Mindlin modal analysis on a composite IGA spline
    surface. Wraps assembly, eigenvalue solve, and XDMF export.
    """

    def __init__(
        self,
        tessellated_iga_spline: TessellatedIGASplineSurface,
        material_config: StructuralMaterialConfig | None = None,
    ) -> None:
        """
        Initialize the modal analysis runner with a tessellated IGA spline
        surface and material properties.

        Material properties are broadcast from the `material_config` to
        all control points as uniform fields.

        :param tessellated_iga_spline: The tessellated IGA spline surface to
            analyze.
        :param material_config: Structural material properties (E, nu, rho, h).
            Defaults to a freshly constructed `StructuralMaterialConfig`. (It is
            built here rather than in the signature so that every runner gets its
            own instance instead of sharing one built at import time.)
        """
        if material_config is None:
            material_config = StructuralMaterialConfig()

        self._tessellated_iga_spline = tessellated_iga_spline
        self._iga_spline = tessellated_iga_spline.composite_spline

        ones = dr.ones(
            dtype=mi.TensorXf,
            shape=(self._iga_spline.global_field_values.shape[0], 1),
        )

        self._youngs_modulus = material_config.youngs_modulus * ones
        self._poisson_ratio = material_config.poisson_ratio * ones
        self._density = material_config.density * ones
        self._thickness = material_config.thickness * ones
        self._shear_correction_factor = material_config.shear_correction_factor

    def run_modal_analysis(
        self,
        num_modes: int,
        output_folder: Path,
        solver_tolerance: float = 1.0e-8,
        max_iterations: int | None = None,
        spectral_shift: float | None = None,
        integration_order: int | None = None,
        drilling_stiffness_factor: float = 1.0,
        max_batch_size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Runs complete modal analysis and saves results to disk.

        This method creates a Reissner-Mindlin modal problem from the current
        material properties and geometry, solves the generalized eigenvalue
        problem to find natural frequencies and mode shapes, and saves the
        results as a `.npz` file.

        :param num_modes: Number of eigenmodes to compute.
        :param output_folder: Directory where results will be saved as
            `modal_analysis_results.npz`.
        :param solver_tolerance: Convergence tolerance for the eigenvalue
            solver.
        :param max_iterations: Maximum iterations for the eigenvalue solver.
            If `None`, the solver default is used.
        :param spectral_shift: Mass-proportional shift added to the stiffness
            matrix to make it positive definite for Cholesky factorization.
            If `None`, an automatic heuristic is used.
        :param integration_order: Quadrature order for patch integration.
            If `None`, defaults to the modal problem heuristic
            (`2 * degree + 1`).
        :param drilling_stiffness_factor: Scale factor for the objective
            drilling penalty.
        :param max_batch_size: Maximum number of knot spans to assemble at
            once. When set, assembly memory is bounded to
            O(max_batch_size * n_dofs_per_span^2). A value of 1024 is
            recommended for large meshes (64x64+). When `None`, all spans
            are assembled at once.

        :returns: A tuple of `(eigenvalues, eigenvectors)` where
            eigenvalues is sorted ascending and eigenvectors has shape
            `(n_dofs, n_modes)`.

        :raises ValueError: If `num_modes` is less than 1, or if the computed
            spectrum contains a significantly negative eigenvalue.
        """
        if num_modes < 1:
            raise ValueError("`num_modes` must be a positive integer.")
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)

        modal_problem = ReissnerMindlinModalProblem(
            composite_iga_spline=self._iga_spline,
            youngs_modulus=self._youngs_modulus,
            poisson_ratio=self._poisson_ratio,
            density=self._density,
            thickness=self._thickness,
            shear_correction_factor=self._shear_correction_factor,
            drilling_stiffness_factor=drilling_stiffness_factor,
            integration_order=integration_order,
            max_batch_size=max_batch_size,
        )

        solver = ModalAnalysisSolver(modal_problem=modal_problem)
        eigenvalues, eigenvectors = solver.solve(
            num_modes=num_modes,
            solver_tolerance=solver_tolerance,
            max_iterations=max_iterations,
            spectral_shift=spectral_shift,
        )

        logger.info(
            f"Modal analysis completed. Found the following eigenvalues:\n{eigenvalues}"
        )
        # The mass-proportional spectral shift is there precisely to keep the
        # spectrum non-negative: the rigid body modes come out at a small
        # positive eigenvalue. A significantly negative eigenvalue therefore
        # means the shift was too small (in practice the Cholesky factorization
        # of K + epsilon * M fails first) and the results cannot be trusted.
        # The threshold is relative to the spectrum so that the near-zero rigid
        # body modes, which straddle zero by round-off, do not trip it.
        negative_threshold = -1.0e-3 * np.max(np.abs(eigenvalues))
        negative_eigenvalues = eigenvalues[eigenvalues < negative_threshold]
        if negative_eigenvalues.size > 0:
            raise ValueError(
                f"Modal analysis produced {negative_eigenvalues.size} significantly "
                f"negative eigenvalues (most negative: "
                f"{np.min(negative_eigenvalues):.4e}, threshold: "
                f"{negative_threshold:.4e}). The spectral shift was insufficient "
                "to regularize the stiffness matrix."
            )
        frequencies = np.sqrt(np.abs(eigenvalues)) / (2.0 * np.pi)
        logger.info(f"Corresponding frequencies (Hz):\n{frequencies}")

        output_path = output_folder / "modal_analysis_results.npz"
        np.savez(
            output_path,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )
        logger.info(f"Modal analysis results saved to {output_path}.")

        return eigenvalues, eigenvectors

    def export_eigenmodes_to_xdmf(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        mode_indices: list[int],
        output_folder: Path,
    ) -> None:
        """
        Export eigenmodes to a temporal XDMF/HDF5 series for ParaView.

        Thin wrapper around `EigenmodeXDMFExporter`, which owns the on-disk
        format, bound to the surface this runner analyzed.

        :param eigenvalues: Array of eigenvalues from modal analysis.
        :param eigenvectors: Eigenvector matrix with shape
            `(n_dofs, n_modes)`.
        :param mode_indices: List of mode indices to export (e.g.,
            `[0, 1, 2]`).
        :param output_folder: Directory where `eigenmodes.xdmf` and
            `eigenmodes.h5` will be saved.

        :raises ValueError: If any mode index is out of range.
        """
        exporter = EigenmodeXDMFExporter(
            tessellated_iga_spline=self._tessellated_iga_spline,
            output_folder=output_folder,
        )
        exporter.export_eigenmodes(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            mode_indices=mode_indices,
        )

    @property
    def youngs_modulus(self) -> "mi.TensorXf":
        """Get the current Young's modulus field."""
        return self._youngs_modulus

    @youngs_modulus.setter
    def youngs_modulus(self, new_youngs_modulus: "mi.TensorXf") -> None:
        """
        Set a new Young's modulus field for heterogeneous materials.

        :param new_youngs_modulus: Tensor with shape `(n_control_points, 1)`.

        :raises ValueError: If the shape does not match the existing field.
        """
        if new_youngs_modulus.shape != self._youngs_modulus.shape:
            raise ValueError(
                "`value` shape must match the existing youngs modulus shape."
            )
        self._youngs_modulus = new_youngs_modulus

    @property
    def poisson_ratio(self) -> "mi.TensorXf":
        """Get the current Poisson's ratio field."""
        return self._poisson_ratio

    @poisson_ratio.setter
    def poisson_ratio(self, new_poisson_ratio: "mi.TensorXf") -> None:
        """
        Set a new Poisson's ratio field for heterogeneous materials.

        :param new_poisson_ratio: Tensor with shape `(n_control_points, 1)`.

        :raises ValueError: If the shape does not match the existing field.
        """
        if new_poisson_ratio.shape != self._poisson_ratio.shape:
            raise ValueError(
                "`value` shape must match the existing poisson ratio shape."
            )
        self._poisson_ratio = new_poisson_ratio

    @property
    def density(self) -> "mi.TensorXf":
        """Get the current density field."""
        return self._density

    @density.setter
    def density(self, new_density: "mi.TensorXf") -> None:
        """
        Set a new density field for heterogeneous materials.

        :param new_density: Tensor with shape `(n_control_points, 1)`.

        :raises ValueError: If the shape does not match the existing field.
        """
        if new_density.shape != self._density.shape:
            raise ValueError("`value` shape must match the existing density shape.")
        self._density = new_density

    @property
    def thickness(self) -> "mi.TensorXf":
        """Get the current thickness field."""
        return self._thickness

    @thickness.setter
    def thickness(self, new_thickness: "mi.TensorXf") -> None:
        """
        Set a new thickness field for heterogeneous shells.

        :param new_thickness: Tensor with shape `(n_control_points, 1)`.

        :raises ValueError: If the shape does not match the existing field.
        """
        if new_thickness.shape != self._thickness.shape:
            raise ValueError("`value` shape must match the existing thickness shape.")
        self._thickness = new_thickness

    @property
    def shear_correction_factor(self) -> float:
        """Get the current shear correction factor."""
        return self._shear_correction_factor

    @shear_correction_factor.setter
    def shear_correction_factor(self, new_value: float) -> None:
        """
        Set a new shear correction factor.

        The shear correction factor accounts for the non-uniform distribution of
        transverse shear stress through the shell thickness. A value of 5/6 is
        commonly used for isotropic materials.

        :param new_value: The new shear correction factor (must be positive).

        :raises ValueError: If the value is not positive.
        """
        if new_value <= 0.0:
            raise ValueError("`shear_correction_factor` must be positive.")
        self._shear_correction_factor = new_value
