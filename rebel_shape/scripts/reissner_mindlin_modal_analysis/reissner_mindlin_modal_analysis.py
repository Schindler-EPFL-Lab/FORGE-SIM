"""
Run Reissner-Mindlin modal analysis on a tessellated IGA spline surface.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import mitsuba as mi
import tyro

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append("../")

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape import logger, tracking
from rebel_shape.scripts.configs.structural_material_config import (
    StructuralMaterialConfig,
)
from rebel_shape.scripts.reissner_mindlin_modal_analysis import (
    reissner_mindlin_modal_runner,
)


def sphere_verification_material() -> StructuralMaterialConfig:
    """Stainless steel, as used for the thin-spherical-shell verification case."""
    return StructuralMaterialConfig(
        youngs_modulus=193.05e9,
        poisson_ratio=0.28,
        density=8025.9,
        thickness=1.5875e-3,
        shear_correction_factor=5.0 / 6.0,
    )


def concrete_shell_material() -> StructuralMaterialConfig:
    """Concrete, as used for the architectural shell."""
    return StructuralMaterialConfig(
        youngs_modulus=40e9,
        poisson_ratio=0.2,
        density=2400.0,
        thickness=0.2,
        shear_correction_factor=5.0 / 6.0,
    )


def wood_shell_material() -> StructuralMaterialConfig:
    """Wood, as used for the wooden shell."""
    return StructuralMaterialConfig(
        youngs_modulus=11e9,
        poisson_ratio=0.35,
        density=600.0,
        thickness=0.02,
        shear_correction_factor=5.0 / 6.0,
    )


def car_body_material() -> StructuralMaterialConfig:
    """Steel sheet, as used for the car-body shell. The default."""
    return StructuralMaterialConfig(
        youngs_modulus=210e9,
        poisson_ratio=0.3,
        density=7850.0,
        thickness=0.0015,
        shear_correction_factor=5.0 / 6.0,
    )


@dataclass
class ReissnerMindlinModalParameters(ReverseCli):
    """
    Configuration parameters for Reissner-Mindlin modal analysis.
    """

    spline_path: Path
    """Path to the reconstructed IGA spline surface mesh."""
    output_folder: Path = Path("./outputs/")
    """Output folder for results."""
    num_modes: int = 20
    """Number of modes to compute."""
    solver_tolerance: float = 1.0e-10
    """Convergence tolerance for the iterative eigensolver."""
    max_iterations: int | None = None
    """Maximum Arnoldi iterations. None uses scipy default."""
    spectral_shift: float | None = None
    """
    Mass-proportional spectral shift for Cholesky factorization.
    None uses an automatic heuristic."""
    integration_order: int | None = None
    """Optional quadrature order for local patch integration."""
    drilling_stiffness_factor: float = 1.0
    """Scale factor for the objective drilling penalty."""
    max_batch_size: int = 16384
    """Maximum knot spans per assembly batch. Bounds peak memory to
    O(max_batch_size * n_dofs_per_span^2). Recommended: 1024 for 64x64+."""

    export_xdmf: bool = True
    """Whether to export eigenmodes to XDMF format for ParaView."""
    export_xdmf_indices: list[int] = field(
        default_factory=lambda: [i for i in range(20)]
    )
    """Mode indices to export to XDMF if `export_xdmf` is True."""

    material_config: StructuralMaterialConfig = field(default_factory=car_body_material)
    """
    Structural material configuration. Defaults to `car_body_material`; the other
    presets in this module (`sphere_verification_material`, `concrete_shell_material`,
    `wood_shell_material`) can be substituted here, or any property overridden from
    the command line.
    """
    target_length_scale: float | None = None
    """
    Target length for the longest axis of the CONTROL-POINT bounding box. When
    set, a uniform scale is applied so that extent
    matches this value. The surface lies inside the control hull, so its own
    bounding box comes out slightly smaller, by an amount set by how finely the
    control net is refined, not by curvature. Rescaling is opt-in: the default
    `None` leaves the geometry at the scale it was loaded.
    """
    h_refinement_factor: int = 2
    """
    Factor for h-refinement of the input spline surface. Each patch will be uniformly
    refined by this factor in both parametric directions. This improves the accuracy
    at the cost of higher computational time and memory usage.
    Recommended values are 1 (no refinement) or 2
    (going beyond 128x128 control points per patch may lead to very high memory usage).
    """
    target_spans: int | None = None
    """
    If set, OVERRIDES `h_refinement_factor`: refine each patch so it has ~this many
    knot spans per direction, computing the factor from the loaded spline's base spans
    (`factor = round(target_spans / base_spans)`). Use to pin a fixed effective
    resolution across splines with DIFFERENT base span counts. Thin RM shells
    (h/L ~ 1e-3) suffer IGA MEMBRANE locking at coarse spans, so 64 spans
    over-stiffens the spectrum ~5-6%; 128 spans lands on the converged FEM
    plateau. Reconstructed splines vary (32-base needs factor 4,
    64-base needs factor 2 for 128), so a fixed factor cannot pin the resolution:
    this does.
    """

    use_gpu: bool = True
    """Whether to use GPU for tensor assembly."""

    def __post_init__(self) -> None:
        if not self.spline_path.exists():
            raise ValueError(f"Spline path {self.spline_path} does not exist.")
        if self.num_modes < 1:
            raise ValueError("`num_modes` must be a positive integer.")
        if self.solver_tolerance <= 0.0:
            raise ValueError("`solver_tolerance` must be positive.")
        if self.integration_order is not None and self.integration_order < 1:
            raise ValueError("`integration_order` must be positive if provided.")

        if self.target_spans is not None and self.target_spans < 1:
            raise ValueError("`target_spans` must be a positive integer if provided.")
        if self.drilling_stiffness_factor < 0.0:
            raise ValueError("`drilling_stiffness_factor` must be non-negative.")
        if self.target_length_scale is not None and self.target_length_scale <= 0.0:
            raise ValueError("`target_length_scale` must be positive if provided.")


def run_modal_analysis(params: ReissnerMindlinModalParameters) -> None:
    """
    Execute Reissner-Mindlin modal analysis on a spline surface.
    """
    if params.use_gpu:
        mi.set_variant("cuda_ad_mono")
        logger.info("Using GPU for modal analysis.")
    else:
        mi.set_variant("llvm_ad_mono")
        logger.info("Using CPU for modal analysis.")

    tessellated_iga_spline = reissner_mindlin_modal_runner.prepare_modal_geometry(
        spline_path=params.spline_path,
        h_refinement_factor=params.h_refinement_factor,
        target_spans=params.target_spans,
        target_length_scale=params.target_length_scale,
    )

    runner = reissner_mindlin_modal_runner.ReissnerMindlinModalRunner(
        tessellated_iga_spline=tessellated_iga_spline,
        material_config=params.material_config,
    )

    eigenvalues, eigenvectors = runner.run_modal_analysis(
        num_modes=params.num_modes,
        output_folder=params.output_folder,
        solver_tolerance=params.solver_tolerance,
        max_iterations=params.max_iterations,
        spectral_shift=params.spectral_shift,
        integration_order=params.integration_order,
        drilling_stiffness_factor=params.drilling_stiffness_factor,
        max_batch_size=params.max_batch_size,
    )

    if params.export_xdmf:
        runner.export_eigenmodes_to_xdmf(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            mode_indices=params.export_xdmf_indices,
            output_folder=params.output_folder / "xdmf",
        )

    logger.info("Modal analysis completed.")


if __name__ == "__main__":
    parameters = tyro.cli(ReissnerMindlinModalParameters)
    with tracking.run(artifact_directory=parameters.output_folder):
        tracking.log_params(parameters.to_loggable_dict())

        run_modal_analysis(params=parameters)
