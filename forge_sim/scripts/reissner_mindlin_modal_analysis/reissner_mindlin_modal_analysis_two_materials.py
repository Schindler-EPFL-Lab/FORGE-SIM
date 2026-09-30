"""
Run Reissner-Mindlin modal analysis with two materials defined by a facade mask.
"""

from dataclasses import dataclass, field
from pathlib import Path

import drjit as dr
import mitsuba as mi
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

from forge_sim import logger, tracking
from forge_sim.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)
from forge_sim.scripts.configs.structural_material_config import (
    StructuralMaterialConfig,
)
from forge_sim.scripts.reissner_mindlin_modal_analysis import (
    reissner_mindlin_modal_runner,
)


@dataclass
class ReissnerMindlinModalParametersTwoMaterials(ReverseCli):
    """
    Configuration parameters for modal analysis with two materials.
    """

    spline_path: Path
    """Path to the reconstructed IGA spline surface mesh."""
    facade_mask_path: Path
    """Path to the facade mask that separates the two materials."""
    output_folder: Path = Path("./outputs/")
    """Output folder for results."""
    num_modes: int = 20
    """Number of modes to compute."""
    solver_tolerance: float = 1.0e-10
    """Tolerance for the sparse eigen solver."""
    max_iterations: int | None = None
    """Maximum iterations for the eigen solver."""
    integration_order: int | None = 10
    """
    Gauss-Legendre quadrature order for local patch integration. `None` would take
    the usual `2 * degree + 1`, which is exactly sufficient for the POLYNOMIAL part
    of the integrand: with the material fields carried in the same spline basis,
    a mass entry is a product of four bi-degree-p polynomials, so bi-degree 4p, and
    an order-(2p+1) rule is exact to 4p+1. (Uniform-material spans only reach 2p,
    where it is generous.)

    Going beyond that buys nothing structurally: the metric from the geometric
    mapping is not polynomial, and per the IGA write-up it "cannot introduce
    underintegration, only reduce the accuracy of the end result", so no amount of
    over-integration changes the locking behaviour. The higher order here simply
    resolves that metric more finely, and hedges the fact that the shell integrands
    (membrane, bending, shear, the second fundamental form and the director terms)
    carry more factors than the heat case for which the degree count was actually
    derived. It is not correcting a discontinuity: the material field is a spline
    and is C^(p-1), so convergence stays exponential either way.
    """
    spectral_shift: float | None = None
    """Mass-proportional spectral shift for Cholesky factorization.
    None uses an automatic heuristic."""
    max_batch_size: int = 8192
    """Maximum knot spans per assembly batch. Bounds peak memory to
    O(max_batch_size * n_dofs_per_span^2). Recommended: 1024 for 64x64+."""

    facade_mask_threshold: float = 0.5
    """Threshold for selecting the secondary material from the mask."""

    base_material_config: StructuralMaterialConfig = field(
        default_factory=lambda: StructuralMaterialConfig(
            youngs_modulus=40e9,
            poisson_ratio=0.2,
            density=2400.0,
            thickness=0.2,
            shear_correction_factor=5.0 / 6.0,
        )
    )
    """Material configuration for the base material."""

    secondary_material_config: StructuralMaterialConfig = field(
        default_factory=lambda: StructuralMaterialConfig(
            youngs_modulus=70.0e9,
            poisson_ratio=0.23,
            density=2500.0,
            thickness=0.01,
            shear_correction_factor=5.0 / 6.0,
        )
    )
    """Material configuration for the secondary material (mask true)."""

    target_length_scale: float | None = None
    """
    Target length for the longest axis of the CONTROL-POINT bounding box. When
    set, a uniform scale is applied so that extent matches this value. The surface
    lies inside the control hull, so its own bounding box comes out slightly
    smaller, by an amount set by how finely the control net is refined, not by
    curvature. Rescaling is opt-in: the default `None` leaves the geometry at the
    scale it was loaded.
    """
    h_refinement_factor: int = 2
    """
    Factor for h-refinement of the input spline surface. Each patch will be uniformly
    refined by this factor in both parametric directions. This improves the accuracy
    at the cost of higher computational time and memory usage.
    Recommended values are 1 (no refinement) or 2
    (going beyond 128x128 control points per patch may lead to very high memory usage).
    The facade mask is refined by the same factor, so that its control points keep
    matching the geometry's.
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
        if not self.facade_mask_path.exists():
            raise ValueError(
                f"Facade mask path {self.facade_mask_path} does not exist."
            )
        if self.num_modes < 1:
            raise ValueError("`num_modes` must be a positive integer.")
        if self.solver_tolerance <= 0.0:
            raise ValueError("`solver_tolerance` must be positive.")
        if self.target_spans is not None and self.target_spans < 1:
            raise ValueError("`target_spans` must be a positive integer if provided.")
        if self.target_length_scale is not None and self.target_length_scale <= 0.0:
            raise ValueError("`target_length_scale` must be positive if provided.")


def run_modal_analysis_two_materials(
    params: ReissnerMindlinModalParametersTwoMaterials,
) -> None:
    """
    Execute Reissner-Mindlin modal analysis with spatially varying materials.
    """
    if params.use_gpu:
        mi.set_variant("cuda_ad_mono")
        logger.info("Using the GPU for tensor assembly.")
    else:
        mi.set_variant("llvm_ad_mono")
        logger.info("Using the CPU for tensor assembly.")

    tessellated_iga_spline = reissner_mindlin_modal_runner.prepare_modal_geometry(
        spline_path=params.spline_path,
        h_refinement_factor=params.h_refinement_factor,
        target_spans=params.target_spans,
        target_length_scale=params.target_length_scale,
    )

    facade_spline = TessellatedIGASplineSurface.from_file(
        composite_spline_surface_path=params.facade_mask_path
    )
    # The mask carries one value per control point, so it has to be refined
    # exactly like the geometry for the two to stay in correspondence. Knot
    # insertion is exact, so the mask field itself is unchanged by this.
    reissner_mindlin_modal_runner.apply_h_refinement(
        tessellated_spline=facade_spline,
        h_refinement_factor=params.h_refinement_factor,
        target_spans=params.target_spans,
    )
    thresholded_facade = (
        facade_spline.global_field_values > params.facade_mask_threshold
    )

    runner = reissner_mindlin_modal_runner.ReissnerMindlinModalRunner(
        tessellated_iga_spline=tessellated_iga_spline,
        material_config=params.base_material_config,
    )

    ones = dr.ones(
        dtype=mi.TensorXf,
        shape=(tessellated_iga_spline.global_field_values.shape[0], 1),
    )

    runner.youngs_modulus = dr.select(
        thresholded_facade,
        params.secondary_material_config.youngs_modulus * ones,
        params.base_material_config.youngs_modulus * ones,
    )
    runner.poisson_ratio = dr.select(
        thresholded_facade,
        params.secondary_material_config.poisson_ratio * ones,
        params.base_material_config.poisson_ratio * ones,
    )
    runner.density = dr.select(
        thresholded_facade,
        params.secondary_material_config.density * ones,
        params.base_material_config.density * ones,
    )
    runner.thickness = dr.select(
        thresholded_facade,
        params.secondary_material_config.thickness * ones,
        params.base_material_config.thickness * ones,
    )
    if (
        params.base_material_config.shear_correction_factor
        != params.secondary_material_config.shear_correction_factor
    ):
        logger.warning(
            "Shear correction factors differ between materials. "
            "Using the base material value."
        )
    runner.shear_correction_factor = params.base_material_config.shear_correction_factor

    eigenvalues, eigenvectors = runner.run_modal_analysis(
        num_modes=params.num_modes,
        integration_order=params.integration_order,
        output_folder=params.output_folder,
        solver_tolerance=params.solver_tolerance,
        max_iterations=params.max_iterations,
        spectral_shift=params.spectral_shift,
        max_batch_size=params.max_batch_size,
    )

    runner.export_eigenmodes_to_xdmf(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        mode_indices=[i for i in range(params.num_modes)],
        output_folder=params.output_folder / "xdmf",
    )

    logger.info("Two-material modal analysis completed.")


if __name__ == "__main__":
    parameters = tyro.cli(ReissnerMindlinModalParametersTwoMaterials)
    with tracking.run(artifact_directory=parameters.output_folder):
        tracking.log_params(parameters.to_loggable_dict())

        run_modal_analysis_two_materials(params=parameters)
