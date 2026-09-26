from dataclasses import dataclass, field

from rebel_shape.scripts.configs.structural_material_config import (
    StructuralMaterialConfig,
)


@dataclass(kw_only=True)
class ModalSimConfig:
    """Reissner-Mindlin shell modal scenario (single steel material, car-like)."""

    steel: StructuralMaterialConfig = field(
        default_factory=lambda: StructuralMaterialConfig(
            youngs_modulus=210.0e9,
            poisson_ratio=0.3,
            density=7850.0,
            thickness=0.002,
            shear_correction_factor=5.0 / 6.0,
        )
    )
    """
    Steel panel. Thickness 2 mm: against the GT geometry's native size (~3.3 m for
    Suzanne) that is h/L ~ 6e-4, a deliberately thin, locking-stressing shell.
    """
    num_modes: int = 20
    """Number of eigenpairs solved for on each side (rigid-body modes included)."""
    iga_modal_spans: int = 128
    """
    Target knot spans per direction for the IGA modal run, pinned across every
    reconstruction spline regardless of its base count. 128 rather than 64 because a
    thin shell (h/L ~ 6e-4) membrane-locks at coarse spans, which the shear-only B-bar
    cannot cure: 64 over-stiffens the spectrum by 5-6 % (Suzanne scale 1.067 -> 1.007).
    """
    fem_remesh_edge: float = 0.025
    """
    Target edge length for the GT FEM modal mesh, giving ~60k triangles. Converged in
    combination with the P3 fields below: across P2 at 0.035 / 0.025 / 0.018 and P3 at
    0.025, P2 at 0.025 is ~3 % too stiff from residual locking, while P3 at 0.025 and
    P2 at 0.018 agree to ~1 %.
    """
    fem_field_degree: int = 3
    """
    Lagrange degree of the FEM displacement and rotation fields. P3 rather than P2
    because at this thinness a B-bar P2 field retains ~3 % locking stiffness that P3
    removes. The GT mesh is faceted (geometry order 1) by design: it is itself the
    reference surface, so only the field is P3.
    """

    def __post_init__(self) -> None:
        if self.num_modes < 1:
            raise ValueError(
                f"`num_modes` must be a positive integer, got {self.num_modes}."
            )
        if self.iga_modal_spans < 1:
            raise ValueError(
                f"`iga_modal_spans` must be a positive integer, got "
                f"{self.iga_modal_spans}."
            )
        if self.fem_remesh_edge <= 0.0:
            raise ValueError(
                f"`fem_remesh_edge` must be a positive edge length, got "
                f"{self.fem_remesh_edge}."
            )
        if self.fem_field_degree < 1:
            raise ValueError(
                f"`fem_field_degree` must be a positive integer, got "
                f"{self.fem_field_degree}."
            )
