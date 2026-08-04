from dataclasses import dataclass

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape import logger


@dataclass(kw_only=True)
class StructuralMaterialConfig(ReverseCli):
    """
    Configuration for structural material properties used in Reissner-Mindlin
    modal analysis.

    The default values are representative of a generic concrete-like material
    and should be adjusted to match the target structure.
    """

    youngs_modulus: float = 30.0e9
    """Young's modulus in Pa."""
    poisson_ratio: float = 0.2
    """Poisson ratio of the material."""
    density: float = 2250.0
    """Density of material in kg/m^3."""
    thickness: float = 0.2
    """Thickness of the material layer in meters."""
    shear_correction_factor: float = 5.0 / 6.0
    """Shear correction factor for Reissner-Mindlin theory."""

    def __post_init__(self) -> None:
        if self.youngs_modulus <= 0.0:
            raise ValueError("`youngs_modulus` must be a positive real number.")
        if self.poisson_ratio <= -1.0 or self.poisson_ratio >= 0.5:
            raise ValueError("`poisson_ratio` must be in the range (-1, 0.5).")
        if self.density <= 0.0:
            raise ValueError("`density` must be a positive real number.")
        if self.thickness <= 0.0:
            raise ValueError("`thickness` must be a positive real number.")
        if self.shear_correction_factor <= 0.0:
            raise ValueError(
                "`shear_correction_factor` must be a positive real number."
            )

        if self.youngs_modulus < 1.0e6 or self.youngs_modulus > 1.0e12:
            logger.warning(
                "Unusual Youngs modulus of "
                f"{self.youngs_modulus} Pa. Consider verifying units."
            )
        if self.density < 10.0 or self.density > 20000.0:
            logger.warning(
                f"Unusual density of {self.density} kg/m^3. "
                + "Consider verifying the material properties and/or units."
            )
        if self.thickness < 0.001 or self.thickness > 1.0:
            logger.warning(
                f"Unusual thickness of {self.thickness} meters. "
                + "Consider verifying the material properties and/or units."
            )
