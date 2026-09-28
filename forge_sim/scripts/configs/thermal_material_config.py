from dataclasses import dataclass

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from forge_sim import logger


@dataclass(kw_only=True)
class ThermalMaterialConfig(ReverseCli):
    """
    Configuration for the thermal material properties used for IGA-based thermal
    simulations.

    By the assumption of the heat conduction model, the geometry is approximated by a
    thin shell with a specified thickness. This class describes the properties of a
    shell region that is made up of a single material with homogenous properties,
    defined by the thermal conductivity, density, and specific heat capacity.

    The default values are given for an "average" concrete material, which can be
    used as a reasonable starting point for our simulations.
    """

    conductivity: float = 2.0
    """Thermal conductivity of material in W/(m*K)"""
    density: float = 2250.0
    """Density of material in kg/m^3"""
    specific_heat: float = 900.0
    """Specific heat capacity of material in J/(kg*K)"""
    thickness: float = 0.2
    """Thickness of the material layer in meters"""

    def __post_init__(self) -> None:
        if self.conductivity <= 0.0:
            raise ValueError("`conductivity` must be a positive real number.")
        if self.density <= 0.0:
            raise ValueError("`density` must be a positive real number.")
        if self.specific_heat <= 0.0:
            raise ValueError("`specific_heat` must be a positive real number.")
        if self.thickness <= 0.0:
            raise ValueError("`thickness` must be a positive real number.")

        # To avoid potential confusion about the units, we log warnings for
        # unusually low/high values for the material properties.
        if self.conductivity < 0.1 or self.conductivity > 1000.0:
            logger.warning(
                f"Unusually low thermal conductivity of {self.conductivity} W/(m*K). "
                + "Consider verifying the material properties and/or units."
            )
        if self.density < 10.0 or self.density > 20000.0:
            logger.warning(
                f"Unusually low density of {self.density} kg/m^3. "
                + "Consider verifying the material properties and/or units."
            )
        if self.specific_heat < 10.0 or self.specific_heat > 5000.0:
            logger.warning(
                f"Unusually low specific heat capacity of {self.specific_heat} "
                + "J/(kg*K). Consider verifying the material properties and/or units."
            )
        if self.thickness < 0.001 or self.thickness > 1.0:
            logger.warning(
                f"Unusual thickness of {self.thickness} meters. "
                + "Consider verifying the material properties and/or units."
            )

    @property
    def diffusivity(self) -> float:
        """
        Computes the thermal diffusivity of the material.

        The thermal diffusivity is given by the formula:

            alpha = k / (rho * c)

        where k is the thermal conductivity, rho is the density, and c is the specific
        heat capacity.
        :returns: Thermal diffusivity in m^2/s
        """
        return self.conductivity / (self.density * self.specific_heat)
