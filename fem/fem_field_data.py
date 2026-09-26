from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, kw_only=True)
class FEMFieldData:
    """
    Field data projected from the spline onto FEM mesh vertices.

    All arrays are vertex-based (one value per P1 node) and in numpy format
    suitable for DOLFINx function creation via interpolation.
    """

    temperature: npt.NDArray[np.float64]
    """Initial temperature field [K], shape (n_vertices,)."""
    conductivity: npt.NDArray[np.float64]
    """Thermal conductivity [W/(m*K)], shape (n_vertices,)."""
    density: npt.NDArray[np.float64]
    """Material density [kg/m^3], shape (n_vertices,)."""
    specific_heat: npt.NDArray[np.float64]
    """Specific heat capacity [J/(kg*K)], shape (n_vertices,)."""
    thickness: npt.NDArray[np.float64]
    """Shell thickness [m], shape (n_vertices,)."""
    source: npt.NDArray[np.float64]
    """Heat source term [W/m^2], shape (n_vertices,)."""
