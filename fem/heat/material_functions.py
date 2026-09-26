from dataclasses import dataclass

from dolfinx import fem


@dataclass(frozen=True, kw_only=True)
class MaterialFunctions:
    """
    The five material coefficients of the transient heat equation, on one mesh.

    Each is a DOLFINx `fem.Function` rather than a scalar, so a coefficient may
    vary from DOF to DOF, which is what a two-material run (a facade of concrete
    and glass, say) is. A homogeneous run simply interpolates a constant.

    `FEMHeatProblem` needs these five spatially varying coefficients as one named
    record rather than by string key in a `dict[str, fem.Function]`, so a misspelt
    or missing coefficient is a `TypeError` at construction rather than a
    `KeyError` deep inside assembly.

    The Functions are held by reference, never copied. `scale_source` rewrites
    `source` in place, so two solvers handed the same record share that mutation;
    build a fresh record per solver rather than reusing one.
    """

    conductivity: fem.Function
    """Thermal conductivity k [W/(m*K)]. Enters the stiffness form as `k d`."""
    density: fem.Function
    """Material density rho [kg/m^3]."""
    specific_heat: fem.Function
    """Specific heat capacity c [J/(kg*K)]."""
    thickness: fem.Function
    """
    Shell thickness d [m].

    It multiplies the mass coefficient `rho c d` and the stiffness coefficient
    `k d` alike, which is why a uniform thickness cancels out of the relaxation
    time of the semi-discrete system.
    """
    source: fem.Function
    """Heat source Q [W/m^2], integrated per unit area (no thickness factor)."""
