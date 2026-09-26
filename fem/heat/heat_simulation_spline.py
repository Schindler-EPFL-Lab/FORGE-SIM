"""
DOLFINx FEM heat simulation on a tessellated spline surface (single material).

This is the FEM counterpart to `rebel_shape.scripts.heat_simulation.heat_simulation`
(the IGA single-material runner). It loads an IGA spline surface, tessellates it
into a P1 triangle mesh, projects the temperature field onto the mesh, and solves
the transient heat equation with DOLFINx using Crank-Nicolson time integration.

The parameters mirror the IGA script (normalized or explicit time step,
`target_delta_temp` source scaling, `target_length_scale` geometry rescaling, the
shared `ThermalMaterialConfig`) so the two solvers can be compared one-to-one.
Everything not specific to the single-material case lives in
`SplineHeatSimulationParameters`, which this shares with the two-material script.
The output XDMF is read back by `compare_iga_fem.py`.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import mitsuba as mi
import tyro

# Repository root: makes `rebel_shape` and this script's own `fem` package
# importable when it is run by path rather than as `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fem.fem_field_data import FEMFieldData
from fem.heat.spline_heat_simulation import SplineHeatSimulationParameters
from fem.spline_to_fem_projector import SplineToFEMProjector
from rebel_shape.scripts.configs.thermal_material_config import ThermalMaterialConfig


@dataclass(kw_only=True)
class HeatSimulationSplineParameters(SplineHeatSimulationParameters):
    """
    Configuration for single-material FEM heat simulation on a spline surface.

    `run` (inherited from `SplineHeatSimulationParameters`) runs the simulation.
    """

    concrete_config: ThermalMaterialConfig = field(
        default_factory=lambda: ThermalMaterialConfig(
            conductivity=2.0,
            density=2250.0,
            specific_heat=900.0,
            thickness=0.2,
        )
    )
    """Thermal material configuration (uniform across the surface)."""

    heat_source: float = 0.0
    """Uniform heat source in W/m^2. Ignored when `target_delta_temp` is set."""

    target_length_scale: float | None = 4.0
    """
    If set, rescales geometry so the longest axis of the control-point
    bounding box equals this value. The surface lies inside the control hull,
    so its own bounding box comes out slightly smaller.
    """

    @property
    def xdmf_name(self) -> str:
        """File name of the XDMF time series written into `output_folder`."""
        return "heat_simulation.xdmf"

    @property
    def heated_material(self) -> ThermalMaterialConfig:
        """The single material of the run carries whatever source there is."""
        return self.concrete_config

    def project_fields(self, projector: SplineToFEMProjector) -> FEMFieldData:
        """
        Projects the temperature field and the uniform material properties.

        When targeting a temperature rise, the source field is unit-valued and
        scaled to the required Q after assembly (mirrors the IGA runner).
        """
        source_value = 1.0 if self.target_delta_temp is not None else self.heat_source
        return projector.project_fields(
            conductivity=self.concrete_config.conductivity,
            density=self.concrete_config.density,
            specific_heat=self.concrete_config.specific_heat,
            thickness=self.concrete_config.thickness,
            source=source_value,
        )


if __name__ == "__main__":
    # Mitsuba variant must be set before any spline loading / Dr.Jit operation,
    # and it is a process-global side effect, so the launching script owns it.
    mi.set_variant("llvm_ad_mono")
    tyro.cli(HeatSimulationSplineParameters).run()
