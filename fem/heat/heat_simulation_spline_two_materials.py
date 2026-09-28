"""
DOLFINx FEM heat simulation on a tessellated spline surface (two materials).

This is the FEM counterpart to
`forge_sim.scripts.heat_simulation.heat_simulation_two_materials` (the IGA
two-material runner). A facade mask spline selects between a base material
(concrete) and windows (glass): the mask is thresholded at the control points and
the per-material properties are interpolated by the spline basis onto the mesh,
reproducing the smooth transition zones the IGA solver integrates.

The parameters mirror the IGA script (normalized or explicit time step,
`target_delta_temp` source scaling, `target_length_scale` geometry rescaling, the
shared `ThermalMaterialConfig` for each material) so the two solvers can be
compared one-to-one. Everything not specific to the two-material case lives in
`SplineHeatSimulationParameters`, which this shares with the single-material
script.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import mitsuba as mi
import tyro

# Repository root: makes `forge_sim` and this script's own `fem` package
# importable when it is run by path rather than as `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fem.fem_field_data import FEMFieldData
from fem.heat.spline_heat_simulation import SplineHeatSimulationParameters
from fem.spline_to_fem_projector import SplineToFEMProjector
from forge_sim.scripts.configs.thermal_material_config import ThermalMaterialConfig


@dataclass(kw_only=True)
class HeatSimulationSplineTwoMaterialsParameters(SplineHeatSimulationParameters):
    """
    Configuration for two-material FEM heat simulation on a spline surface.

    Material properties are spatially varied from a facade mask: control points
    whose mask value exceeds `facade_mask_threshold` are treated as glass
    (windows), the rest as concrete. `run` (inherited from
    `SplineHeatSimulationParameters`) runs the simulation.
    """

    facade_mask_path: Path
    """Path to the window mask spline used to set facade material properties."""
    facade_mask_threshold: float = 0.5
    """Threshold distinguishing concrete (below) from glass (above) in the mask."""

    concrete_config: ThermalMaterialConfig = field(
        default_factory=lambda: ThermalMaterialConfig(
            conductivity=0.15, density=600.0, specific_heat=2400.0, thickness=0.02
        )
    )
    """Thermal material configuration for the base (non-window) region."""
    glass_config: ThermalMaterialConfig = field(
        default_factory=lambda: ThermalMaterialConfig(
            conductivity=0.9, density=2500.0, specific_heat=840.0, thickness=0.01
        )
    )
    """Thermal material configuration for the window region."""

    facade_heat_source: float = 0.1
    """
    Heat source applied to glass regions in W/m^2. Ignored when
    `target_delta_temp` is set.
    """
    target_delta_temp: float | None = 0.5
    """
    Target temperature rise (K) for glass regions over the simulation duration.
    When set, overrides `facade_heat_source`; the required source is computed
    after assembly as Q = target_delta_temp * rho_glass * c_glass * h_glass /
    t_sim (the rise an insulated glass element would achieve).
    """

    target_length_scale: float | None = 2.5
    """
    If set, rescales geometry so the longest axis of the control-point
    bounding box equals this value. The surface lies inside the control hull,
    so its own bounding box comes out slightly smaller.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.facade_mask_path.exists():
            raise ValueError(
                f"Facade mask path {self.facade_mask_path} does not exist."
            )

    @property
    def xdmf_name(self) -> str:
        """File name of the XDMF time series written into `output_folder`."""
        return "heat_simulation_two_materials.xdmf"

    @property
    def heated_material(self) -> ThermalMaterialConfig:
        """The windows are what heat up, so the glass carries the source."""
        return self.glass_config

    def project_fields(self, projector: SplineToFEMProjector) -> FEMFieldData:
        """
        Projects the temperature field and both materials, split by the mask.

        When targeting a temperature rise, the glass source is unit-valued and
        scaled to the required Q after assembly (mirrors the IGA runner).
        """
        glass_source = (
            1.0 if self.target_delta_temp is not None else self.facade_heat_source
        )
        return projector.project_two_material_fields(
            material_mask_spline_path=self.facade_mask_path,
            material_threshold=self.facade_mask_threshold,
            concrete_conductivity=self.concrete_config.conductivity,
            concrete_density=self.concrete_config.density,
            concrete_specific_heat=self.concrete_config.specific_heat,
            concrete_thickness=self.concrete_config.thickness,
            glass_conductivity=self.glass_config.conductivity,
            glass_density=self.glass_config.density,
            glass_specific_heat=self.glass_config.specific_heat,
            glass_thickness=self.glass_config.thickness,
            glass_source=glass_source,
        )


if __name__ == "__main__":
    # Mitsuba variant must be set before any spline loading / Dr.Jit operation,
    # and it is a process-global side effect, so the launching script owns it.
    mi.set_variant("llvm_ad_mono")
    tyro.cli(HeatSimulationSplineTwoMaterialsParameters).run()
