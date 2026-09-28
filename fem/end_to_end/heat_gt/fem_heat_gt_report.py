from dataclasses import asdict, dataclass
from typing import Any

from forge_sim.scripts.configs.thermal_material_config import ThermalMaterialConfig
from forge_sim.scripts.end_to_end.heat.thermal_sim_config import ThermalSimConfig


@dataclass(frozen=True, kw_only=True)
class FEMHeatGTReport:
    """
    What the FEM heat run on the ground-truth mesh reports about itself.

    Serialised as `fem_heat_gt_log.json` beside the XDMF series, and read back by
    the heat comparison (through `HeatEnergyLog`) and by
    `end_to_end.heat.iga_heat_simulation`, which reuses `dt` so both solvers march on
    one physical clock. It carries the same quantities as its IGA counterpart
    `IGAHeatReport` in a different shape: parallel series rather than a list of
    per-step records, because the two sides are deliberately independent, and
    `HeatEnergyLog.from_json` selects between the two shapes on exactly that.

    The resolved materials, the source region and the element degree are recorded so
    that a run can be reproduced from its own artifact: they are the physics the
    numbers below integrate and the space it was integrated in, and they otherwise
    survive only in the command line that launched it. The output cadence
    (`save_interval`, `write_xdmf`) is deliberately not among them: it selects
    which steps are written, not what is solved.
    """

    time: list[float]
    """Physical time of each saved step [s]."""
    total_heat: list[float]
    """Total thermal energy of the surface at each saved step [J]."""
    heat_generation_rate: float
    """Source power `integral(Q) dA` of the run [W]."""
    relaxation_time: float
    """This mesh's own diffusive relaxation time [s], from its assembled (K, M)."""
    dt: float
    """Absolute physical time step the run marched [s]."""
    num_time_steps: int
    """Crank-Nicolson steps taken."""
    fem_degree: int
    """Lagrange degree of the temperature field (1 = P1)."""
    target_delta_temp: float | None
    """Temperature rise the source was auto-scaled to produce [K], or None."""
    source_Q: float | None
    """Source magnitude that scaling resolved to [W/m^2], or None when unscaled."""
    heat_source: float
    """Configured absolute source magnitude [W/m^2], used when no target was set."""
    heat_source_region: ThermalSimConfig.HeatSourceRegion
    """Which material carried the source."""
    steel: ThermalMaterialConfig
    """Resolved properties of the active (`material > 0.5`) slot."""
    wood: ThermalMaterialConfig
    """Resolved properties of the passive slot."""
    num_vertices: int
    """Vertices of the analysis mesh the run integrated."""
    num_triangles: int
    """Triangles of the analysis mesh the run integrated."""

    def to_json_dict(self) -> dict[str, Any]:
        """
        The report as the JSON object `fem_heat_gt_log.json` records.

        `heat_source_region` crosses the file boundary as its `.value` string,
        which is the form the IGA log writes and `HeatEnergyLog.from_json` resolves
        back; the two materials cross as their own field dictionaries.
        """
        return {**asdict(self), "heat_source_region": self.heat_source_region.value}
