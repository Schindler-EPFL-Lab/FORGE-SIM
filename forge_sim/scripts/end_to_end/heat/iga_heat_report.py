from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from forge_sim.scripts.end_to_end.heat.thermal_sim_config import ThermalSimConfig


@dataclass(frozen=True, kw_only=True)
class IGAHeatReport:
    """
    What the IGA heat run on the reconstructed spline reports about itself.

    Serialised as `iga_heat_log.json` beside the XDMF series, and read back by
    the heat comparison (through `HeatEnergyLog`) and by the recompute stage, which
    takes `lambda_1 = 1 / relaxation_time` from it.
    """

    class TimeStepSource(Enum):
        """Where the run's absolute time step came from."""

        GT_FEM = "gt_fem"
        """Reused from the FEM ground-truth run, so both share one physical clock."""
        SELF_NORMALIZED = "self_normalized"
        """Derived from this model's own relaxation time, the FEM run being absent."""

    @dataclass(frozen=True, kw_only=True)
    class EnergyEntry:
        """The energy diagnostics of one saved step."""

        step: int
        """Step index; 0 is the initial state."""
        time: float
        """Physical time of the step [s]."""
        total_heat: float
        """Total thermal energy of the surface [J]."""
        heat_generation_rate: float
        """Source power `integral(Q) dA` [W]."""

    xdmf: str
    """Path of the written XDMF temperature series."""
    dt: float
    """Absolute physical time step the run marched [s]."""
    dt_source: TimeStepSource
    """How `dt` was resolved."""
    relaxation_time: float
    """
    This model's own diffusive relaxation time [s]. Reported as an independent
    geometric metric even when `dt` came from the FEM run instead.
    """
    num_time_steps: int
    """Crank-Nicolson steps taken."""
    steel_cp_fraction: float
    """Fraction of control points the inpainted mask assigned to the active material."""
    target_delta_temp: float | None
    """Temperature rise the source was auto-scaled to produce [K], or None."""
    source_Q: float | None
    """Source magnitude that scaling resolved to [W/m^2], or None when unscaled."""
    heat_source: float
    """Configured absolute source magnitude [W/m^2], used when no target was set."""
    heat_source_region: ThermalSimConfig.HeatSourceRegion
    """Which material carries the source."""
    total_heat_initial: float
    """Total thermal energy at the first saved step [J]."""
    total_heat_final: float
    """Total thermal energy at the last saved step [J]."""
    heat_generation_rate: float
    """Source power at the last saved step [W]."""
    energy_log: list[EnergyEntry]
    """Every saved step's diagnostics, in time order."""

    def to_json_dict(self) -> dict[str, Any]:
        """
        The report as the JSON object `iga_heat_log.json` records.

        The two enum-valued fields cross the file boundary as their `.value`
        strings, which is the form the recorded logs carry. `HeatEnergyLog.from_json`
        resolves `heat_source_region` back by that value, and the heat comparison
        rejects a pair of runs whose regions disagree, so this spelling is load
        bearing rather than decorative; `dt_source` is read the same way by
        `HeatFieldMetricsRunner._require_gt_fem_dt_source`, which rejects a
        space-time field comparison built from a run whose `dt` did not come from
        the FEM ground truth.
        """
        return {
            **asdict(self),
            "dt_source": self.dt_source.value,
            "heat_source_region": self.heat_source_region.value,
        }
