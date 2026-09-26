import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from rebel_shape.scripts.end_to_end.heat.thermal_sim_config import ThermalSimConfig


@dataclass(frozen=True, kw_only=True)
class HeatEnergyLog:
    """
    The per-saved-step energy diagnostics a heat run logs, in either solver's format.

    The two solvers write the same quantities in different shapes: the IGA run nests
    them as a list of per-step records under `energy_log`, the FEM driver writes
    parallel arrays, because the sides are deliberately independent. Normalising them
    here is what lets the comparison read one shape.
    """

    time: npt.NDArray[np.float64]
    """Physical time of each saved step [s]."""
    total_heat: npt.NDArray[np.float64]
    """Total thermal energy of the surface at each saved step [J]."""
    gen_rate: npt.NDArray[np.float64]
    """Heat generated per unit time by the source at each saved step [W]."""
    dt: float | None
    """The run's time step [s], or None if it was not logged."""
    relaxation_time: float | None
    """The model's own diffusive relaxation time [s], or None if it was not logged."""
    heat_source_region: ThermalSimConfig.HeatSourceRegion | None
    """
    Which material the run put the source on, or None if it was not logged.

    Optional for the same reason `dt` is: artifacts recorded before the key existed
    must still parse, and no recorded run can be re-stamped with the region it used.
    """

    @classmethod
    def from_json(cls, path: Path) -> "HeatEnergyLog":
        """
        Read either solver's log, normalising both formats to arrays.

        :raises ValueError: If `heat_source_region` is present but names no
            `ThermalSimConfig.HeatSourceRegion` member.
        """
        raw = json.loads(path.read_text())
        # Both logs write the region as its `.value` string; `dt` and the relaxation
        # time are read the same tolerant way, so an older artifact still parses.
        region = raw.get("heat_source_region")
        heat_source_region = (
            None if region is None else ThermalSimConfig.HeatSourceRegion(region)
        )
        if "energy_log" in raw:  # IGA step-6 format
            log = raw["energy_log"]
            return cls(
                time=np.array([e["time"] for e in log]),
                total_heat=np.array([e["total_heat"] for e in log]),
                gen_rate=np.array([e["heat_generation_rate"] for e in log]),
                dt=raw.get("dt"),
                relaxation_time=raw.get("relaxation_time"),
                heat_source_region=heat_source_region,
            )
        return cls(  # FEM fem_heat_gt_log format (arrays)
            time=np.atleast_1d(np.array(raw["time"], dtype=float)),
            total_heat=np.atleast_1d(np.array(raw["total_heat"], dtype=float)),
            gen_rate=np.atleast_1d(np.array(raw["heat_generation_rate"], dtype=float)),
            dt=raw.get("dt"),
            relaxation_time=raw.get("relaxation_time"),
            heat_source_region=heat_source_region,
        )
