from dataclasses import dataclass

from forge_sim.scripts.configs.thermal_material_config import ThermalMaterialConfig


@dataclass(kw_only=True)
class FEMSimulationConfig:
    """
    Time-stepping, output, and discretization options for an FEM heat run.

    The time step is resolved in this precedence order: `final_time` (dt =
    final_time / num_time_steps) > `time_step_size` (explicit dt) > the
    `dt_over_t_relax` normalized step. When `final_time` is None, exactly one of
    `time_step_size` and `dt_over_t_relax` must be set. `final_time` is the knob
    for matching a reference run's end time exactly (e.g. an IGA XDMF), which
    also skips the relaxation eigensolve.
    """

    num_time_steps: int
    """Number of Crank-Nicolson steps."""
    save_interval: int
    """Steps between XDMF snapshots / drift reports."""
    time_step_size: float | None
    """Explicit physical time step [s], or None."""
    dt_over_t_relax: float | None
    """Time step normalized by `T_relax`, or None."""
    final_time: float | None = None
    """Total simulated time [s] to hit exactly, or None."""
    target_delta_temp: float | None = None
    """
    Target temperature rise [K]; when set the source is scaled so an insulated
    source element rises by this much over the run.
    """
    heat_source_material: ThermalMaterialConfig | None = None
    """
    Material whose (rho, c, h) define the source scaling; required when
    `target_delta_temp` is set.
    """
    fem_degree: int = 1
    """Lagrange polynomial degree (1 = P1)."""
    write_xdmf: bool = True
    """Whether to write the solution to XDMF."""

    def __post_init__(self) -> None:
        if self.num_time_steps < 1:
            raise ValueError(
                f"`num_time_steps` must be a positive integer, got "
                f"{self.num_time_steps}."
            )
        if self.save_interval < 1:
            raise ValueError(
                f"`save_interval` must be a positive integer, got "
                f"{self.save_interval}."
            )
        if self.final_time is not None and self.final_time <= 0.0:
            raise ValueError(
                f"`final_time` must be a positive real number, got "
                f"{self.final_time}."
            )
        if self.final_time is None and (self.time_step_size is None) == (
            self.dt_over_t_relax is None
        ):
            raise ValueError(
                "Exactly one of `time_step_size` and `dt_over_t_relax` must be set "
                "when `final_time` is not given."
            )
        if self.target_delta_temp is not None and self.heat_source_material is None:
            raise ValueError(
                "`heat_source_material` must be provided when `target_delta_temp` "
                "is set."
            )
        if self.fem_degree < 1:
            raise ValueError(
                f"`fem_degree` must be a positive integer, got {self.fem_degree}."
            )
