from pathlib import Path
from typing import Callable

import drjit as dr
import mitsuba as mi

from forge_sim import logger
from forge_sim.isogeometric_analysis.crank_nicolson_scheme import CrankNicolsonScheme
from forge_sim.isogeometric_analysis.heat_simulation import HeatSimulation
from forge_sim.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)
from forge_sim.scene_optimizers.optimization_events import OptimizationEvents
from forge_sim.scripts.configs.thermal_material_config import ThermalMaterialConfig


class HeatSimulationRunner:
    """
    Orchestrates transient heat simulations on tessellated IGA spline surfaces.

    The runner manages the complete workflow for simulating heat transfer on
    parametric surfaces: initialization of thermal material properties and
    execution of the time-stepping heat equation solver.

    Temperature fields are stored as scalar field values on the control points
    of the underlying IGA spline surface.

    Material properties (conductivity, density, specific heat, thickness) and
    the source term are initialized as uniform fields but can be modified through
    their respective property setters to enable spatially-varying heterogeneous
    materials.

    The simulation uses the Crank-Nicolson time integration scheme internally,
    which provides second-order temporal accuracy and unconditional stability.
    """

    def __init__(
        self,
        tessellated_iga_spline: TessellatedIGASplineSurface,
        material_config: ThermalMaterialConfig = ThermalMaterialConfig(),
        per_step_events: OptimizationEvents | None = None,
    ) -> None:
        """
        Initializes the heat simulation runner for the given tessellated
        IGA spline surface.

        This constructor sets up all components needed for running heat
        simulations. The `tessellated_iga_spline` provides both the geometric
        surface representation and the underlying IGA basis for solving the
        heat equation. Material properties are initialized as uniform fields
        based on `material_config` values.

        The `per_step_events` object allows custom callbacks to be executed
        at specific time steps during simulation, enabling features like
        logging, checkpointing, or adaptive parameter adjustment. A fresh
        `OptimizationEvents` is created per instance when omitted, rather than
        a shared default, since it is mutated by every `run_heat_simulation*`
        call.
        """
        self._tessellated_iga_spline = tessellated_iga_spline
        self._iga_spline = tessellated_iga_spline.composite_spline
        self._per_step_events = (
            per_step_events if per_step_events is not None else OptimizationEvents()
        )

        ones = dr.ones(
            dtype=mi.TensorXf, shape=(self._iga_spline.global_field_values.shape[0], 1)
        )

        self._conductivity = material_config.conductivity * ones
        self._density = material_config.density * ones
        self._specific_heat = material_config.specific_heat * ones
        self._thickness = material_config.thickness * ones
        self._source = 0.0 * ones

        self._heat_simulation: HeatSimulation | None = None
        """`HeatSimulation` solved by the most recent run, or None before any run."""
        self._cn_scheme: CrankNicolsonScheme | None = None
        """`CrankNicolsonScheme` used by the most recent run, or None before any run."""

    @staticmethod
    def _build_cn_scheme(
        heat_simulation: HeatSimulation,
        per_step_events: OptimizationEvents,
        time_step_size: float | None,
        dt_over_t_relax: float | None,
    ) -> CrankNicolsonScheme:
        """
        Builds a `CrankNicolsonScheme` from either a physical or a normalized
        time step. Exactly one of `time_step_size` and `dt_over_t_relax` must
        be specified; the normalized path runs an eigsolve on the assembled
        (K, M) to derive the physical dt (see
        `CrankNicolsonScheme.from_normalized_time_step`).
        """
        if (time_step_size is None) == (dt_over_t_relax is None):
            raise ValueError(
                "Exactly one of `time_step_size` and `dt_over_t_relax` must "
                "be specified."
            )
        if time_step_size is not None:
            return CrankNicolsonScheme(
                heat_simulation=heat_simulation,
                time_step_size=time_step_size,
                per_step_events=per_step_events,
            )
        assert dt_over_t_relax is not None
        return CrankNicolsonScheme.from_normalized_time_step(
            heat_simulation=heat_simulation,
            dt_over_t_relax=dt_over_t_relax,
            per_step_events=per_step_events,
        )

    @staticmethod
    def _apply_target_delta_temp_scaling(
        cn_scheme: CrankNicolsonScheme,
        num_time_steps: int,
        target_delta_temp: float | None,
        heat_source_material: ThermalMaterialConfig | None,
    ) -> None:
        """
        When `target_delta_temp` is set, compute the heat source strength Q
        (W/m²) that would raise an insulated element of
        `heat_source_material` by `target_delta_temp` over the full
        simulation duration, then scale the already-assembled source vector.

        The formula (insulated thin-shell element, no lateral conduction):
            Q = ΔT · ρ · c · h / t_sim
        where t_sim = num_time_steps × dt.
        """
        if target_delta_temp is None:
            return
        if heat_source_material is None:
            raise ValueError(
                "`heat_source_material` must be provided when "
                "`target_delta_temp` is set."
            )
        dt = cn_scheme.time_step_size
        t_sim = num_time_steps * dt
        rho = heat_source_material.density
        c = heat_source_material.specific_heat
        h = heat_source_material.thickness
        Q = target_delta_temp * rho * c * h / t_sim
        cn_scheme.scale_source(Q)
        logger.info(
            f"target_delta_temp={target_delta_temp:.4g} K -> "
            f"Q={Q:.6g} W/m² (t_sim={t_sim:.6g} s, "
            f"thermal_mass=ρch={rho * c * h:.6g} J/(m²·K))"
        )

    def run_heat_simulation(
        self,
        num_time_steps: int,
        time_step_size: float | None = None,
        dt_over_t_relax: float | None = None,
        target_delta_temp: float | None = None,
        heat_source_material: ThermalMaterialConfig | None = None,
        drift_reporting_interval: int = 30,
    ) -> None:
        """
        Executes the transient heat simulation for the specified number of
        time steps.

        This method is the main entry point for running heat simulations. It
        constructs a `HeatSimulation` problem from the current material
        properties and geometry, then solves it using the Crank-Nicolson
        time integration scheme.

        The `num_time_steps` parameter controls the total simulation duration
        (physical time = num_time_steps * time_step_size).

        The time step is specified by exactly one of:

        - `time_step_size`: temporal discretization in seconds. Use when an
          explicit physical dt is desired.
        - `dt_over_t_relax`: temporal discretization normalized by the
          natural relaxation time `T_relax` of the discretized system.
          The physical dt is then `dt_over_t_relax * T_relax`, where
          `T_relax` is derived from the smallest non-zero generalized
          eigenvalue of the assembled (K, M); see
          `CrankNicolsonScheme.from_normalized_time_step`. This makes the
          Crank-Nicolson temporal error `~ (dt/T_relax)^2` the explicit
          accuracy knob, independent of geometry, units, and material
          heterogeneity.

        The heat drift (deviation from the energy balance the Crank-Nicolson
        scheme should conserve) is logged every `drift_reporting_interval`
        steps.

        The temperature field in the underlying IGA spline is updated
        in-place during simulation and can be accessed via the
        `temperature_field` property after completion. The `HeatSimulation`
        and `CrankNicolsonScheme` this call solved with are available via the
        `heat_simulation` and `cn_scheme` properties.

        :raises ValueError: If neither / both of `time_step_size` and
            `dt_over_t_relax` are provided, or if `drift_reporting_interval`
            is not a positive integer.
        """
        if drift_reporting_interval < 1:
            raise ValueError("`drift_reporting_interval` must be a positive integer.")

        heat_simulation = HeatSimulation(
            composite_iga_spline=self._iga_spline,
            conductivity=self._conductivity,
            density=self._density,
            specific_heat=self._specific_heat,
            thickness=self._thickness,
            source_term=self._source,
        )
        self._heat_simulation = heat_simulation

        # Build the CN scheme up front so the physical dt is known whether
        # it was supplied directly or derived from `dt_over_t_relax`.
        cn_scheme_solver = self._build_cn_scheme(
            heat_simulation=heat_simulation,
            per_step_events=self._per_step_events,
            time_step_size=time_step_size,
            dt_over_t_relax=dt_over_t_relax,
        )
        self._cn_scheme = cn_scheme_solver

        self._apply_target_delta_temp_scaling(
            cn_scheme_solver, num_time_steps, target_delta_temp,
            heat_source_material,
        )

        # Result is discarded as it can be accessed via the temperature_field
        # property (updated in cn_scheme_solver).
        _ = cn_scheme_solver.simulate(
            num_iterations=num_time_steps,
            drift_reporting_interval=drift_reporting_interval,
        )

    def run_heat_simulation_with_xdmf(
        self,
        num_time_steps: int,
        xdmf_path: Path,
        time_step_size: float | None = None,
        dt_over_t_relax: float | None = None,
        xdmf_save_interval: int = 30,
        tessellation_subdivisions: int = 512,
        target_delta_temp: float | None = None,
        heat_source_material: ThermalMaterialConfig | None = None,
        on_snapshot: Callable[[int, float], None] | None = None,
    ) -> None:
        """
        Executes the heat simulation with XDMF export for comparison with FEM.

        This method extends `run_heat_simulation` with XDMF file export
        functionality. The temperature field is evaluated on a tessellated
        mesh and written to XDMF format at regular intervals, enabling
        direct comparison with DOLFINx FEM solutions.

        The `tessellation_subdivisions` parameter should match the FEM mesh
        resolution for fair comparison. The XDMF file contains temperature
        snapshots at intervals specified by `xdmf_save_interval`.

        `on_snapshot`, when given, is called at the same cadence as the XDMF
        export (including the initial state at time 0) with the step index
        and its physical time. Because `heat_simulation` returns the live,
        in-place-updated instance, a caller can read `total_heat` or
        `heat_generation_rate` off it from inside the callback without this
        method needing a forwarding property per diagnostic.

        :param num_time_steps: Total number of simulation time steps
        :param xdmf_path: Path for the output XDMF file
        :param time_step_size: Explicit dt in seconds (exactly one of this
            and `dt_over_t_relax` must be provided).
        :param dt_over_t_relax: dt normalized by `T_relax` (see
            `run_heat_simulation` for details).
        :param xdmf_save_interval: Interval for saving to XDMF (in time steps)
        :param tessellation_subdivisions: Mesh resolution for XDMF export
        :param on_snapshot: Callback invoked with `(step_index, time)` at
            each XDMF snapshot.

        :raises ValueError: If parameters are invalid
        """
        from forge_sim.isogeometric_analysis.iga_xdmf_exporter import IGAXDMFExporter

        heat_simulation = HeatSimulation(
            composite_iga_spline=self._iga_spline,
            conductivity=self._conductivity,
            density=self._density,
            specific_heat=self._specific_heat,
            thickness=self._thickness,
            source_term=self._source,
        )
        self._heat_simulation = heat_simulation

        # Build the CN scheme first so we have the resolved physical dt for
        # XDMF time labeling, even when the caller passes `dt_over_t_relax`.
        cn_scheme_solver = self._build_cn_scheme(
            heat_simulation=heat_simulation,
            per_step_events=self._per_step_events,
            time_step_size=time_step_size,
            dt_over_t_relax=dt_over_t_relax,
        )
        self._cn_scheme = cn_scheme_solver

        self._apply_target_delta_temp_scaling(
            cn_scheme_solver, num_time_steps, target_delta_temp,
            heat_source_material,
        )

        resolved_dt = cn_scheme_solver.time_step_size

        # Initialize XDMF exporter
        xdmf_exporter = IGAXDMFExporter(
            tessellated_iga_spline=self._tessellated_iga_spline,
            output_path=xdmf_path,
            tessellation_subdivisions=tessellation_subdivisions,
        )

        # Schedule post-step XDMF snapshots. Skip step 0 because the initial
        # state is written explicitly before the solver loop.
        def write_xdmf_callback(step_index: int, dt: float) -> None:
            # Events fire after a step update, so use the 1-based step index
            # for physical time labeling (step 1 -> t=dt, ..., step N -> t=N*dt).
            time = step_index * dt
            xdmf_exporter.write_temperature(time=time)
            if on_snapshot is not None:
                on_snapshot(step_index, time)

        for step_idx in range(1, num_time_steps + 1):
            if step_idx % xdmf_save_interval == 0 or step_idx == num_time_steps:
                self._per_step_events.add_event(
                    iteration=step_idx - 1,
                    callback=lambda s=step_idx, dt=resolved_dt: write_xdmf_callback(
                        s, dt
                    ),
                )

        # Initial XDMF write
        xdmf_exporter.write_temperature(time=0.0)
        if on_snapshot is not None:
            on_snapshot(0, 0.0)

        # Run simulation
        _ = cn_scheme_solver.simulate(
            num_iterations=num_time_steps,
            drift_reporting_interval=xdmf_save_interval,
        )

        # Close XDMF exporter
        xdmf_exporter.close()
        logger.info(f"IGA XDMF export complete: {xdmf_path}")

    @property
    def iga_spline(self) -> "mi.TensorXf":
        """The underlying composite IGA spline surface."""
        return self._iga_spline

    @property
    def heat_simulation(self) -> "HeatSimulation | None":
        """
        The `HeatSimulation` solved by the most recent run.

        This is the live object, not a snapshot: `CrankNicolsonScheme.
        scale_source` (triggered by `target_delta_temp`) rebinds this same
        instance's `source_term` to a new, scaled tensor rather than
        mutating a copy, so reading `total_heat` or `heat_generation_rate`
        here always reflects the scaled source. `None` before
        `run_heat_simulation` or `run_heat_simulation_with_xdmf` has been
        called.
        """
        return self._heat_simulation

    @property
    def cn_scheme(self) -> "CrankNicolsonScheme | None":
        """
        The `CrankNicolsonScheme` used by the most recent run.

        Exposes the resolved `time_step_size` and `relaxation_time` as the
        live object rather than as a forwarding property per diagnostic.
        `None` before `run_heat_simulation` or `run_heat_simulation_with_xdmf`
        has been called.
        """
        return self._cn_scheme

    @property
    def temperature_field(self) -> "mi.TensorXf":
        """
        Temperature field values at the control points of the IGA spline.

        The temperature field is stored as a vector of scalar values, one for
        each control point in the global IGA spline. This field is updated
        in-place during simulation by the Crank-Nicolson solver.

        :returns: Temperature field in Kelvin or Celsius depending on
            initial conditions.
        """
        return self._iga_spline.global_field_values

    @temperature_field.setter
    def temperature_field(self, new_temperature_field: "mi.TensorXf") -> None:
        """
        Sets the temperature field values at the control points.

        This setter allows for custom initialization of temperature
        distributions or for applying boundary conditions. The shape of
        the provided tensor must match the existing field shape.

        :raises ValueError: If the shape of `value` does not match the
            existing temperature field shape.
        """
        if new_temperature_field.shape != self._iga_spline.global_field_values.shape:
            raise ValueError(
                "`value` shape must match the existing temperature field shape."
            )
        self._iga_spline.global_field_values = new_temperature_field

    @property
    def conductivity(self) -> "mi.TensorXf":
        """
        Thermal conductivity field in W/(m*K).

        The conductivity can be spatially varying with one value per control
        point, enabling heterogeneous material simulations.
        """
        return self._conductivity

    @conductivity.setter
    def conductivity(self, new_conductivity: "mi.TensorXf") -> None:
        """
        Sets the thermal conductivity field.

        :raises ValueError: If the shape of `value` does not match the
            existing conductivity field shape.
        """
        if new_conductivity.shape != self._conductivity.shape:
            raise ValueError(
                "`value` shape must match the existing conductivity shape."
            )
        self._conductivity = new_conductivity

    @property
    def density(self) -> "mi.TensorXf":
        """
        Material density field in kg/m^3.

        The density can be spatially varying with one value per control
        point, enabling heterogeneous material simulations.
        """
        return self._density

    @density.setter
    def density(self, new_density: "mi.TensorXf") -> None:
        """
        Sets the material density field.

        :raises ValueError: If the shape of `value` does not match the
            existing density field shape.
        """
        if new_density.shape != self._density.shape:
            raise ValueError("`value` shape must match the existing density shape.")
        self._density = new_density

    @property
    def specific_heat(self) -> "mi.TensorXf":
        """
        Specific heat capacity field in J/(kg*K).

        The specific heat can be spatially varying with one value per control
        point, enabling heterogeneous material simulations.
        """
        return self._specific_heat

    @specific_heat.setter
    def specific_heat(self, new_specific_heat: "mi.TensorXf") -> None:
        """
        Sets the specific heat capacity field.

        :raises ValueError: If the shape of `value` does not match the
            existing specific heat field shape.
        """
        if new_specific_heat.shape != self._specific_heat.shape:
            raise ValueError(
                "`value` shape must match the existing specific heat shape."
            )
        self._specific_heat = new_specific_heat

    @property
    def thickness(self) -> "mi.TensorXf":
        """
        Material thickness field in meters.

        The thickness can be spatially varying with one value per control
        point, enabling shell structures with variable thickness.
        """
        return self._thickness

    @thickness.setter
    def thickness(self, new_thickness: "mi.TensorXf") -> None:
        """
        Sets the material thickness field.

        :raises ValueError: If the shape of `value` does not match the
            existing thickness field shape.
        """
        if new_thickness.shape != self._thickness.shape:
            raise ValueError("`value` shape must match the existing thickness shape.")
        self._thickness = new_thickness

    @property
    def source(self) -> "mi.TensorXf":
        """
        Heat source term field in W/m^2.

        The source term can be spatially varying with one value per control
        point, enabling localized heating or cooling effects.
        """
        return self._source

    @source.setter
    def source(self, new_source: "mi.TensorXf") -> None:
        """
        Sets the heat source term field.

        :raises ValueError: If the shape of `value` does not match the
            existing source term field shape.
        """
        if new_source.shape != self._source.shape:
            raise ValueError("`value` shape must match the existing source term shape.")
        self._source = new_source
