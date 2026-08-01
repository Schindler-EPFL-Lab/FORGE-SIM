import tempfile
import time
from pathlib import Path

import drjit as dr
import mitsuba as mi

from rebel_shape import logger
from rebel_shape.isogeometric_analysis.crank_nicolson_scheme import CrankNicolsonScheme
from rebel_shape.isogeometric_analysis.heat_simulation import HeatSimulation
from rebel_shape.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from rebel_shape.property_optimizers.field_texture_optimizer import (
    FieldTextureOptimizer,
)
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents
from rebel_shape.scripts.configs.camera_ring_config import CameraRingConfig
from rebel_shape.scripts.configs.thermal_material_config import ThermalMaterialConfig
from rebel_shape.utils.image_utils import show_rendered_views


class HeatSimulationRunner:
    """
    This class orchestrates the execution of transient heat simulations on
    tessellated IGA spline surfaces with visualization capabilities.

    The runner manages the complete workflow for simulating heat transfer on
    parametric surfaces including: initialization of thermal material properties,
    creation of Mitsuba scenes for visualization, periodic rendering of
    temperature distributions as false-color images, and execution of the
    time-stepping heat equation solver.

    Temperature fields are stored as scalar field values on the control points
    of the underlying IGA spline surface and are visualized by mapping them to
    emissive textures on tessellated meshes. The scene contains multiple sensors
    positioned in a ring around the geometry to capture the temperature evolution
    from different viewpoints.

    Material properties (conductivity, density, specific heat, thickness) and
    the source term are initialized as uniform fields but can be modified through
    their respective property setters to enable spatially-varying heterogeneous
    materials.

    The simulation uses the Crank-Nicolson time integration scheme internally,
    which provides second-order temporal accuracy and unconditional stability.
    Rendering occurs at intervals controlled by `render_interval` to balance
    visualization frequency with computational cost.
    """

    def __init__(
        self,
        tessellated_iga_spline: TessellatedIGASplineSurface,
        texture_resolution: int = 512,
        render_interval: int = 30,
        camera_config: CameraRingConfig = CameraRingConfig(),
        material_config: ThermalMaterialConfig = ThermalMaterialConfig(),
        per_step_events: OptimizationEvents = OptimizationEvents(),
    ) -> None:
        """
        Initializes the heat simulation runner for the given tessellated
        IGA spline surface.

        This constructor sets up all components needed for running and
        visualizing heat simulations. The `tessellated_iga_spline` provides
        both the geometric surface representation and the underlying IGA basis
        for solving the heat equation. Material properties are initialized as
        uniform fields based on `material_config` values.

        The `texture_resolution` determines the dimensions of the temperature
        textures used for visualization (texture_resolution x texture_resolution
        pixels per patch). Higher resolutions provide smoother visualization
        but require more memory and rendering time.

        Rendering frequency is controlled by `render_interval`, which specifies
        how many time steps occur between visualization outputs. For example,
        a value of 30 means the temperature field is rendered every 30 time
        steps. The first and last time steps are always rendered regardless
        of this interval.

        The `camera_config` defines the sensor arrangement for capturing views
        of the temperature distribution. Sensors are positioned in a ring
        around the geometry's center with viewing directions pointing inward.

        The `per_step_events` object allows custom callbacks to be executed
        at specific time steps during simulation, enabling features like
        logging, checkpointing, or adaptive parameter adjustment.

        :raises ValueError: If `texture_resolution` or `render_interval`
            is not a positive integer.
        """
        if texture_resolution < 1:
            raise ValueError("`texture_resolution` must be a positive integer.")
        if render_interval < 1:
            raise ValueError("`render_interval` must be a positive integer.")

        self._texture_resolution = texture_resolution
        self._render_interval = render_interval

        self._tessellated_iga_spline = tessellated_iga_spline
        self._iga_spline = tessellated_iga_spline.composite_spline
        self._per_step_events = per_step_events

        ones = dr.ones(
            dtype=mi.TensorXf, shape=(self._iga_spline.global_field_values.shape[0], 1)
        )

        self._conductivity = material_config.conductivity * ones
        self._density = material_config.density * ones
        self._specific_heat = material_config.specific_heat * ones
        self._thickness = material_config.thickness * ones
        self._source = 0.0 * ones

        self._scene, self._sensors = self._initialize_scene(camera_config=camera_config)
        self._scene_params = mi.traverse(self._scene)

        self._property_key = "emitter.radiance.data"
        self._texture_keys = [
            f"{mesh.name}.{self._property_key}" for mesh in self._tessellated_iga_spline
        ]

    def _initialize_scene(
        self, camera_config: CameraRingConfig
    ) -> tuple["mi.Scene", list["mi.Sensor"]]:
        """
        Creates and loads a Mitsuba scene containing the tessellated IGA
        spline surface with temperature-carrying textures and sensor views.

        This method initializes a Mitsuba scene by converting the
        `_tessellated_iga_spline` into PLY meshes with monochromatic
        emissive textures. Each texture has resolution
        `_texture_resolution` x `_texture_resolution` pixels and will carry
        temperature data evaluated at specific parametric coordinates during
        the simulation.

        Sensors are created according to `camera_config` in a radial pattern
        around the origin, where the origin is defined as the mean of the
        global control points of the spline surface. The sensors look inward
        toward this origin. See `CameraRingConfig` for detailed configuration
        options including sensor count, resolution, field of view, and
        viewing angles.

        :returns: A tuple containing the Mitsuba scene object with meshes
            that represent the spline surface with temperature distribution
            textures, and a list of Mitsuba sensor objects positioned in a
            radial pattern around the origin.
        """
        scene_dict = {"type": "scene", "integrator": {"type": "path"}}

        with tempfile.TemporaryDirectory() as temp_dir:
            for mesh in self._tessellated_iga_spline:
                initial_texture_path = (
                    Path(temp_dir) / f"initial_texture_{mesh.name}.exr"
                )

                FieldTextureOptimizer.generate_empty_texture(
                    base_color=mi.Color3f(0.5, 0.5, 0.5),
                    texture_output_path=initial_texture_path,
                    width=self._texture_resolution,
                    height=self._texture_resolution,
                )

                mesh.save_to_ply(output_dir=Path(temp_dir), save_name=mesh.name)

                scene_dict[mesh.name] = {
                    "type": "ply",
                    "filename": temp_dir + f"/{mesh.name}.ply",
                    "bsdf": DiffuseBSDF(
                        reflectance=DiffuseBSDF.Albedo(albedo=0.0),
                    ).mitsuba_dict,
                    "emitter": {
                        "type": "area",
                        "radiance": {
                            "type": "bitmap",
                            "filename": str(initial_texture_path),
                        },
                    },
                }

            scene_source: "mi.Scene" = mi.load_dict(scene_dict)

        target_coords = self._iga_spline.global_control_points.numpy().mean(axis=0)
        camera_config.origin_position = target_coords

        return scene_source, camera_config.sensors

    def render_temperature(self, iteration: int, output_folder: Path) -> None:
        """
        Renders the current temperature field as textures on the spline surface.

        This method updates the temperature textures in `_scene_params` based
        on the current temperature field of the `_iga_spline` surface. For
        each patch in the composite spline, the temperature field is evaluated
        at texture coordinates and assigned to the corresponding texture in
        the scene. The scene is then rendered from all sensor viewpoints and
        the resulting images are saved as a composite figure.

        The `_texture_keys` list corresponds to the textures in `_scene_params`
        that will be updated with the new temperature data. The `iteration`
        number is used in the output filename along with a timestamp. The
        maximum temperature value across all textures is logged for monitoring
        purposes.

        Images are saved to `output_folder` with the naming pattern:
        `views_{timestamp}_step_{iteration:04}.png`

        :raises ValueError: If `iteration` is negative or if `output_folder`
            does not exist.
        """
        if iteration < 0:
            raise ValueError("`iteration` must be a non-negative integer.")
        if not output_folder.exists():
            raise ValueError(f"`output_folder` {output_folder} does not exist.")

        max_value = mi.TensorXf([0.0])

        for texture_key, patch in zip(self._texture_keys, self._iga_spline):
            width, height = self._scene_params[texture_key].shape[:2]
            new_field_texture = patch.evaluate_field_texture(width=width, height=height)
            self._scene_params[texture_key] = new_field_texture

            max_value = dr.maximum(max_value, dr.max(new_field_texture))

        self._scene_params.update()
        logger.debug(
            f"Maximum temperature value in textures: {max_value.numpy()[0]:.7f}"
        )

        rendered_images = [
            mi.render(scene=self._scene, sensor=sensor) for sensor in self._sensors
        ]

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        show_rendered_views(
            views=[mi.util.convert_to_bitmap(image) for image in rendered_images],
            image_dpi=150,
            save_path=output_folder / f"views_{timestamp}_step_{iteration:04}.png",
        )

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
        When ``target_delta_temp`` is set, compute the heat source strength Q
        (W/m²) that would raise an insulated element of
        ``heat_source_material`` by ``target_delta_temp`` over the full
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
        output_folder: Path,
        time_step_size: float | None = None,
        dt_over_t_relax: float | None = None,
        target_delta_temp: float | None = None,
        heat_source_material: ThermalMaterialConfig | None = None,
    ) -> None:
        """
        Executes the transient heat simulation for the specified number of
        time steps and saves visualization outputs.

        This method is the main entry point for running heat simulations. It
        constructs a `HeatSimulation` problem from the current material
        properties and geometry, then solves it using the Crank-Nicolson
        time integration scheme.

        The `num_time_steps` parameter controls the total simulation duration
        (physical time = num_time_steps * time_step_size), and must be a
        positive integer.

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

        Visualization events are automatically scheduled according to the
        `render_interval` set during initialization. An initial rendering at
        time step 0 captures the initial temperature distribution, and
        subsequent renderings occur every `render_interval` steps plus the
        final time step.

        All rendered images are saved to `output_folder`, which must exist
        before calling this method. The temperature field in the underlying
        IGA spline is updated in-place during simulation and can be accessed
        via the `temperature_field` property after completion.

        :raises ValueError: If `num_time_steps` is not a positive integer,
            if neither / both of `time_step_size` and `dt_over_t_relax` are
            provided, or if `output_folder` does not exist.
        """
        heat_simulation = HeatSimulation(
            composite_iga_spline=self._iga_spline,
            conductivity=self._conductivity,
            density=self._density,
            specific_heat=self._specific_heat,
            thickness=self._thickness,
            source_term=self._source,
        )

        # Build the CN scheme up front so the physical dt is known whether
        # it was supplied directly or derived from `dt_over_t_relax`.
        cn_scheme_solver = self._build_cn_scheme(
            heat_simulation=heat_simulation,
            per_step_events=self._per_step_events,
            time_step_size=time_step_size,
            dt_over_t_relax=dt_over_t_relax,
        )

        self._apply_target_delta_temp_scaling(
            cn_scheme_solver, num_time_steps, target_delta_temp,
            heat_source_material,
        )

        for it in range(num_time_steps):
            if it % self._render_interval == 0 or it == num_time_steps - 1:
                self._per_step_events.add_event(
                    iteration=it,
                    callback=lambda i=it: self.render_temperature(
                        iteration=i, output_folder=output_folder
                    ),
                )

        # Initial rendering of the temperature distribution
        self.render_temperature(iteration=0, output_folder=output_folder)

        # Main simulation loop, result is discarded as it can be accessed via the
        # temperature_field property (updated in cn_scheme_solver)
        _ = cn_scheme_solver.simulate(
            num_iterations=num_time_steps,
            drift_reporting_interval=self._render_interval,
        )

    def run_heat_simulation_with_xdmf(
        self,
        num_time_steps: int,
        output_folder: Path,
        xdmf_path: Path,
        time_step_size: float | None = None,
        dt_over_t_relax: float | None = None,
        xdmf_save_interval: int = 30,
        tessellation_subdivisions: int = 512,
        target_delta_temp: float | None = None,
        heat_source_material: ThermalMaterialConfig | None = None,
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

        :param num_time_steps: Total number of simulation time steps
        :param output_folder: Directory for rendered images
        :param xdmf_path: Path for the output XDMF file
        :param time_step_size: Explicit dt in seconds (exactly one of this
            and `dt_over_t_relax` must be provided).
        :param dt_over_t_relax: dt normalized by `T_relax` (see
            `run_heat_simulation` for details).
        :param xdmf_save_interval: Interval for saving to XDMF (in time steps)
        :param tessellation_subdivisions: Mesh resolution for XDMF export

        :raises ValueError: If parameters are invalid
        """
        from rebel_shape.isogeometric_analysis.iga_xdmf_exporter import IGAXDMFExporter

        heat_simulation = HeatSimulation(
            composite_iga_spline=self._iga_spline,
            conductivity=self._conductivity,
            density=self._density,
            specific_heat=self._specific_heat,
            thickness=self._thickness,
            source_term=self._source,
        )

        # Build the CN scheme first so we have the resolved physical dt for
        # XDMF time labeling, even when the caller passes `dt_over_t_relax`.
        cn_scheme_solver = self._build_cn_scheme(
            heat_simulation=heat_simulation,
            per_step_events=self._per_step_events,
            time_step_size=time_step_size,
            dt_over_t_relax=dt_over_t_relax,
        )

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

        # Schedule rendering events
        for it in range(num_time_steps):
            if it % self._render_interval == 0 or it == num_time_steps - 1:
                self._per_step_events.add_event(
                    iteration=it,
                    callback=lambda i=it: self.render_temperature(
                        iteration=i, output_folder=output_folder
                    ),
                )

        # Schedule XDMF export events
        def write_xdmf_callback(step_index: int, dt: float) -> None:
            # Events fire after a step update, so use the 1-based step index
            # for physical time labeling (step 1 -> t=dt, ..., step N -> t=N*dt).
            xdmf_exporter.write_temperature(time=step_index * dt)

        # Schedule post-step XDMF snapshots. Skip step 0 because the initial
        # state is written explicitly before the solver loop.
        for step_idx in range(1, num_time_steps + 1):
            if step_idx % xdmf_save_interval == 0 or step_idx == num_time_steps:
                self._per_step_events.add_event(
                    iteration=step_idx - 1,
                    callback=lambda s=step_idx, dt=resolved_dt: write_xdmf_callback(
                        s, dt
                    ),
                )

        # Initial rendering and XDMF write
        self.render_temperature(iteration=0, output_folder=output_folder)
        xdmf_exporter.write_temperature(time=0.0)

        # Run simulation
        _ = cn_scheme_solver.simulate(
            num_iterations=num_time_steps,
            drift_reporting_interval=self._render_interval,
        )

        # Close XDMF exporter
        xdmf_exporter.close()
        logger.info(f"IGA XDMF export complete: {xdmf_path}")

    @property
    def iga_spline(self) -> "mi.TensorXf":
        """The underlying composite IGA spline surface."""
        return self._iga_spline

    @property
    def scene(self) -> "mi.Scene":
        """The Mitsuba scene containing the tessellated geometry."""
        return self._scene

    @property
    def scene_params(self) -> "mi.SceneParams":
        """The Mitsuba scene parameters for updating scene properties."""
        return self._scene_params

    @property
    def sensors(self) -> list["mi.Sensor"]:
        """List of Mitsuba sensors positioned around the geometry."""
        return self._sensors

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
