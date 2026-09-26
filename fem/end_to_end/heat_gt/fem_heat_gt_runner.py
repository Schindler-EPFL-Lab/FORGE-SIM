import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from fem.end_to_end.heat_gt.fem_heat_gt_report import FEMHeatGTReport
from fem.fem_field_data import FEMFieldData
from fem.fem_mesh_data import FEMMeshData
from fem.heat.fem_heat_runner import FEMHeatRun
from fem.heat.fem_simulation_config import FEMSimulationConfig
from fem.heat.fem_simulation_diagnostics import FEMSimulationDiagnostics
from rebel_shape import logger
from rebel_shape.scripts.configs.thermal_material_config import ThermalMaterialConfig
from rebel_shape.scripts.end_to_end.heat.thermal_sim_config import ThermalSimConfig


class FEMHeatGTRunner:
    """
    Runs the shared FEM heat simulation on a baked ground-truth mesh, and
    writes the diagnostics sidecar alongside its XDMF output.

    Holds the resolved run configuration every step of `run` reads (never the
    script's own `FEMHeatGTMeshParameters`), so that this module never imports
    from the entry script that imports it. `run` is the entry point a caller
    reaches for. `load_baked_mesh` needs no instance and is a classmethod;
    `write_log_json` reads the instance's own configuration, so a caller
    reaches it off a constructed runner; both are public because the
    driver's own tests exercise them directly, without a solve.

    The material mask is cut at `ThermalSimConfig.ACTIVE_MATERIAL_THRESHOLD`, the
    value `iga_heat_simulation` thresholds the IGA side at, so the two follow one
    shared convention, though not to identical partitions: the IGA side cuts at
    its coarse control points and interpolates the transition via the spline
    basis, while this mask is already baked onto the analysis mesh's own vertices
    and is cut there directly.
    """

    @dataclass(frozen=True, kw_only=True)
    class BakedMesh:
        """What `load_baked_mesh` reads out of one baked `.npz`."""

        mesh_data: FEMMeshData
        """The vertices and faces of the baked ground-truth mesh."""
        field_data: FEMFieldData
        """The per-vertex temperature and material fields the shared runner takes."""
        num_active_vertices: int
        """Vertices the material mask assigned to `active` rather than `passive`."""

    _XDMF_NAME: str = "fem_heat_gt.xdmf"
    """Name of the solution time series written into the output folder."""

    _LOG_NAME: str = "fem_heat_gt_log.json"
    """Name of the physical-diagnostics sidecar written into the output folder."""

    def __init__(
        self,
        *,
        npz_path: Path,
        output_folder: Path,
        steel_config: ThermalMaterialConfig,
        wood_config: ThermalMaterialConfig,
        heat_source: float,
        heat_source_region: ThermalSimConfig.HeatSourceRegion,
        target_delta_temp: float | None,
        num_time_steps: int,
        save_interval: int,
        dt_over_t_relax: float,
        fem_degree: int,
        write_xdmf: bool,
    ) -> None:
        """Stores the resolved run configuration; does no I/O."""
        self._npz_path = npz_path
        self._output_folder = output_folder
        self._steel_config = steel_config
        self._wood_config = wood_config
        self._heat_source = heat_source
        self._heat_source_region = heat_source_region
        self._target_delta_temp = target_delta_temp
        self._num_time_steps = num_time_steps
        self._save_interval = save_interval
        self._dt_over_t_relax = dt_over_t_relax
        self._fem_degree = fem_degree
        self._write_xdmf = write_xdmf

    def run(self) -> None:
        """
        Loads the baked mesh and fields, runs the shared FEM heat simulation on
        them, and writes the diagnostics sidecar alongside its XDMF output.
        """
        comm = MPI.COMM_WORLD
        source_material = self._source_material()
        # With `target_delta_temp` the source field is assembled at unit magnitude
        # and rescaled by the runner once dt (and hence the simulated span) is
        # known. A region carrying no source has no thermal mass to derive that
        # scaling from, which is the same guard the IGA run applies.
        use_target_delta_temp = (
            self._target_delta_temp is not None and source_material is not None
        )
        baked = self.load_baked_mesh(
            npz_path=self._npz_path,
            active=self._steel_config,
            passive=self._wood_config,
            source=1.0 if use_target_delta_temp else self._heat_source,
            region=self._heat_source_region,
        )
        if comm.rank == 0:
            self._log_loaded_mesh(
                baked.mesh_data,
                baked.num_active_vertices,
                source_material,
                use_target_delta_temp,
            )

        config = self._build_config(source_material, use_target_delta_temp)
        diagnostics = FEMHeatRun(
            mesh_data=baked.mesh_data,
            field_data=baked.field_data,
            config=config,
            comm=comm,
        ).run_with_xdmf(
            output_folder=self._output_folder, xdmf_name=self._XDMF_NAME
        )

        if comm.rank == 0:
            self.write_log_json(diagnostics, baked.mesh_data)
            first, last = diagnostics.snapshots[0], diagnostics.snapshots[-1]
            logger.info(
                f"Total heat: H[0]={first.total_heat:.6e} J -> "
                f"H[-1]={last.total_heat:.6e} J (expected rise = gen_rate * T_final = "
                f"{diagnostics.heat_generation_rate * last.time:.6e} J)."
            )

    def _source_material(self) -> ThermalMaterialConfig | None:
        """
        The material `heat_source_region` places the source on.

        :returns: `steel_config` or `wood_config`, or None under
            `HeatSourceRegion.NONE`, which names no material, and therefore
            no thermal mass for `target_delta_temp` to scale a source against.
        """
        if self._heat_source_region is ThermalSimConfig.HeatSourceRegion.STEEL:
            return self._steel_config
        if self._heat_source_region is ThermalSimConfig.HeatSourceRegion.WOOD:
            return self._wood_config
        return None

    def _log_loaded_mesh(
        self,
        mesh_data: FEMMeshData,
        n_active: int,
        source_material: ThermalMaterialConfig | None,
        use_target_delta_temp: bool,
    ) -> None:
        """Logs the loaded mesh and where the configured heat source lands on it."""
        magnitude = (
            "auto (target delta T)"
            if use_target_delta_temp
            else f"{self._heat_source:.6g} W/m^2"
        )
        source_description = (
            "none (pure diffusion)"
            if source_material is None
            else f"{magnitude} on the {self._heat_source_region.value} material"
        )
        n_vertices = mesh_data.vertices.shape[0]
        logger.info(
            f"Loaded GT mesh from {self._npz_path}: {n_vertices} vertices, "
            f"{mesh_data.faces.shape[0]} triangles ({n_active} active, "
            f"{n_vertices - n_active} passive). Heat source: {source_description}."
        )

    def _build_config(
        self,
        source_material: ThermalMaterialConfig | None,
        use_target_delta_temp: bool,
    ) -> FEMSimulationConfig:
        """Builds the shared runner's config for one heat run."""
        return FEMSimulationConfig(
            num_time_steps=self._num_time_steps,
            save_interval=self._save_interval,
            time_step_size=None,
            dt_over_t_relax=self._dt_over_t_relax,
            target_delta_temp=(
                self._target_delta_temp if use_target_delta_temp else None
            ),
            heat_source_material=source_material if use_target_delta_temp else None,
            fem_degree=self._fem_degree,
            write_xdmf=self._write_xdmf,
        )

    @classmethod
    def load_baked_mesh(
        cls,
        npz_path: Path,
        active: ThermalMaterialConfig,
        passive: ThermalMaterialConfig,
        source: float,
        region: ThermalSimConfig.HeatSourceRegion,
    ) -> BakedMesh:
        """
        Reads the baked `.npz` and builds the mesh- and field-data the shared
        runner takes.

        Per-vertex material properties are picked by thresholding the material mask
        at `ThermalSimConfig.ACTIVE_MATERIAL_THRESHOLD`, which is the FEM analogue
        of the `dr.select` the two-material spline projector applies at the
        control points. The source field is then placed on the vertices `region`
        names, the other side getting 0.

        `npz_path` is the path to the baked `.npz`. `active` and `passive` are
        the properties of the material the mask selects and rejects,
        respectively. `source` is the source value [W/m^2] written onto the
        source region's vertices. `region` is which material carries the
        source, mirroring the three-way select the IGA run makes on the same
        shared configuration.

        :returns: The mesh data, field data and active-vertex count, as a
            `BakedMesh`.
        :raises ValueError: If any array has the wrong rank or a shape
            inconsistent with the vertex count.
        """
        data = np.load(npz_path)

        vertices = np.ascontiguousarray(data["vertices"], dtype=np.float64)
        faces = np.ascontiguousarray(data["faces"], dtype=np.uint32)
        temperature = np.ascontiguousarray(data["temperature"], dtype=np.float64)
        material = np.ascontiguousarray(data["material"], dtype=np.float64)

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"`vertices` must be (n, 3); got {vertices.shape}.")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"`faces` must be (m, 3); got {faces.shape}.")
        n_vertices = vertices.shape[0]
        for name, array in (("temperature", temperature), ("material", material)):
            if array.shape != (n_vertices,):
                raise ValueError(
                    f"`{name}` must be ({n_vertices},); got {array.shape}."
                )

        is_active = material > ThermalSimConfig.ACTIVE_MATERIAL_THRESHOLD

        def select(
            active_value: float, passive_value: float
        ) -> npt.NDArray[np.float64]:
            """Per-vertex float64 field holding one of two constants, by `is_active`."""
            return np.where(is_active, active_value, passive_value).astype(np.float64)

        if region is ThermalSimConfig.HeatSourceRegion.STEEL:
            source_field = select(source, 0.0)
        elif region is ThermalSimConfig.HeatSourceRegion.WOOD:
            source_field = select(0.0, source)
        else:
            source_field = np.zeros(n_vertices, dtype=np.float64)

        field_data = FEMFieldData(
            temperature=temperature,
            conductivity=select(active.conductivity, passive.conductivity),
            density=select(active.density, passive.density),
            specific_heat=select(active.specific_heat, passive.specific_heat),
            thickness=select(active.thickness, passive.thickness),
            source=source_field,
        )
        mesh_data = FEMMeshData(vertices=vertices, faces=faces)
        return FEMHeatGTRunner.BakedMesh(
            mesh_data=mesh_data,
            field_data=field_data,
            num_active_vertices=int(np.count_nonzero(is_active)),
        )

    def write_log_json(
        self,
        diagnostics: FEMSimulationDiagnostics,
        mesh_data: FEMMeshData,
    ) -> None:
        """
        Writes the physical-diagnostics sidecar that the comparison stages read back.

        The schema is `FEMHeatGTReport`, and it is fixed by its consumers:
        `end_to_end.heat.compare_heat` diffs this run against the IGA one using
        the per-snapshot `time` and `total_heat` series together with the
        run-wide scalar `heat_generation_rate`, and
        `end_to_end.heat.iga_heat_simulation` reuses `dt` so the IGA run
        integrates the very time step this run derived from its own assembled
        (K, M).
        """
        report = FEMHeatGTReport(
            time=[snapshot.time for snapshot in diagnostics.snapshots],
            total_heat=[snapshot.total_heat for snapshot in diagnostics.snapshots],
            heat_generation_rate=diagnostics.heat_generation_rate,
            relaxation_time=diagnostics.relaxation_time,
            dt=diagnostics.dt,
            num_time_steps=self._num_time_steps,
            fem_degree=self._fem_degree,
            target_delta_temp=self._target_delta_temp,
            source_Q=diagnostics.applied_heat_source,
            heat_source=self._heat_source,
            heat_source_region=self._heat_source_region,
            steel=self._steel_config,
            wood=self._wood_config,
            num_vertices=int(mesh_data.vertices.shape[0]),
            num_triangles=int(mesh_data.faces.shape[0]),
        )
        log_path = self._output_folder / self._LOG_NAME
        log_path.write_text(json.dumps(report.to_json_dict(), indent=2) + "\n")
        logger.info(f"Diagnostics log written to {log_path}.")
