import xml.etree.ElementTree as ElementTree
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from fem.end_to_end.heat_comparison.heat_comparison_scene_config import (
    HeatComparisonSceneConfig,
)
from fem.end_to_end.heat_comparison.iga_reference import IGAReference
from fem.end_to_end.heat_comparison.scene_comparison import SceneComparison
from fem.end_to_end.heat_comparison.snapshot_comparison import SnapshotComparison
from fem.fem_field_data import FEMFieldData
from fem.heat.fem_heat_runner import FEMHeatRun
from fem.heat.fem_simulation_config import FEMSimulationConfig
from fem.matching_snapshots_guard import MatchingSnapshotsGuard
from fem.spline_to_fem_projector import SplineToFEMProjector
from fem.xdmf_time_series import XDMFTimeSeries
from forge_sim import logger


class SceneHeatComparison:
    """
    One scene's IGA-vs-FEM heat comparison: re-solve, diff, write the artifacts.

    Construction resolves the scene's directories from `scene`; it touches no
    disk, so a scene rejected here leaves no empty `fem_comparison/` folder behind
    in the dataset tree to be read later as an empty result. `run` makes the
    directory. `SceneModalComparison`
    (`fem/end_to_end/modal_comparison/scene_modal_comparison.py`) is built the
    same way for the same reason.

    Takes the values it needs directly, as `HeatComparisonSceneConfig`, rather than
    a scene identity resolved through a registry: the registry of published scenes
    is `main_scripts`' own subject (`CLAUDE.md` SS3), and a comparison for one
    scene is ordinary library functionality that has no need to know the registry
    exists.
    """

    OUTPUT_SUBDIR: Path = Path("fem_comparison")
    """Per-scene subdirectory this comparison writes its artifacts into."""
    TABLE_NAME: str = "comparison_L2.txt"
    """Human-readable per-timestep error table."""
    FEM_SOLUTION_NAME: str = "fem_solution.xdmf"
    """The FEM series, on the IGA export mesh."""
    DIFFERENCE_NAME: str = "difference.xdmf"
    """The IGA-minus-FEM difference field, on the same mesh."""
    _PEAK_FLOOR: float = 1e-30
    """
    Floor on the IGA peak that normalizes `SnapshotComparison.relative_linf`. Not a
    derived tolerance and not claimed as one: it is a bare guard against dividing by an
    all-zero reference field, and it stands unchanged because the published numbers were
    produced with it. It sits far below any temperature these scenes reach, so it never
    participates in a real division. `fem/compare_iga_fem.py` solves the same problem
    properly, with a derived threshold and an absolute-norm fallback.
    """
    _UNIT_WINDOW_SOURCE_MAGNITUDE: float = 1.0
    """
    Magnitude the window source field is assembled at before it is rescaled. Unity by
    construction, not by choice: the load vector is linear in the source, so assembling
    at this value and scaling afterwards by `_apply_target_delta_temp_scaling` is how
    the target-rise path reaches its Q once dt is known, the same two-step route the
    spline scripts take. A value other than 1.0 here would silently break that two-step
    scaling, which is why it is a class constant rather than a caller-facing knob.
    """

    def __init__(
        self,
        *,
        scene: HeatComparisonSceneConfig,
        dataset_root: Path,
        scene_subdir: Path,
        tessellation_subdivisions: int,
        num_time_steps: int,
        save_interval: int,
        window_target_delta_temp: float | None,
    ) -> None:
        """
        Pins `scene`'s directories and materials; touches no disk.

        `dataset_root` and `scene_subdir` locate the scene on disk;
        `tessellation_subdivisions`, `num_time_steps` and `save_interval` are the
        FEM discretization knobs `run` builds its solve from. `window_target_delta_temp`
        is the temperature rise a two-material scene's window source is scaled to
        reach; it is not read off `scene` because it must match what the compared
        IGA run used, so the caller passes it explicitly rather than this class
        deriving it on its own.
        """
        self._scene = scene
        self._tessellation_subdivisions = tessellation_subdivisions
        self._num_time_steps = num_time_steps
        self._save_interval = save_interval
        self._window_target_delta_temp = window_target_delta_temp
        self._scene_directory = dataset_root / scene.directory / scene_subdir
        self._output_directory = self._scene_directory / self.OUTPUT_SUBDIR

    def run(self) -> SceneComparison:
        """
        Re-solves the scene under the FEM and reports its agreement with the IGA run.

        The FEM run is pinned to the IGA run's final time (dt = T_final /
        `num_time_steps`), so the two integrate the same physical span and their
        snapshots line up. FEM values are gathered onto the IGA vertex ordering by
        nearest-dof matching before the difference is taken.

        :returns: The per-snapshot rows, also written to disk.
        :raises ValueError: If the FEM run's snapshot count does not match the IGA
            reference's, which `require_matching_snapshot_counts` rejects.
        """
        self._output_directory.mkdir(parents=True, exist_ok=True)

        iga = self.load_iga_reference(self._scene_directory, self._scene.iga_base_name)
        final_time = iga.times[-1]

        projector = SplineToFEMProjector(
            thermal_spline_path=self._scene_directory / "inpainted_thermal_data.json",
            tessellation_subdivisions=self._tessellation_subdivisions,
            target_length_scale=self._scene.target_length_scale,
        )
        field_data = self._project_scene_fields(projector)
        mesh_data = projector.get_mesh_data()

        config = FEMSimulationConfig(
            num_time_steps=self._num_time_steps,
            save_interval=self._save_interval,
            time_step_size=None,
            dt_over_t_relax=None,
            final_time=final_time,
            target_delta_temp=(
                self._window_target_delta_temp
                if self._scene.is_two_material
                else None
            ),
            heat_source_material=self._scene.window_material,
        )
        run = FEMHeatRun(
            mesh_data=mesh_data,
            field_data=field_data,
            config=config,
            comm=MPI.COMM_WORLD,
        )
        solution = run.march()
        fem_fields = solution.gather_at(iga.vertices)

        self.require_matching_snapshot_counts(
            str(self._scene.directory),
            fem_times=solution.times,
            iga_times=iga.times,
        )

        differences = []
        rows = []
        for index in range(len(iga.snapshots)):
            iga_field = iga.snapshots[index]
            difference = iga_field - fem_fields[index]
            differences.append(difference)
            reference_norm = float(np.sqrt((iga_field**2).sum()))
            reference_peak = float(np.abs(iga_field).max())
            rows.append(
                SnapshotComparison(
                    time=iga.times[index],
                    relative_l2=(
                        float(np.sqrt((difference**2).sum()) / reference_norm)
                        if reference_norm > 0.0
                        else 0.0
                    ),
                    relative_linf=float(
                        np.abs(difference).max()
                        / max(reference_peak, self._PEAK_FLOOR)
                    ),
                    fem_peak=float(fem_fields[index].max()),
                    iga_peak=float(iga_field.max()),
                )
            )

        comparison = SceneComparison(
            name=str(self._scene.directory), snapshots=tuple(rows)
        )
        summary = comparison.summary()
        self._write_table(rows, summary)
        self._write_comparison_xdmf(
            self._output_directory / self.FEM_SOLUTION_NAME, fem_fields, iga
        )
        self._write_comparison_xdmf(
            self._output_directory / self.DIFFERENCE_NAME, differences, iga
        )

        logger.info(
            f"[{self._scene.directory}] {summary}  "
            f"(artifacts -> {self._output_directory})"
        )
        return comparison

    @staticmethod
    def load_iga_reference(scene_directory: Path, base_name: str) -> IGAReference:
        """
        Reads an IGA heat run's exported mesh, snapshot times and temperature fields.

        The times live in the `.xdmf` sidecar (one `<Time>` element per snapshot)
        while the arrays live in the `.h5`; the two are paired by the `Step_<i>`
        index.

        `scene_directory` is the directory holding the `<base_name>.h5` /
        `.xdmf` pair, and `base_name` is the stem shared by the two files.
        """
        with h5py.File(scene_directory / f"{base_name}.h5", "r") as h5_file:
            vertices = np.array(h5_file["Mesh/Vertices"]).astype(np.float64)
            faces = np.array(h5_file["Mesh/Topology"]).astype(np.int32)
            step_names = sorted(
                (key for key in h5_file["Function"] if key.startswith("Step_")),
                key=lambda key: int(key.split("_")[1]),
            )
            snapshots = [
                np.array(h5_file[f"Function/{name}"]).ravel().astype(np.float64)
                for name in step_names
            ]

        root = ElementTree.parse(scene_directory / f"{base_name}.xdmf").getroot()
        times = tuple(float(element.attrib["Value"]) for element in root.iter("Time"))
        return IGAReference(
            vertices=vertices, faces=faces, times=times, snapshots=tuple(snapshots)
        )

    @classmethod
    def require_matching_snapshot_counts(
        cls,
        name: str,
        *,
        fem_times: Sequence[float],
        iga_times: Sequence[float],
    ) -> None:
        """
        Rejects a scene whose two runs were not saved at the same instants.

        Delegates to `MatchingSnapshotsGuard.require`, the check this class
        shares with `ComparisonReport.require_matching_time_steps`; `name`
        labels the scene in the raised message, since a batch run over many
        scenes needs to say which one failed. That is a structural mismatch of
        the two runs, not a magnitude anything could explain, so it is
        rejected rather than truncated to the shorter series, the same
        verdict `fem/compare_iga_fem.py` reaches for the same pairing.

        :raises ValueError: If the two runs carry different snapshot counts,
            or the same count at times that disagree by more than
            `MatchingSnapshotsGuard._TIME_RTOL` (relative to the IGA time) at
            some snapshot.
        """
        MatchingSnapshotsGuard.require(
            iga_times=iga_times,
            fem_times=fem_times,
            remediation=(
                "Rerun with a `--save-interval` that reproduces the IGA "
                "run's own cadence, over the same simulated span."
            ),
            context=name,
        )

    def _project_scene_fields(self, projector: SplineToFEMProjector) -> FEMFieldData:
        """
        Projects the scene's temperature and material fields onto the FEM mesh vertices.

        Two-material scenes threshold the facade mask at the control points and let the
        spline basis interpolate, exactly as the IGA run does; the window source is
        laid down at `_UNIT_WINDOW_SOURCE_MAGNITUDE` and rescaled once dt is known.
        """
        window = self._scene.window_material
        if window is None:
            return projector.project_fields(
                conductivity=self._scene.base_material.conductivity,
                density=self._scene.base_material.density,
                specific_heat=self._scene.base_material.specific_heat,
                thickness=self._scene.base_material.thickness,
                source=0.0,
            )

        return projector.project_two_material_fields(
            material_mask_spline_path=(
                self._scene_directory / "inpainted_facade_data.json"
            ),
            concrete_conductivity=self._scene.base_material.conductivity,
            concrete_density=self._scene.base_material.density,
            concrete_specific_heat=self._scene.base_material.specific_heat,
            concrete_thickness=self._scene.base_material.thickness,
            glass_conductivity=window.conductivity,
            glass_density=window.density,
            glass_specific_heat=window.specific_heat,
            glass_thickness=window.thickness,
            glass_source=self._UNIT_WINDOW_SOURCE_MAGNITUDE,
        )

    @staticmethod
    def _write_comparison_xdmf(
        path: Path, fields: list[npt.NDArray[np.float64]], iga: IGAReference
    ) -> None:
        """
        Writes one artifact in this comparison's recorded XDMF layout.

        `XDMFTimeSeries.Layout.COMPACT` with `Precision="8"` is the layout of every
        recorded artifact (verified byte for byte against all fourteen sidecars
        under the VGGT scenes), so ParaView and anything else already reading them
        keeps working.

        The attribute name is `temperature`, and the difference field inherits it.
        Not every recorded artifact carries that name; buildingA-spring, car, lion
        and woodshed carry `Name="field"`, so a reader keyed on the name must
        accept both.
        """
        XDMFTimeSeries.write(
            path,
            iga.vertices,
            iga.faces,
            iga.times,
            fields,
            attribute_name="temperature",
            layout=XDMFTimeSeries.Layout.COMPACT,
            declare_float_precision=True,
        )

    def _write_table(self, rows: list[SnapshotComparison], summary: str) -> None:
        """
        Writes the human-readable per-timestep error table for one scene.

        The `x6` in the header is the patch count of the surface, which is 6 for
        every registered scene: they are all cubed-sphere reconstructions, checked
        against each scene's own spline. It is written rather than computed so the
        header stays identical to the recorded tables.
        """
        subdivisions = self._tessellation_subdivisions
        lines = [
            f"# IGA vs FEM (P1, {subdivisions}x{subdivisions}x6, direct MUMPS): "
            f"{self._scene.directory}",
            f"# {'time[s]':>12} {'relL2[%]':>10} {'relLinf[%]':>11} "
            f"{'FEMpeak':>9} {'IGApeak':>9}",
        ]
        for row in rows:
            lines.append(
                f"{row.time:>14.2f} {100 * row.relative_l2:>10.4f} "
                f"{100 * row.relative_linf:>11.4f} {row.fem_peak:>9.4f} "
                f"{row.iga_peak:>9.4f}"
            )
        lines.append("# " + summary)
        (self._output_directory / self.TABLE_NAME).write_text("\n".join(lines) + "\n")
