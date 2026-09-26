import dataclasses
import json
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from unittest import mock

import numpy as np

from fem.end_to_end.modal_comparison.spline_modal_comparison_report import (
    SplineModalComparisonReport,
)

# The comparison itself needs DOLFINx (`fem.end_to_end.modal_comparison.
# SceneModalComparison`, which this environment does not carry), but the scene
# and profile resolution in front of it (`SceneModalComparisonRun`) is plain
# Python and stays reachable, the same way `SceneHeatComparisonRun` does on the
# heat side. `SplineModalComparisonReport` moved beside `SceneModalComparison`
# but carries no DOLFINx dependency of its own, so it is reachable too.
from main_scripts.compare_all_scenes import CompareAllScenesParameters
from main_scripts.compare_modal_fem_iga import ModalComparisonParameters
from main_scripts.conda_python import CondaPython
from main_scripts.heat_comparison.scene_heat_comparison_run import (
    SceneHeatComparisonRun,
)
from main_scripts.modal_comparison.scene_modal_comparison_run import (
    SceneModalComparisonRun,
)
from main_scripts.scenes.e2e_scene import E2EScene
from main_scripts.scenes.e2e_scene_profile import E2ESceneProfile
from main_scripts.scenes.vggt_scene import VGGTScene
from main_scripts.scenes.vggt_scene_profile import VGGTSceneProfile
from main_scripts.scenes.vggt_scene_thermal_profile import VGGTSceneThermalProfile
from rebel_shape.scripts.configs.thermal_material_config import ThermalMaterialConfig

# `_require_matching_source_region` is private, but it is the whole of the guard and
# its public caller needs both solvers' XDMF series and a stage directory to run.
from rebel_shape.scripts.end_to_end.end_to_end_config import EndToEndConfig
from rebel_shape.scripts.end_to_end.heat.compare_heat import (
    _require_matching_source_region,
    run_heat_comparison,
)
from rebel_shape.scripts.end_to_end.heat.heat_comparison_report import (
    HeatComparisonReport,
)
from rebel_shape.scripts.end_to_end.heat.heat_energy_log import HeatEnergyLog
from rebel_shape.scripts.end_to_end.heat.iga_heat_report import IGAHeatReport
from rebel_shape.scripts.end_to_end.heat.thermal_sim_config import ThermalSimConfig
from rebel_shape.scripts.end_to_end.metrics.recorded_runs_parameters import (
    RecordedRunsParameters,
)
from rebel_shape.scripts.end_to_end.modal.align_modal_xdmf import (
    AlignModalXdmfParameters,
)
from rebel_shape.scripts.end_to_end.modal.modal_comparison_report import (
    ModalComparisonReport,
)
from rebel_shape.scripts.end_to_end.modal.modal_frequencies import modal_frequencies


class TestEndToEndReports(unittest.TestCase):
    """
    Serialization contract of the end-to-end run reports, checked against what
    reads them back.

    `IGAHeatReport.to_json_dict` is the one report boundary no other test reaches:
    `iga_heat_log.json` is produced by a stage that needs a solved spline, and
    consumed by two others: `HeatEnergyLog.from_json` for the heat comparison, and
    the recompute stage for `lambda_1 = 1 / relaxation_time`. A renamed or dropped
    field therefore surfaces only on a full pipeline run, against recorded logs that
    a reader has no way to regenerate cheaply. Pinning the schema here closes that gap.

    The same holds one artifact over, for `modal_analysis_results.npz`: the archives
    behind the published tables were written by solvers that have since gained a key,
    and nothing rewrites them, so what has to be pinned is the readers' tolerance of
    the older layouts rather than the writers' current one.

    `modal_comparison_fem_iga.json` is the third such artifact, and the one that goes
    furthest past a schema: nothing else recomputes the FEM-versus-IGA diff, so its
    rank pairing, its rigid-body cutoff and its three summary statistics are pinned
    here alongside the keys, as are the two guards that decide a scene gets no such
    file at all, since the driver writes into the dataset tree and a refused scene
    must leave nothing there.

    The guard `compare_heat` puts between those two logs is pinned here for the same
    reason: it exists to reject a pair of runs that integrated different physics, and
    the only evidence it has of what each run integrated is the key each log carries.

    `SceneModalComparisonRun` and `SceneHeatComparisonRun` are pinned alongside the
    reports for the same underlying reason: each spawns its FEM comparison from
    `VGGTSceneProfile` values that must agree with what the compared IGA run used, and
    the argv each builds is the only place that agreement is checkable without a
    DOLFINx environment. The heat side once forwarded no window rise at all, silently
    falling back to the spawned script's own default while the driver's own comparison
    forwarded the profile's; the tests here are the ones that would have caught it.
    """

    _IGA_HEAT_LOG_KEYS = frozenset(
        {
            "xdmf",
            "dt",
            "dt_source",
            "relaxation_time",
            "num_time_steps",
            "steel_cp_fraction",
            "target_delta_temp",
            "source_Q",
            "heat_source",
            "heat_source_region",
            "total_heat_initial",
            "total_heat_final",
            "heat_generation_rate",
            "energy_log",
        }
    )
    """
    Every key `iga_heat_log.json` carries, spelled out rather than derived.

    Taking the expectation from `IGAHeatReport`'s own fields would make the
    test agree with any rename, which is the failure it exists to catch;
    these are the names the recorded logs use and the consumers index by.
    """

    _ENERGY_ENTRY_KEYS = frozenset(
        {"step", "time", "total_heat", "heat_generation_rate"}
    )
    """Every key one element of the `energy_log` series carries."""

    _FEM_HEAT_LOG = {
        "time": [0.0, 48.99474351565758, 97.98948703131516],
        "total_heat": [262586.4397160252, 265556.6170268467, 268526.79433761473],
        "heat_generation_rate": 60.62236676099091,
        "relaxation_time": 48994.743515657574,
        "dt": 0.4899474351565758,
        "num_time_steps": 1000,
        "fem_degree": 1,
        "target_delta_temp": 0.15,
        "source_Q": 7.439573591879592,
        "heat_source": 0.05,
        "heat_source_region": "steel",
        "steel": {
            "conductivity": 237.0,
            "density": 2700.0,
            "specific_heat": 900.0,
            "thickness": 0.01,
        },
        "wood": {
            "conductivity": 50.0,
            "density": 7850.0,
            "specific_heat": 500.0,
            "thickness": 0.01,
        },
        "num_vertices": 129526,
        "num_triangles": 259048,
    }
    """
    A `fem_heat_gt_log.json` as `fem/end_to_end/heat_gt/fem_heat_gt_mesh.py` writes
    it: Suzanne's recorded log, the eleven-sample series cut to three, and the
    physics the run integrated (source region and the two resolved materials)
    filled in as that run received them. Its producer is in the DOLFINx environment
    and cannot be imported here, so the shape is written out in full even though
    only the region is read back below.
    """

    _MODAL_ARCHIVE_SCHEMAS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("iga", ("eigenvalues", "eigenvectors")),
        ("fem_recorded", ("eigenvalues", "eigenvectors", "frequencies")),
        (
            "fem_current",
            ("eigenvalues", "eigenvectors", "frequencies", "mode_energies"),
        ),
    )
    """
    Every key set a `modal_analysis_results.npz` carries, oldest first.

    Read off the twenty-four archives behind the published tables: the IGA
    runner writes eigenpairs alone, and both FEM drivers wrote three keys
    when those runs were made. `mode_energies` joined the FEM schema
    afterwards and no archive is rewritten, so a reader indexing it
    unconditionally would break on every recorded file.
    """

    _MODAL_COMPARISON_KEYS = frozenset(
        {
            "scene",
            "num_rb_fem",
            "num_rb_iga",
            "num_compared_elastic_modes",
            "field_degree",
            "subdivisions_per_patch",
            "mean_rel_error",
            "median_rel_error",
            "max_rel_error",
            "modes",
        }
    )
    """
    Every key `modal_comparison_fem_iga.json` carries, spelled out rather
    than derived.

    These are the names the seven recorded comparisons behind the published
    modal table use. Taking the expectation from
    `SplineModalComparisonReport`'s own fields would make the test agree with
    any rename, which is the failure it exists to catch.
    """

    _MODAL_COMPARISON_MODE_KEYS = frozenset(
        {"elastic_mode", "fem_spline_hz", "iga_spline_hz", "rel_error"}
    )
    """
    Every key one entry of that file's `modes` array carries.

    `fem_spline_hz` is what separates this schema from the GT-mesh
    comparison's `fem_gt_hz`: the two records are the same shape for
    different pairings, and neither name may drift into the other.
    """

    @staticmethod
    def _report(
        *, target_delta_temp: float | None, source_Q: float | None
    ) -> IGAHeatReport:
        """
        An `IGAHeatReport` with distinct values in every field.

        The values are arbitrary but pairwise distinct, so a field that serialises
        into the wrong key cannot pass by coincidence. The two auto-scaling fields
        are parameters because they are the report's only optional ones.
        """
        return IGAHeatReport(
            xdmf="iga_heat.xdmf",
            dt=0.25,
            dt_source=IGAHeatReport.TimeStepSource.GT_FEM,
            relaxation_time=1234.5,
            num_time_steps=400,
            steel_cp_fraction=0.375,
            target_delta_temp=target_delta_temp,
            source_Q=source_Q,
            heat_source=7.5,
            heat_source_region=ThermalSimConfig.HeatSourceRegion.STEEL,
            total_heat_initial=101.0,
            total_heat_final=202.0,
            heat_generation_rate=3.5,
            energy_log=[
                IGAHeatReport.EnergyEntry(
                    step=0, time=0.0, total_heat=101.0, heat_generation_rate=3.5
                ),
                IGAHeatReport.EnergyEntry(
                    step=8, time=2.0, total_heat=202.0, heat_generation_rate=3.25
                ),
            ],
        )

    def test_the_written_keys_are_exactly_the_recorded_ones(self):
        """
        The key set is the schema: a consumer indexes it by name, so a rename is a
        silent break rather than a type error. Checked against the literal names the
        recorded logs carry.

        The second assertion pins the same literal against the dataclass's own fields.
        It cannot catch a bug inside `to_json_dict` (that method returns `asdict`
        plus two value overrides, so the two key sets coincide by construction); it
        keeps the literal above from drifting away from the record it describes.
        """
        written = self._report(target_delta_temp=0.5, source_Q=12.0).to_json_dict()

        self.assertEqual(frozenset(written), self._IGA_HEAT_LOG_KEYS)
        self.assertEqual(
            frozenset(field.name for field in fields(IGAHeatReport)),
            self._IGA_HEAT_LOG_KEYS,
        )
        for entry in written["energy_log"]:
            self.assertEqual(frozenset(entry), self._ENERGY_ENTRY_KEYS)

    def test_every_field_survives_the_round_trip_through_json(self):
        """
        Values, not just keys: the two enum-valued fields must cross as their
        `.value` strings (an `Enum` is not JSON-serialisable, so leaving one in
        place would raise at write time), and everything else must come back
        unchanged through a real `json.dumps`/`json.loads`.
        """
        report = self._report(target_delta_temp=0.5, source_Q=12.0)

        restored = json.loads(json.dumps(report.to_json_dict()))

        self.assertEqual(restored["dt_source"], "gt_fem")
        self.assertEqual(restored["heat_source_region"], "steel")
        self.assertEqual(restored["xdmf"], report.xdmf)
        self.assertEqual(restored["dt"], report.dt)
        self.assertEqual(restored["relaxation_time"], report.relaxation_time)
        self.assertEqual(restored["num_time_steps"], report.num_time_steps)
        self.assertEqual(restored["steel_cp_fraction"], report.steel_cp_fraction)
        self.assertEqual(restored["target_delta_temp"], report.target_delta_temp)
        self.assertEqual(restored["source_Q"], report.source_Q)
        self.assertEqual(restored["heat_source"], report.heat_source)
        self.assertEqual(restored["total_heat_initial"], report.total_heat_initial)
        self.assertEqual(restored["total_heat_final"], report.total_heat_final)
        self.assertEqual(
            restored["heat_generation_rate"], report.heat_generation_rate
        )
        self.assertEqual(
            restored["energy_log"],
            [
                {
                    "step": 0,
                    "time": 0.0,
                    "total_heat": 101.0,
                    "heat_generation_rate": 3.5,
                },
                {
                    "step": 8,
                    "time": 2.0,
                    "total_heat": 202.0,
                    "heat_generation_rate": 3.25,
                },
            ],
        )

    def test_an_unscaled_run_keeps_its_optional_keys_as_nulls(self):
        """
        A run that set no target temperature leaves both auto-scaling fields None.
        They must still be written: dropping a key rather than nulling it is the
        shape that breaks a consumer indexing by name.
        """
        written = self._report(target_delta_temp=None, source_Q=None).to_json_dict()

        self.assertEqual(frozenset(written), self._IGA_HEAT_LOG_KEYS)
        self.assertIsNone(written["target_delta_temp"])
        self.assertIsNone(written["source_Q"])

    def test_the_heat_comparison_reads_back_what_the_report_writes(self):
        """
        The consumer's half of the contract: `HeatEnergyLog.from_json` picks the IGA
        format by the presence of `energy_log` and then indexes four more names.
        Reading a freshly written file proves the producer and that consumer still
        agree, which comparing key sets alone would not.
        """
        report = self._report(target_delta_temp=0.5, source_Q=12.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "iga_heat_log.json"
            path.write_text(json.dumps(report.to_json_dict()))

            log = HeatEnergyLog.from_json(path)

        self.assertEqual(log.dt, report.dt)
        self.assertEqual(log.relaxation_time, report.relaxation_time)
        self.assertEqual(list(log.time), [0.0, 2.0])
        self.assertEqual(list(log.total_heat), [101.0, 202.0])
        self.assertEqual(list(log.gen_rate), [3.5, 3.25])
        self.assertIs(log.heat_source_region, report.heat_source_region)

    def test_the_fem_log_s_parallel_arrays_normalise_to_the_same_fields(self):
        """
        `HeatEnergyLog.from_json`'s other branch, selected by the absence of
        `energy_log`: the FEM driver writes `time` and `total_heat` as parallel
        series but `heat_generation_rate` as a single run-wide scalar.

        That asymmetry is what `np.atleast_1d` absorbs, and the reason it is load
        bearing is that `compare_heat` reads `gen_rate[-1]` on both sides without
        knowing which solver wrote which: dropping the call leaves the FEM side a 0-d
        array, which cannot be indexed at all.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "fem_heat_gt_log.json"
            path.write_text(json.dumps(self._FEM_HEAT_LOG))

            log = HeatEnergyLog.from_json(path)

        self.assertEqual(log.dt, self._FEM_HEAT_LOG["dt"])
        self.assertEqual(log.relaxation_time, self._FEM_HEAT_LOG["relaxation_time"])
        self.assertEqual(list(log.time), self._FEM_HEAT_LOG["time"])
        self.assertEqual(list(log.total_heat), self._FEM_HEAT_LOG["total_heat"])
        self.assertEqual(
            list(log.gen_rate), [self._FEM_HEAT_LOG["heat_generation_rate"]]
        )
        self.assertEqual(log.gen_rate[-1], self._FEM_HEAT_LOG["heat_generation_rate"])
        self.assertIs(
            log.heat_source_region, ThermalSimConfig.HeatSourceRegion.STEEL
        )

    def test_a_log_missing_the_two_optional_scalars_reads_them_as_none(self):
        """
        `dt` and `relaxation_time` are read with `.get`, and both fields document the
        tolerance that buys ("or None if it was not logged"). Every producer writes
        both today, so nothing else would notice the day one stopped, or the day the
        `.get` became an index, which is the same break seen from the other side.
        """
        sparse_log = {
            key: value
            for key, value in self._FEM_HEAT_LOG.items()
            if key not in ("dt", "relaxation_time")
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "fem_heat_gt_log.json"
            path.write_text(json.dumps(sparse_log))

            log = HeatEnergyLog.from_json(path)

        self.assertIsNone(log.dt)
        self.assertIsNone(log.relaxation_time)
        self.assertEqual(list(log.time), self._FEM_HEAT_LOG["time"])

    def test_a_log_predating_the_source_region_reads_it_as_none(self):
        """
        Every log recorded before the region was written carries no such key, and no
        recorded artifact can be re-stamped with the physics it actually integrated.
        Reading it must therefore yield None rather than raise or assume a default:
        assuming one is exactly the silent agreement the guard below exists to refuse.
        """
        older_log = {
            key: value
            for key, value in self._FEM_HEAT_LOG.items()
            if key != "heat_source_region"
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "fem_heat_gt_log.json"
            path.write_text(json.dumps(older_log))

            log = HeatEnergyLog.from_json(path)

        self.assertIsNone(log.heat_source_region)
        self.assertEqual(list(log.time), self._FEM_HEAT_LOG["time"])

    @staticmethod
    def _log_with_region(
        region: ThermalSimConfig.HeatSourceRegion | None,
    ) -> HeatEnergyLog:
        """A minimal energy log carrying `region`; only that field is read."""
        return HeatEnergyLog(
            time=np.array([0.0, 1.0]),
            total_heat=np.array([1.0, 2.0]),
            gen_rate=np.array([0.5, 0.5]),
            dt=0.5,
            relaxation_time=100.0,
            heat_source_region=region,
        )

    def test_two_runs_that_heated_the_same_material_are_compared(self):
        """
        The agreeing case must pass silently: the guard is on the path of every
        comparison, so a false rejection would block the whole stage.
        """
        region = ThermalSimConfig.HeatSourceRegion.WOOD

        _require_matching_source_region(
            self._log_with_region(region), self._log_with_region(region)
        )

    def test_two_runs_that_heated_different_materials_are_rejected(self):
        """
        The failure the two independent solvers make possible: with the region honoured
        on one side only, a `--thermal.heat-source-region wood` run heats the passive
        half on the IGA side and the active half on the FEM side, and every energy the
        comparison prints is a number for two different problems.

        The message is asserted on the guard's own wording and on both region names, so
        that an unrelated failure mentioning the same materials cannot stand in for it.
        """
        with self.assertRaises(ValueError) as raised:
            _require_matching_source_region(
                self._log_with_region(ThermalSimConfig.HeatSourceRegion.STEEL),
                self._log_with_region(ThermalSimConfig.HeatSourceRegion.WOOD),
            )

        message = str(raised.exception)
        self.assertIn("different materials", message)
        self.assertIn("steel", message)
        self.assertIn("wood", message)

    def test_a_run_that_recorded_no_region_is_warned_about_not_rejected(self):
        """
        One side silent is the recorded-artifact case, not a mismatch: the older log
        cannot say what it integrated, so the comparison proceeds and says so. Failing
        here instead would make every archived run unreadable by the current stage.
        """
        with self.assertLogs("rebel_shape", level="WARNING") as captured:
            _require_matching_source_region(
                self._log_with_region(None),
                self._log_with_region(ThermalSimConfig.HeatSourceRegion.STEEL),
            )

        self.assertIn("recorded on only one side", "\n".join(captured.output))

    def test_the_comparison_stage_refuses_a_mismatched_pair(self):
        """
        The guard on the path it guards, since a guard nothing calls is no guard.

        The two logs are written as the two stages write them, and the stage is run
        against them: it must raise before writing any of its outputs, so a mismatched
        pair leaves no comparison table behind to be read as a result.
        """
        iga_report = dataclasses.replace(
            self._report(target_delta_temp=0.5, source_Q=12.0),
            heat_source_region=ThermalSimConfig.HeatSourceRegion.WOOD,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = EndToEndConfig(output_root=Path(tmp_dir))
            (config.fem_heat_dir / "fem_heat_gt_log.json").write_text(
                json.dumps(self._FEM_HEAT_LOG)
            )
            (config.iga_heat_dir / "iga_heat_log.json").write_text(
                json.dumps(iga_report.to_json_dict())
            )

            with self.assertRaises(ValueError) as raised:
                run_heat_comparison(config)

            self.assertEqual(list(config.heat_compare_dir.iterdir()), [])

        self.assertIn("different materials", str(raised.exception))

    @staticmethod
    def _modal_comparison() -> SplineModalComparisonReport:
        """
        A comparison of two deliberately mismatched spectra.

        Every quantity the report derives is made distinguishable by the fixture:
        the two sides drop a different number of rigid-body modes (3 and 2, the
        FEM side's third sitting exactly on the cutoff), the FEM side carries one
        elastic mode more than the IGA side, one pair errs low and two err high, and
        the three relative errors 0.25 / 0.5 / 1.5 give a mean, a median and a
        maximum that are pairwise distinct and exact in binary floating point. They
        are assigned out of rank order: (0.25, 1.5, 0.5), not ascending, so that
        neither the median nor the maximum sits at the array position a naive
        stand-in would read: `relative_errors[1]` or `relative_errors[-1]` would
        pass for `np.median(...)` or `.max()` on an ascending fixture, undetected.
        The discretisation knobs are set away from their CLI defaults so a field
        that is never forwarded cannot pass by coincidence.
        """
        return SplineModalComparisonReport.from_spectra(
            scene="fixture-scene",
            fem_frequencies=np.array([0.0, 1.0, 2.5, 4.5, 50.0, 60.0, 200.0]),
            iga_frequencies=np.array([0.5, 2.0, 6.0, 20.0, 40.0]),
            rigid_body_cutoff_hz=2.5,
            field_degree=2,
            subdivisions_per_patch=48,
        )

    def test_the_modal_comparison_writes_exactly_the_recorded_keys(self):
        """
        The key set is the schema: the seven recorded `modal_comparison_fem_iga.json`
        files behind the published modal table are indexed by name, so a rename is a
        silent break rather than a type error.

        The second assertion pins the same literal against the dataclass's own fields,
        which is what keeps the literal above from drifting away from the record it
        describes.
        """
        written = json.loads(self._modal_comparison().to_json_text())

        self.assertEqual(frozenset(written), self._MODAL_COMPARISON_KEYS)
        self.assertEqual(
            frozenset(field.name for field in fields(SplineModalComparisonReport)),
            self._MODAL_COMPARISON_KEYS,
        )
        for mode in written["modes"]:
            self.assertEqual(frozenset(mode), self._MODAL_COMPARISON_MODE_KEYS)

    def test_the_modal_comparison_pairs_the_elastic_modes_by_rank(self):
        """
        The whole of the diff, on spectra built to separate every way it can go wrong.

        A rigid-body mode is one at or below the cutoff, counted per side: the FEM
        fixture's 2.5 Hz mode sits exactly on it, and the two counts differ, so
        reporting one side's count for both, or making the cutoff exclusive, changes
        the answer. The shorter spectrum sets how many modes are compared, so the
        FEM side's fourth elastic mode must not appear. Errors are relative to the
        IGA frequency and unsigned, which the 4.5 Hz-versus-6 Hz pair pins in both
        respects: dividing by the FEM frequency, or dropping the absolute value,
        moves it. The three per-mode errors are also out of ascending order, so
        the mean, median and max cannot be stood in for by a fixed list index.
        """
        report = self._modal_comparison()

        self.assertEqual(report.num_rb_fem, 3)
        self.assertEqual(report.num_rb_iga, 2)
        self.assertEqual(report.num_compared_elastic_modes, 3)
        self.assertEqual(len(report.modes), 3)
        self.assertEqual(report.field_degree, 2)
        self.assertEqual(report.subdivisions_per_patch, 48)

        self.assertEqual(
            [(mode.elastic_mode, mode.fem_spline_hz, mode.iga_spline_hz)
             for mode in report.modes],
            [(1, 4.5, 6.0), (2, 50.0, 20.0), (3, 60.0, 40.0)],
        )
        self.assertEqual([mode.rel_error for mode in report.modes], [0.25, 1.5, 0.5])
        self.assertEqual(report.mean_rel_error, 0.75)
        self.assertEqual(report.median_rel_error, 0.5)
        self.assertEqual(report.max_rel_error, 1.5)

    def test_a_modal_comparison_with_no_elastic_mode_is_rejected(self):
        """
        Two spectra that are entirely rigid-body leave nothing to pair, and every
        statistic the report exists to carry undefined: a mean over an empty array
        is NaN and a maximum over one raises from inside NumPy. The comparison must
        say so itself, before a file full of nulls is written.
        """
        with self.assertRaises(ValueError) as raised:
            SplineModalComparisonReport.from_spectra(
                scene="fixture-scene",
                fem_frequencies=np.array([0.1, 0.2]),
                iga_frequencies=np.array([0.3]),
                rigid_body_cutoff_hz=1.0,
                field_degree=3,
                subdivisions_per_patch=64,
            )

        self.assertIn("no elastic mode to compare", str(raised.exception))
        self.assertIn("fixture-scene", str(raised.exception))

    def test_the_modal_markdown_carries_the_geometry_order_it_is_given(self):
        """
        `geometry_order` is the one FEM knob the JSON does not record, so the markdown
        banner is the only place it is written down; it is passed in rather than held,
        which is exactly the wiring that can silently go missing.

        Each compared mode's row is pinned whole (the table is the artifact a reader
        opens), and the FEM side's uncompared fourth elastic mode must not appear in
        it, since the summary above the table counts three.
        """
        markdown = self._modal_comparison().to_markdown_text(geometry_order=4)

        self.assertIn("FEM: geom_order=4, field_deg=2, S=48.", markdown)
        self.assertIn("fixture-scene", markdown)
        self.assertIn("RB modes 3/2 (FEM/IGA).", markdown)
        for row in (
            "| 1 | 4.500 | 6.000 | 25.000% |",
            "| 2 | 50.000 | 20.000 | 150.000% |",
            "| 3 | 60.000 | 40.000 | 50.000% |",
        ):
            self.assertIn(row, markdown)
        self.assertNotIn("| 4 |", markdown)

    def test_every_comparison_knob_is_validated_by_the_name_it_carries(self):
        """
        Six near-identical validators are exactly where a copy-pasted message names
        the wrong field, which would send a reader to a knob that is perfectly fine.
        Each rejection must name its own field and print the value it refused.

        The refused values are distinct and none of them is zero, because `str(0)`
        is a substring of nearly any such message: a hardcoded `got 0.` would
        satisfy the printed-value assertion while reporting nothing. Zero is the
        boundary each validator actually turns on, so it is checked separately.
        """
        offenders = {
            "subdivisions_per_patch": -3,
            "field_degree": -5,
            "geometry_order": -7,
            "num_modes": -11,
            "rb_freq_hz": -2.5,
        }
        for name, value in offenders.items():
            with self.subTest(knob=name):
                with self.assertRaises(ValueError) as raised:
                    ModalComparisonParameters(**{name: value})

                self.assertIn(f"`{name}`", str(raised.exception))
                self.assertIn(str(value), str(raised.exception))

        for name, zero in (
            ("subdivisions_per_patch", 0),
            ("field_degree", 0),
            ("geometry_order", 0),
            ("num_modes", 0),
            ("rb_freq_hz", 0.0),
        ):
            with self.subTest(knob=name, boundary=zero):
                with self.assertRaises(ValueError):
                    ModalComparisonParameters(**{name: zero})

    def test_every_recomputation_knob_is_validated_by_the_name_it_carries(self):
        """
        The three knobs that moved off module scope onto the two parameter classes.
        Each rejection must name its own field and print the value it refused.

        The refused values are distinct and none of them is zero, because `str(0)`
        is a substring of nearly any such message.
        """
        for parameters, name, value in (
            (RecordedRunsParameters, "cluster_gap", -0.37),
            (RecordedRunsParameters, "geometry_sample_points", -9),
            (AlignModalXdmfParameters, "degeneracy_rtol", -0.41),
        ):
            with self.subTest(knob=name):
                with self.assertRaises(ValueError) as raised:
                    parameters(**{name: value})

                self.assertIn(f"`{name}`", str(raised.exception))
                self.assertIn(str(value), str(raised.exception))

        for parameters, name in (
            (RecordedRunsParameters, "cluster_gap"),
            (RecordedRunsParameters, "geometry_sample_points"),
            (AlignModalXdmfParameters, "degeneracy_rtol"),
        ):
            with self.subTest(knob=name, boundary=0):
                with self.assertRaises(ValueError):
                    parameters(**{name: 0})

    def test_the_two_comparison_reports_read_back_what_they_write(self):
        """
        Both comparison stages serialise their record with `dataclasses.asdict` and
        two later stages read it again. Reading a freshly written file proves the
        writer and the reader still agree, and that the nested entries come back as
        their own records rather than as the raw mappings JSON hands over.
        """
        heat = HeatComparisonReport(
            total_heat_initial=HeatComparisonReport.Quantity(
                fem_gt=41.5, iga_spline=39.25, rel_error=0.0542
            ),
            total_heat_final=HeatComparisonReport.Quantity(
                fem_gt=63.75, iga_spline=61.125, rel_error=0.0412
            ),
            heat_generation_rate=HeatComparisonReport.Quantity(
                fem_gt=7.125, iga_spline=7.375, rel_error=0.0351
            ),
            relaxation_time=HeatComparisonReport.RelaxationTimes(
                fem_gt=4821.5, iga_spline=None
            ),
        )
        modal = ModalComparisonReport(
            num_rb_modes_fem=6,
            num_rb_modes_iga=5,
            num_compared_elastic_modes=2,
            mean_rel_error=0.0731,
            median_rel_error=0.0625,
            max_rel_error=0.1875,
            modes=[
                ModalComparisonReport.Mode(
                    elastic_mode=1,
                    fem_gt_hz=118.5,
                    iga_spline_hz=126.25,
                    rel_error=0.0654,
                ),
                ModalComparisonReport.Mode(
                    elastic_mode=2,
                    fem_gt_hz=204.75,
                    iga_spline_hz=196.5,
                    rel_error=0.0403,
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            heat_path = Path(tmp_dir) / "heat_comparison.json"
            modal_path = Path(tmp_dir) / "modal_comparison.json"
            heat_path.write_text(json.dumps(dataclasses.asdict(heat), indent=2))
            modal_path.write_text(json.dumps(dataclasses.asdict(modal), indent=2))

            heat_back = HeatComparisonReport.from_json(heat_path)
            modal_back = ModalComparisonReport.from_json(modal_path)

        self.assertEqual(heat_back, heat)
        self.assertEqual(modal_back, modal)
        self.assertIsInstance(
            heat_back.total_heat_initial, HeatComparisonReport.Quantity
        )
        self.assertIsInstance(
            heat_back.relaxation_time, HeatComparisonReport.RelaxationTimes
        )
        self.assertIsNone(heat_back.relaxation_time.iga_spline)
        self.assertIsInstance(modal_back.modes[1], ModalComparisonReport.Mode)
        self.assertEqual(modal_back.modes[1].fem_gt_hz, 204.75)

    def test_a_scene_with_no_profile_is_rejected_before_anything_is_written(self):
        """
        The comparison is against an IGA run on one specific shell at one specific
        scale, both of which live in `SCENE_PROFILES`; a scene with no entry has
        neither, so there is nothing to compare against.

        Selecting a scene that was never published is no longer expressible: the
        driver takes `VGGTScene`, a closed enum, so the state this reaches is the
        registry having fallen out of sync with the enum, which is what
        `VGGTSceneProfile.for_scene` raises on. It is reached here by dropping a
        registered entry.

        The rejection has to happen in construction, before the output directory is
        made: the driver writes into the dataset tree, and a scene it refuses must
        leave no `fem_comparison/` folder behind to be read later as an empty result.
        """
        registry_without_car = {
            scene: profile
            for scene, profile in VGGTSceneProfile.SCENE_PROFILES.items()
            if scene is not VGGTScene.CAR
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)

            with mock.patch.object(
                VGGTSceneProfile, "SCENE_PROFILES", registry_without_car
            ):
                with self.assertRaises(ValueError) as raised:
                    SceneModalComparisonRun(
                        VGGTScene.CAR,
                        dataset_root=dataset_root,
                        params=ModalComparisonParameters(dataset_root=dataset_root),
                    )

            self.assertEqual(list(dataset_root.iterdir()), [])

        self.assertIn("No VGGTSceneProfile registered", str(raised.exception))
        self.assertIn("CAR", str(raised.exception))

    def test_every_end_to_end_target_states_its_temperature_field_rule(self):
        """The rule used to be inferred from whether a target's mesh happened to carry
        UVs, so a loader change silently moved every heat number for one target while
        leaving its modal ones untouched. Each target now states the rule, and only
        Spot's field comes from its unwrap; Suzanne carries one that is deliberately
        not used, so a registry that simply followed the meshes would get it wrong.
        """
        sources = {
            scene: E2ESceneProfile.for_scene(scene).temperature_field_source
            for scene in E2EScene
        }

        self.assertEqual(set(sources), set(E2EScene))
        uv = ThermalSimConfig.TemperatureFieldSource.UV_TEXTURE
        self.assertEqual(
            {scene for scene, src in sources.items() if src is uv}, {E2EScene.SPOT}
        )

    def test_the_field_source_enum_offers_no_inferring_member(self):
        """The footgun this replaced: any member meaning "work it out from the mesh"
        restores the silent switch, so the enum must stay exactly the two explicit
        rules."""
        self.assertEqual(
            [m.name for m in ThermalSimConfig.TemperatureFieldSource],
            ["WORLD_CHECKER", "UV_TEXTURE"],
        )

    def test_the_default_selection_is_every_registered_scene_in_registry_order(self):
        """
        The driver used to carry its own list of eight scene directory names, one of
        which spelled `buildnet-4`'s deeper nesting by hand. That was a second copy
        of what `SCENE_PROFILES` already holds, and the copy is what drifts: this
        comparison's whole purpose is agreement between two solvers, so a scene set
        that quietly diverges from the registry's yields a plausible number nobody
        can distinguish from the result being measured.

        An empty selection now means every registered scene, matching
        `CompareAllScenesParameters.selected_scenes` on the heat side.
        """
        params = ModalComparisonParameters(dataset_root=Path("unused"))

        self.assertEqual(
            params.selected_scenes, list(VGGTSceneProfile.SCENE_PROFILES)
        )
        self.assertEqual(
            [scene.value for scene in params.selected_scenes],
            ["car", "buildinga_spring", "lion", "woodshed",
             "buildnet-1", "buildnet-2", "buildnet-3", "buildnet-4"],
        )

    def test_a_selection_is_filtered_into_registry_order_not_argument_order(self):
        """
        Two scenes named back-to-front still run in the registry's order, so the run
        log reads the same whatever order the command line listed them in.
        """
        params = ModalComparisonParameters(
            dataset_root=Path("unused"),
            scenes=[VGGTScene.BUILDNET_2, VGGTScene.CAR],
        )

        self.assertEqual(
            params.selected_scenes, [VGGTScene.CAR, VGGTScene.BUILDNET_2]
        )

    def test_a_registered_scene_with_no_shell_material_is_rejected_the_same_way(self):
        """
        The other half of the same guard, and the one no registered scene can reach:
        every published profile names a `structural_material`, so it takes a
        fabricated one to run this path at all. The fabrication now has to patch a
        real scene's entry rather than invent a new key: `SCENE_PROFILES` is keyed by
        `VGGTScene`, a closed enum with no member an arbitrary label like
        `"shell-less"` could construct.

        It is checked in construction rather than beside the solve so that a FEM
        archive surviving from an earlier run is reported rather than skipped past:
        an archive nothing can tie to a shell is exactly the state the guard exists
        to surface. `run_vggt_pipeline._run_heat_sim` orders its window-material
        check the same way. The skip-if-exists check itself is no longer in this
        class: it moved with the solve into `SceneModalComparison`
        (`fem/end_to_end/modal_comparison/scene_modal_comparison.py`), which needs
        DOLFINx and cannot be constructed here, so the archive's name is the
        literal that class's own `ARCHIVE_NAME` carries, not a reference to it.
        """
        shell_less = VGGTSceneProfile(
            directory=Path("car"),
            thermal=VGGTSceneProfile.SCENE_PROFILES[VGGTScene.CAR].thermal,
            structural_material=None,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            solved = dataset_root / "car" / "modal_analysis" / "fem_comparison"
            solved.mkdir(parents=True)
            (solved / "modal_analysis_results.npz").write_bytes(b"")

            patched = {VGGTScene.CAR: shell_less}
            with mock.patch.dict(VGGTSceneProfile.SCENE_PROFILES, patched):
                with self.assertRaises(ValueError) as raised:
                    SceneModalComparisonRun(
                        VGGTScene.CAR,
                        dataset_root=dataset_root,
                        params=ModalComparisonParameters(dataset_root=dataset_root),
                    )

        self.assertIn("no `structural_material`", str(raised.exception))
        self.assertIn("car", str(raised.exception))

    def test_every_recorded_modal_archive_schema_still_reads_as_frequencies(self):
        """
        `modal_frequencies` against all three key sets on disk, including the two
        that predate `mode_energies`. It may not index a key the recorded
        archives lack.

        The eigenvalues are exact squares of `2 pi f`, one of them negated: an
        archive carrying no `frequencies` must be converted through
        `sqrt(|lambda|) / (2 pi)`, and that sign is how the rigid-body modes arrive,
        so a conversion dropping the absolute value would yield `nan` instead of the
        frequency the FEM schemas store outright.

        They are also stored out of order, which `modal_frequencies` must preserve:
        `align_modal_xdmf` pairs its result positionally with the mode shapes in the
        eigenmode HDF5, so a sort here would relabel every mode. The other reader this
        fixture used to exercise side by side, `SceneModalComparison.sorted_frequencies`
        (which sorts, because it pairs two independent spectra by rank rather than
        indexing one), moved to `fem/tests/test_scene_modal_comparison.py` along with
        the DOLFINx-only class it belongs to; that module runs the same three schemas
        through it, checking the sorted counterpart of this contract.
        """
        as_stored = [3.0, 0.5, 7.0]
        arrays = {
            "eigenvalues": np.array([
                (2.0 * np.pi * 3.0) ** 2,
                -((2.0 * np.pi * 0.5) ** 2),
                (2.0 * np.pi * 7.0) ** 2,
            ]),
            "eigenvectors": np.eye(3),
            "frequencies": np.array(as_stored),
            "mode_energies": np.zeros((3, 4)),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            for label, keys in self._MODAL_ARCHIVE_SCHEMAS:
                path = Path(tmp_dir) / f"{label}.npz"
                np.savez(path, **{key: arrays[key] for key in keys})

                with self.subTest(schema=label):
                    frequencies = modal_frequencies(path)

                    self.assertEqual(len(frequencies), 3)
                    for got, want in zip(frequencies, as_stored):
                        self.assertAlmostEqual(got, want, places=12)

    def test_the_heat_command_carries_the_profile_s_values_not_a_coincidental_default(
        self,
    ):
        """
        Every thermal value in `SceneHeatComparisonRun`'s argv must trace back to the
        scene's `VGGTSceneProfile`, not to a script default that happens to agree
        with it. The real defect this closes forwarded no `--target-delta-temp` at
        all, so the IGA side fell back to the heat script's own default while this
        FEM comparison read the profile's: a fixture whose numbers equal an
        existing default cannot tell "read from the profile" apart from "fell back
        and got lucky", so every field here is a value nothing else in the system
        defaults to.
        """
        car = VGGTSceneProfile.SCENE_PROFILES[VGGTScene.CAR]
        fixture = dataclasses.replace(
            car,
            directory=Path("fixture_single"),
            iga_base_name="heat_simulation_fixture_single",
            thermal=VGGTSceneThermalProfile(
                base_material=ThermalMaterialConfig(
                    conductivity=1.111,
                    density=3333.0,
                    specific_heat=444.0,
                    thickness=0.0222,
                ),
                target_length_scale=12.5,
                window_target_delta_temp=3.75,
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            with mock.patch.dict(
                VGGTSceneProfile.SCENE_PROFILES, {VGGTScene.CAR: fixture}
            ), mock.patch.object(
                CondaPython,
                "resolve_from_environment",
                return_value=Path("conda-python"),
            ), mock.patch(
                "main_scripts.heat_comparison.scene_heat_comparison_run."
                "subprocess.run"
            ) as mock_run:
                run = SceneHeatComparisonRun(
                    VGGTScene.CAR,
                    params=CompareAllScenesParameters(dataset_root=dataset_root),
                )
                run.run()

        command = mock_run.call_args.args[0]

        self.assertEqual(
            command,
            [
                "conda-python",
                str(SceneHeatComparisonRun.COMPARISON_SCRIPT),
                "--dataset-root", str(dataset_root),
                "--directory", "fixture_single",
                "--scene-subdir", "heat_simulation",
                "--iga-base-name", "heat_simulation_fixture_single",
                "--tessellation-subdivisions", "512",
                "--num-time-steps", "500",
                "--save-interval", "50",
                "--target-length-scale", "12.5",
                "--base-material.conductivity", "1.111",
                "--base-material.density", "3333.0",
                "--base-material.specific-heat", "444.0",
                "--base-material.thickness", "0.0222",
                "--window-target-delta-temp", "3.75",
            ],
        )

    def test_a_window_material_on_the_profile_adds_two_material_and_its_own_glass(
        self,
    ):
        """
        The other half of the same forwarding contract: a profile naming a
        `window_material` must add `--two-material` and the four window-material
        flags, carrying that material's own values rather than the base material's
        or a script default. Every number here is distinct from the single-material
        fixture above and from every registered scene, so a swapped or dropped field
        cannot pass by coincidence either.
        """
        car = VGGTSceneProfile.SCENE_PROFILES[VGGTScene.CAR]
        fixture = dataclasses.replace(
            car,
            directory=Path("fixture_two"),
            iga_base_name="heat_simulation_fixture_two",
            thermal=VGGTSceneThermalProfile(
                base_material=ThermalMaterialConfig(
                    conductivity=5.5,
                    density=6666.0,
                    specific_heat=777.0,
                    thickness=0.0333,
                ),
                window_material=ThermalMaterialConfig(
                    conductivity=9.99,
                    density=8888.0,
                    specific_heat=111.0,
                    thickness=0.0099,
                ),
                target_length_scale=8.25,
                window_target_delta_temp=2.25,
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            with mock.patch.dict(
                VGGTSceneProfile.SCENE_PROFILES, {VGGTScene.CAR: fixture}
            ), mock.patch.object(
                CondaPython,
                "resolve_from_environment",
                return_value=Path("conda-python"),
            ), mock.patch(
                "main_scripts.heat_comparison.scene_heat_comparison_run."
                "subprocess.run"
            ) as mock_run:
                run = SceneHeatComparisonRun(
                    VGGTScene.CAR,
                    params=CompareAllScenesParameters(dataset_root=dataset_root),
                )
                run.run()

        command = mock_run.call_args.args[0]

        self.assertIn("--two-material", command)
        tail = command[command.index("--two-material") :]
        self.assertEqual(
            tail,
            [
                "--two-material",
                "--window-material-config.conductivity", "9.99",
                "--window-material-config.density", "8888.0",
                "--window-material-config.specific-heat", "111.0",
                "--window-material-config.thickness", "0.0099",
            ],
        )
        self.assertEqual(
            command[command.index("--target-length-scale") + 1], "8.25"
        )
        self.assertEqual(
            command[command.index("--window-target-delta-temp") + 1], "2.25"
        )

    def test_a_scene_with_no_window_material_emits_no_two_material_flag(self):
        """
        A single-material profile must not add `--two-material` or any of the four
        window-material flags: the branch a profile carrying no `window_material`
        must not take.
        """
        car = VGGTSceneProfile.SCENE_PROFILES[VGGTScene.CAR]
        fixture = dataclasses.replace(
            car,
            thermal=dataclasses.replace(car.thermal, window_material=None),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            with mock.patch.dict(
                VGGTSceneProfile.SCENE_PROFILES, {VGGTScene.CAR: fixture}
            ), mock.patch.object(
                CondaPython,
                "resolve_from_environment",
                return_value=Path("conda-python"),
            ), mock.patch(
                "main_scripts.heat_comparison.scene_heat_comparison_run."
                "subprocess.run"
            ) as mock_run:
                run = SceneHeatComparisonRun(
                    VGGTScene.CAR,
                    params=CompareAllScenesParameters(dataset_root=dataset_root),
                )
                run.run()

        command = mock_run.call_args.args[0]

        self.assertNotIn("--two-material", command)
        self.assertNotIn("--window-material-config.conductivity", command)

    def test_a_heat_comparison_scene_with_no_profile_is_rejected(self):
        """
        The heat side's own version of the guard `SceneModalComparisonRun` carries: a
        scene with no `SCENE_PROFILES` entry names no materials and no length scale
        to compare against, so construction must refuse it rather than build a
        command from a default that was never recorded as this scene's.
        """
        registry_without_car = {
            scene: profile
            for scene, profile in VGGTSceneProfile.SCENE_PROFILES.items()
            if scene is not VGGTScene.CAR
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            with mock.patch.object(
                VGGTSceneProfile, "SCENE_PROFILES", registry_without_car
            ):
                with self.assertRaises(ValueError) as raised:
                    SceneHeatComparisonRun(
                        VGGTScene.CAR,
                        params=CompareAllScenesParameters(dataset_root=dataset_root),
                    )

        self.assertIn("No VGGTSceneProfile registered", str(raised.exception))
        self.assertIn("CAR", str(raised.exception))

    def test_the_default_heat_selection_is_every_registered_scene_in_registry_order(
        self,
    ):
        """
        An empty `--scenes` selects every registered scene in the registry's own
        order, matching `ModalComparisonParameters.selected_scenes` on the modal
        side: a driver that instead iterated some other order would run a
        different sweep than the one the registry records, silently.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            params = CompareAllScenesParameters(dataset_root=Path(tmp_dir))

            self.assertEqual(
                params.selected_scenes, list(VGGTSceneProfile.SCENE_PROFILES)
            )
            self.assertEqual(
                [scene.value for scene in params.selected_scenes],
                ["car", "buildinga_spring", "lion", "woodshed",
                 "buildnet-1", "buildnet-2", "buildnet-3", "buildnet-4"],
            )

    def test_a_heat_selection_is_filtered_into_registry_order_not_argument_order(
        self,
    ):
        """
        Two scenes named back-to-front on the command line still run in the
        registry's order, so a re-run of a subset reads the same regardless of how
        the scenes happened to be listed.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            params = CompareAllScenesParameters(
                dataset_root=Path(tmp_dir),
                scenes=[VGGTScene.BUILDNET_2, VGGTScene.CAR],
            )

            self.assertEqual(
                params.selected_scenes, [VGGTScene.CAR, VGGTScene.BUILDNET_2]
            )
