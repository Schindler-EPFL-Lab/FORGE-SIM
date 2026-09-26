import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from _fem_fixtures import (
    ACTIVE,
    INITIAL_TEMPERATURE,
    PASSIVE,
    SQUARE_FACES,
    SQUARE_VERTICES,
    write_fixture_npz,
)

from fem.end_to_end.heat_gt import fem_heat_gt_runner
from fem.end_to_end.heat_gt.fem_heat_gt_mesh import (
    FEMHeatGTMeshParameters,
    run_fem_heat,
)
from fem.end_to_end.heat_gt.fem_heat_gt_runner import FEMHeatGTRunner
from fem.fem_mesh_data import FEMMeshData
from fem.heat.fem_simulation_config import FEMSimulationConfig
from fem.heat.fem_simulation_diagnostics import FEMSimulationDiagnostics
from fem.heat.snapshot_diagnostics import SnapshotDiagnostics
from rebel_shape.scripts.end_to_end.heat.heat_energy_log import HeatEnergyLog
from rebel_shape.scripts.end_to_end.heat.thermal_sim_config import ThermalSimConfig


class TestFEMHeatGTRunner(unittest.TestCase):
    """
    Where the heat source lands, the `FEMSimulationConfig` `run` assembles for
    the shared solve (with the solve itself mocked away), and the
    `fem_heat_gt_log.json` sidecar a finished run leaves behind: the only
    thing the comparison stage and the IGA run ever see of this run, and both
    live in the other environment.

    The solve itself is the shared `FEMHeatRun.run_with_xdmf`, pinned by
    `test_fem_heat_runner.py`.

    The source placement is the load-bearing one. Which material carries
    the source is a three-way choice on the configuration both solvers
    share, and the IGA run honours all three; a driver that hardcoded one
    of them would integrate different physics from the run it is compared
    against, and the comparison would report a plausible number for it.

    The sidecar is checked by reading it back with the consumer that
    actually reads it, `HeatEnergyLog.from_json`. That consumer lives in
    the other environment's package but imports cleanly here, so the
    producer and its reader can be pinned against each other in one
    process rather than by a hand-written fixture on each side.
    """

    SOURCE = 7.5
    """Source magnitude handed to `FEMHeatGTRunner.load_baked_mesh`, in W/m^2."""

    @classmethod
    def setUpClass(cls) -> None:
        # A real record rather than a mock, because it is serialised into the
        # sidecar; the values are arbitrary and pairwise distinct, so a field
        # written into the wrong key cannot pass by coincidence.
        cls.stub_diagnostics = FEMSimulationDiagnostics(
            dt=0.25,
            relaxation_time=2.5e4,
            heat_generation_rate=3.0,
            applied_heat_source=1.5,
            snapshots=(
                SnapshotDiagnostics(time=0.0, total_heat=10.0, relative_drift=0.0),
                SnapshotDiagnostics(
                    time=0.5, total_heat=11.5, relative_drift=-1.0e-14
                ),
            ),
        )

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self._temporary.name)
        self.npz_path = self.folder / "heat_gt.npz"
        write_fixture_npz(self.npz_path)
        self.parameters = FEMHeatGTMeshParameters(
            npz_path=self.npz_path,
            output_folder=self.folder,
            steel_config=ACTIVE,
            wood_config=PASSIVE,
            heat_source_region=ThermalSimConfig.HeatSourceRegion.WOOD,
        )
        self.runner = FEMHeatGTRunner(
            npz_path=self.parameters.npz_path,
            output_folder=self.parameters.output_folder,
            steel_config=self.parameters.steel_config,
            wood_config=self.parameters.wood_config,
            heat_source=self.parameters.heat_source,
            heat_source_region=self.parameters.heat_source_region,
            target_delta_temp=self.parameters.target_delta_temp,
            num_time_steps=self.parameters.num_time_steps,
            save_interval=self.parameters.save_interval,
            dt_over_t_relax=self.parameters.dt_over_t_relax,
            fem_degree=self.parameters.fem_degree,
            write_xdmf=self.parameters.write_xdmf,
        )
        self.log_path = self.folder / "fem_heat_gt_log.json"

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def load(self, region: ThermalSimConfig.HeatSourceRegion):
        """`FEMHeatGTRunner.load_baked_mesh` on the fixture, source on `region`."""
        return FEMHeatGTRunner.load_baked_mesh(
            npz_path=self.npz_path,
            active=ACTIVE,
            passive=PASSIVE,
            source=self.SOURCE,
            region=region,
        )

    def test_the_source_lands_on_the_region_the_configuration_names(self) -> None:
        """
        The regression test for a driver that placed the source by itself.

        Hardcoding the active material makes `--heat-source-region wood` a silent
        no-op: the run stays well-posed, heats the wrong half of the surface, and the
        heat comparison reports its energies against an IGA run that heated the other
        half.
        """
        expected_by_region = {
            ThermalSimConfig.HeatSourceRegion.STEEL: [
                self.SOURCE, self.SOURCE, 0.0, 0.0
            ],
            ThermalSimConfig.HeatSourceRegion.WOOD: [
                0.0, 0.0, self.SOURCE, self.SOURCE
            ],
            ThermalSimConfig.HeatSourceRegion.NONE: [0.0, 0.0, 0.0, 0.0],
        }

        for region, expected in expected_by_region.items():
            with self.subTest(region=region.name):
                baked = self.load(region)

                np.testing.assert_array_equal(
                    baked.field_data.source, np.array(expected)
                )

    def test_the_material_properties_follow_the_mask_whatever_the_region(self) -> None:
        """
        The region moves the source and nothing else.

        The two materials' conductivities differ by ~300x here, so a region that
        leaked into the property selection would swap the diffusivities of the two
        halves and change every energy the run reports, not just where it is generated.
        """
        for region in ThermalSimConfig.HeatSourceRegion:
            with self.subTest(region=region.name):
                baked = self.load(region)

                self.assertEqual(baked.num_active_vertices, 2)
                np.testing.assert_array_equal(
                    baked.field_data.conductivity,
                    np.array([ACTIVE.conductivity] * 2 + [PASSIVE.conductivity] * 2),
                )
                np.testing.assert_array_equal(
                    baked.field_data.thickness,
                    np.array([ACTIVE.thickness] * 2 + [PASSIVE.thickness] * 2),
                )
                np.testing.assert_array_equal(
                    baked.field_data.temperature, INITIAL_TEMPERATURE
                )

    def solver_configuration(self, **overrides) -> FEMSimulationConfig:
        """Runs the driver with the runner mocked out; returns what it was handed."""
        parameters = FEMHeatGTMeshParameters(
            npz_path=self.npz_path,
            output_folder=self.folder,
            steel_config=ACTIVE,
            wood_config=PASSIVE,
            **overrides,
        )
        with mock.patch.object(fem_heat_gt_runner, "FEMHeatRun") as runner:
            runner.return_value.run_with_xdmf.return_value = self.stub_diagnostics
            run_fem_heat(parameters)
        return runner.call_args.kwargs["config"]

    def test_the_target_rise_is_scaled_against_the_material_the_region_names(
        self,
    ) -> None:
        """
        Q = dT rho c h / t_sim is divided by the source material's thermal mass, so
        routing the wrong slot mis-scales every source by the ratio of the two (here
        39x) while leaving the run perfectly well-posed.
        """
        for region, expected in (
            (ThermalSimConfig.HeatSourceRegion.STEEL, ACTIVE),
            (ThermalSimConfig.HeatSourceRegion.WOOD, PASSIVE),
        ):
            with self.subTest(region=region.name):
                config = self.solver_configuration(
                    target_delta_temp=2.0, heat_source_region=region
                )

                self.assertIs(config.heat_source_material, expected)
                self.assertEqual(config.target_delta_temp, 2.0)

    def test_a_source_free_run_carries_no_target_rise_into_the_solver(self) -> None:
        """
        Under `NONE` the target rise must be dropped, not merely unused: the runner
        rejects a rise with no material behind it, so forwarding one would abort a run
        whose only fault is that the pipeline's default rise was left set.
        """
        config = self.solver_configuration(
            target_delta_temp=2.0,
            heat_source_region=ThermalSimConfig.HeatSourceRegion.NONE,
        )

        self.assertIsNone(config.heat_source_material)
        self.assertIsNone(config.target_delta_temp)

    def write_log(self) -> None:
        """Writes the sidecar for the fixture mesh and `stub_diagnostics`."""
        self.runner.write_log_json(
            self.stub_diagnostics,
            FEMMeshData(vertices=SQUARE_VERTICES, faces=SQUARE_FACES),
        )

    def test_the_comparison_reads_back_what_the_run_writes(self) -> None:
        """
        The consumer's half: `HeatEnergyLog.from_json` selects this format because
        the IGA log's `energy_log` key is missing, then indexes five names. Reading
        a freshly written file proves the two still agree, which comparing key sets on
        either side alone would not.
        """
        self.write_log()

        log = HeatEnergyLog.from_json(self.log_path)

        self.assertEqual(list(log.time), [0.0, 0.5])
        self.assertEqual(list(log.total_heat), [10.0, 11.5])
        self.assertEqual(
            list(log.gen_rate), [self.stub_diagnostics.heat_generation_rate]
        )
        self.assertEqual(log.dt, self.stub_diagnostics.dt)
        self.assertEqual(log.relaxation_time, self.stub_diagnostics.relaxation_time)
        self.assertIs(log.heat_source_region, ThermalSimConfig.HeatSourceRegion.WOOD)

    def test_the_run_records_the_physics_it_integrated(self) -> None:
        """
        The region, both resolved materials and the element degree, so a recorded run
        can be reproduced from its own artifact rather than from the command line that
        launched it.

        The region crosses as its `.value` string, which is the spelling the IGA log
        carries and the one `HeatEnergyLog` resolves; the member name would parse as
        nothing on the way back.
        """
        self.write_log()

        written = json.loads(self.log_path.read_text())

        self.assertEqual(written["heat_source_region"], "wood")
        self.assertEqual(
            written["steel"],
            {
                "conductivity": ACTIVE.conductivity,
                "density": ACTIVE.density,
                "specific_heat": ACTIVE.specific_heat,
                "thickness": ACTIVE.thickness,
            },
        )
        self.assertEqual(
            written["wood"],
            {
                "conductivity": PASSIVE.conductivity,
                "density": PASSIVE.density,
                "specific_heat": PASSIVE.specific_heat,
                "thickness": PASSIVE.thickness,
            },
        )
        self.assertEqual(written["fem_degree"], self.parameters.fem_degree)
        self.assertEqual(written["num_vertices"], len(SQUARE_VERTICES))
        self.assertEqual(written["num_triangles"], len(SQUARE_FACES))
        self.assertEqual(
            written["source_Q"], self.stub_diagnostics.applied_heat_source
        )
