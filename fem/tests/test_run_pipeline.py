import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Imported for its module-level side effect: `_fem_fixtures` puts the
# repository root on `sys.path`, so `fem` and `forge_sim` resolve.
import _fem_fixtures  # noqa: F401
import tyro

from fem.end_to_end.heat_gt.fem_heat_gt_mesh import FEMHeatGTMeshParameters
from fem.end_to_end.modal_gt.fem_shell_modal_gt_mesh import (
    FEMShellModalGTMeshParameters,
)
from forge_sim.scripts.configs.structural_material_config import (
    StructuralMaterialConfig,
)
from forge_sim.scripts.configs.thermal_material_config import ThermalMaterialConfig
from forge_sim.scripts.end_to_end.end_to_end_config import EndToEndConfig
from forge_sim.scripts.end_to_end.heat.thermal_sim_config import ThermalSimConfig
from forge_sim.scripts.end_to_end.modal.modal_sim_config import ModalSimConfig
from main_scripts.pipeline.stage import Stage
from main_scripts.pipeline.stage_commands import StageCommands
from main_scripts.run_pipeline import RunPipelineParameters, main
from main_scripts.scenes.e2e_scene import E2EScene


class TestPipelineForwarding(unittest.TestCase):
    """
    What `StageCommands` sends the two conda-side drivers, as they parse it.

    `main_scripts/run_pipeline.py` builds each stage's command line as
    a list of strings and hands it to a subprocess in the other
    environment. Nothing checks the two ends against each other, and every
    way they can drift apart fails the same silent way: a flag the driver
    renamed, a value crossing in a spelling the parser rejects (an enum's
    `.value` where tyro accepts only its member name), or a knob the
    launcher stopped forwarding, all leave the stage running on the
    driver's own defaults and the comparison downstream reporting a
    plausible agreement number for two different problems.

    Parsing succeeds only if every forwarded flag exists on the driver:
    tyro rejects an unknown one, so these tests pin both halves of the
    contract: that the argv is accepted at all, and that the values
    arriving are the ones the shared configuration holds.

    This lives on the conda side because it is the only environment that
    can import the drivers, both of which reach DOLFINx. It can import the
    launcher because `main_scripts` is a plain directory on
    `sys.path`, belonging to neither package.
    """

    FIXTURE_NUM_MODES: int = 7
    """Mode count, diverged from the 20 both the driver and the configuration
    default to."""

    RECONSTRUCT_MODULE: str = (
        "forge_sim.scripts.end_to_end.reconstruction.reconstruct_surface"
    )
    """The module the `reconstruct` stage runs, which `run_pipeline` skips when a
    pre-built spline is in use."""

    DEDICATED_DRIVER_SCRIPTS: tuple[str, str] = (
        "fem_heat_gt_mesh.py",
        "fem_shell_modal_gt_mesh.py",
    )
    """The two stages whose argv is built from their own drivers' flags rather than
    from `EndToEndConfig` and the forwarded dataset overrides."""

    SUZANNE_SPLINE: str = "data/splines/spline_suzanne.json"
    """Suzanne's recovered spline, which the repository ships."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.output_root = Path(self._temporary.name)

        # Every forwarded value is diverged from the driver's own default, so a
        # test built on the shared configuration's defaults cannot see a flag
        # the launcher has stopped sending. The two sides agree on more than
        # they disagree: both thermal slots are 1 cm thick, both modal
        # materials are 210 GPa, 0.3, 7850 steel at a 5/6 correction, and both
        # ask for 20 modes.
        self._fixture_steel = ThermalMaterialConfig(
            conductivity=201.0, density=2710.0, specific_heat=910.0, thickness=0.011
        )
        self._fixture_wood = ThermalMaterialConfig(
            conductivity=45.0, density=7830.0, specific_heat=505.0, thickness=0.012
        )
        self._fixture_structural = StructuralMaterialConfig(
            youngs_modulus=195.0e9,
            poisson_ratio=0.28,
            density=7700.0,
            thickness=0.0021,
            shear_correction_factor=0.85,
        )

        self.config = EndToEndConfig(
            output_root=self.output_root,
            thermal=ThermalSimConfig(
                steel=self._fixture_steel, wood=self._fixture_wood
            ),
            modal=ModalSimConfig(
                steel=self._fixture_structural, num_modes=self.FIXTURE_NUM_MODES
            ),
        )

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def stage_arguments(self, config: EndToEndConfig, stage: Stage) -> list[str]:
        """
        The argv `StageCommands` builds for `stage`, without its interpreter and
        script.

        Both drivers' `__post_init__` reject a mesh path that does not exist, so the
        input each stage names is created first; they only check for its presence, so
        an empty file is enough and no mesh has to be baked.
        """
        for directory, name in (
            (EndToEndConfig.FEM_HEAT_DIR_NAME, "heat_gt.npz"),
            (EndToEndConfig.FEM_MODAL_DIR_NAME, "modal_gt.npz"),
        ):
            path = config.dir_path(directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / name).touch()
        commands = StageCommands(
            cfg=config, conda_python="conda-python", use_gpu=True, overrides=[]
        ).commands
        return commands[stage][2:]

    def test_the_heat_driver_receives_the_configured_materials(self) -> None:
        """
        Both material slots arrive whole, field by field.

        The two solvers are compared against each other, so a property that fails to
        cross, and silently falls back to the driver's own standalone default, puts
        them on different physics: the driver's passive slot is a softwood 300x less
        conductive than the steel the pipeline sends it.
        """
        arguments = self.stage_arguments(self.config, Stage.FEM_HEAT)

        parameters = tyro.cli(FEMHeatGTMeshParameters, args=arguments)

        self.assertEqual(parameters.steel_config, self.config.thermal.steel)
        self.assertEqual(parameters.wood_config, self.config.thermal.wood)

    def test_the_heat_driver_receives_the_configured_time_stepping(self) -> None:
        """
        The scalars that fix which problem is integrated, and for how long.

        `dt_over_t_relax` and `num_time_steps` are the load-bearing pair: the IGA run
        reuses the absolute dt this run derives from them, so a stale value here moves
        both sides at once and leaves no disagreement to notice.
        """
        arguments = self.stage_arguments(self.config, Stage.FEM_HEAT)
        thermal = self.config.thermal

        parameters = tyro.cli(FEMHeatGTMeshParameters, args=arguments)

        self.assertEqual(parameters.num_time_steps, thermal.num_time_steps)
        self.assertEqual(parameters.dt_over_t_relax, thermal.dt_over_t_relax)
        self.assertEqual(parameters.save_interval, thermal.save_interval)
        self.assertEqual(parameters.heat_source, thermal.heat_source)
        self.assertEqual(parameters.target_delta_temp, thermal.target_delta_temp)

    def test_every_source_region_crosses_as_the_member_the_iga_side_reads(self) -> None:
        """
        All three regions, because the crossing is where they can be lost.

        tyro accepts only an enum's member name, so forwarding `.value` (the
        spelling the JSON logs carry, and the obvious thing to reach for) is rejected
        outright at the stage boundary. Every member is parsed rather than just the
        configured one: a forwarding that happens to work for today's default is how
        this drifts back unnoticed.
        """
        for region in ThermalSimConfig.HeatSourceRegion:
            with self.subTest(region=region.name):
                config = dataclasses.replace(
                    self.config,
                    thermal=dataclasses.replace(
                        self.config.thermal, heat_source_region=region
                    ),
                )
                arguments = self.stage_arguments(config, Stage.FEM_HEAT)

                parameters = tyro.cli(FEMHeatGTMeshParameters, args=arguments)

                self.assertIs(parameters.heat_source_region, region)

    def test_the_modal_driver_receives_the_configured_material(self) -> None:
        """
        The structural material, whole.

        Thickness shows what a missed field costs: a shell's bending frequencies scale
        with h, so the 1.5 mm the driver falls back to against the 2 mm a recorded run
        configures would return the whole spectrum a quarter low and still look like a
        converged solve.
        """
        arguments = self.stage_arguments(self.config, Stage.FEM_MODAL)

        parameters = tyro.cli(FEMShellModalGTMeshParameters, args=arguments)

        self.assertEqual(parameters.material_config, self.config.modal.steel)

    def test_the_modal_driver_receives_the_configured_discretisation(self) -> None:
        """
        The mode count, the field degree and the locking cure the recorded runs used.

        `fem_field_degree` is P3 on the configuration against the driver's own P2, and
        the difference is ~3 % of residual locking stiffness, the same size as the
        errors the comparison reports, so it would be read as a result.

        The B-bar cure is asserted on the argv rather than on the parsed value: the
        launcher passes it as a bare flag and the driver already defaults to it, so
        only the flag's presence distinguishes a launcher that asks for it from one
        that has stopped.
        """
        arguments = self.stage_arguments(self.config, Stage.FEM_MODAL)

        parameters = tyro.cli(FEMShellModalGTMeshParameters, args=arguments)

        self.assertEqual(parameters.num_modes, self.config.modal.num_modes)
        self.assertEqual(parameters.field_degree, self.config.modal.fem_field_degree)
        self.assertIn("--use-bbar", arguments)
        self.assertFalse(parameters.use_psri)

    @staticmethod
    def executed_commands(**parameters: object) -> list[list[str]]:
        """
        Every argv `main` hands to `subprocess.run` for a full run, in order.

        The subprocess is replaced by a stub that reports success, so no stage runs
        and the argv is exactly what each stage would have been started with.
        """
        with mock.patch(
            "main_scripts.run_pipeline.subprocess.run",
            return_value=mock.Mock(returncode=0),
        ) as run:
            main(
                RunPipelineParameters(
                    conda_python=Path(sys.executable), use_gpu=False, **parameters
                )
            )
        return [call.args[0] for call in run.call_args_list]

    def stage_configs(self, commands: list[list[str]]) -> list[EndToEndConfig]:
        """
        The `EndToEndConfig` each dataset-config stage would parse from its argv.

        Each stage reads the whole configuration from its own command line, so what a
        stage simulates is what this parse returns, whatever `main` held itself. The
        two DOLFINx ground-truth drivers take their own flags and are left out.
        """
        configs = []
        for argv in commands:
            if argv[1].endswith(self.DEDICATED_DRIVER_SCRIPTS):
                continue
            arguments = argv[3:] if argv[1] == "-m" else argv[2:]
            configs.append(tyro.cli(EndToEndConfig, args=arguments))
        return configs

    def test_a_scene_without_a_spline_reconstructs_instead_of_using_suzannes(
        self,
    ) -> None:
        """
        `--scene BUNNY` must neither skip reconstruction nor hand any stage Suzanne's
        spline.

        The config default is Suzanne's spline, and the launcher only replaced it when
        `--prebuilt-spline-path` was given, so a bunny run kept it: reconstruction was
        skipped and every stage simulated Suzanne's surface against the bunny's ground
        truth, with nothing to show for it in the outputs but plausible numbers. The
        absence of a spline has to reach each stage's own configuration as an explicit
        `None`, since a stage that is not told falls back to the same default.
        """
        commands = self.executed_commands(scene=E2EScene.BUNNY)

        self.assertTrue(any(self.RECONSTRUCT_MODULE in argv for argv in commands))
        configs = self.stage_configs(commands)
        self.assertEqual(len(configs), len(Stage) - 2)
        for config in configs:
            self.assertIsNone(config.prebuilt_spline_path)

    def test_the_suzanne_scene_forwards_its_own_spline_to_every_stage(self) -> None:
        """
        `--scene SUZANNE` skips reconstruction and names Suzanne's spline to every
        stage explicitly, rather than leaving each to agree by default.
        """
        commands = self.executed_commands(scene=E2EScene.SUZANNE)

        self.assertFalse(any(self.RECONSTRUCT_MODULE in argv for argv in commands))
        forwarding = [
            argv
            for argv in commands
            if not argv[1].endswith(self.DEDICATED_DRIVER_SCRIPTS)
        ]
        self.assertEqual(len(forwarding), len(Stage) - 3)
        for argv in forwarding:
            self.assertIn("--prebuilt-spline-path", argv)
            self.assertEqual(
                argv[argv.index("--prebuilt-spline-path") + 1], self.SUZANNE_SPLINE
            )

    def test_an_explicit_spline_outranks_the_scenes_own(self) -> None:
        """
        `--prebuilt-spline-path` wins over what `--scene` registers, in the run's own
        skip decision and in every stage's configuration.
        """
        explicit = Path("data/splines/spline_sphere_16.json")

        commands = self.executed_commands(
            scene=E2EScene.BUNNY, prebuilt_spline_path=explicit
        )

        self.assertFalse(any(self.RECONSTRUCT_MODULE in argv for argv in commands))
        for config in self.stage_configs(commands):
            self.assertEqual(config.prebuilt_spline_path, explicit)


class TestRunPipelineParametersStageOrder(unittest.TestCase):
    """`stages` runs in the canonical `Stage` order regardless of how it was
    passed, so `--stages CMP_HEAT GT` cannot run CMP_HEAT before GT."""

    def test_stages_are_reordered_to_canonical_order(self) -> None:
        parameters = RunPipelineParameters(
            stages=(Stage.CMP_HEAT, Stage.GT, Stage.GT)
        )

        self.assertEqual(parameters.stages, (Stage.GT, Stage.CMP_HEAT))
