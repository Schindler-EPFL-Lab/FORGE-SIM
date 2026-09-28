import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from forge_sim.scripts.configs.thermal_material_config import ThermalMaterialConfig
from main_scripts.run_vggt_pipeline import (
    VGGTPipelineParameters,
    _detect_scenes,
    main,
)
from main_scripts.scenes.vggt_scene import VGGTScene
from main_scripts.scenes.vggt_scene_layout import VGGTSceneLayout
from main_scripts.scenes.vggt_scene_profile import VGGTSceneProfile
from main_scripts.scenes.vggt_scene_thermal_profile import VGGTSceneThermalProfile
from main_scripts.vggt_pipeline.inpaint_pass import InpaintPass
from main_scripts.vggt_pipeline.pipeline_stage import PipelineStage
from main_scripts.vggt_pipeline.pipeline_steps import PipelineSteps
from main_scripts.vggt_pipeline.scene_pipeline_run import ScenePipelineRun


class TestVGGTPipelineRunner(unittest.TestCase):
    """
    Tests for the VGGT main runner: the commands it emits per stage, and its dry
    run's freedom from side effects.

    `main_scripts/run_vggt_pipeline.py` runs every step of the real-dataset pipeline
    in a subprocess, so the set of commands it builds is its entire observable output,
    and the only thing a change to it can move. Those commands are pinned here against
    the published scenes, which makes the material constants and length scales that
    reached the paper's runs part of the suite rather than a property of one script
    nobody exercises. Those values now live in
    `main_scripts.scenes.vggt_scene_thermal_profile`, the registry the FEM
    cross-validation's driver reads a scene's plain values from, so the runner's
    contract with that registry is pinned here too, including the four checks moved
    in from `fem/tests/test_scene_profile.py` when the registry itself moved out of
    `fem/`'s reach.

    `--dry-run` builds the same commands without running them, and is the mode a
    reader uses to inspect a run against a dataset they must not write to. That it
    writes nothing is therefore a contract rather than a convenience, and is pinned
    separately: the modal step used to create its output directory unconditionally.

    What the runner does itself rather than by subprocess is small but not nothing,
    and is pinned outside dry-run with the spawn mocked out: it moves each
    inpainting pass's outputs up under pass-labelled names, clears the folder a
    crashed pass left behind, skips a step whose output is already there, and
    rejects a step whose inputs are not.
    """

    _INTERPRETER = "python"
    """
    Placeholder standing in for `sys.executable` in an expected command.

    The runner spawns the interpreter running it, whose path is a property of the
    environment rather than of the runner, so a comparison substitutes this for the
    command's first token instead of asserting a machine-specific path.
    """

    _TWO_MATERIAL_SCENES = (
        VGGTScene.CAR,
        VGGTScene.BUILDINGA_SPRING,
        VGGTScene.WOODSHED,
        VGGTScene.BUILDNET_1,
        VGGTScene.BUILDNET_2,
        VGGTScene.BUILDNET_3,
        VGGTScene.BUILDNET_4,
    )
    """
    The published scenes whose dataset carries a facade mask, so they are solved
    with a window material distinct from the surface material: seven of the
    eight recorded datasets reference a `masks_facade` in at least one frame and
    carry a `heat_simulation_two_materials.xdmf` beside it; only `lion` carries
    neither. Written out here rather than derived from `window_material`, so a
    profile that drops the field cannot slip past this test.
    """

    _GLASS_FLAGS = (
        "--glass-config.conductivity",
        "--glass-config.density",
        "--glass-config.specific-heat",
        "--glass-config.thickness",
    )
    """The four flags a two-material heat command must carry to pin its window
    material."""

    @staticmethod
    def _build_scene(scene_dir: Path, *, facade_mask: bool) -> Path:
        """
        Writes the least a scene needs for the runner to reach its subprocess calls:
        a reconstruction the modal step resolves, and a `transforms.json` whose one
        frame does or does not reference a facade mask. Nothing else is read in
        dry-run mode, so the files are otherwise empty.

        :returns: Path of the written reconstruction.
        """
        (scene_dir / "dataset").mkdir(parents=True)
        reconstruction = (
            scene_dir / "result" / "shape_20260601-023350_iteration-249.json"
        )
        reconstruction.parent.mkdir(parents=True)
        reconstruction.write_text("{}")

        frame: dict = {"rgb": {"file_path": "images/f0000.jpeg"}}
        if facade_mask:
            frame["rgb"]["windows_mask_file_path"] = "masks_facade/f0000.png"
        (scene_dir / "dataset" / "transforms.json").write_text(
            json.dumps({"frames": [frame]})
        )
        return reconstruction

    @staticmethod
    def _tree(root: Path) -> list[str]:
        """Every path under `root`, relative and sorted, for a before/after
        comparison."""
        return sorted(str(path.relative_to(root)) for path in root.rglob("*"))

    @staticmethod
    def _parse_command(message: str) -> list[str] | None:
        """
        Parses one logged line into a token list with the interpreter path replaced by
        `_INTERPRETER`, or `None` if the line is not a subprocess command.

        `ScenePipelineRun._spawn` logs every command as `[dry-run] <argv joined by
        spaces>`, with the interpreter as the first token.
        `ScenePipelineRun.move_inpaint_outputs` logs its own line under the same
        `[dry-run] ` prefix (`[dry-run] would move ...`) without spawning anything,
        so the prefix alone does not identify a command - the first token after it
        must also be the interpreter that `_spawn` always puts there.
        """
        if not message.startswith("[dry-run] "):
            return None
        tokens = message[len("[dry-run] ") :].split(" ")
        if not tokens or tokens[0] != sys.executable:
            return None
        return [TestVGGTPipelineRunner._INTERPRETER, *tokens[1:]]

    def _emitted_commands(self, params: VGGTPipelineParameters) -> list[list[str]]:
        """
        Runs the pipeline and returns one token list per subprocess command it logged.

        Every other logged line - progress reporting and
        `ScenePipelineRun.move_inpaint_outputs`'s own `[dry-run] would move ...`
        line alike - is dropped here; see `_parse_command`.
        """
        with self.assertLogs("forge_sim", level="INFO") as captured:
            main(params)

        commands = []
        for record in captured.records:
            command = self._parse_command(record.getMessage())
            if command is not None:
                commands.append(command)
        return commands

    def _failed_run(self, params: VGGTPipelineParameters) -> tuple[str, list[str]]:
        """
        Runs a pipeline expected to lose at least one scene, and returns the text of
        the `RuntimeError` it ends with alongside every message it logged.

        The raise is asserted here rather than at each call site because it is easy to
        leave out: `main` collects a scene's failure into its end-of-run summary and
        carries on, so the only thing separating a run that lost two of eight scenes
        from one that succeeded is that final raise.
        """
        with self.assertLogs("forge_sim", level="INFO") as captured:
            with self.assertRaises(RuntimeError) as raised:
                main(params)
        return str(raised.exception), [
            record.getMessage() for record in captured.records
        ]

    def _resumed_run(self, params: VGGTPipelineParameters) -> list[str]:
        """
        Runs a pipeline outside dry-run whose every enabled step should recognise its
        own finished output, and returns the messages it logged.

        A skip is a property of the real path: each guard is written as
        `output.exists() and not dry_run`, so these are the only tests that leave
        dry-run. `subprocess.run` is patched out and asserted unused, which pins the
        skip and keeps a guard that stopped working from launching a solver on a
        directory of empty files.
        """
        with mock.patch.object(subprocess, "run") as spawn:
            with self.assertLogs("forge_sim", level="INFO") as captured:
                main(params)

        spawn.assert_not_called()
        return [record.getMessage() for record in captured.records]

    def test_dry_run_of_the_modal_stage_creates_no_directories(self):
        """
        The modal step created `modal_analysis/` before building its command, with no
        dry-run guard, so inspecting a run against a read-only dataset wrote into it.
        Seven of the eight published scenes already held that directory, which is why
        nobody saw it. A dry run must leave the tree exactly as it found it.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            self._build_scene(dataset_root / "buildnet-1", facade_mask=True)
            before = self._tree(dataset_root)

            main(
                VGGTPipelineParameters(
                    dataset_root=dataset_root,
                    scenes=[VGGTScene.BUILDNET_1],
                    stage=PipelineStage.MODAL_ONLY,
                    dry_run=True,
                )
            )

            self.assertFalse((dataset_root / "buildnet-1" / "modal_analysis").exists())
            self.assertEqual(self._tree(dataset_root), before)

    def test_dry_run_of_every_stage_creates_no_directories(self):
        """
        The same guarantee for the stages that inpaint and simulate as well: each has
        its own output directory to create, and none of them may.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            self._build_scene(dataset_root / "car", facade_mask=True)
            before = self._tree(dataset_root)

            for stage in PipelineStage:
                main(
                    VGGTPipelineParameters(
                        dataset_root=dataset_root,
                        scenes=[VGGTScene.CAR],
                        stage=stage,
                        dry_run=True,
                    )
                )

                self.assertEqual(self._tree(dataset_root), before, stage.value)

    def test_modal_command_carries_the_unrescaled_scene_s_concrete(self):
        """
        buildnet-1's published modal run: concrete at 0.2 m, refined to 128 spans, and
        kept at its natural scale - which reaches the solver as the literal string
        `None`, since tyro renders an optional float as a value rather than as a
        `--no-` flag.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            reconstruction = self._build_scene(
                dataset_root / "buildnet-1", facade_mask=True
            )
            modal_dir = dataset_root / "buildnet-1" / "modal_analysis"

            commands = self._emitted_commands(
                VGGTPipelineParameters(
                    dataset_root=dataset_root,
                    scenes=[VGGTScene.BUILDNET_1],
                    stage=PipelineStage.MODAL_ONLY,
                    dry_run=True,
                )
            )

            self.assertEqual(
                commands,
                [
                    [
                        self._INTERPRETER,
                        "-m",
                        "forge_sim.scripts.reissner_mindlin_modal_analysis"
                        ".reissner_mindlin_modal_analysis",
                        "--spline-path",
                        str(reconstruction.resolve()),
                        "--output-folder",
                        str(modal_dir.resolve()),
                        "--target-spans",
                        "128",
                        "--target-length-scale",
                        "None",
                        "--material-config.youngs-modulus",
                        "40000000000.0",
                        "--material-config.poisson-ratio",
                        "0.2",
                        "--material-config.density",
                        "2400.0",
                        "--material-config.thickness",
                        "0.2",
                        "--material-config.shear-correction-factor",
                        "0.8333333333333334",
                        "--use-gpu",
                    ]
                ],
            )

    def test_heat_command_carries_the_car_s_steel_and_glazing(self):
        """
        The car's published heat run: 2 mm steel over the body and 1 mm glass over the
        windows, rescaled to 4 m, reading the two inpainted fields the earlier stages
        leave in the output directory.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            self._build_scene(dataset_root / "car", facade_mask=True)
            heat_dir = dataset_root / "car" / "heat_simulation"

            commands = self._emitted_commands(
                VGGTPipelineParameters(
                    dataset_root=dataset_root,
                    scenes=[VGGTScene.CAR],
                    stage=PipelineStage.HEAT_SIM_ONLY,
                    dry_run=True,
                )
            )

            self.assertEqual(
                commands,
                [
                    [
                        self._INTERPRETER,
                        "-m",
                        "forge_sim.scripts.heat_simulation"
                        ".heat_simulation_two_materials",
                        "--spline-path",
                        str(heat_dir / "inpainted_thermal_data.json"),
                        "--facade-mask-path",
                        str(heat_dir / "inpainted_facade_data.json"),
                        "--output-folder",
                        str(heat_dir),
                        "--num-time-steps",
                        "500",
                        "--concrete-config.conductivity",
                        "50.0",
                        "--concrete-config.density",
                        "7850.0",
                        "--concrete-config.specific-heat",
                        "500.0",
                        "--concrete-config.thickness",
                        "0.002",
                        "--target-length-scale",
                        "4.0",
                        "--glass-config.conductivity",
                        "0.9",
                        "--glass-config.density",
                        "2500.0",
                        "--glass-config.specific-heat",
                        "840.0",
                        "--glass-config.thickness",
                        "0.001",
                        "--target-delta-temp",
                        "0.5",
                        "--use-gpu",
                    ]
                ],
            )

    def test_the_window_rise_is_read_from_the_profile_not_the_script_default(self):
        """
        The IGA and the FEM side of this comparison must integrate the same window
        source, and the FEM side reads `window_target_delta_temp` off the profile.
        Leaving the IGA side on the heat script's own default gave the two solvers
        two independent sources for one quantity, agreeing only while both literals
        said 0.5, and a drifted shared input here does not produce an obviously
        wrong answer, it produces a plausible one the harness reports as ordinary
        solver disagreement.

        A profile value the script's default does not share is what separates
        forwarding from coincidence, so the assertion uses one.
        """
        car = VGGTSceneProfile.SCENE_PROFILES[VGGTScene.CAR]
        retuned = dataclasses.replace(
            car,
            thermal=dataclasses.replace(car.thermal, window_target_delta_temp=1.25),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            self._build_scene(dataset_root / "car", facade_mask=True)

            with mock.patch.dict(
                VGGTSceneProfile.SCENE_PROFILES, {VGGTScene.CAR: retuned}
            ):
                commands = self._emitted_commands(
                    VGGTPipelineParameters(
                        dataset_root=dataset_root,
                        scenes=[VGGTScene.CAR],
                        stage=PipelineStage.HEAT_SIM_ONLY,
                        dry_run=True,
                    )
                )

        argv = commands[0]
        self.assertIn("--target-delta-temp", argv)
        self.assertEqual(argv[argv.index("--target-delta-temp") + 1], "1.25")

    def test_a_run_level_rise_overrides_the_profile_s(self):
        """
        The profile is the source, not a ceiling: a run that sets the rise explicitly
        still wins, which is what `heat_target_delta_temp` exists for.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            self._build_scene(dataset_root / "car", facade_mask=True)

            commands = self._emitted_commands(
                VGGTPipelineParameters(
                    dataset_root=dataset_root,
                    scenes=[VGGTScene.CAR],
                    stage=PipelineStage.HEAT_SIM_ONLY,
                    dry_run=True,
                    heat_target_delta_temp=2.0,
                )
            )

        argv = commands[0]
        self.assertEqual(argv[argv.index("--target-delta-temp") + 1], "2.0")

    def test_every_two_material_scene_pins_its_own_glass(self):
        """
        Six of the seven passed no glass configuration at all, so their window material
        came from whatever default `heat_simulation_two_materials` happened to carry -
        published runs whose parameters lived in a script that nothing tied them to, and
        that any edit would have moved silently. The property is about the class of
        two-material scenes rather than about one material: a scene solved with two
        materials states both of them.

        The four values are pinned exactly, not merely as positive: the car's 1 mm
        glazing and the other six scenes' 10 mm glazing share every other constant
        (0.9 W/(m*K), 2500 kg/m^3, 840 J/(kg*K)), so a scene silently wired to the wrong
        one of the two glass materials would still clear a bare positivity check on all
        four flags. The expectations below are `(conductivity, density, specific_heat,
        thickness)` in `_GLASS_FLAGS` order.
        """
        expected_glass_by_scene: dict[VGGTScene, tuple[str, str, str, str]] = {
            VGGTScene.CAR: ("0.9", "2500.0", "840.0", "0.001"),
            VGGTScene.BUILDINGA_SPRING: ("0.9", "2500.0", "840.0", "0.01"),
            VGGTScene.WOODSHED: ("0.9", "2500.0", "840.0", "0.01"),
            VGGTScene.BUILDNET_1: ("0.9", "2500.0", "840.0", "0.01"),
            VGGTScene.BUILDNET_2: ("0.9", "2500.0", "840.0", "0.01"),
            VGGTScene.BUILDNET_3: ("0.9", "2500.0", "840.0", "0.01"),
            VGGTScene.BUILDNET_4: ("0.9", "2500.0", "840.0", "0.01"),
        }

        for scene in self._TWO_MATERIAL_SCENES:
            with self.subTest(scene=scene):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    dataset_root = Path(tmp_dir)
                    # `buildnet-4`'s registered directory is nested; every other
                    # scene's is its own identity string.
                    self._build_scene(
                        dataset_root / VGGTSceneProfile.SCENE_PROFILES[scene].directory,
                        facade_mask=True,
                    )

                    commands = self._emitted_commands(
                        VGGTPipelineParameters(
                            dataset_root=dataset_root,
                            scenes=[scene],
                            stage=PipelineStage.HEAT_SIM_ONLY,
                            dry_run=True,
                        )
                    )

                    self.assertEqual(len(commands), 1)
                    command = commands[0]
                    self.assertEqual(
                        command[2],
                        "forge_sim.scripts.heat_simulation"
                        ".heat_simulation_two_materials",
                    )
                    for flag, expected in zip(
                        self._GLASS_FLAGS, expected_glass_by_scene[scene]
                    ):
                        self.assertIn(flag, command)
                        self.assertEqual(
                            command[command.index(flag) + 1],
                            expected,
                            f"{scene} {flag}",
                        )

    def test_a_masked_scene_whose_profile_names_no_glass_is_rejected(self):
        """
        Two-material-ness is decided by the dataset (a facade mask reference) while
        both materials come from the profile, so the two can disagree. They did:
        a masked scene whose profile carried no `window_material` ran anyway, on
        whatever window material `heat_simulation_two_materials` happened to default
        to - the same defect the building profiles were fixed for, one level up.

        Both ways of reaching that state are checked. `mystery_scene` is unregistered
        and falls back to `_DEFAULT_PROFILE`, reached here through auto-detection
        since an unregistered scene is no longer expressible through the typed
        `scenes` field; `lion` is registered and single-material by design, so
        handing it a mask is equally a contradiction, reached through an explicit
        `--scenes`. Neither may emit a command, and the error has to name the scene
        and what to do about it, since a run over eight scenes reports it in a
        summary rather than stopping.
        """
        cases: tuple[tuple[str, list[VGGTScene]], ...] = (
            ("mystery_scene", []),
            ("lion", [VGGTScene.LION]),
        )
        for scene_name, scenes in cases:
            with self.subTest(scene=scene_name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    dataset_root = Path(tmp_dir)
                    self._build_scene(dataset_root / scene_name, facade_mask=True)

                    _, messages = self._failed_run(
                        VGGTPipelineParameters(
                            dataset_root=dataset_root,
                            scenes=scenes,
                            stage=PipelineStage.HEAT_SIM_ONLY,
                            dry_run=True,
                        )
                    )
                    commands = [
                        command
                        for command in (
                            self._parse_command(message) for message in messages
                        )
                        if command is not None
                    ]

                    self.assertEqual(commands, [])
                    self.assertTrue(
                        any(
                            scene_name in message
                            and "window_material" in message
                            and "VGGT_SCENE_THERMAL_PROFILES" in message
                            for message in messages
                        ),
                        msg=f"No actionable rejection logged for {scene_name}: "
                        f"{messages}",
                    )

    def test_heat_command_without_a_facade_mask_is_the_single_material_one(self):
        """
        A scene whose `transforms.json` references no facade mask cannot be solved with
        two materials, so the runner falls back to the single-material script - which
        takes no glass configuration and no facade mask path.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            self._build_scene(dataset_root / "car", facade_mask=False)

            commands = self._emitted_commands(
                VGGTPipelineParameters(
                    dataset_root=dataset_root,
                    scenes=[VGGTScene.CAR],
                    stage=PipelineStage.HEAT_SIM_ONLY,
                    dry_run=True,
                )
            )

            self.assertEqual(len(commands), 1)
            self.assertEqual(
                commands[0][2],
                "forge_sim.scripts.heat_simulation.heat_simulation",
            )
            self.assertNotIn("--facade-mask-path", commands[0])
            self.assertNotIn("--glass-config.conductivity", commands[0])

    def test_inpaint_commands_route_the_enum_name_to_the_cli_and_its_value_to_the_label(
        self,
    ):
        """
        `InpaintPass`'s member NAME (`THERMAL`/`WINDOWS`) is what `--inpainting-images-
        type` must receive, since that is the choice `inpaint_data_spline` parses; its
        VALUE (`thermal`/`facade`) is the label the moved outputs are filed under and
        the tmp directory is named for. Swapping the two would hand the CLI a value it
        rejects, or silently mislabel every inpainted output - either way a bug this
        test would catch, since both crossings are pinned independently below.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            reconstruction = self._build_scene(dataset_root / "car", facade_mask=True)
            heat_dir = dataset_root / "car" / "heat_simulation"
            dataset_dir = dataset_root / "car" / "dataset"

            with self.assertLogs("forge_sim", level="INFO") as captured:
                main(
                    VGGTPipelineParameters(
                        dataset_root=dataset_root,
                        scenes=[VGGTScene.CAR],
                        stage=PipelineStage.INPAINTING_ONLY,
                        dry_run=True,
                        skip_mask_fix=True,
                    )
                )
            messages = [record.getMessage() for record in captured.records]
            commands = [
                command
                for command in (self._parse_command(message) for message in messages)
                if command is not None
            ]

            def _inpaint_command(images_type: str, tmp_name: str) -> list[str]:
                return [
                    self._INTERPRETER,
                    "-m",
                    "forge_sim.scripts.inpaint_data_spline",
                    "--thermoscenes-scene-path",
                    str(dataset_dir.resolve()),
                    "--inpainting-images-type",
                    images_type,
                    "--spline-path",
                    str(reconstruction.resolve()),
                    "--h-refinement-factor",
                    "4",
                    "--initial-color",
                    "0.1",
                    "--optimization-iterations",
                    "20",
                    "--output-folder",
                    str((heat_dir / f"_tmp_inpaint_{tmp_name}").resolve()),
                    "--use-gpu",
                ]

            self.assertEqual(
                commands,
                [
                    _inpaint_command("THERMAL", "thermal"),
                    _inpaint_command("WINDOWS", "facade"),
                ],
            )
            self.assertTrue(any("(rename thermal)" in m for m in messages))
            self.assertTrue(any("(rename facade)" in m for m in messages))

    def test_moving_an_inpaint_pass_s_outputs_labels_them_and_clears_the_tmp(self):
        """
        The move a dry run can only describe. Each pass writes into a tmp folder of its
        own and its outputs are then moved up under pass-labelled names, because the two
        passes write the same file names: `inpainted_data.json` becomes
        `inpainted_facade_data.json`, which is the name the heat step reads back, and
        every `views_*.png` is renamed rather than overwritten, so the windows pass's
        previews do not land on top of the thermal pass's.

        `Initial_state.png` is left out of the tmp folder deliberately: a missing output
        is warned about and the remaining ones are still moved, since a pass that wrote
        no preview must not strand the field the heat step needs. The tmp folder is gone
        afterwards, and only the renamed files are left.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            heat_sim_dir = Path(tmp_dir)
            pass_dir = heat_sim_dir / "_tmp_inpaint_facade"
            pass_dir.mkdir()
            for name in (
                "inpainted_data.json",
                "Final_comparison.png",
                "views_0000.png",
                "views_0007.png",
            ):
                (pass_dir / name).write_text(name)

            with self.assertLogs("forge_sim", level="INFO") as captured:
                ScenePipelineRun.move_inpaint_outputs(
                    pass_dir, heat_sim_dir, InpaintPass.WINDOWS, dry_run=False
                )
            messages = [record.getMessage() for record in captured.records]

            self.assertEqual(
                self._tree(heat_sim_dir),
                [
                    "Final_comparison_facade.png",
                    "inpainted_facade_data.json",
                    "views_inpaint_facade_0000.png",
                    "views_inpaint_facade_0007.png",
                ],
            )
            self.assertEqual(
                (heat_sim_dir / "inpainted_facade_data.json").read_text(),
                "inpainted_data.json",
            )
            self.assertTrue(
                any(
                    "Initial_state.png" in message and "missing" in message
                    for message in messages
                ),
                msg=f"No warning for the absent output: {messages}",
            )

    def test_an_inpaint_pass_whose_output_is_already_there_is_skipped(self):
        """
        Inpainting is the expensive step, so a rerun of a partly finished scene must
        recognise the field it already recovered - by the name the move above files it
        under, which is the same name the heat step reads.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            self._build_scene(dataset_root / "car", facade_mask=True)
            heat_sim_dir = dataset_root / "car" / "heat_simulation"
            heat_sim_dir.mkdir()
            (heat_sim_dir / "inpainted_thermal_data.json").write_text("{}")

            messages = self._resumed_run(
                VGGTPipelineParameters(
                    dataset_root=dataset_root,
                    scenes=[VGGTScene.CAR],
                    stage=PipelineStage.THERMAL_INPAINT_ONLY,
                    skip_mask_fix=True,
                    dry_run=False,
                )
            )

            self.assertTrue(
                any(
                    "Skipping THERMAL inpaint for car" in message
                    for message in messages
                ),
                msg=f"The finished pass was not recognised: {messages}",
            )

    def test_a_two_material_heat_sim_whose_output_is_already_there_is_skipped(self):
        """
        The same resume for the two-material solve. Its guard sits after the two
        checks that its inputs are present, so both inpainted fields are written here
        as well: a scene reaching the skip with a missing input would be reporting the
        wrong thing about a run it is not going to do.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            self._build_scene(dataset_root / "car", facade_mask=True)
            heat_sim_dir = dataset_root / "car" / "heat_simulation"
            heat_sim_dir.mkdir()
            (heat_sim_dir / "inpainted_thermal_data.json").write_text("{}")
            (heat_sim_dir / "inpainted_facade_data.json").write_text("{}")
            (heat_sim_dir / "heat_simulation_two_materials.xdmf").write_text("")

            messages = self._resumed_run(
                VGGTPipelineParameters(
                    dataset_root=dataset_root,
                    scenes=[VGGTScene.CAR],
                    stage=PipelineStage.HEAT_SIM_ONLY,
                    dry_run=False,
                )
            )

            self.assertTrue(
                any("Skipping heat sim for car" in message for message in messages),
                msg=f"The finished solve was not recognised: {messages}",
            )

    def test_a_single_material_heat_sim_whose_output_is_already_there_is_skipped(self):
        """
        The unmasked scene's resume, which turns on a different output name
        (`heat_simulation.xdmf`, not `heat_simulation_two_materials.xdmf`): checking
        for the other script's output here would re-solve a finished scene every run.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            self._build_scene(dataset_root / "car", facade_mask=False)
            heat_sim_dir = dataset_root / "car" / "heat_simulation"
            heat_sim_dir.mkdir()
            (heat_sim_dir / "inpainted_thermal_data.json").write_text("{}")
            (heat_sim_dir / "heat_simulation.xdmf").write_text("")

            messages = self._resumed_run(
                VGGTPipelineParameters(
                    dataset_root=dataset_root,
                    scenes=[VGGTScene.CAR],
                    stage=PipelineStage.HEAT_SIM_ONLY,
                    dry_run=False,
                )
            )

            self.assertTrue(
                any(
                    "Skipping single-material heat sim for car" in message
                    for message in messages
                ),
                msg=f"The finished solve was not recognised: {messages}",
            )

    def test_a_masked_scene_with_no_glass_is_rejected_even_when_its_output_exists(self):
        """
        The window-material check runs before the already-finished skip, and this is
        what pins that order. A masked scene whose profile names no window material,
        carrying a `heat_simulation_two_materials.xdmf` from an earlier and differently
        configured run, must still be reported: an output nothing can tie to a window
        material is exactly the state the check exists to surface, and the other order
        lets a resumed run inherit it as finished work. Both inpainted inputs are
        written, so the missing-input checks between the two cannot be what fires.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            self._build_scene(dataset_root / "lion", facade_mask=True)
            heat_sim_dir = dataset_root / "lion" / "heat_simulation"
            heat_sim_dir.mkdir()
            (heat_sim_dir / "inpainted_thermal_data.json").write_text("{}")
            (heat_sim_dir / "inpainted_facade_data.json").write_text("{}")
            (heat_sim_dir / "heat_simulation_two_materials.xdmf").write_text("")

            with mock.patch.object(subprocess, "run") as spawn:
                _, messages = self._failed_run(
                    VGGTPipelineParameters(
                        dataset_root=dataset_root,
                        scenes=[VGGTScene.LION],
                        stage=PipelineStage.HEAT_SIM_ONLY,
                        dry_run=False,
                    )
                )

            spawn.assert_not_called()
            self.assertTrue(
                any(
                    "lion" in message and "window_material" in message
                    for message in messages
                ),
                msg=f"The finished output was taken for a done scene: {messages}",
            )

    def test_a_heat_sim_whose_inpainted_field_is_missing_is_rejected(self):
        """
        Each heat step checks the fields it reads before spawning anything, so a heat
        sim run before its inpainting finished is reported by the runner rather than
        from inside a solver several layers down. Both entry points carry the check and
        both are pinned: the unmasked scene misses the thermal field, and the masked one
        - given its thermal field - misses the facade field, which is the second of the
        two-material step's checks and the one a single check would swallow.
        """
        for facade_mask, missing in (
            (False, "inpainted_thermal_data.json"),
            (True, "inpainted_facade_data.json"),
        ):
            with self.subTest(facade_mask=facade_mask):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    dataset_root = Path(tmp_dir)
                    self._build_scene(dataset_root / "car", facade_mask=facade_mask)
                    heat_sim_dir = dataset_root / "car" / "heat_simulation"
                    heat_sim_dir.mkdir()
                    if facade_mask:
                        (heat_sim_dir / "inpainted_thermal_data.json").write_text("{}")

                    with mock.patch.object(subprocess, "run") as spawn:
                        error, messages = self._failed_run(
                            VGGTPipelineParameters(
                                dataset_root=dataset_root,
                                scenes=[VGGTScene.CAR],
                                stage=PipelineStage.HEAT_SIM_ONLY,
                                dry_run=False,
                            )
                        )

                    spawn.assert_not_called()
                    self.assertIn("car", error)
                    self.assertTrue(
                        any(
                            missing in message and "inpainting stage first" in message
                            for message in messages
                        ),
                        msg=f"The missing field was not named: {messages}",
                    )

    def test_a_tmp_folder_left_by_a_crashed_pass_is_cleared_before_the_next(self):
        """
        The pass's tmp folder is emptied before the pass starts, so a crashed earlier
        attempt cannot contribute to the move that follows: that move takes every
        `views_*.png` it finds, and would file the dead run's preview under this run's
        label. Keeping the folder around is the tempting shortcut, since the pass would
        overwrite its own outputs anyway - the previews are what it would not.

        The cleanup is inside the `not dry_run` branch, so the folder is inspected at
        the moment the pass would have been spawned, with the spawn mocked out.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            self._build_scene(dataset_root / "car", facade_mask=True)
            heat_sim_dir = dataset_root / "car" / "heat_simulation"
            stale = heat_sim_dir / "_tmp_inpaint_thermal" / "views_0000.png"
            stale.parent.mkdir(parents=True)
            stale.write_text("left by a crashed pass")

            trees_at_spawn: list[list[str]] = []
            with mock.patch.object(subprocess, "run") as spawn:
                spawn.side_effect = lambda *_, **__: trees_at_spawn.append(
                    self._tree(heat_sim_dir)
                )
                main(
                    VGGTPipelineParameters(
                        dataset_root=dataset_root,
                        scenes=[VGGTScene.CAR],
                        stage=PipelineStage.THERMAL_INPAINT_ONLY,
                        skip_mask_fix=True,
                        dry_run=False,
                    )
                )

            self.assertEqual(trees_at_spawn, [["_tmp_inpaint_thermal"]])

    def test_use_gpu_false_emits_the_no_use_gpu_flag(self):
        """
        `--use-gpu` and `--no-use-gpu` are appended as plain strings, not by tyro, so
        the branch is this file's to pin: `use_gpu=False` must reach the child command
        as `--no-use-gpu`, never as `--use-gpu` with a value tyro would have to parse.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            self._build_scene(dataset_root / "buildnet-1", facade_mask=True)

            commands = self._emitted_commands(
                VGGTPipelineParameters(
                    dataset_root=dataset_root,
                    scenes=[VGGTScene.BUILDNET_1],
                    stage=PipelineStage.MODAL_ONLY,
                    dry_run=True,
                    use_gpu=False,
                )
            )

            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0][-1], "--no-use-gpu")
            self.assertNotIn("--use-gpu", commands[0])

    def test_heat_target_delta_temp_set_inserts_the_target_delta_temp_flag(self):
        """
        `heat_target_delta_temp` defaults to `None`, in which case no flag is emitted
        (the baseline already pinned by the two heat-command tests above); setting it
        must thread the value through as `--target-delta-temp <value>`, immediately
        before the GPU flag.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            self._build_scene(dataset_root / "car", facade_mask=True)

            commands = self._emitted_commands(
                VGGTPipelineParameters(
                    dataset_root=dataset_root,
                    scenes=[VGGTScene.CAR],
                    stage=PipelineStage.HEAT_SIM_ONLY,
                    dry_run=True,
                    heat_target_delta_temp=12.5,
                )
            )

            self.assertEqual(len(commands), 1)
            self.assertEqual(
                commands[0][-3:], ["--target-delta-temp", "12.5", "--use-gpu"]
            )

    def test_heat_target_delta_temp_reaches_the_single_material_command_too(self):
        """
        The flag is threaded through both heat entry points and only the two-material
        one was pinned above, so an unmasked scene could have lost its source scaling
        with nothing to notice - and silently, since the heat script simply falls back
        to its absolute source magnitude when the flag is absent.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            self._build_scene(dataset_root / "car", facade_mask=False)

            commands = self._emitted_commands(
                VGGTPipelineParameters(
                    dataset_root=dataset_root,
                    scenes=[VGGTScene.CAR],
                    stage=PipelineStage.HEAT_SIM_ONLY,
                    dry_run=True,
                    heat_target_delta_temp=12.5,
                )
            )

            self.assertEqual(len(commands), 1)
            self.assertEqual(
                commands[0][2],
                "forge_sim.scripts.heat_simulation.heat_simulation",
            )
            self.assertEqual(
                commands[0][-3:], ["--target-delta-temp", "12.5", "--use-gpu"]
            )

    def test_unregistered_scene_falls_back_to_the_default_profile(self):
        """
        A scene name absent from `SCENE_PROFILES` gets `_DEFAULT_PROFILE`: concrete
        thermal properties and a 4 m rescale, which is the whole of what its unmasked
        heat command carries here, plus a warning naming the scene. That profile names
        neither a window nor a shell material on purpose; the glass test above and the
        shell test below pin those two runs as rejected rather than defaulted.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            self._build_scene(dataset_root / "mystery_scene", facade_mask=False)
            heat_dir = dataset_root / "mystery_scene" / "heat_simulation"

            # Empty `scenes` auto-detects: an unregistered scene is no longer
            # expressible through the typed field.
            with self.assertLogs("forge_sim", level="INFO") as captured:
                main(
                    VGGTPipelineParameters(
                        dataset_root=dataset_root,
                        scenes=[],
                        stage=PipelineStage.HEAT_SIM_ONLY,
                        dry_run=True,
                    )
                )
            messages = [record.getMessage() for record in captured.records]
            commands = [
                command
                for command in (self._parse_command(message) for message in messages)
                if command is not None
            ]

            self.assertEqual(
                commands,
                [
                    [
                        self._INTERPRETER,
                        "-m",
                        "forge_sim.scripts.heat_simulation.heat_simulation",
                        "--spline-path",
                        str(heat_dir / "inpainted_thermal_data.json"),
                        "--output-folder",
                        str(heat_dir),
                        "--num-time-steps",
                        "500",
                        "--concrete-config.conductivity",
                        "2.0",
                        "--concrete-config.density",
                        "2250.0",
                        "--concrete-config.specific-heat",
                        "900.0",
                        "--concrete-config.thickness",
                        "0.2",
                        "--target-length-scale",
                        "4.0",
                        "--use-gpu",
                    ]
                ],
            )
            self.assertTrue(
                any(
                    "No VGGTSceneProfile registered for scene 'mystery_scene'" in m
                    for m in messages
                )
            )

    def test_every_scene_draws_its_thermal_profile_from_the_shared_registry(self):
        """
        The materials and length scale are the problem statement both solvers are
        handed, so the runner holds the registry's own object rather than an equal
        copy: a copy would pass a value comparison here and then drift the moment the
        registry was edited for the FEM side alone.
        """
        for scene_name, profile in VGGTSceneProfile.SCENE_PROFILES.items():
            with self.subTest(scene=scene_name):
                self.assertIs(
                    profile.thermal,
                    VGGTSceneThermalProfile.VGGT_SCENE_THERMAL_PROFILES[scene_name],
                )

    def test_the_two_scene_key_sets_cover_each_other(self):
        """
        A scene carrying a shell material but no thermal profile - or the reverse -
        fails only when its stage is reached, hours into a sweep over eight scenes.
        """
        self.assertEqual(
            set(VGGTSceneProfile.SCENE_PROFILES),
            set(VGGTSceneThermalProfile.VGGT_SCENE_THERMAL_PROFILES),
        )

    def test_each_recorded_stem_agrees_with_the_materials_beside_it(self):
        """
        The eight recorded scenes do all follow the derivation, so a stem transposed
        between two rows shows up as a disagreement with the profile beside it. This
        recomputes the expected value, so it says nothing about whether the field is
        held or derived; that is the next test's job, and neither stands alone.
        """
        for scene_name, profile in VGGTSceneProfile.SCENE_PROFILES.items():
            with self.subTest(scene=scene_name):
                expected = (
                    "heat_simulation_two_materials"
                    if profile.thermal.is_two_material
                    else "heat_simulation"
                )
                self.assertEqual(profile.iga_base_name, expected)

    def test_a_window_material_must_state_the_rise_it_is_scaled_to(self):
        """
        A profile with a window material and no target rise is rejected.

        None means "leave it alone" for `target_length_scale` and the opposite here:
        the source keeps its raw unit magnitude rather than the scaling being
        skipped, so the two solvers would integrate different sources and the
        harness would report the gap as solver error.
        """
        two_material = VGGTSceneThermalProfile.for_scene(
            next(
                scene_name
                for scene_name, profile in (
                    VGGTSceneThermalProfile.VGGT_SCENE_THERMAL_PROFILES.items()
                )
                if profile.is_two_material
            )
        )

        with self.assertRaises(ValueError):
            dataclasses.replace(two_material, window_target_delta_temp=None)

    def test_every_registered_two_material_scene_states_its_rise(self):
        """
        No registered profile relies on the validation above to catch it, which is
        what makes the recorded runs comparable to the IGA reference in the first
        place.
        """
        for (
            scene_name,
            profile,
        ) in VGGTSceneThermalProfile.VGGT_SCENE_THERMAL_PROFILES.items():
            with self.subTest(scene=scene_name):
                if profile.is_two_material:
                    self.assertIsNotNone(profile.window_target_delta_temp)

    def test_the_reference_stem_is_held_rather_than_derived(self):
        """
        The stem must survive contradicting the materials. The producer routes on
        whether the dataset itself carries a facade mask, so a scene with a
        window material and no mask is written under the single-material stem,
        which a derived `iga_base_name` would silently overrule, sending the
        comparison to a file that does not exist.
        """
        contradictory = VGGTSceneProfile(
            directory=Path("buildinga_spring"),
            iga_base_name="heat_simulation",
            thermal=VGGTSceneThermalProfile.VGGT_SCENE_THERMAL_PROFILES[
                VGGTScene.BUILDINGA_SPRING
            ],
        )

        self.assertTrue(contradictory.thermal.is_two_material)
        self.assertEqual(contradictory.iga_base_name, "heat_simulation")

    def test_a_thermal_profile_rejects_a_non_positive_length_scale(self):
        """
        One of the shared record's validated fields. No bounding box has a
        non-positive longest extent, and a zero would divide the rescaling by zero
        several stages downstream of where it was written.
        """
        for length_scale in (0.0, -4.0):
            with self.subTest(target_length_scale=length_scale):
                with self.assertRaises(ValueError) as raised:
                    VGGTSceneThermalProfile(
                        base_material=ThermalMaterialConfig(),
                        target_length_scale=length_scale,
                    )
                self.assertIn("target_length_scale", str(raised.exception))
                self.assertIn(str(length_scale), str(raised.exception))

    def test_declining_to_rescale_reaches_the_cli_as_the_literal_none(self):
        """
        tyro renders a `float | None` parameter as `{None}|FLOAT`, so the downstream
        CLIs read the word rather than a `--no-...` flag. A profile that kept its
        natural scale must therefore spell it out, not omit the argument.
        """
        natural_scale = VGGTSceneThermalProfile(
            base_material=ThermalMaterialConfig(), target_length_scale=None
        )
        rescaled = VGGTSceneThermalProfile(
            base_material=ThermalMaterialConfig(), target_length_scale=2.5
        )

        self.assertEqual(natural_scale.target_length_scale_argument, "None")
        self.assertEqual(rescaled.target_length_scale_argument, "2.5")
        self.assertFalse(natural_scale.is_two_material)

    def test_a_scene_whose_profile_names_no_shell_material_is_rejected(self):
        """
        The modal step's half of the contradiction the glass test covers above, and the
        half that was silent: `_DEFAULT_PROFILE` names no `structural_material`, the
        `--material-config.*` flags were then simply omitted, and the modal script's own
        default - a 1.5 mm steel car body - answered for what the same profile treats as
        a concrete building. Nothing in the command or the log said so, since the
        fallback warning speaks of the thermal defaults.

        No command may be emitted, and the message has to name the scene and the field
        to fill in, since a run over eight scenes reports it in a summary rather than
        stopping.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            self._build_scene(dataset_root / "mystery_scene", facade_mask=False)

            # Empty `scenes` auto-detects: an unregistered scene is no longer
            # expressible through the typed field.
            error, messages = self._failed_run(
                VGGTPipelineParameters(
                    dataset_root=dataset_root,
                    scenes=[],
                    stage=PipelineStage.MODAL_ONLY,
                    dry_run=True,
                )
            )
            commands = [
                command
                for command in (self._parse_command(message) for message in messages)
                if command is not None
            ]

            self.assertEqual(commands, [])
            self.assertIn("mystery_scene", error)
            self.assertTrue(
                any(
                    "mystery_scene" in message
                    and "structural_material" in message
                    and "SCENE_PROFILES" in message
                    for message in messages
                ),
                msg=f"No actionable rejection logged: {messages}",
            )

    def test_buildnet_4_resolves_to_its_nested_directory(self):
        """
        `buildnet-4` is one of the eight published scenes, and it genuinely lives one
        directory deeper on disk than the rest (`buildnet-4/buildnet-4`). Before
        `VGGTScene` existed, a bare `--scenes buildnet-4/buildnet-4` spelled that
        nested path directly and `_profile_for`'s leading-component fallback matched
        it to the `buildnet-4` profile. `scenes: list[VGGTScene]` closes `--scenes`
        to registered identities, none of which can spell a path, so the nested
        directory is now resolved through `VGGTSceneProfile.directory` instead:
        selecting `VGGTScene.BUILDNET_4` must still reach the same on-disk location
        and emit the same command, just through the identity rather than a
        hand-typed path.

        This is not a weaker replacement for the old assertion: it pins the more
        realistic entry point (a user naming the scene, not its directory layout) to
        the same outcome. What it no longer exercises is `_profile_for`'s
        leading-component split itself, which no longer fires on this path now that
        `main` resolves the nested directory before `_profile_for` ever sees it (an
        exact match on the full name suffices); that function is otherwise untested
        directly.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            reconstruction = self._build_scene(
                dataset_root / "buildnet-4" / "buildnet-4", facade_mask=False
            )
            modal_dir = dataset_root / "buildnet-4" / "buildnet-4" / "modal_analysis"

            with self.assertLogs("forge_sim", level="INFO") as captured:
                main(
                    VGGTPipelineParameters(
                        dataset_root=dataset_root,
                        scenes=[VGGTScene.BUILDNET_4],
                        stage=PipelineStage.MODAL_ONLY,
                        dry_run=True,
                    )
                )
            messages = [record.getMessage() for record in captured.records]
            commands = [
                command
                for command in (self._parse_command(message) for message in messages)
                if command is not None
            ]

            self.assertEqual(
                commands,
                [
                    [
                        self._INTERPRETER,
                        "-m",
                        "forge_sim.scripts.reissner_mindlin_modal_analysis"
                        ".reissner_mindlin_modal_analysis",
                        "--spline-path",
                        str(reconstruction.resolve()),
                        "--output-folder",
                        str(modal_dir.resolve()),
                        "--target-spans",
                        "128",
                        "--target-length-scale",
                        "None",
                        "--material-config.youngs-modulus",
                        "40000000000.0",
                        "--material-config.poisson-ratio",
                        "0.2",
                        "--material-config.density",
                        "2400.0",
                        "--material-config.thickness",
                        "0.2",
                        "--material-config.shear-correction-factor",
                        "0.8333333333333334",
                        "--use-gpu",
                    ]
                ],
            )
            self.assertFalse(
                any("No VGGTSceneProfile registered" in m for m in messages)
            )

    def test_auto_detection_cannot_reach_the_nested_buildnet_4_directory(self):
        """
        A gap distinct from the one the test above closes: `_detect_scenes` lists
        only `dataset_root`'s own top-level directory names and asks
        `find_reconstruction` to find a reconstruction directly inside each, one
        level short of where `buildnet-4`'s reconstruction actually sits. Naming
        the scene explicitly (`--scenes BUILDNET_4`) is therefore the only way to
        reach it; an unqualified run silently skips it rather than failing loudly,
        which is worth pinning so a future change to `_detect_scenes` is a
        deliberate fix rather than an accidental one.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            self._build_scene(
                dataset_root / "buildnet-4" / "buildnet-4", facade_mask=False
            )

            self.assertEqual(_detect_scenes(dataset_root), [])

    def test_find_reconstruction_searches_every_result_layout(self):
        """
        The four layouts the v2/v3 exports use - `result/`, `results/`,
        `result/ours/`, `results/ours/` - must each be found on its own; the case
        under test is named by `subTest` so a failure names which layout broke.
        """
        layouts = ("result", "results", "result/ours", "results/ours")
        for layout in layouts:
            with self.subTest(layout=layout):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    scene_dir = Path(tmp_dir)
                    target = scene_dir / layout / "shape_only.json"
                    target.parent.mkdir(parents=True)
                    target.write_text("{}")

                    self.assertEqual(VGGTSceneLayout(scene_dir).reconstruction, target)

    def test_find_reconstruction_picks_the_most_recently_modified_candidate(self):
        """
        With candidates in two different layouts, `find_reconstruction` must return
        the one with the latest mtime, not the first one `glob` happens to yield -
        which is why the older candidate is placed in the layout searched first
        (`results/`) and the newer one in the layout searched second (`result/ours/`):
        a search that just returned its first match would return the wrong one here.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_dir = Path(tmp_dir)
            older = scene_dir / "results" / "shape_older.json"
            newer = scene_dir / "result" / "ours" / "shape_newer.json"
            older.parent.mkdir(parents=True)
            newer.parent.mkdir(parents=True)
            older.write_text("{}")
            newer.write_text("{}")
            os.utime(older, (1_000_000_000, 1_000_000_000))
            os.utime(newer, (1_000_000_100, 1_000_000_100))

            self.assertEqual(VGGTSceneLayout(scene_dir).reconstruction, newer)

    def test_find_reconstruction_with_no_shape_json_raises_file_not_found(self):
        """
        A scene with no `shape_*.json` under any of the four layouts - even one that
        holds an unrelated `dataset/` directory - must raise, not return a stale or
        empty result.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_dir = Path(tmp_dir)
            (scene_dir / "dataset").mkdir()

            with self.assertRaises(FileNotFoundError):
                VGGTSceneLayout(scene_dir).reconstruction

    def test_pipeline_stage_steps_match_the_documented_table(self):
        """
        Each `PipelineStage` member's five-step selection, hardcoded here from what
        the member's name promises rather than read off `PipelineStage.steps` itself -
        a stage silently turning on (or forgetting to turn on) a step it should not
        would otherwise ship as a set of commands nobody asked for.
        """
        expected_by_stage = {
            PipelineStage.MASK_FIX_ONLY: PipelineSteps(
                mask_fix=True,
                thermal_inpaint=False,
                windows_inpaint=False,
                heat_sim=False,
                modal=False,
            ),
            PipelineStage.THERMAL_INPAINT_ONLY: PipelineSteps(
                mask_fix=True,
                thermal_inpaint=True,
                windows_inpaint=False,
                heat_sim=False,
                modal=False,
            ),
            PipelineStage.WINDOWS_INPAINT_ONLY: PipelineSteps(
                mask_fix=True,
                thermal_inpaint=False,
                windows_inpaint=True,
                heat_sim=False,
                modal=False,
            ),
            PipelineStage.INPAINTING_ONLY: PipelineSteps(
                mask_fix=True,
                thermal_inpaint=True,
                windows_inpaint=True,
                heat_sim=False,
                modal=False,
            ),
            PipelineStage.HEAT_SIM_ONLY: PipelineSteps(
                mask_fix=False,
                thermal_inpaint=False,
                windows_inpaint=False,
                heat_sim=True,
                modal=False,
            ),
            PipelineStage.INPAINTING_AND_HEAT: PipelineSteps(
                mask_fix=True,
                thermal_inpaint=True,
                windows_inpaint=True,
                heat_sim=True,
                modal=False,
            ),
            PipelineStage.MODAL_ONLY: PipelineSteps(
                mask_fix=False,
                thermal_inpaint=False,
                windows_inpaint=False,
                heat_sim=False,
                modal=True,
            ),
            PipelineStage.ALL: PipelineSteps(
                mask_fix=True,
                thermal_inpaint=True,
                windows_inpaint=True,
                heat_sim=True,
                modal=True,
            ),
        }

        self.assertEqual(set(expected_by_stage), set(PipelineStage))
        for stage, expected in expected_by_stage.items():
            with self.subTest(stage=stage.value):
                self.assertEqual(stage.steps, expected)

    def test_a_scene_without_a_reconstruction_runs_nothing_at_all(self):
        """
        The reconstruction is resolved before any step is spawned, so a scene that has
        not been reconstructed yet fails before the mask fix rewrites its
        `transforms.json`. Resolving it lazily instead - inside the block that inpaints
        - would let that one real subprocess run first, which a dry run cannot show and
        the published scenes cannot catch, since all eight are reconstructed.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            scene_dir = dataset_root / "car"
            (scene_dir / "dataset").mkdir(parents=True)
            (scene_dir / "dataset" / "transforms.json").write_text(
                json.dumps({"frames": []})
            )

            _, messages = self._failed_run(
                VGGTPipelineParameters(
                    dataset_root=dataset_root,
                    scenes=[VGGTScene.CAR],
                    stage=PipelineStage.INPAINTING_ONLY,
                    dry_run=True,
                )
            )
            commands = [
                command
                for command in (self._parse_command(message) for message in messages)
                if command is not None
            ]

            self.assertEqual(commands, [])

    def test_a_failed_scene_spares_the_others_but_still_fails_the_run(self):
        """
        The two halves of the failure contract, which pull against each other: one
        scene's failure must not cost the remaining scenes their run, and the run must
        nonetheless fail. `main` used to satisfy only the first - it tallied the
        failures, logged them, and returned `None` regardless - so an automated caller
        reading the exit status of an eight-scene run was told it succeeded even when
        every scene had failed.

        `mystery_scene` carries a facade mask that its fallback profile has no window
        material for; `car` is registered and emits its two-material command all the
        same. Both are auto-detected (`scenes=[]`) rather than named explicitly, since
        an unregistered scene is no longer expressible through the typed field; which
        of the two runs first is then alphabetical rather than chosen, so the
        assertions below do not depend on order. The error names the scene that
        failed and how many of how many.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            self._build_scene(dataset_root / "mystery_scene", facade_mask=True)
            self._build_scene(dataset_root / "car", facade_mask=True)

            error, messages = self._failed_run(
                VGGTPipelineParameters(
                    dataset_root=dataset_root,
                    scenes=[],
                    stage=PipelineStage.HEAT_SIM_ONLY,
                    dry_run=True,
                )
            )
            commands = [
                command
                for command in (self._parse_command(message) for message in messages)
                if command is not None
            ]

            self.assertIn("mystery_scene", error)
            self.assertIn("1/2", error)
            self.assertEqual(len(commands), 1)
            self.assertEqual(
                commands[0][2],
                "forge_sim.scripts.heat_simulation.heat_simulation_two_materials",
            )

    def test_mask_fix_command_targets_the_scene_s_dataset_directory(self):
        """
        The one step with no other test pinning it. It is handed the scene's `dataset/`
        directory, resolved, and the fixed `thermal` folder name - never the scene
        directory itself, which is what the underlying script would then scan for
        images that are not there.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            self._build_scene(dataset_root / "car", facade_mask=True)

            commands = self._emitted_commands(
                VGGTPipelineParameters(
                    dataset_root=dataset_root,
                    scenes=[VGGTScene.CAR],
                    stage=PipelineStage.MASK_FIX_ONLY,
                    dry_run=True,
                )
            )

            self.assertEqual(
                commands,
                [
                    [
                        self._INTERPRETER,
                        "-m",
                        "forge_sim.scripts.fix_vggt_thermal_paths",
                        "--scene-path",
                        str((dataset_root / "car" / "dataset").resolve()),
                        "--thermal-folder",
                        "thermal",
                    ]
                ],
            )

    def test_mask_fix_on_a_scene_with_no_dataset_directory_is_rejected(self):
        """
        The step rewrites paths inside `dataset/transforms.json`, so a scene that has
        no `dataset/` at all is reported rather than handed to the underlying script,
        which would be pointed at a directory that is not there. The scene itself
        exists and holds a reconstruction, so nothing earlier stops the run: without
        this check the failure would surface from a subprocess, one level further away
        from what is actually wrong.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            scene_dir = dataset_root / "car"
            reconstruction = scene_dir / "result" / "shape_only.json"
            reconstruction.parent.mkdir(parents=True)
            reconstruction.write_text("{}")

            error, messages = self._failed_run(
                VGGTPipelineParameters(
                    dataset_root=dataset_root,
                    scenes=[VGGTScene.CAR],
                    stage=PipelineStage.MASK_FIX_ONLY,
                    dry_run=True,
                )
            )
            commands = [
                command
                for command in (self._parse_command(message) for message in messages)
                if command is not None
            ]

            self.assertEqual(commands, [])
            self.assertIn("car", error)
            self.assertTrue(
                any(
                    "No dataset/ directory" in message and str(scene_dir) in message
                    for message in messages
                ),
                msg=f"No actionable rejection logged: {messages}",
            )

    def test_every_facade_mask_spelling_selects_the_two_material_heat_command(self):
        """
        Three dataset exports spell the facade mask reference differently, and the key
        decides which heat script runs: miss the spelling and a masked scene silently
        gets the single-material command instead, with no window source at all. The
        VGGT layout nests the reference under `frame["rgb"]`; the ThermoScenes layout
        puts it directly on the frame, so both shapes are checked for each spelling.
        """
        two_material = "forge_sim.scripts.heat_simulation.heat_simulation_two_materials"
        spellings = ("windows_mask_file_path", "masks_facade", "masks_facades")

        for key in spellings:
            for nested in (True, False):
                with self.subTest(key=key, nested=nested):
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        dataset_root = Path(tmp_dir)
                        scene_dir = dataset_root / "car"
                        (scene_dir / "dataset").mkdir(parents=True)
                        frame = {key: "m/f0000.png"}
                        (scene_dir / "dataset" / "transforms.json").write_text(
                            json.dumps(
                                {"frames": [{"rgb": frame} if nested else frame]}
                            )
                        )

                        commands = self._emitted_commands(
                            VGGTPipelineParameters(
                                dataset_root=dataset_root,
                                scenes=[VGGTScene.CAR],
                                stage=PipelineStage.HEAT_SIM_ONLY,
                                dry_run=True,
                            )
                        )

                        self.assertEqual(commands[0][2], two_material)

    def test_a_mask_reference_on_any_frame_is_enough(self):
        """
        Mask coverage is often partial, so the scene counts as masked when any frame
        carries a reference - not only the first, which is the frame a short-circuiting
        check would look at.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            scene_dir = dataset_root / "car"
            (scene_dir / "dataset").mkdir(parents=True)
            (scene_dir / "dataset" / "transforms.json").write_text(
                json.dumps(
                    {
                        "frames": [
                            {"rgb": {"file_path": "images/f0000.jpeg"}},
                            {"rgb": {"masks_facade": "m/f0001.png"}},
                        ]
                    }
                )
            )

            commands = self._emitted_commands(
                VGGTPipelineParameters(
                    dataset_root=dataset_root,
                    scenes=[VGGTScene.CAR],
                    stage=PipelineStage.HEAT_SIM_ONLY,
                    dry_run=True,
                )
            )

            self.assertEqual(
                commands[0][2],
                "forge_sim.scripts.heat_simulation.heat_simulation_two_materials",
            )

    def test_an_unreadable_transforms_json_reads_as_no_facade_mask(self):
        """
        A `transforms.json` that is missing or is not JSON must fall back to the
        single-material heat command rather than raising, so a scene the reconstruction
        stage has not reached yet is reported as a skip and does not stop the run.
        """
        for content in (None, "{not json"):
            with self.subTest(content=content):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    dataset_root = Path(tmp_dir)
                    scene_dir = dataset_root / "car"
                    (scene_dir / "dataset").mkdir(parents=True)
                    if content is not None:
                        (scene_dir / "dataset" / "transforms.json").write_text(content)

                    commands = self._emitted_commands(
                        VGGTPipelineParameters(
                            dataset_root=dataset_root,
                            scenes=[VGGTScene.CAR],
                            stage=PipelineStage.HEAT_SIM_ONLY,
                            dry_run=True,
                        )
                    )

                    self.assertEqual(
                        commands[0][2],
                        "forge_sim.scripts.heat_simulation.heat_simulation",
                    )

    def test_scenes_are_auto_detected_by_their_reconstruction(self):
        """
        Naming no scene processes every subdirectory that holds a reconstruction, in
        the sorted order `_detect_scenes` walks the root in. What must not be picked up
        is anything else the root carries: a directory with no `shape_*.json` (a scene
        the reconstruction stage has not reached, whose steps would all fail) and a
        plain file (the summary tables the comparison drivers write beside the scenes).
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            self._build_scene(dataset_root / "woodshed", facade_mask=True)
            self._build_scene(dataset_root / "car", facade_mask=True)
            (dataset_root / "not_reconstructed" / "dataset").mkdir(parents=True)
            (dataset_root / "modal_fem_iga_summary.md").write_text("")

            commands = self._emitted_commands(
                VGGTPipelineParameters(
                    dataset_root=dataset_root,
                    stage=PipelineStage.MASK_FIX_ONLY,
                    dry_run=True,
                )
            )

            self.assertEqual(
                [command[command.index("--scene-path") + 1] for command in commands],
                [
                    str((dataset_root / "car" / "dataset").resolve()),
                    str((dataset_root / "woodshed" / "dataset").resolve()),
                ],
            )

    def test_a_root_with_no_reconstructed_scene_is_rejected(self):
        """
        Auto-detection finding nothing must be reported, not read as an empty list of
        work: a run pointed at the wrong root would otherwise log that it had processed
        zero scenes and exit as a success.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            (dataset_root / "not_reconstructed").mkdir()

            with self.assertRaises(ValueError) as raised:
                main(
                    VGGTPipelineParameters(
                        dataset_root=dataset_root,
                        stage=PipelineStage.MASK_FIX_ONLY,
                        dry_run=True,
                    )
                )

            self.assertIn(str(dataset_root), str(raised.exception))
