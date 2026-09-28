import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from forge_sim import logger
from main_scripts.run_vggt_pipeline import VGGTPipelineParameters
from main_scripts.scenes.vggt_scene import VGGTScene
from main_scripts.scenes.vggt_scene_layout import VGGTSceneLayout
from main_scripts.scenes.vggt_scene_profile import VGGTSceneProfile
from main_scripts.vggt_pipeline.inpaint_pass import InpaintPass
from main_scripts.vggt_pipeline.pipeline_steps import PipelineSteps


class ScenePipelineRun:
    """
    One scene's pass through the post-processing pipeline.

    Construction resolves what every step needs and would otherwise be threaded
    through each of them (the scene directory, its `VGGTSceneProfile`, the output
    directory, and whether its dataset carries a facade mask) and validates that
    the scene directory exists. It reads but never writes: `process` creates the
    output directory, and only outside `--dry-run`, which is what makes a dry run
    against a read-only dataset genuinely read-only.

    `params` carries `dry_run` and `use_gpu`, so no step method takes either.
    """

    def __init__(self, scene_name: str, params: VGGTPipelineParameters) -> None:
        """
        Resolves `scene_name`'s profile, directory, output directory and
        facade-mask state, reading the dataset but writing nothing.

        :raises FileNotFoundError: If the resolved scene directory does not exist
            under `params.dataset_root`.
        """
        self._scene_name = scene_name
        self._params = params
        self._profile = self._resolve_profile(scene_name)
        # `profile.directory` is the resolved on-disk directory for a registered
        # scene (nested for `buildnet-4`); `VGGTSceneProfile.DEFAULT_PROFILE` names
        # no directory of its own, since it is shared by every unregistered scene,
        # so `scene_name` (the literal name this scene was selected or
        # auto-detected under) is used instead.
        self._scene_dir = params.dataset_root / (
            scene_name if self._profile.directory is None else self._profile.directory
        )
        if not self._scene_dir.is_dir():
            raise FileNotFoundError(f"Scene directory missing: {self._scene_dir}")
        self._heat_sim_dir = self._scene_dir / params.output_subdir
        self._scene_layout = VGGTSceneLayout(self._scene_dir)
        self._has_facade_mask = self._scene_layout.has_facade_mask

    @staticmethod
    def _resolve_profile(scene_name: str) -> VGGTSceneProfile:
        """
        The registered profile for `scene_name`, or `DEFAULT_PROFILE` with a
        warning.

        `scene_name` is a raw directory name (explicit or auto-detected), not
        necessarily a registered `VGGTScene` identity, so it is resolved through
        `VGGTScene`'s own value parsing before `VGGTSceneProfile.for_scene`, which
        takes the typed enum, is ever reached.
        """
        scene = VGGTScene.for_value(scene_name)
        if scene is not None:
            return VGGTSceneProfile.for_scene(scene)

        logger.warning(
            f"No VGGTSceneProfile registered for scene '{scene_name}'. Falling "
            f"back to concrete over the whole surface, rescaled to 4 m; it names "
            f"no window or shell material, so a masked heat run and any modal "
            f"run are rejected."
        )
        return VGGTSceneProfile.DEFAULT_PROFILE

    def process(self, steps: PipelineSteps) -> None:
        """
        Run the enabled `steps` for this scene, in pipeline order.

        The reconstruction is resolved up front, before any step runs, so that a
        scene which has not been reconstructed yet fails before the mask fix
        rewrites its `transforms.json`. It is `None` exactly when no step needs it.

        :raises FileNotFoundError: If a step needs a reconstruction or an earlier
            step's output that is not there.
        :raises ValueError: If the scene carries a facade mask but its profile
            names no window material, or a modal run is asked for and it names no
            shell material; see `_run_heat_sim` and `_run_modal`.
        """
        if not self._params.dry_run:
            self._heat_sim_dir.mkdir(exist_ok=True)

        reconstruction: Path | None = None
        if steps.thermal_inpaint or steps.windows_inpaint:
            reconstruction = self._scene_layout.reconstruction
            logger.info(f"[{self._scene_name}] reconstruction: {reconstruction}")

        logger.info(
            f"=== Scene: {self._scene_name} (profile={self._profile}) "
            f"has_facade_mask={self._has_facade_mask} ==="
        )

        if steps.mask_fix and not self._params.skip_mask_fix:
            self._run_mask_fix()

        if reconstruction is not None:
            if steps.thermal_inpaint:
                self._run_inpaint(InpaintPass.THERMAL, reconstruction)

            if steps.windows_inpaint:
                if not self._has_facade_mask:
                    logger.warning(
                        f"[{self._scene_name}] Skipping WINDOWS inpaint: "
                        f"transforms.json has no facade mask reference "
                        f"({' / '.join(VGGTSceneLayout.FACADE_MASK_KEYS)}). "
                        "Provide masks to enable this step."
                    )
                else:
                    self._run_inpaint(InpaintPass.WINDOWS, reconstruction)

        if steps.heat_sim:
            if self._has_facade_mask:
                self._run_heat_sim()
            else:
                logger.info(
                    f"[{self._scene_name}] No facade mask, running "
                    "single-material heat sim."
                )
                self._run_heat_sim_single_material()

        if steps.modal:
            self._run_modal()

    @staticmethod
    def move_inpaint_outputs(
        tmp_dir: Path, dest_dir: Path, inpaint_pass: InpaintPass, *, dry_run: bool
    ) -> None:
        """
        Move inpainting outputs from a per-pass tmp folder up to the shared
        heat_simulation/ folder with pass-labelled filenames, then remove the tmp.
        """
        if dry_run:
            logger.info(
                f"[dry-run] would move {tmp_dir}/* into {dest_dir}/ "
                f"(rename {inpaint_pass.value}) and rmtree {tmp_dir}"
            )
            return

        label = inpaint_pass.value
        renames = {
            "inpainted_data.json": f"inpainted_{label}_data.json",
            "Final_comparison.png": f"Final_comparison_{label}.png",
            "Initial_state.png": f"Initial_state_{label}.png",
        }
        for src_name, dst_name in renames.items():
            src = tmp_dir / src_name
            if not src.exists():
                logger.warning(f"Expected inpaint output missing: {src}")
                continue
            dst = dest_dir / dst_name
            shutil.move(str(src), str(dst))

        for src in sorted(tmp_dir.glob("views_*.png")):
            dst = dest_dir / f"views_inpaint_{label}_{src.name[len('views_') :]}"
            shutil.move(str(src), str(dst))

        shutil.rmtree(tmp_dir, ignore_errors=True)

    def _spawn(self, command: Sequence[str | Path | float]) -> None:
        """
        Spawn `command` as a checked subprocess, or print it in dry-run mode.

        Every element is stringified here rather than at each call site, which is
        the one boundary where a `Path` or a number becomes an argument.

        :raises subprocess.CalledProcessError: If the step exits non-zero.
        """
        argv = [str(x) for x in command]
        label = "[dry-run] " if self._params.dry_run else ""
        logger.info(f"{label}{' '.join(argv)}")
        if self._params.dry_run:
            return

        subprocess.run(argv, check=True)

    def _run_mask_fix(self) -> None:
        """
        Fix `thermal_file_path` entries in the scene's `transforms.json`.

        :raises FileNotFoundError: If the scene has no `dataset/` directory.
        """
        dataset_dir = self._scene_dir / "dataset"
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"No dataset/ directory in {self._scene_dir}")

        cmd = [
            sys.executable,
            "-m",
            "forge_sim.scripts.fix_vggt_thermal_paths",
            "--scene-path",
            dataset_dir.resolve(),
            "--thermal-folder",
            "thermal",
        ]
        self._spawn(cmd)

    def _run_inpaint(self, inpaint_pass: InpaintPass, reconstruction: Path) -> None:
        """
        Run one inpainting pass, into a tmp folder whose outputs are then moved up.

        Skipped silently if the final output already exists in `_heat_sim_dir`.
        """
        final_output = self._heat_sim_dir / f"inpainted_{inpaint_pass.value}_data.json"
        if final_output.exists() and not self._params.dry_run:
            logger.info(
                f"Skipping {inpaint_pass.name} inpaint for {self._scene_dir.name}: "
                f"{final_output} already exists."
            )
            return

        tmp_dir = self._heat_sim_dir / f"_tmp_inpaint_{inpaint_pass.value}"
        if not self._params.dry_run:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "forge_sim.scripts.inpaint_data_spline",
            "--thermoscenes-scene-path",
            (self._scene_dir / "dataset").resolve(),
            "--inpainting-images-type",
            inpaint_pass.name,
            "--spline-path",
            reconstruction.resolve(),
            "--h-refinement-factor",
            self._profile.h_refinement_factor,
            "--initial-color",
            self._profile.inpaint_initial_color,
            "--optimization-iterations",
            self._params.inpaint_optimization_iterations,
            "--output-folder",
            tmp_dir.resolve(),
        ]
        cmd.append("--use-gpu" if self._params.use_gpu else "--no-use-gpu")

        self._spawn(cmd)
        self.move_inpaint_outputs(
            tmp_dir, self._heat_sim_dir, inpaint_pass, dry_run=self._params.dry_run
        )

    def _run_heat_sim(self) -> None:
        """
        Simulate heat over both inpainted fields, the window regions carrying the
        source.

        A scene is routed here by its dataset carrying a facade mask, while both
        materials come from its profile; the two are reconciled by the raise below
        rather than by letting the window material fall through to the heat
        script's own default. That check runs before the skip, so a scene whose
        output survives from an earlier run is reported rather than passed over:
        an output nothing can tie to a window material is the state this exists to
        surface.

        The window's target temperature rise is sourced the same way, and for the
        same reason: `profile.thermal.window_target_delta_temp` unless the run
        overrides it. The FEM side of this comparison already reads that field, so
        leaving the IGA side on the heat script's own default would give the two
        solvers being compared two independent sources for one physical quantity,
        agreeing only while both literals happen to match.

        Skipped silently if the expected output already exists in `_heat_sim_dir`.

        :raises ValueError: If `profile.thermal` names no `window_material`. The
            dataset selected the two-material solver, so there is no window
            material to run it with.
        :raises FileNotFoundError: If either inpainting pass's output is missing.
        """
        glass = self._profile.thermal.window_material
        if glass is None:
            raise ValueError(
                f"Scene '{self._scene_name}' has a facade mask, so it is solved "
                f"with two materials, but its thermal profile names no "
                f"`window_material`. Give its entry in "
                f"`VGGTSceneThermalProfile.VGGT_SCENE_THERMAL_PROFILES` the window "
                f"material the run should use, adding the entry if the scene has "
                f"none."
            )

        spline_path = self._heat_sim_dir / "inpainted_thermal_data.json"
        facade_path = self._heat_sim_dir / "inpainted_facade_data.json"

        if not self._params.dry_run:
            if not spline_path.exists():
                raise FileNotFoundError(
                    f"Heat sim needs {spline_path} but it does not exist. "
                    "Run the thermal inpainting stage first."
                )
            if not facade_path.exists():
                raise FileNotFoundError(
                    f"Heat sim needs {facade_path} but it does not exist. "
                    "Run the windows inpainting stage first."
                )

        expected_output = self._heat_sim_dir / "heat_simulation_two_materials.xdmf"
        if expected_output.exists() and not self._params.dry_run:
            logger.info(
                f"Skipping heat sim for {self._heat_sim_dir.parent.name}: "
                f"{expected_output} already exists."
            )
            return

        material = self._profile.thermal.base_material
        cmd = [
            sys.executable,
            "-m",
            "forge_sim.scripts.heat_simulation.heat_simulation_two_materials",
            "--spline-path",
            spline_path,
            "--facade-mask-path",
            facade_path,
            "--output-folder",
            self._heat_sim_dir,
            "--num-time-steps",
            self._params.heat_num_time_steps,
            "--concrete-config.conductivity",
            material.conductivity,
            "--concrete-config.density",
            material.density,
            "--concrete-config.specific-heat",
            material.specific_heat,
            "--concrete-config.thickness",
            material.thickness,
            "--target-length-scale",
            self._profile.thermal.target_length_scale_argument,
            "--glass-config.conductivity",
            glass.conductivity,
            "--glass-config.density",
            glass.density,
            "--glass-config.specific-heat",
            glass.specific_heat,
            "--glass-config.thickness",
            glass.thickness,
        ]
        delta_temp = (
            self._params.heat_target_delta_temp
            if self._params.heat_target_delta_temp is not None
            else self._profile.thermal.window_target_delta_temp
        )
        if delta_temp is not None:
            cmd.extend(["--target-delta-temp", delta_temp])
        cmd.append("--use-gpu" if self._params.use_gpu else "--no-use-gpu")
        self._spawn(cmd)

    def _run_heat_sim_single_material(self) -> None:
        """
        Simulate heat over the thermal field alone, for a scene with no facade mask.

        Skipped silently if the expected output already exists in `_heat_sim_dir`.

        :raises FileNotFoundError: If the thermal inpainting pass's output is
            missing.
        """
        spline_path = self._heat_sim_dir / "inpainted_thermal_data.json"

        if not self._params.dry_run:
            if not spline_path.exists():
                raise FileNotFoundError(
                    f"Heat sim needs {spline_path} but it does not exist. "
                    "Run the thermal inpainting stage first."
                )

        expected_output = self._heat_sim_dir / "heat_simulation.xdmf"
        if expected_output.exists() and not self._params.dry_run:
            logger.info(
                f"Skipping single-material heat sim for "
                f"{self._heat_sim_dir.parent.name}: {expected_output} already "
                f"exists."
            )
            return

        material = self._profile.thermal.base_material
        cmd = [
            sys.executable,
            "-m",
            "forge_sim.scripts.heat_simulation.heat_simulation",
            "--spline-path",
            spline_path,
            "--output-folder",
            self._heat_sim_dir,
            "--num-time-steps",
            self._params.heat_num_time_steps,
            "--concrete-config.conductivity",
            material.conductivity,
            "--concrete-config.density",
            material.density,
            "--concrete-config.specific-heat",
            material.specific_heat,
            "--concrete-config.thickness",
            material.thickness,
            "--target-length-scale",
            self._profile.thermal.target_length_scale_argument,
        ]
        if self._params.heat_target_delta_temp is not None:
            cmd.extend(["--target-delta-temp", self._params.heat_target_delta_temp])
        cmd.append("--use-gpu" if self._params.use_gpu else "--no-use-gpu")
        self._spawn(cmd)

    def _run_modal(self) -> None:
        """
        Solve the Reissner-Mindlin modal problem on the scene's reconstruction.

        The raw 3-DOF reconstruction is used, not the inpainted spline: that one
        carries 6 DOFs per control point (position and field) and has already been
        h-refined for inpainting. The shell solver refines it to
        `profile.modal_target_spans` from its original base instead.

        The shell material is reconciled with the solver the way `_run_heat_sim`
        reconciles the window material: by the raise below rather than by letting
        it fall through to the modal script's own default, which is a 1.5 mm steel
        car body and would answer a masonry facade with the spectrum of something
        else entirely.

        :raises ValueError: If `profile` names no `structural_material`, leaving
            the shell the scene is solved as unstated.
        :raises FileNotFoundError: If the scene holds no reconstruction.
        """
        material = self._profile.structural_material
        if material is None:
            raise ValueError(
                f"Scene '{self._scene_dir.name}' has no `structural_material` in "
                f"its profile, so its modal run would inherit the modal script's "
                f"own default shell. Give its entry in `SCENE_PROFILES` the shell "
                f"material the run should use, adding the entry if the scene has "
                f"none."
            )

        spline_path = self._scene_layout.reconstruction.resolve()

        modal_dir = self._scene_dir / "modal_analysis"
        if not self._params.dry_run:
            modal_dir.mkdir(exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "forge_sim.scripts.reissner_mindlin_modal_analysis"
            ".reissner_mindlin_modal_analysis",
            "--spline-path",
            spline_path,
            "--output-folder",
            modal_dir.resolve(),
            "--target-spans",
            self._profile.modal_target_spans,
            "--target-length-scale",
            self._profile.thermal.target_length_scale_argument,
            "--material-config.youngs-modulus",
            material.youngs_modulus,
            "--material-config.poisson-ratio",
            material.poisson_ratio,
            "--material-config.density",
            material.density,
            "--material-config.thickness",
            material.thickness,
            "--material-config.shear-correction-factor",
            material.shear_correction_factor,
        ]
        cmd.append("--use-gpu" if self._params.use_gpu else "--no-use-gpu")
        self._spawn(cmd)
