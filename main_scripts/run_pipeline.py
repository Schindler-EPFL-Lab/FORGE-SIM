"""End-to-end orchestrator for the synthetic digital-twin validation experiment.

Runs every stage in order, each in its own subprocess (the stages set different
global Mitsuba variants, so they must not share a process). The FEM-side stages
(mesh preparation and the two DOLFINx drivers, all of which live in
`fem/end_to_end/`) run in the separate conda environment; everything else runs
in the active .venv.

Stages (see `Stage`): mesh, dataset, reconstruct, inpaint, gt, fem_heat, iga_heat,
cmp_heat, fem_modal, iga_modal, cmp_modal, summarize.

It lives outside both packages on purpose: it spans two environments and belongs to
neither, so it is a plain script rather than a module of either one.

Examples:
    python main_scripts/run_pipeline.py
    python main_scripts/run_pipeline.py --stages GT FEM_MODAL IGA_MODAL CMP_MODAL
    python main_scripts/run_pipeline.py --conda-python /path/to/python
    python main_scripts/run_pipeline.py --dry-run

This driver uses the default (Suzanne preset) configuration; for parameter sweeps,
run the individual stage modules directly (each is a `tyro` CLI over
`EndToEndConfig`) or edit the defaults in `end_to_end_config.py` and its
four sub-config modules.
"""

import importlib.util
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main_scripts.conda_python import CondaPython  # noqa: E402
from main_scripts.pipeline.dataset_overrides import DatasetOverrides  # noqa: E402
from main_scripts.pipeline.stage import Stage  # noqa: E402
from main_scripts.pipeline.stage_commands import StageCommands  # noqa: E402
from main_scripts.scenes.e2e_scene import E2EScene  # noqa: E402
from main_scripts.scenes.e2e_scene_profile import E2ESceneProfile  # noqa: E402
from rebel_shape import logger  # noqa: E402
from rebel_shape.scripts.end_to_end.data_paths import REPO_ROOT  # noqa: E402
from rebel_shape.scripts.end_to_end.end_to_end_config import (  # noqa: E402
    EndToEndConfig,
)
from rebel_shape.scripts.end_to_end.heat.thermal_sim_config import (  # noqa: E402
    ThermalSimConfig,
)


@dataclass(kw_only=True)
class RunPipelineParameters(ReverseCli):
    """Which pipeline stages to run, against which dataset and interpreters."""

    stages: Sequence[Stage] = tuple(Stage)
    """Subset of stages to run. Reordered to the canonical `Stage` order in
    `__post_init__`, so the flag only selects which stages run, not their order."""
    conda_python: Path | None = None
    """Python interpreter of the DOLFINx conda env (for the FEM stages). Defaults to
    the `REBEL_SHAPE_DOLFINX_PYTHON` environment variable."""
    use_gpu: bool = True
    """Run the Mitsuba (venv) stages on the GPU. Set False to use the CPU/LLVM
    variant, e.g. to avoid the cuda_ad_mono PTX-compilation bug seen on some
    driver/toolkit combinations during inpainting/IGA modal."""
    scene: E2EScene | None = None
    """One of the four registered targets. Fills `name`, `mesh_path` and the
    ground-truth temperature rule from `E2ESceneProfile.SCENE_PROFILES`, so a target
    is named once rather than restated as three flags that can disagree. Any of those
    flags still wins where it is set. None -> the `EndToEndConfig` defaults."""
    name: str | None = None
    """Dataset identifier. None -> `scene`'s own name, else the `EndToEndConfig`
    default (Suzanne)."""
    mesh_path: Path | None = None
    """Ground-truth source mesh. None -> the `EndToEndConfig` default."""
    prebuilt_spline_path: Path | None = None
    """Pre-optimised recovered spline, skipping reconstruction. None -> the default."""
    output_root: Path | None = None
    """Root directory for all stage outputs. None -> the `EndToEndConfig` default."""
    temperature_field_source: ThermalSimConfig.TemperatureFieldSource | None = None
    """Which coordinates the ground-truth temperature checkerboard is evaluated in.
    None -> `scene`'s registered rule, else the `EndToEndConfig` default. Two runs
    being compared must agree on this: it is the experiment's controlled field, and a
    silent difference in it reads exactly like a solver result."""
    binary_temperature: bool | None = None
    """Override thermal.binary_temperature: True = sharp 0/1 checkerboard field,
    False = smooth (band-limited) field. None -> EndToEndConfig default (smooth).
    Forwarded to every venv stage; the FEM heat side inherits it via the GT npz."""
    dry_run: bool = False
    """Build every stage's command and check that its target resolves (a `-m`
    module's import spec, or a subprocess script's path) without running anything.
    Prints what it checked and raises on the first target that does not resolve."""

    def __post_init__(self) -> None:
        """Puts `stages` in `Stage` order and drops duplicates, so a caller cannot
        change the canonical order by passing `--stages` out of sequence."""
        self.stages = tuple(stage for stage in Stage if stage in set(self.stages))

    @property
    def resolved_conda_python(self) -> Path:
        """
        Resolve `conda_python`, falling back to `CondaPython.ENV_VAR`.

        :raises ValueError: If neither `conda_python` nor the environment variable
            is set, or if the resolved interpreter path does not exist.
        """
        return CondaPython.resolve_with_override(self.conda_python)


def _dry_run(commands: dict[Stage, list[str]]) -> None:
    """
    Check that every stage's command target resolves, without running any of
    them.

    A `-m module` command is checked with `importlib.util.find_spec` (which locates
    the module without importing/executing it); a subprocess script command (the FEM
    stages) is checked with `Path.exists()`.

    :raises ValueError: If a stage's `-m` module has no import spec, or its script
        path does not exist. Reports the first such stage, in canonical order.
    """
    for stage, command in commands.items():
        if len(command) >= 3 and command[1] == "-m":
            module = command[2]
            if importlib.util.find_spec(module) is None:
                raise ValueError(f"stage '{stage.value}': module not found: {module}")
            logger.info(f"[dry-run] {stage.value}: module resolves -> {module}")
        else:
            script = Path(command[1])
            if not script.exists():
                raise ValueError(f"stage '{stage.value}': script not found: {script}")
            logger.info(f"[dry-run] {stage.value}: script exists -> {script}")


def main(params: RunPipelineParameters) -> None:
    """
    Run the selected stages in canonical order, each in its own subprocess.

    :raises RuntimeError: If a stage exits non-zero; later stages are not run.
    """
    profile = (
        None if params.scene is None else E2ESceneProfile.for_scene(params.scene)
    )
    name = params.name
    if name is None and params.scene is not None:
        name = params.scene.value
    mesh_path = params.mesh_path
    if mesh_path is None and profile is not None:
        mesh_path = profile.mesh_path
    field_source = params.temperature_field_source
    if field_source is None and profile is not None:
        field_source = profile.temperature_field_source

    overrides = DatasetOverrides(
        name=name,
        mesh_path=mesh_path,
        prebuilt_spline_path=params.prebuilt_spline_path,
        output_root=params.output_root,
    )
    flags = overrides.flags
    if field_source is not None:
        flags += ["--thermal.temperature-field-source", field_source.name]
    if params.binary_temperature is not None:
        flags.append(
            "--thermal.binary-temperature"
            if params.binary_temperature
            else "--thermal.no-binary-temperature"
        )
    cfg = overrides.apply(EndToEndConfig())
    commands = StageCommands(
        cfg=cfg,
        conda_python=str(params.resolved_conda_python),
        use_gpu=params.use_gpu,
        overrides=flags,
    ).commands

    if params.dry_run:
        _dry_run(commands)
        logger.info("===== [run_pipeline] dry-run OK =====")
        return

    # Resolved against the repository root, not the launcher's cwd: every stage runs
    # with `cwd=REPO_ROOT` below, so a relative configured path is only meaningful
    # from there, and an unresolved miss here silently reconstructs from scratch.
    use_prebuilt = (
        cfg.prebuilt_spline_path is not None
        and (REPO_ROOT / cfg.prebuilt_spline_path).exists()
    )
    for stage in params.stages:
        if stage is Stage.RECONSTRUCT and use_prebuilt:
            logger.info(
                f"===== [run_pipeline] skipping 'reconstruct' (using pre-built spline "
                f"{cfg.prebuilt_spline_path.name}) ====="
            )
            continue
        logger.info(f"===== [run_pipeline] stage '{stage.value}' =====")
        result = subprocess.run(commands[stage], cwd=str(REPO_ROOT))
        if result.returncode != 0:
            raise RuntimeError(
                f"Stage '{stage.value}' failed (exit {result.returncode}). Aborting."
            )
    completed = [s.value for s in params.stages]
    logger.info(f"===== [run_pipeline] completed stages: {completed} =====")


if __name__ == "__main__":
    main(tyro.cli(RunPipelineParameters))
