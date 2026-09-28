"""
Master runner for the post-processing pipeline over a VGGT-reconstructed dataset.

Per scene it runs some subset of five steps, in this order: repairing the thermal image
paths in `transforms.json`, inpainting the thermal images onto the reconstruction,
inpainting the facade mask onto it, simulating heat over the two recovered fields, and
solving the Reissner-Mindlin modal problem on the raw reconstruction. `PipelineStage`
names the subsets worth asking for.

Each step is spawned as a subprocess rather than imported, because each is a `tyro` CLI
in its own right and all but the path fix set a process-global Mitsuba variant, so they
must not share a process. `--dry-run` builds every command and prints it without running
anything and without touching the dataset, which is what makes it usable against a
dataset that must not be written to.

Outputs land in `<scene>/heat_simulation/` and `<scene>/modal_analysis/`, leaving the
reconstruction folder (`result/` or `results/`) untouched.

Like `run_pipeline.py` beside it, this lives outside both packages: it drives
`forge_sim` from the outside rather than belonging to it, so it is a plain script
rather than a module of either.
"""

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge_sim import logger  # noqa: E402
from main_scripts.scenes.vggt_scene import VGGTScene  # noqa: E402
from main_scripts.scenes.vggt_scene_layout import VGGTSceneLayout  # noqa: E402
from main_scripts.vggt_pipeline.pipeline_stage import PipelineStage  # noqa: E402
from main_scripts.vggt_pipeline.scene_failure import SceneFailure  # noqa: E402


@dataclass(kw_only=True)
class VGGTPipelineParameters(ReverseCli):
    """Which subset of the pipeline to run, over which scenes of which dataset."""

    dataset_root: Path
    """Root directory containing per-scene folders (e.g., VGGT-SAM3-v3)."""

    stage: PipelineStage = PipelineStage.INPAINTING_AND_HEAT
    """Which subset of the pipeline to execute."""

    scenes: list[VGGTScene] = field(default_factory=list)
    """
    Explicit list of scenes to process. If empty, auto-detect by finding any subdir
    of `dataset_root` that has a `shape_*.json` in `result/` or `results/`, which,
    unlike this field, can find a scene not registered in `SCENE_PROFILES`.
    """

    skip_mask_fix: bool = False
    """Skip the mask-fix step (the underlying script is idempotent anyway)."""

    dry_run: bool = False
    """Print the subprocess commands that would run; do not execute them, and do not
    create any of the directories they would write into."""

    use_gpu: bool = True
    """Whether downstream scripts should use the GPU."""

    output_subdir: Path = Path("heat_simulation")
    """Per-scene subdirectory the inpaint + heat outputs are written to. Use a new
    name (e.g. `heat_simulation_fixed`) to keep a prior run's results intact."""

    inpaint_optimization_iterations: int = 20
    """`--optimization-iterations` passed to inpainting."""

    heat_num_time_steps: int = 500
    """`--num-time-steps` passed to heat sim."""

    heat_target_delta_temp: float | None = None
    """
    Target temperature rise (K) for heated regions. When set, the heat
    source strength Q is computed after matrix assembly so that an
    insulated element rises by this amount over the full simulation.
    Passed as `--target-delta-temp` to the heat sim scripts.
    """

    def __post_init__(self) -> None:
        if not self.dataset_root.is_dir():
            raise ValueError(f"`dataset_root` is not a directory: {self.dataset_root}")


def _detect_scenes(dataset_root: Path) -> list[str]:
    """Auto-detect scenes that have a reconstruction."""
    detected: list[str] = []
    for sub in sorted(dataset_root.iterdir()):
        if not sub.is_dir():
            continue
        if VGGTSceneLayout(sub).has_reconstruction:
            detected.append(sub.name)
    return detected


def main(params: VGGTPipelineParameters) -> None:
    """
    Run the selected stage over every selected scene, one scene at a time.

    A scene that fails does not stop the others: its exception is logged and repeated in
    a summary at the end, so a long run over eight scenes reports everything it managed.
    The run as a whole then fails, so a caller reading only the exit status is not told
    that a run which lost two scenes succeeded.

    :raises ValueError: If no scenes were named and none could be auto-detected.
    :raises RuntimeError: If any scene failed, once every other scene has been run.
    """
    # Deferred: `ScenePipelineRun` type-hints its `params` argument with
    # `VGGTPipelineParameters`, so importing it at module scope here would cycle.
    from main_scripts.vggt_pipeline.scene_pipeline_run import ScenePipelineRun

    scenes = [scene.value for scene in params.scenes] or _detect_scenes(
        params.dataset_root
    )
    if not scenes:
        raise ValueError(
            f"No scenes selected and none auto-detected under {params.dataset_root}."
        )

    steps = params.stage.steps
    logger.info(
        f"Running stage={params.stage.value} steps={steps} for {len(scenes)} scene(s): "
        f"{scenes}"
    )

    failures: list[SceneFailure] = []
    for scene_name in scenes:
        try:
            ScenePipelineRun(scene_name, params).process(steps)
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            ValueError,
        ) as exc:
            logger.error(f"[{scene_name}] FAILED: {exc}")
            failures.append(SceneFailure(scene_name=scene_name, message=str(exc)))

    if failures:
        logger.warning(f"{len(failures)}/{len(scenes)} scene(s) failed:")
        for failure in failures:
            logger.warning(f"  - {failure.scene_name}: {failure.message}")
        failed_scenes = ", ".join(failure.scene_name for failure in failures)
        raise RuntimeError(
            f"{len(failures)}/{len(scenes)} scene(s) failed: {failed_scenes}."
        )

    logger.info(f"All {len(scenes)} scene(s) completed successfully.")


if __name__ == "__main__":
    main(tyro.cli(VGGTPipelineParameters))
