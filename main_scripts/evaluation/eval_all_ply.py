"""
Batch PLY-mesh Chamfer evaluation of splines reconstructed by the proposed
rebel-shape method, across the PLY reference scenes.

`--scenes-names` takes `PlyScene` members; tyro renders them by member NAME, not
by value, so a command line spells `--scenes-names ARMADILLO` where it used to
spell `--scenes-names armadillo`.
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from main_scripts.evaluation.eval_all_common import (  # noqa: E402
    prepare_metrics_output_folder,
)
from main_scripts.evaluation.ply_scene import PlyScene  # noqa: E402
from rebel_shape import logger  # noqa: E402


@dataclass(kw_only=True)
class EvalAllPlyParameters(ReverseCli):
    """
    Which PLY scenes to evaluate, and where their reconstructions live.

    For each scene, spawns `rebel_shape/scripts/evaluation/eval_ply.py` once per
    reconstructed spline under `validation_inputs_root`/splines/<scene>/, then
    `rebel_shape/scripts/evaluation/aggregate_metrics.py` once to average the
    per-spline metrics into that scene's mean.
    """

    DATASET_ROOT: ClassVar[Path] = Path("data/meshes")
    """Directory holding each scene's reference `.ply`."""

    scenes_names: list[PlyScene]
    """PLY mesh scenes to validate on."""
    validation_inputs_root: Path
    """
    Root directory holding the reconstructions to validate. Must contain a
    `splines/` subdirectory with one folder per scene name, each holding that
    scene's spline reconstructions as `.json` files.
    """
    output_folder_name: str = "splines-metrics"
    """Name of the output folder created under `validation_inputs_root`."""

    def __post_init__(self) -> None:
        """
        :raises FileNotFoundError: If `validation_inputs_root`/splines or
            `DATASET_ROOT` does not exist.
        """
        splines_root = self.validation_inputs_root / "splines"
        if not splines_root.exists():
            raise FileNotFoundError(
                f"The `splines_root` {splines_root} does not exist."
            )
        if not self.DATASET_ROOT.exists():
            raise FileNotFoundError(
                f"The `dataset_root` {self.DATASET_ROOT} does not exist."
            )


def main(params: EvalAllPlyParameters) -> None:
    """
    Evaluates every selected scene, writing each scene's mean metrics.

    :raises FileExistsError: If a scene's output folder already holds the
        metrics of an earlier run.
    :raises subprocess.CalledProcessError: If an evaluation or aggregation
        subprocess fails.
    """
    splines_root = params.validation_inputs_root / "splines"
    output_root = params.validation_inputs_root / params.output_folder_name
    output_root.mkdir(exist_ok=True, parents=True)

    for scene in params.scenes_names:
        splines_paths = sorted((splines_root / scene.value).glob("*.json"))

        output_folders: list[Path] = []
        for idx, spline_path in enumerate(splines_paths):
            logger.info(f"Evaluating {spline_path}")
            output_folder = output_root / f"{scene.value}_{idx}"
            prepare_metrics_output_folder(output_folder, "eval_shape.json")
            output_folders.append(output_folder)
            reference_shape_path = params.DATASET_ROOT / f"{scene.value}.ply"

            cmd = [
                "python3",
                "rebel_shape/scripts/evaluation/eval_ply.py",
                "--reference-shape-path",
                str(reference_shape_path),
                "--spline-path",
                str(spline_path),
                "--output-file-path",
                str(output_folder / "eval_shape.json"),
            ]
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

        cmd = [
            "python3",
            "rebel_shape/scripts/evaluation/aggregate_metrics.py",
            "--metrics-folders",
            *map(str, output_folders),
            "--metrics-file-names",
            "eval_shape.json",
            "--output-file-path",
            str(output_root / f"{scene.value}_aggregated.json"),
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main(params=tyro.cli(EvalAllPlyParameters))
