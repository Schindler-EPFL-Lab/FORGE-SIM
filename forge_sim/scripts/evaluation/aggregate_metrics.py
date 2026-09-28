"""
Averages the per-target metrics files written by the evaluation scripts into one
metrics file, metric by metric.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli


@dataclass(kw_only=True)
class AggregateMetricsParameters(ReverseCli):
    """
    Which evaluation outputs to average, and where to write the average.

    Each folder in `metrics_folders` is one evaluated target (one reconstruction),
    holding the metrics files named by `metrics_file_names` as the single-target
    evaluation scripts beside this one (`eval_ply.py`, `eval_real.py`,
    `eval_buildnet3d.py` and their `_mesh` variants) wrote them. A single folder is
    a valid input: its metrics are then written through unchanged.
    """

    metrics_folders: list[Path]
    """One output folder per evaluated target, all of them averaged together."""
    metrics_file_names: list[Path]
    """
    Metrics files to read from every folder, such as `eval_nvs.json` and
    `eval_shape.json`. Their metrics are merged, so no two may share a metric name.
    """
    output_file_path: Path
    """Where to write the averaged metrics (its folder is created if absent)."""

    def __post_init__(self) -> None:
        """
        :raises ValueError: If `metrics_folders` or `metrics_file_names` is empty.
        :raises FileNotFoundError: If a folder in `metrics_folders` lacks one of the
            `metrics_file_names`.
        """
        if not self.metrics_folders:
            raise ValueError(
                "`metrics_folders` is empty, so there is nothing to average: a mean "
                "over no evaluated target is undefined."
            )
        if not self.metrics_file_names:
            raise ValueError("`metrics_file_names` must name at least one file.")
        missing = [
            folder / name
            for folder in self.metrics_folders
            for name in self.metrics_file_names
            if not (folder / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Metrics files do not exist: {', '.join(map(str, missing))}."
            )


def read_target_metrics(
    folder: Path, metrics_file_names: list[Path]
) -> dict[str, float]:
    """
    Reads the metrics files `metrics_file_names` from `folder` and merges them into
    one mapping from metric name to value, in file order and then in each file's own
    key order.

    :returns: Every metric of the target evaluated into `folder`.
    :raises ValueError: If two of the files report a metric under the same name,
        which would leave one of the two values silently discarded.
    """
    metrics: dict[str, float] = {}
    for name in metrics_file_names:
        with open(folder / name) as f:
            file_metrics: dict[str, float] = json.load(f)
        duplicated = metrics.keys() & file_metrics.keys()
        if duplicated:
            raise ValueError(
                f"{folder / name} reports {', '.join(sorted(duplicated))}, which an "
                f"earlier file in `metrics_file_names` already reported."
            )
        metrics.update(file_metrics)
    return metrics


def aggregate_metrics(params: AggregateMetricsParameters) -> None:
    """
    Writes to `params.output_file_path` the mean of every metric over the targets
    in `params.metrics_folders`.

    Each target's metrics are read with `read_target_metrics`. Every mean is taken
    over all targets, in the order of `params.metrics_folders`, and the metrics
    appear in the output in the first target's order. With a single folder, each
    mean is the target's own value unchanged.

    :raises ValueError: If the targets do not all report the same metrics, since a
        mean over the subset of targets reporting a metric would read as a mean over
        all of them.
    """
    values_per_metric: dict[str, list[float]] = {}
    for folder in params.metrics_folders:
        target_metrics = read_target_metrics(folder, params.metrics_file_names)
        if values_per_metric and target_metrics.keys() != values_per_metric.keys():
            raise ValueError(
                f"{folder} reports the metrics {sorted(target_metrics)}, but "
                f"{params.metrics_folders[0]} reports {sorted(values_per_metric)}."
            )
        for metric_name, value in target_metrics.items():
            values_per_metric.setdefault(metric_name, []).append(value)

    mean_metrics = {
        metric_name: np.mean(values).item()
        for metric_name, values in values_per_metric.items()
    }

    params.output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(params.output_file_path, "w") as f:
        json.dump(mean_metrics, f, indent=4)


if __name__ == "__main__":
    parameters = tyro.cli(AggregateMetricsParameters)
    aggregate_metrics(params=parameters)
