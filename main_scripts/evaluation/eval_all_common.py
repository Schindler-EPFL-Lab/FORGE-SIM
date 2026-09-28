"""
What the six `eval_all_*` batch drivers share.

Each driver shells out to a per-scene render script and a per-scene metrics script,
then to `forge_sim/scripts/evaluation/aggregate_metrics.py`, which reads back the
metrics files the per-scene metrics script wrote. The guard here is what keeps that
read honest when a driver is re-run into an output root that is not empty.
"""

from pathlib import Path


def prepare_metrics_output_folder(
    output_folder: Path, *metrics_file_names: str
) -> None:
    """
    Creates `output_folder`, refusing to reuse one that already holds any of the
    metrics files named by `metrics_file_names`.

    The aggregation reads whatever the per-scene metrics script left in
    `output_folder`, and it cannot tell a file this run wrote from one an earlier run
    left behind. So re-running into a populated output root would let the earlier
    run's metrics stand in wherever this run's render or evaluation failed, and the
    aggregate would look complete while describing numbers nobody produced this time.
    Refusing up front makes that impossible. Nothing is deleted: the earlier run stays
    where it is, to inspect or to move aside.

    :raises FileExistsError: If `output_folder` already holds one of the named metrics
        files.
    """
    stale = [name for name in metrics_file_names if (output_folder / name).is_file()]
    if stale:
        raise FileExistsError(
            f"{output_folder} already holds {', '.join(stale)} from an earlier run. "
            "Aggregating those again would report the earlier run's numbers even "
            "where this run failed. Move or remove that folder, or point "
            "`output_folder_name` at a fresh output root."
        )
    output_folder.mkdir(parents=True, exist_ok=True)
