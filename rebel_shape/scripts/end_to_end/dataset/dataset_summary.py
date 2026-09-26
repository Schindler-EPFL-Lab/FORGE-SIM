from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class DatasetSummary:
    """
    What the synthetic dataset stage produced, for a caller that did not watch it.

    The dataset itself is on disk; this only says where it went and how the views
    were split, which is what a driver needs in order to report the stage.
    """

    dataset_dir: Path
    """Directory holding the three modality folders and `transforms.json`."""
    num_train: int
    """Views written with a `train` stem, the ones reconstruction fits."""
    num_eval: int
    """Views written with an `eval` stem, held out for novel-view checks."""
