from dataclasses import dataclass

from rebel_shape.data_processing.masked_pose_image import MaskedPoseImage


@dataclass(frozen=True, kw_only=True)
class SyntheticReferenceImages:
    """
    The synthetic dataset, loaded back and split by
    `DatasetGenerationRunner.load_synthetic_references`.

    Both lists are paired with the same deterministic cameras that rendered them, in
    the order `run_dataset_generation` wrote the frames.
    """

    train_images: list[MaskedPoseImage]
    """Views written with a `train` stem, the ones reconstruction/inpainting fit."""
    eval_images: list[MaskedPoseImage]
    """Views written with an `eval` stem, held out for novel-view checks."""
