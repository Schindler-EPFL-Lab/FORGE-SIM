from dataclasses import dataclass
from pathlib import Path


@dataclass
class SAMConfig:
    """
    Configuration structure for Segment Anything Model (SAM).

    Specifies the path to the model weights, the type of the model, and the filename
    of the checkpoint. The model type can be one of "vit_b", "vit_l", or "vit_h".
    The model weights must be found in the specified directory, and the checkpoint
    filename must match the model type.
    """

    model_dir_path: Path
    """Path to the folder that contains the weights checkpoint"""
    model_type: str = "vit_h"
    """Of which size is the SAM ViT model (vit_h is the largest)"""
    checkpoint_filename: str = "sam_vit_h_4b8939.pth"
    """SAM model checkpoint filename"""

    def __post_init__(self) -> None:
        if not self.model_dir_path.exists():
            raise ValueError(
                f"Segment Anything model path {self.model_dir_path} does not exist"
            )
        if self.model_type not in ["vit_b", "vit_l", "vit_h"]:
            raise ValueError(
                f"Model type {self.model_type} is not supported. "
                f"Supported types are: vit_b, vit_l, vit_h"
            )
        if not self.checkpoint_filename.endswith(".pth"):
            raise ValueError(
                f"Segment Anything model checkpoint {self.checkpoint_filename} "
                f"should have .pth extension"
            )
        if self.model_type not in self.checkpoint_filename:
            raise ValueError(
                f"Segment Anything model checkpoint {self.checkpoint_filename} "
                f"should contain the model type {self.model_type}"
            )

    @property
    def checkpoint_path(self) -> Path:
        """
        Returns the full path to the SAM model checkpoint.
        """
        sam_model_checkpoint_path = self.model_dir_path / self.checkpoint_filename
        if not sam_model_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Segment Anything model {sam_model_checkpoint_path} not found. "
                f"Make sure it is downloaded and placed "
                f"in {self.model_dir_path} directory."
            )

        return sam_model_checkpoint_path
