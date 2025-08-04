import importlib.resources
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GroundingDINOConfig:
    """
    Configuration structure for GroundingDINO.

    Specifies the path to the cloned GroundingDINO repository folder, the type of
    the model, and the thresholds for box and text confidence.
    GroundingDINO is specific in that it requires a script from the repository
    to be used for loading the model, alongside the model weights.
    """

    model_dir_path: Path
    """Path to the cloned GroundingDINO repository folder"""
    box_threshold: float = 0.35
    """How certain GroundingDINO has to be to produce a box around the target object"""
    text_threshold: float = 0.25
    """
    How certain GroundingDINO has to be to include tokens from the 
    prompt in the final label for the object
    """

    def __post_init__(self) -> None:
        if not self.model_dir_path.exists():
            raise ValueError(f"GroundingDINO path {self.model_dir_path} does not exist")
        if self.box_threshold < 0 or self.box_threshold > 1:
            raise ValueError(
                f"Box threshold {self.box_threshold} has to be in [0, 1] range"
            )
        if self.text_threshold < 0 or self.text_threshold > 1:
            raise ValueError(
                f"Text threshold {self.text_threshold} has to be in [0, 1] range"
            )

    @property
    def _script_relative_path(self) -> str:
        """
        Returns the relative path to the GroundingDINO script from root of the
        GroundingDINO repository.
        """
        return "config/GroundingDINO_SwinT_OGC.py"

    @property
    def model_type(self) -> str:
        """
        Returns the type of the GroundingDINO model.

        The appropriate checkpoint file is expected to be named this way
        (with a `.pth` suffix).
        """
        return "groundingdino_swint_ogc"

    @property
    def checkpoint_path(self) -> Path:
        """
        Returns the full path to the GroundingDINO model checkpoint.

        :raises FileNotFoundError: If the model weights are not found in the
            `model_dir_path` directory.
        """
        groundingdino_model_path = self.model_dir_path / (self.model_type + ".pth")
        if not groundingdino_model_path.exists():
            raise FileNotFoundError(
                f"GroundingDINO model weights not found at {groundingdino_model_path}. "
                f"Make sure that they are downloaded and placed `weights` directory."
            )

        return groundingdino_model_path

    @property
    def script_path(self) -> Path:
        """
        Returns the full path to the GroundingDINO script that is needed as a config
        argument for loading the model.

        :raises FileNotFoundError: If the script is not found in the `model_dir_path`
            directory at the predefined relative path.
        """
        groundingdino_script_path = Path(
            str(
                importlib.resources.files("groundingdino").joinpath(
                    self._script_relative_path
                )
            )
        )
        if not groundingdino_script_path.exists():
            raise FileNotFoundError(
                f"GroundingDINO config file not found at {groundingdino_script_path}."
                f"Make sure the path "
            )

        return groundingdino_script_path
