import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tyro
from azure.ai.ml import Input, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure_ml_utils.azure_connect import get_client

from rebel_shape.scripts.segmentation.segment_foreground_and_background_scene import (
    SceneForegroundBackgroundSegmentation,
)


@dataclass
class TrainingParametersAzure:
    """
    Send depth estimation of a scene job on azure

    :raises ValueError if `input_scene_name` == `output_scene_name` and
    `input_scene_version` == `output_scene_version` since it is more
    reasonable to have a new version of the dataset
    """

    script_params: SceneForegroundBackgroundSegmentation
    """
    Estimate foreground and background of a scene using depths and objects masks
    """

    input_scene_name: str = "extracted-buildingA-spring"
    """
    Name of the input scene which images are inferenced by a depth estimation model
    """
    input_scene_version: str = "1"
    """Version of the scene"""

    output_scene_name: str = "extracted-buildingA-spring"
    """
    Name of the output scene which will store the depth predictions
    """
    output_scene_version: str = "2"

    experiment_name: str = "Estimate foreground and background of a scene"
    environment_version: str = "latest"
    """Version of the environment"""
    compute: str = "davor-cpu"
    """Compute target"""

    @property
    def environment(self) -> str:
        return "rebel-shape"

    def __post_init__(self) -> None:
        if (self.output_scene_name == self.input_scene_name) and (
            self.output_scene_version == self.input_scene_version
        ):
            raise ValueError(
                "Input scene and output scene have the same name and version"
            )


def main() -> None:
    logging.basicConfig()

    parameters = tyro.cli(TrainingParametersAzure)

    ml_client = get_client()
    input_scene = ml_client.data.get(
        name=parameters.input_scene_name, version=parameters.input_scene_version
    )
    output_scene = ml_client.data.get(
        name=parameters.output_scene_name, version=parameters.output_scene_version
    )

    job_inputs = dict(
        input_scene=Input(
            type=AssetTypes.URI_FOLDER,
            path=input_scene.id,
            mode=InputOutputModes.RO_MOUNT,
        ),
        output_scene=Input(
            type=AssetTypes.URI_FOLDER,
            path=output_scene.id,
            mode=InputOutputModes.RW_MOUNT,
        ),
    )

    job_name = "segment-fgbg" + "-" + datetime.now().strftime("%d-%m-%Y-%H%M%S")

    parameters.experiment_name = "segment-fgbg-scene"
    parameters.script_params.input_scene_path = Path("${{inputs.input_scene}}")
    parameters.script_params.output_scene_path = Path("${{inputs.output_scene}}")

    cmd = (
        "python3.12 rebel_shape/scripts/segmentation/"
        + "segment_foreground_and_background_scene.py "
    )
    cmd += parameters.script_params.to_command_string("")

    if parameters.environment_version.isnumeric():
        env_version_string = ":" + parameters.environment_version
    else:
        env_version_string = "@" + parameters.environment_version

    job = command(
        inputs=job_inputs,
        code=".",  # location of source code
        command=cmd,
        environment=parameters.environment + env_version_string,
        compute=parameters.compute,
        experiment_name=parameters.experiment_name,
        display_name=job_name,
        name=job_name,
    )

    job = ml_client.create_or_update(job)

    assert job.name is not None


if __name__ == "__main__":
    main()
