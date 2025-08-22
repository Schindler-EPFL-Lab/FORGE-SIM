import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tyro
from azure.ai.ml import Input, command
from azure.ai.ml.constants import AssetTypes
from azure_ml_utils.azure_connect import get_client

from rebel_shape.scripts.optimize_thermoscenes_spline_manual_initial_pose import (
    ThermoSceneSplineOptimizationParameters,
)


@dataclass
class TrainingParametersAzure:
    script_params: ThermoSceneSplineOptimizationParameters
    """Parameters for the training script"""
    scene_name: str = "extracted-buildingA-spring"
    """Name of the scene that is to be optimized"""
    experiment_name: str = (
        "Spline optimization of diffuse "
        "ThermoScenes mesh spline and manual initialization"
    )
    scene_version: str = "1"
    """Version of the scene"""
    environment_version: str = "latest"
    """Version of the environment"""
    compute: str = "rebel-nerf-t4-ssh"
    """Compute target"""

    @property
    def environment(self) -> str:
        return "rebel-shape"


def main() -> None:
    model_type = "PRB_Projective_Spline"
    logging.basicConfig()

    parameters = tyro.cli(TrainingParametersAzure)

    ml_client = get_client()
    dataset = ml_client.data.get(
        name=parameters.scene_name, version=parameters.scene_version
    )

    job_inputs = dict(
        scene_name=parameters.scene_name,
        data=Input(
            type=AssetTypes.URI_FOLDER,  # type: ignore
            path=dataset.id,
        ),
        model_type=model_type,
    )

    job_name = (
        parameters.scene_name
        + "-"
        + model_type
        + "-"
        + datetime.now().strftime("%d-%m-%Y-%H%M%S")
    )

    parameters.experiment_name = parameters.scene_name + "-training-job"
    parameters.script_params.thermoscenes_scene_path = Path("${{inputs.data}}")

    cmd = (
        "python3.10 rebel_shape/scripts/"
        "optimize_thermoscenes_spline_manual_initial_pose.py "
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
        experiment_name=model_type + "-" + parameters.experiment_name,
        display_name=job_name,
        name=job_name,
    )

    job = ml_client.create_or_update(job)

    assert job.name is not None


if __name__ == "__main__":
    main()
