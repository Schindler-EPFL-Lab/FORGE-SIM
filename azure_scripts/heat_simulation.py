import logging
from dataclasses import dataclass
from datetime import datetime

import tyro
from azure.ai.ml import command
from azure_ml_utils.azure_connect import get_client

from rebel_shape.scripts.run_heat_simulation import HeatSimulationParameters


@dataclass
class TrainingParametersAzure:
    script_params: HeatSimulationParameters
    """Parameters for simulating heat transfer on a spline surface"""
    scene_name: str = "extracted-buildingA-spring"
    """Name of the scene/object"""
    experiment_name: str = "Heat_simulation_of_thermoscenes_spline"
    environment_version: str = "latest"
    """Version of the environment"""
    compute: str = "davor-cpu"
    """Compute target"""

    @property
    def environment(self) -> str:
        return "rebel-shape"


def main() -> None:
    model_type = "IGA_Crank_Nicolson"
    logging.basicConfig()

    parameters = tyro.cli(TrainingParametersAzure)

    ml_client = get_client()

    job_name = (
        parameters.scene_name
        + "-"
        + model_type
        + "-"
        + datetime.now().strftime("%d-%m-%Y-%H%M%S")
    )

    parameters.experiment_name = parameters.scene_name + "-heat-simulation"

    cmd = "python3.12 rebel_shape/scripts/run_heat_simulation.py "
    cmd += parameters.script_params.to_command_string("")

    if parameters.environment_version.isnumeric():
        env_version_string = ":" + parameters.environment_version
    else:
        env_version_string = "@" + parameters.environment_version

    job = command(
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
