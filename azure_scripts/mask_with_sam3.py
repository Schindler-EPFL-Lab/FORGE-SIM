import logging
from dataclasses import dataclass
from datetime import datetime

import tyro
from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure_ml_utils.azure_connect import get_client

from rebel_shape.scripts.segmentation.mask_with_sam3 import Sam3ThermoScenesParams


@dataclass
class Sam3AzureJobConfig:
    script_params: Sam3ThermoScenesParams
    """Specific parameters for the SAM 3 segmentation script"""
    input_data_name: str = "ThermoScenes"
    """Name of the UriFolder data asset"""
    input_data_version: str = "4"
    """Version of the data asset """
    scene_name: str = "reflect-laptop"
    """Name of the scene"""
    bpe_asset_name: str = "sam3_bpe_vocab"
    """Name of the BPE vocabulary asset"""
    bpe_asset_version: str = "1"
    """Version of the BPE asset"""
    checkpoint_asset_name: str = "SAM3_pretrained_weights"
    """Name of the SAM 3 pretrained weights asset"""
    checkpoint_asset_version: str = "1"
    """Version of the checkpoint"""
    experiment_name: str = "sam3-segmentation"
    """Name of the experiment in Azure ML Studio"""
    compute: str = "t4-dedicated-small"
    """Compute target"""
    environment_name: str = "sam3"
    """Name of the environment"""
    environment_version: str = "29"
    """Version of the environment"""


def main() -> None:
    logging.basicConfig()

    config = tyro.cli(Sam3AzureJobConfig)
    ml_client = get_client()
    input_data = ml_client.data.get(
        name=config.input_data_name, version=config.input_data_version
    )

    bpe_data = ml_client.data.get(
        name=config.bpe_asset_name, version=config.bpe_asset_version
    )

    checkpoint_data = ml_client.data.get(
        name=config.checkpoint_asset_name, version=config.checkpoint_asset_version
    )

    job_inputs = {
        "thermoscene_root": Input(
            type=AssetTypes.URI_FOLDER,
            path=input_data.id,
            mode=InputOutputModes.RO_MOUNT,
        ),
        "bpe_resources": Input(
            type=AssetTypes.URI_FOLDER,
            path=bpe_data.id,
            mode=InputOutputModes.RO_MOUNT,
        ),
        "checkpoint_resources": Input(
            type=AssetTypes.URI_FOLDER,
            path=checkpoint_data.id,
            mode=InputOutputModes.RO_MOUNT,
        ),
    }

    job_outputs = {
        "scene_output": Output(
            type=AssetTypes.URI_FOLDER,
            mode=InputOutputModes.UPLOAD,
        )
    }

    job_name = (
        f"sam3-seg-{config.scene_name}-{datetime.now().strftime('%d-%m-%Y-%H%M%S')}"
    )

    cmd_base = (
        "python rebel_shape/scripts/segmentation/mask_with_sam3.py "
        f"--scene-dir ${{inputs.thermoscene_root}}/{config.scene_name} "
        f"--sam3-checkpoint-dir ${{inputs.checkpoint_resources}} "
        f"--output-scene-dir ${{outputs.scene_output}} "
        f"--bpe-path ${{inputs.bpe_resources}} "
    )

    cmd_args = config.script_params.to_command_string("")
    full_command = f"set -euo pipefail && {cmd_base} {cmd_args}"

    if config.environment_version.isnumeric():
        env_version_string = ":" + config.environment_version
    else:
        env_version_string = "@" + config.environment_version

    env_string = f"azureml:{config.environment_name}{env_version_string}"

    job = command(
        name=job_name,
        display_name=job_name,
        experiment_name=config.experiment_name,
        code=".",
        command=full_command,
        inputs=job_inputs,
        outputs=job_outputs,
        environment=env_string,
        compute=config.compute,
    )

    ml_client.create_or_update(job)


if __name__ == "__main__":
    main()
