import logging
import subprocess
import tempfile
from pathlib import Path

import tyro
from azure_ml_utils.azure_connect import get_client


def create_gif(
    output_folder: Path,
    save_path: Path,
) -> None:
    """
    Creates a GIF from PNGs in `output_folder` matching 'views_*_iteration-*.png'.
    Saves to `save_path` (default: `output_folder`/out.gif).
    """

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-pattern_type",
            "glob",
            "-i",
            f"{str(output_folder)}/views_*_iteration-*.png",
            str(save_path),
        ]
    )


def create_gif_from_output(
    job_name: str,
    save_path: Path,
    retry_num: int = 0,
    workspace: str = "building-mind-workspace",
) -> None:
    """Create a gif from job output images

    Args:
        job_name: name of the job.
        save_path: path to store gif.
        retry_num: Final images are in which of "retry_???" directory. Passing 0 means
            that the images are not in "retry_???" directory.
        workspace: the name of the workspace used to create the job.
    """
    logging.basicConfig()

    ml_client = get_client(workspace_name=workspace)

    with tempfile.TemporaryDirectory() as download_path:
        ml_client.jobs.download(
            name=job_name.lower(),
            download_path=download_path,
            all=True,
        )

        data_folder = Path(download_path) / "artifacts/outputs"
        if retry_num > 0:
            data_folder = data_folder / f"retry_{retry_num:03}"

        create_gif(
            output_folder=data_folder,
            save_path=save_path,
        )


if __name__ == "__main__":
    tyro.cli(create_gif_from_output)
