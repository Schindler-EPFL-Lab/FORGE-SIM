"""
A script to create a black or white environmental map and save it
"""

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli


@dataclass
class CreateEnvironmentalMapConfig(ReverseCli):
    """
    Configuration for creation of an environmental map with resolution `resolution` and
    to save it at `save_path`

    :raises ValueError if parent of save_path does not exist or
        if the extension of save_path is not .exr
    """

    save_path: Path
    """Where to store the output path of the file"""

    black: bool = True
    """black if true, white otherwise"""

    resolution: tuple[int, int] = (1024, 2048)
    """Resolution of the environmental map"""

    def __post_init__(self):
        if not self.save_path.parent.exists():
            raise ValueError(f"Parent of save_path {self.save_path} does not exist")
        if self.save_path.suffix != ".exr":
            raise ValueError(f"Extension of save_path {self.save_path} is not .exr")


def create_environmental_map(params: CreateEnvironmentalMapConfig):
    """
    Creates a black environmental map with resolution `params.resolution` and
    store it at `params.save_path`
    """
    if params.black:
        image = np.zeros(params.resolution, dtype=np.float32)
    else:
        image = np.ones(params.resolution, dtype=np.float32)
    iio.imwrite(params.save_path, image)


if __name__ == "__main__":
    params = tyro.cli(CreateEnvironmentalMapConfig)
    create_environmental_map(params)
