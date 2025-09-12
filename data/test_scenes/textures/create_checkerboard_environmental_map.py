"""
A script to create a checkerboard environmental map and save it
"""

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tyro


def create_black_environmental_map(
    save_path: Path,
    resolution: tuple[int, int] = (1024, 2048),
    ratios: tuple[float, float] = (0.005, 0.0025),
    values: tuple[float, float] = (0.0, 2.0),
) -> None:
    """
    Creates checkerboard environmental map with `resolution`, where each cell ratio are
    `ratios`, and value of each "black" and "white" are `values`. It stores the
    result at `save_path`.

    :raises ValueError
        if parent of save_path does not exist or
        if the extension of save_path is not .exr or
        if values[0] > values[1]
        if ratios are too small (s.t. resolution[i] * ratios[i] < 1)
    """
    if not save_path.parent.exists():
        raise ValueError(f"Parent of save_path {save_path} does not exist")
    if save_path.suffix != ".exr":
        raise ValueError(f"Extension of save_path {save_path} is not .exr")
    if values[0] > values[1]:
        raise ValueError(
            f"Minimum should not be greater than maximum but get values = {values}"
        )
    if resolution[0] * ratios[0] < 1:
        raise ValueError(
            "The first ratio is too small, resolution[0] * ratios[0] = "
            + f"{resolution[0] * ratios[0]} < 1"
        )
    if resolution[1] * ratios[1] < 1:
        raise ValueError(
            "The second ratio is too small, resolution[1] * ratios[1] = "
            + f"{resolution[1] * ratios[1]} < 1"
        )

    image = np.empty(resolution, dtype=np.float32)
    for i in range(resolution[0]):
        for j in range(resolution[1]):
            dim_0_is_even = int(i / (resolution[0] * ratios[0])) % 2 == 0
            dim_1_is_even = int(j / (resolution[1] * ratios[1])) % 2 == 0
            image[i, j] = values[0] if dim_0_is_even == dim_1_is_even else values[1]
    image = image[:, :, None].repeat(3, axis=2)

    iio.imwrite(save_path, image)


if __name__ == "__main__":
    params = tyro.cli(create_black_environmental_map)
