import json
from pathlib import Path

import mitsuba as mi
import numpy as np
from scipy.spatial.transform import Rotation


def load_bounding_bbox_transforms(bbox_transforms_path: Path) -> "mi.Transform4f":
    """
    Creates a 4x4 transformation matrix based on scale, translation and rotation
    values from JSON file `bbox_transforms_path`.

    :returns
        a 4x4 tranformation matrix from scale, translation and rotation.

    :raises
        RuntimeError: If any of the required keys ("scale", "translation", "rotation")
        are missing in the file.
    """

    with open(bbox_transforms_path, "r") as f:
        bbox_transforms = json.load(f)

    required_keys = {"scale", "translation", "rotation"}
    if not required_keys.issubset(bbox_transforms):
        raise RuntimeError(
            f"Missing required keys in bbox_transforms: "
            f"{required_keys - set(bbox_transforms)}"
        )

    tf = np.eye(4)
    tf[:3, :3] = np.diag(bbox_transforms["scale"])
    tf[:3, :3] = (
        Rotation.from_euler(
            "xyz", bbox_transforms["rotation"], degrees=True
        ).as_matrix()
        @ tf[:3, :3]
    )
    tf[:3, 3] = bbox_transforms["translation"]
    return mi.Transform4f(tf)
