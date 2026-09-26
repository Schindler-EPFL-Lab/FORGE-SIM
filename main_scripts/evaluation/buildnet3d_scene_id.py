from enum import Enum


class BuildNet3DSceneID(Enum):
    """
    One of the four BuildNet3D realistic scenes the shape/NVS sweeps evaluate,
    by identity alone.

    `BuildNet3DSceneTarget.SCENE_TARGETS` keys off it to resolve the dataset
    directory and the ground-truth mesh a scene id names. Distinct from
    `main_scripts.scenes.vggt_scene.VGGTScene`'s own `BUILDNET_1` through
    `BUILDNET_4`, which register the same four scenes under a different
    naming convention for the heat and modal comparisons.

    `value` is the digit these sweeps have always taken as a
    `--scenes-ids`/`--scenes-names` entry, and does not move.
    """

    SCENE_1 = "1"
    """BuildNet3D synthetic facade 1."""
    SCENE_2 = "2"
    """BuildNet3D synthetic facade 2."""
    SCENE_3 = "3"
    """BuildNet3D synthetic facade 3."""
    SCENE_4 = "4"
    """BuildNet3D synthetic facade 4."""
