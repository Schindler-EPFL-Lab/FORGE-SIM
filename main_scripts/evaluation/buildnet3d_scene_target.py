from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from main_scripts.evaluation.buildnet3d_scene_id import BuildNet3DSceneID


@dataclass(frozen=True, kw_only=True)
class BuildNet3DSceneTarget:
    """
    What a `BuildNet3DSceneID` names: the scene's directory under the dataset
    root, and the ground-truth mesh its Chamfer step reads against.

    `SCENE_TARGETS` replaces the two parallel, index-matched dicts the
    original `eval_all_buildnet3d*` scripts each carried: one id-keyed
    table for the dataset directory, a second for the ground-truth mesh name,
    edited in lockstep to stay consistent.
    """

    SCENE_TARGETS: ClassVar[dict[BuildNet3DSceneID, "BuildNet3DSceneTarget"]]
    """The target every registered `BuildNet3DSceneID` resolves to."""

    directory: Path
    """
    Scene's directory name, relative to the dataset root. Used doubled, as
    `<dataset_root>/<directory>/<directory>`, for the thermoscenes path, and
    singly for the splines/large-steps reconstruction folder.
    """
    ground_truth_mesh_filename: Path
    """Filename of the ground-truth mesh under `BuildNet3DCorrectedMeshes/`."""

    @classmethod
    def for_id(cls, scene_id: BuildNet3DSceneID) -> "BuildNet3DSceneTarget":
        """
        The registered target for `scene_id`.

        :raises ValueError: If `scene_id` has no entry in `SCENE_TARGETS`.
            Every `BuildNet3DSceneID` member is registered, so this signals
            the registry has fallen out of sync with the enum, not a bad
            `scene_id`.
        """
        target = cls.SCENE_TARGETS.get(scene_id)
        if target is None:
            raise ValueError(
                f"No BuildNet3DSceneTarget registered for {scene_id.name}. "
                f"Registered scenes: "
                f"{sorted(member.name for member in cls.SCENE_TARGETS)}."
            )
        return target


BuildNet3DSceneTarget.SCENE_TARGETS = {
    BuildNet3DSceneID.SCENE_1: BuildNet3DSceneTarget(
        directory=Path("extracted-buildnet3d-realistic-1"),
        ground_truth_mesh_filename=Path("building1.ply"),
    ),
    BuildNet3DSceneID.SCENE_2: BuildNet3DSceneTarget(
        directory=Path("extracted-buildnet3d-realistic-2"),
        ground_truth_mesh_filename=Path("building2.ply"),
    ),
    BuildNet3DSceneID.SCENE_3: BuildNet3DSceneTarget(
        directory=Path("extracted-buildnet3d-realistic-3"),
        ground_truth_mesh_filename=Path("building3.ply"),
    ),
    BuildNet3DSceneID.SCENE_4: BuildNet3DSceneTarget(
        directory=Path("extracted-buildnet3d-realistic-4"),
        ground_truth_mesh_filename=Path("building4.ply"),
    ),
}
