import json
from pathlib import Path
from typing import ClassVar

from forge_sim import logger


class VGGTSceneLayout:
    """
    The on-disk conventions of one VGGT scene directory.

    `__init__` stores the directory and does no I/O; `reconstruction` and
    `has_facade_mask` each do their own lookup when asked, so a caller that wants
    the answer fixed across several reads (`ScenePipelineRun`'s facade-mask
    check, resolved once at construction) reads the property once and holds
    the result itself.
    """

    FACADE_MASK_KEYS: ClassVar[tuple[str, str, str]] = (
        "windows_mask_file_path",
        "masks_facade",
        "masks_facades",
    )
    """
    The keys under which a `transforms.json` frame may reference its facade mask.

    Three spellings are in circulation because the VGGT, ThermoScenes and BuildNet
    exports each chose their own; a scene is taken to have a mask if any frame
    carries any of them.
    """

    def __init__(self, scene_directory: Path) -> None:
        """
        Stores `scene_directory`; no I/O happens until a property is read.
        """
        self._scene_directory = scene_directory

    def _reconstruction_candidates(self) -> list[Path]:
        """
        Every `shape_*.json` under `result(s)/[ours/]`, in glob order.

        Searches the four shapes the v2 / v3 layouts use:
            scene/results/shape_*.json
            scene/result/shape_*.json
            scene/results/ours/shape_*.json
            scene/result/ours/shape_*.json
        """
        candidates: list[Path] = []
        for sub in ("results", "result"):
            d = self._scene_directory / sub
            if d.is_dir():
                candidates.extend(d.glob("shape_*.json"))
                ours = d / "ours"
                if ours.is_dir():
                    candidates.extend(ours.glob("shape_*.json"))
        return candidates

    @property
    def reconstruction(self) -> Path:
        """
        Latest `shape_*.json` under `result(s)/[ours/]`.

        :raises FileNotFoundError: If none of the four layouts holds a `shape_*.json`.
        """
        candidates = self._reconstruction_candidates()
        if not candidates:
            raise FileNotFoundError(
                f"No `shape_*.json` reconstruction found under "
                f"{self._scene_directory}/result(s)/[ours/]"
            )
        return max(candidates, key=lambda p: p.stat().st_mtime)

    @property
    def has_reconstruction(self) -> bool:
        """
        Whether this scene directory holds a reconstruction `reconstruction` can return.

        Auto-detection asks this of every candidate directory, most of which are
        not scenes at all, so the absence of one is an ordinary answer here rather
        than the error it is when a named scene is asked for its reconstruction.
        """
        return bool(self._reconstruction_candidates())

    @property
    def has_facade_mask(self) -> bool:
        """
        Whether the scene's `transforms.json` references a facade mask in at
        least one frame (not necessarily the first; partial coverage is fine).

        A missing or unparseable `transforms.json` reads as no mask, so a scene
        the reconstruction stage has not reached yet falls back to the
        single-material path rather than failing here.
        """
        transforms_path = self._scene_directory / "dataset" / "transforms.json"
        if not transforms_path.is_file():
            return False
        try:
            data = json.loads(transforms_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.debug(
                f"{transforms_path} is not valid JSON; reading as no facade mask."
            )
            return False

        for frame in data.get("frames", []):
            if not isinstance(frame, dict):
                continue
            # VGGT layout nests inside frame["rgb"]; ThermoScenes layout is flat.
            rgb = frame.get("rgb")
            entry = rgb if isinstance(rgb, dict) else frame
            if any(key in entry for key in self.FACADE_MASK_KEYS):
                return True
        return False
