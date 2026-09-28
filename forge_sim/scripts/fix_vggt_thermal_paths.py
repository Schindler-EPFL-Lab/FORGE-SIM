"""
Repair utility for VGGT datasets missing `thermal_file_path` entries.

VGGT scene exports store camera and RGB metadata under `frame["rgb"]` entries
in `transforms.json`. Thermal images are expected to be located in a `thermal/`
folder and referenced by `thermal_file_path` within the same `rgb` dict.

For each frame, the script resolves the thermal image by matching the RGB image
stem from `frame["rgb"]["file_path"]` with exactly one thermal image file in the
chosen thermal folder.

Validation performed before writing:
1. Scene path exists and is a directory.
2. `transforms.json` exists and has a valid VGGT-like structure.
3. Thermal folder exists and contains supported image files.
4. Every frame has a valid `rgb.file_path`.
5. Every frame has a valid `rgb.thermal_file_path` after repair, and every
   referenced thermal file exists on disk.

The script edits `transforms.json` in place.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

from forge_sim import logger

SUPPORTED_IMAGE_EXTENSIONS = (
    ".JPG",
    ".jpg",
    ".JPEG",
    ".jpeg",
    ".PNG",
    ".png",
    ".TIF",
    ".tif",
    ".TIFF",
    ".tiff",
)
THERMAL_KEY = "thermal_file_path"


@dataclass
class FixVggtThermalPathsParams(ReverseCli):
    """
    Parameters for repairing missing `thermal_file_path` entries in a VGGT dataset.

    Expected scene structure:
    ```
    scene_path/
    ├── transforms.json
    ├── images*/
    └── thermal/
    ```

    The script updates `scene_path/transforms.json` in place.
    """

    scene_path: Path
    """Path to a VGGT scene directory containing `transforms.json`."""

    thermal_folder: str = "thermal"
    """Folder name containing thermal images (relative to `scene_path`)."""

    def __post_init__(self) -> None:
        if not self.scene_path.exists() or not self.scene_path.is_dir():
            raise ValueError(
                f"The provided scene_path {self.scene_path} does not exist or "
                "is not a directory."
            )

        transforms_path = self.scene_path / "transforms.json"
        if not transforms_path.exists() or not transforms_path.is_file():
            raise ValueError(f"The file {transforms_path} does not exist.")

        thermal_path = self.scene_path / self.thermal_folder
        if not thermal_path.exists() or not thermal_path.is_dir():
            raise ValueError(
                "Thermal folder does not exist. Expected a directory at "
                f"{thermal_path}."
            )


def _load_image_paths_from_folder(folder: Path) -> dict[str, Path]:
    """Load supported image files from a folder as `filename -> absolute_path`."""
    image_filenames = sorted(
        [
            image.name
            for image in folder.iterdir()
            if image.is_file() and image.suffix in SUPPORTED_IMAGE_EXTENSIONS
        ]
    )
    return {image_name: folder / image_name for image_name in image_filenames}


def _validate_transforms_structure(transforms_data: dict) -> None:
    """Validate the minimum structure expected for VGGT `transforms.json`."""
    if not isinstance(transforms_data, dict):
        raise ValueError("transforms.json must contain a JSON object at top-level.")

    frames = transforms_data.get("frames")
    if not isinstance(frames, list):
        raise ValueError("`frames` must be present in transforms.json as a list.")

    for i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"Frame at index {i} must be a JSON object.")
        if "rgb" not in frame or not isinstance(frame["rgb"], dict):
            raise ValueError(f"Frame at index {i} is missing a `rgb` object.")
        if "file_path" not in frame["rgb"]:
            raise ValueError(
                f"Frame at index {i} is missing required key `rgb.file_path`."
            )
        rgb_path = frame["rgb"]["file_path"]
        if not isinstance(rgb_path, str) or not rgb_path.strip():
            raise ValueError(
                f"Frame at index {i} has invalid `rgb.file_path` value: {rgb_path!r}."
            )


def _build_candidates_by_stem(image_paths: dict[str, Path]) -> dict[str, list[str]]:
    """Build `stem -> [filename, ...]` map for deterministic resolution."""
    candidates: dict[str, list[str]] = {}
    for filename in sorted(image_paths.keys()):
        stem = Path(filename).stem
        candidates.setdefault(stem, []).append(filename)
    return candidates


def fix_vggt_thermal_paths(params: FixVggtThermalPathsParams) -> int:
    """
    Repair missing `thermal_file_path` entries in a VGGT `transforms.json`.

    For each frame without `thermal_file_path`, the script resolves a thermal image
    by matching the stem of `frame["rgb"]["file_path"]` against filenames in the
    thermal folder. Exactly one match is required.

    Existing `thermal_file_path` values are preserved. After repair, the function
    validates that every frame has `thermal_file_path` and that each referenced
    file exists.

    :returns: Number of frames that were updated.
    :raises ValueError: If transforms structure is invalid or stem matching is
        ambiguous/missing.
    :raises FileNotFoundError: If referenced thermal files are missing.
    """
    transforms_path = params.scene_path / "transforms.json"

    with open(transforms_path, "r", encoding="utf-8") as file:
        transforms_data = json.load(file)

    _validate_transforms_structure(transforms_data)

    thermal_folder = params.scene_path / params.thermal_folder
    thermal_paths = _load_image_paths_from_folder(thermal_folder)
    if not thermal_paths:
        raise ValueError(
            f"Thermal folder {thermal_folder} does not contain any supported images."
        )

    thermal_candidates_by_stem = _build_candidates_by_stem(thermal_paths)

    updated_frames_count = 0
    frames = transforms_data["frames"]

    for frame_idx, frame in enumerate(frames):
        rgb_frame = frame["rgb"]
        if THERMAL_KEY in rgb_frame:
            continue

        rgb_filename_stem = Path(rgb_frame["file_path"]).stem
        candidate_thermal_filenames = thermal_candidates_by_stem.get(
            rgb_filename_stem, []
        )

        if len(candidate_thermal_filenames) == 0:
            raise FileNotFoundError(
                "Cannot resolve a thermal image for frame index "
                f"{frame_idx} with rgb.file_path={rgb_frame['file_path']!r}. "
                f"No file with stem {rgb_filename_stem!r} found in {thermal_folder}."
            )
        if len(candidate_thermal_filenames) > 1:
            raise ValueError(
                "Ambiguous thermal resolution for frame index "
                f"{frame_idx} with rgb.file_path={rgb_frame['file_path']!r}. "
                f"Found multiple matches: {candidate_thermal_filenames}."
            )

        resolved_filename = candidate_thermal_filenames[0]
        relative_thermal_path = f"{params.thermal_folder}/{resolved_filename}"
        absolute_thermal_path = params.scene_path / relative_thermal_path

        if not absolute_thermal_path.exists() or not absolute_thermal_path.is_file():
            raise FileNotFoundError(
                f"Resolved thermal path does not exist: {absolute_thermal_path}"
            )

        rgb_frame[THERMAL_KEY] = relative_thermal_path
        updated_frames_count += 1

    # Validate all entries after patching to fail fast on stale/malformed values.
    for frame_idx, frame in enumerate(frames):
        rgb_frame = frame["rgb"]
        if THERMAL_KEY not in rgb_frame:
            raise ValueError(
                f"Frame at index {frame_idx} still has no `{THERMAL_KEY}` after repair."
            )

        thermal_path = rgb_frame[THERMAL_KEY]
        if not isinstance(thermal_path, str) or not thermal_path.strip():
            raise ValueError(
                f"Frame at index {frame_idx} has invalid `{THERMAL_KEY}` value: "
                f"{thermal_path!r}."
            )

        absolute_thermal_path = params.scene_path / thermal_path
        if not absolute_thermal_path.exists() or not absolute_thermal_path.is_file():
            raise FileNotFoundError(
                "Frame index "
                f"{frame_idx} references a non-existing thermal file: "
                f"{absolute_thermal_path}"
            )

    with open(transforms_path, "w", encoding="utf-8") as file:
        json.dump(transforms_data, file, indent=4)

    logger.info(
        f"Repair complete for {transforms_path}. "
        f"Added `{THERMAL_KEY}` to {updated_frames_count} frame(s)."
    )

    return updated_frames_count


if __name__ == "__main__":
    parameters = tyro.cli(FixVggtThermalPathsParams)
    fix_vggt_thermal_paths(params=parameters)
