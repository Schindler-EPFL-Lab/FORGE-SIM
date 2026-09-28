"""
Renames a VGGT-style dataset's image and depth files with `train_`/`eval_` prefixes
(80/20 split by default) and updates `transforms.json` to reference the new names.

Frames are assigned to the training split via a `random.seed(42)`-seeded shuffle, so
the split is deterministic across runs.

Validation performed before writing anything to disk:
1. `dataset_path` exists and `train_ratio` is in `(0, 1)`.
2. `transforms.json` exists and `images/` contains at least one `.jpeg` image.
3. No image is already prefixed, so a second run cannot prefix it twice and reshuffle
   the split underneath it.
4. Every planned rename source (image and depth file) exists, and no planned rename
   target already exists.
5. Every `rgb.file_path` referenced in `transforms.json` corresponds to a known frame.

The script edits `transforms.json` in place and performs the file renames last, once
every check above has passed, so a validation failure cannot leave the dataset
half-renamed.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

from forge_sim import logger

SPLIT_PREFIXES = ("train_", "eval_")
"""The prefixes this script puts in front of a frame name, one per split."""


@dataclass
class RenameDatasetParams(ReverseCli):
    """
    Parameters for renaming a VGGT-style dataset's image/depth files with
    `train_`/`eval_` prefixes and updating `transforms.json` to match.
    """

    dataset_path: Path = Path("dataset/buildinga_spring_vggt")
    """Path to the dataset directory."""

    train_ratio: float = 0.8
    """Ratio of frames assigned to the training split, must be in `(0, 1)`."""

    def __post_init__(self) -> None:
        if not self.dataset_path.exists() or not self.dataset_path.is_dir():
            raise ValueError(
                f"`dataset_path` {self.dataset_path} does not exist or is not a "
                "directory."
            )
        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError(
                f"`train_ratio` must be in (0, 1), got {self.train_ratio}."
            )

    @property
    def image_dir(self) -> Path:
        """Directory containing image files."""
        return self.dataset_path / "images"

    @property
    def depth_dir(self) -> Path:
        """Directory containing depth files."""
        return self.dataset_path / "depths"

    @property
    def transforms_file(self) -> Path:
        """Path to the `transforms.json` file."""
        return self.dataset_path / "transforms.json"


def _compute_train_frames(frame_names: list[str], train_ratio: float) -> set[str]:
    """
    Deterministically assigns a subset of `frame_names` to the training split.

    Shuffles a copy of `frame_names` with `random.seed(42)`, then takes the first
    `round(len(frame_names) * train_ratio)` names as the training split.

    :returns: The set of frame names assigned to the training split.
    """
    shuffled_frame_names = list(frame_names)
    random.seed(42)
    random.shuffle(shuffled_frame_names)
    split_idx = int(len(shuffled_frame_names) * train_ratio)
    return set(shuffled_frame_names[:split_idx])


def _build_rename_plan(
    frame_names: list[str],
    train_frames: set[str],
    directory: Path,
    suffix: str,
) -> list[tuple[Path, Path]]:
    """
    Builds the list of `(old_path, new_path)` renames for every frame in
    `frame_names`, located under `directory` with file extension `suffix` (e.g.
    `.jpeg` or `.npy`). Frames present in `train_frames` are prefixed with
    `train_`, the rest with `eval_`.

    :returns: The list of `(old_path, new_path)` pairs, in `frame_names` order.
    """
    renames: list[tuple[Path, Path]] = []
    for frame in frame_names:
        prefix = "train_" if frame in train_frames else "eval_"
        renames.append(
            (directory / f"{frame}{suffix}", directory / f"{prefix}{frame}{suffix}")
        )
    return renames


def _reject_already_renamed(frame_names: list[str], image_dir: Path) -> None:
    """
    Refuses to run on a dataset that has already been renamed.

    A second run would treat the prefixed names as frame names of their own, reshuffle
    the split over them, and produce `eval_eval_<frame>` / `train_train_<frame>`, since
    every check in this script passes for a plan built on the wrong names. `frame_names`
    are the stems found in `image_dir`, which is named in the error.

    :raises ValueError: If any of `frame_names` already begins with a split prefix.
    """
    already_renamed = [name for name in frame_names if name.startswith(SPLIT_PREFIXES)]
    if not already_renamed:
        return

    listed = ", ".join(already_renamed[:5])
    if len(already_renamed) > 5:
        listed += f", ... ({len(already_renamed)} in total)"
    raise ValueError(
        f"{image_dir} has already been renamed: {listed} already carry a "
        f"{' / '.join(SPLIT_PREFIXES)} prefix. Renaming again would prefix them a "
        "second time and reshuffle the train/eval split; nothing has been changed."
    )


def _validate_rename_plan(renames: list[tuple[Path, Path]]) -> None:
    """
    Checks that every source path in `renames` exists and that no target path
    already exists, without renaming anything. Meant to be called on every planned
    rename before any of them are performed.

    :raises FileNotFoundError: If a planned rename source does not exist.
    :raises FileExistsError: If a planned rename target already exists.
    """
    for old_path, new_path in renames:
        if not old_path.exists():
            raise FileNotFoundError(
                f"Expected file {old_path} to exist before renaming."
            )
        if new_path != old_path and new_path.exists():
            raise FileExistsError(
                f"Rename target {new_path} already exists; refusing to overwrite it."
            )


def rename_dataset(params: RenameDatasetParams) -> None:
    """
    Renames image and depth files in `params.dataset_path` with `train_`/`eval_`
    prefixes according to `params.train_ratio`, and rewrites
    `params.transforms_file` to reference the new filenames.

    The whole rename plan (images, depth files, and the `transforms.json` frame
    references) is validated before any file on disk is renamed or overwritten, so
    a validation failure never leaves the dataset partially renamed.

    :raises ValueError: If `params.transforms_file` does not exist, if
        `params.image_dir` contains no `.jpeg` images, if it has already been renamed,
        or if `transforms.json` references a frame with no corresponding image.
    :raises FileNotFoundError: If a planned rename source (image or depth file)
        does not exist.
    :raises FileExistsError: If a planned rename target already exists.
    """
    if not params.transforms_file.exists():
        raise ValueError(f"`{params.transforms_file}` does not exist.")

    image_files = sorted(f for f in params.image_dir.iterdir() if f.suffix == ".jpeg")
    if not image_files:
        raise ValueError(f"No `.jpeg` images found in {params.image_dir}.")

    frame_names = [f.stem for f in image_files]
    _reject_already_renamed(frame_names, params.image_dir)

    train_frames = _compute_train_frames(frame_names, params.train_ratio)

    image_renames = _build_rename_plan(
        frame_names, train_frames, params.image_dir, ".jpeg"
    )
    depth_renames = _build_rename_plan(
        frame_names, train_frames, params.depth_dir, ".npy"
    )

    # Validate every planned rename before touching the filesystem, so a missing
    # or colliding file cannot leave the dataset half-renamed.
    _validate_rename_plan(image_renames)
    _validate_rename_plan(depth_renames)

    with open(params.transforms_file, "r") as file:
        transforms_data = json.load(file)

    prefix_by_frame = {
        frame: ("train_" if frame in train_frames else "eval_") for frame in frame_names
    }
    for frame_data in transforms_data.get("frames", []):
        rgb = frame_data.get("rgb")
        if rgb is None:
            continue

        frame_name = Path(rgb["file_path"]).stem
        if frame_name not in prefix_by_frame:
            raise ValueError(
                f"`transforms.json` references frame {frame_name!r} which has no "
                f"corresponding image in {params.image_dir}."
            )

        prefix = prefix_by_frame[frame_name]
        rgb["file_path"] = f"images/{prefix}{frame_name}.jpeg"
        if "depth_file_path" in rgb:
            rgb["depth_file_path"] = f"depths/{prefix}{frame_name}.npy"

    # Every check has passed - perform the renames, then persist transforms.json.
    for old_path, new_path in image_renames + depth_renames:
        old_path.rename(new_path)

    with open(params.transforms_file, "w") as file:
        json.dump(transforms_data, file, indent=4)

    logger.info(
        f"Renamed {len(image_renames)} image(s) and {len(depth_renames)} depth "
        f"file(s) in {params.dataset_path} "
        f"({len(train_frames)} train / {len(frame_names) - len(train_frames)} eval)."
    )


if __name__ == "__main__":
    parameters = tyro.cli(RenameDatasetParams)
    rename_dataset(params=parameters)
