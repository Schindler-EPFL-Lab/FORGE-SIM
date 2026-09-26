from dataclasses import astuple, dataclass
from pathlib import Path


@dataclass
class SplineTexturePaths:
    """Paths to a reconstructed spline's six cube-face textures."""

    back: Path
    """Path to the .exr back texture."""
    bottom: Path
    """Path to the .exr bottom texture."""
    front: Path
    """Path to the .exr front texture."""
    left: Path
    """Path to the .exr left texture."""
    right: Path
    """Path to the .exr right texture."""
    top: Path
    """Path to the .exr top texture."""

    def as_list_str(self) -> list[str]:
        """The six paths, in field order, each converted to `str`."""
        return [str(path) for path in astuple(self)]

    @classmethod
    def discover_all(cls, directory: Path) -> list["SplineTexturePaths"]:
        """
        Every reconstruction's `SplineTexturePaths` found directly under
        `directory`, glob-matched by face name and paired by sort order.

        A directory holding N reconstructions carries N files whose name
        contains `back`, N containing `bottom`, and so on for the other
        four faces; sorting each face's matches and zipping across faces pairs
        the Nth path of each face together, which is the convention every
        spline reconstruction run writes its texture filenames to satisfy.

        :raises ValueError: If the faces do not all match the same number of files,
            since pairing by sort order would then put one run's face beside
            another run's.
        """
        faces = {
            face: sorted(directory.glob(f"*{face}*.exr"))
            for face in ("back", "bottom", "front", "left", "right", "top")
        }
        counts = {face: len(paths) for face, paths in faces.items()}
        if len(set(counts.values())) > 1:
            raise ValueError(
                f"The cube-face textures in {directory} cannot be paired by sort "
                f"order: the faces match different numbers of files, {counts}."
            )
        return [
            cls(back=back, bottom=bottom, front=front, left=left, right=right, top=top)
            for back, bottom, front, left, right, top in zip(
                faces["back"],
                faces["bottom"],
                faces["front"],
                faces["left"],
                faces["right"],
                faces["top"],
            )
        ]
