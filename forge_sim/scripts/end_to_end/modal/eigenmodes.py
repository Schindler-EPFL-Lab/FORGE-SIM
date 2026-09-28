from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, kw_only=True)
class Eigenmodes:
    """
    A modal run's mode shapes on their own undeformed surface.

    Both solvers export the same HDF5 layout: one undeformed mesh under `/Mesh`,
    and per-mode `displacement`/`rotation` fields under `/Modes/mode_<k>`. The
    modes are held in the file's own order, which is frequency rank: the ordering
    the comparison pairs by.
    """

    vertices: npt.NDArray[np.float64]
    """Undeformed vertex positions, shape `(n_v, 3)`."""
    faces: npt.NDArray[np.int64]
    """Triangle vertex indices, shape `(n_f, 3)`."""
    displacement: npt.NDArray[np.float64]
    """Per-mode translational mode shape, shape `(n_modes, n_v, 3)`."""
    rotation: npt.NDArray[np.float64]
    """Per-mode rotational mode shape, shape `(n_modes, n_v, 3)`."""

    @classmethod
    def from_h5(cls, h5_path: Path) -> "Eigenmodes":
        """
        Read an `eigenmodes.h5` written by either the FEM or the IGA exporter.

        Modes are ordered by the integer in their `mode_<k>` group name rather than
        lexically, so mode 10 follows mode 9 instead of mode 1.
        """
        with h5py.File(h5_path, "r") as f:
            keys = sorted(
                (k for k in f["/Modes"].keys() if k.startswith("mode_")),
                key=lambda s: int(s.split("_")[1]),
            )
            return cls(
                vertices=np.asarray(f["/Mesh/vertices"], dtype=np.float64),
                faces=np.asarray(f["/Mesh/topology"], dtype=np.int64),
                displacement=np.stack([f[f"/Modes/{k}/displacement"][:] for k in keys]),
                rotation=np.stack([f[f"/Modes/{k}/rotation"][:] for k in keys]),
            )

    @property
    def num_modes(self) -> int:
        """How many modes the file carried."""
        return self.displacement.shape[0]
