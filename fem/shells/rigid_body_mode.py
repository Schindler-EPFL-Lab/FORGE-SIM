from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, kw_only=True)
class RigidBodyMode:
    """
    One analytic rigid-body field: a uniform translation or an infinitesimal
    rotation about one Cartesian axis.

    Used by
    :meth:`fem.shells.geometrically_exact_shell_modal_solver.GeometricallyExactShellModalSolver.rigid_body_energies`
    to probe each of the six zero-energy motions of a closed shell directly,
    which isolates the strain measures from the eigensolver. Replaces a pair of
    factory closures that each captured an axis: the DOLFINx interpolation
    callback a caller needs is `displacement`, a bound method rather than a
    closure, so no state is captured anywhere but on `self`.
    """

    class Kind(Enum):
        """Whether a mode is a translation or an infinitesimal rotation."""

        TRANSLATION = "translation"
        ROTATION = "rotation"

    kind: Kind
    """Whether this mode is a translation or a rotation."""
    axis: npt.NDArray[np.float64]
    """The translation direction, or the rotation axis, shape `(3,)`."""
    label: str
    """Short tag (`"Tx"`, ..., `"Rz"`) this mode is reported under."""

    @property
    def rotation(self) -> npt.NDArray[np.float64]:
        """
        The constant rotation field `omega` interpolated onto the rotation
        DOFs: `axis` for a rotation mode, zero for a translation.
        """
        if self.kind is RigidBodyMode.Kind.ROTATION:
            return self.axis
        return np.zeros(3)

    def displacement(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        DOLFINx interpolation callback for the displacement field of this mode.

        `x` is the point coordinates, shape `(gdim, n)`.

        :returns: The field values at those points, shape `(vdim, n)`: a
            uniform translation along `axis` for a translation mode, or
            `axis x point` for a rotation mode.
        """
        if self.kind is RigidBodyMode.Kind.TRANSLATION:
            return np.tile(self.axis[:, None], (1, x.shape[1]))
        return np.cross(self.axis, x.T).T

    @staticmethod
    def standard_basis() -> tuple["RigidBodyMode", ...]:
        """
        The six canonical rigid-body modes of a free-free shell.

        :returns: Translation then rotation about each of x, y, z, in that
            order.
        """
        axes = np.eye(3)
        tags = ("x", "y", "z")
        translations = tuple(
            RigidBodyMode(kind=RigidBodyMode.Kind.TRANSLATION, axis=axes[i],
                         label=f"T{tags[i]}")
            for i in range(3)
        )
        rotations = tuple(
            RigidBodyMode(kind=RigidBodyMode.Kind.ROTATION, axis=axes[i],
                         label=f"R{tags[i]}")
            for i in range(3)
        )
        return translations + rotations
