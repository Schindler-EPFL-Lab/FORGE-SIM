from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True, kw_only=True)
class ShellElement:
    """Finite-element / integration choice for the shell."""

    class LockingCure(Enum):
        """
        Treatment of the membrane and transverse-shear stiffness.

        The three are mutually exclusive, which is what makes them an enum: the
        pair of booleans this replaced could be set together, and the precedence
        rule that resolved it silently ignored one of them. Each member's value
        is the tag it contributes to :meth:`ShellElement.label`.

        PSRI down-weights the spurious thin-limit membrane/shear constraints by
        `alpha=(t/h)^2` rather than removing them, so it relieves locking only
        partially. BBAR removes them exactly instead, which is why it is the stronger,
        parameter-free cure: it restores better-than-O(h^2) eigenvalue convergence at P3
        on thin bending-dominated plates, where FULL integration locks (rate ~0) and
        PSRI only partially relieves it.
        """

        FULL = "full"
        """
        No locking cure: both terms are integrated at the full quadrature
        degree. It locks on thin bending-dominated geometry (at P3 on a thin
        plate the eigenvalue convergence rate falls to ~0), and is kept as the
        uncured reference the two cures are measured against on a
        membrane-dominated problem, where all three must agree.
        """

        PSRI = "PSRI"
        """
        Partial selective reduced integration (fenics-shells style): membrane
        and shear are each split `alpha*full + (1-alpha)*reduced` with
        `alpha = (t/h_cell)^2` and the order-aware reduced quadrature degree
        of `reduced_quad`. It down-weights the spurious thin-limit
        constraints rather than removing them, so it relieves locking only
        partially.
        """

        BBAR = "Bbar"
        """
        B-bar: element-local L2 projection of the membrane and transverse-shear strains
        onto a per-cell DG_{degree-1} space (assumed-strain method, Hu et al.
        arXiv:1709.00402). Bending is left fully integrated (projecting it would cause
        rank deficiency). The projection is a symmetric orthogonal projector, so the
        stiffness stays symmetric PSD with the correct 6-dim rigid-body nullspace
        (required by the shift-invert solver).
        """

    class NormalKind(Enum):
        """
        Source of the surface normal the strain measures are built from.

        Each member's value is the tag it contributes to
        :meth:`ShellElement.label`.
        """

        GEOMETRIC = "geometric"
        """
        From the Jacobian of the element map. General, and consistent with
        the discrete geometry at every mesh order.
        """

        ANALYTIC_SPHERE = "analytic_sphere"
        """
        `x/|x|`: valid only on curved geometry that really lies on a
        sphere centred at the origin. A validation aid; on a flat (order-1)
        mesh it is inconsistent with the discrete surface gradient and corrupts
        the rigid-body modes, which the constructor warns about.
        """

    degree: int = 2
    """Lagrange degree of the displacement and rotation fields."""

    locking_cure: LockingCure = LockingCure.PSRI
    """Locking cure applied to the membrane and transverse-shear stiffness."""

    reduced_quad: int | None = None
    """
    Quadrature degree of the PSRI reduced term; `None` selects the
    order-aware default 2*(degree-1).
    """

    drilling: bool = True
    """Add the drilling stabilization stiffness."""

    drilling_stiffness_factor: float = 1.0
    """Scale on the drilling stabilization stiffness."""

    project_rotary_inertia: bool = True
    """
    Zero the inertia of the drilling rotation (theta.n), matching the IGA's (I - D(x)D)
    projection. This sends drilling modes to +inf regardless of mesh, so they cannot
    intrude into the physical spectrum on graded/complex meshes. Unprojected, they
    instead sit at a finite, mesh-size-dependent frequency ~ (1/h_cell) sqrt(12 E / rho)
    that can drop into the spectrum on large elements, a risk a uniform sphere mesh
    hides. Strongly recommended True for complex geometry.
    """

    normal: NormalKind = NormalKind.GEOMETRIC
    """Source of the surface normal the strain measures are built from."""

    def label(self) -> str:
        """
        Short human-readable tag of this element choice, for run logs.

        :returns: A string of the form `"P2/P2+Bbar+geometric"`, naming the
            displacement/rotation degrees, the locking cure and the normal.
        """
        return (
            f"P{self.degree}/P{self.degree}"
            f"+{self.locking_cure.value}+{self.normal.value}"
        )

    @classmethod
    def locking_cure_from_flags(
        cls, use_bbar: bool, use_psri: bool
    ) -> ShellElement.LockingCure:
        """
        Translates the two boolean flags the entry points expose into the cure
        they select.

        The flags are a serialization boundary: the archived runs and command
        lines are written in them, so they stay booleans on the command line
        and become an enum here, once, rather than in each driver. `use_bbar`
        is the `--use-bbar` flag and `use_psri` is the `--use-psri` flag.

        :returns: B-bar or PSRI when the matching flag is set, and full
            integration when neither is.
        :raises ValueError: If both flags are set, which would name two cures.
        """
        if use_bbar and use_psri:
            raise ValueError(
                f"`use_bbar` and `use_psri` select different locking cures and "
                f"cannot both be set, got use_bbar={use_bbar} and "
                f"use_psri={use_psri}."
            )
        if use_bbar:
            return cls.LockingCure.BBAR
        if use_psri:
            return cls.LockingCure.PSRI
        return cls.LockingCure.FULL
