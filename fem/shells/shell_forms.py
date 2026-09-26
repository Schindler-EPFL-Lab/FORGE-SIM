from dataclasses import dataclass

from dolfinx import fem


@dataclass(frozen=True, kw_only=True)
class ShellForms:
    """
    The compiled bilinear forms of one shell configuration.

    Assembled from the strain measures once, in
    :meth:`fem.shells.geometrically_exact_shell_problem.GeometricallyExactShellProblem._build_forms`,
    and thereafter read-only: which of them are present is fixed by the element
    choice, and the two solve paths differ only in which they use. `membrane`
    and `shear` are `None` exactly when the B-bar cure replaces them by the
    element-local triple products of `bbar_membrane` / `bbar_shear`, and
    `drilling` is `None` exactly when the drilling stabilization is off.
    """

    @dataclass(frozen=True, kw_only=True)
    class BBar:
        """
        The forms projecting one strain (membrane or shear) onto the B-bar
        reduced space, as
        :meth:`GeometricallyExactShellProblem._build_bbar_forms` derives them.
        """

        coupling: fem.Form
        """
        `B = <Q, s(u, theta)>`, coupling the reduced space to the primal
        DOFs. Rectangular: its test and trial spaces differ.
        """
        reduced_mass: fem.Form
        """`M_W = <Q, R>`, the reduced-space mass; block-diagonal per cell."""
        material_mass: fem.Form | None
        """
        `G = <D Q, R>`, the material-weighted reduced-space mass, or
        `None` when `scalar_modulus` stands in for it.
        """
        scalar_modulus: float | None
        """
        The material modulus, when `D` is that scalar times the identity
        and `G` therefore reduces to `scalar_modulus * M_W`; `None` when
        `material_mass` is given instead.
        """

        def __post_init__(self) -> None:
            if (self.material_mass is None) == (self.scalar_modulus is None):
                raise ValueError(
                    "Exactly one of `material_mass` and `scalar_modulus` "
                    f"describes the material, got material_mass="
                    f"{self.material_mass} and scalar_modulus="
                    f"{self.scalar_modulus}."
                )

    membrane: fem.Form | None
    """Membrane stiffness, or `None` under the B-bar cure."""
    bending: fem.Form
    """Bending stiffness; always fully integrated."""
    shear: fem.Form | None
    """Transverse-shear stiffness, or `None` under the B-bar cure."""
    drilling: fem.Form | None
    """Drilling stabilization stiffness, or `None` when it is switched off."""
    mass: fem.Form
    """Mass form, with either the full or the projected rotary inertia."""
    stiffness: fem.Form
    """
    The blocks above summed into one form, for the PETSc/SLEPc path. Under
    B-bar it holds only bending (+drilling); the membrane and shear blocks are
    assembled separately as matrices.
    """
    bbar_membrane: BBar | None
    """Membrane projection forms, present only under the B-bar cure."""
    bbar_shear: BBar | None
    """Transverse-shear projection forms, present only under the B-bar cure."""

    def __post_init__(self) -> None:
        for strain, form, bbar in (("membrane", self.membrane, self.bbar_membrane),
                                   ("shear", self.shear, self.bbar_shear)):
            if (form is None) == (bbar is None):
                raise ValueError(
                    f"The {strain} stiffness must come either from a UFL form or "
                    f"from B-bar projection forms, got form={form} and "
                    f"bbar={bbar}."
                )
