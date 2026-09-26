from dataclasses import dataclass
from enum import Enum

import scipy.sparse as sp
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix
from mpi4py import MPI
from petsc4py import PETSc

from fem.shells.shell_forms import ShellForms


class ShellSystemAssembler:
    """
    Assembles one shell configuration's compiled forms into either scipy or
    PETSc matrices, memoizing the scipy system it has already built.

    The two solve paths of
    :class:`fem.shells.geometrically_exact_shell_modal_solver.GeometricallyExactShellModalSolver`
    want different representations of the same physics: the small-scale
    diagnostic solve needs the four stiffness blocks separately, as scipy CSR,
    for its per-mode energy split; the memory-scalable SLEPc solve needs one
    combined stiffness and one mass matrix, as PETSc AIJ, and never holds five
    scipy matrices at once. `Backend` selects between the two at construction,
    mirroring
    `rebel_shape.isogeometric_analysis.sparse_cholesky.SparseCholeskySolver`:
    one class, one entry point (`assemble`), the choice readable afterwards.

    Under the B-bar locking cure the membrane and transverse-shear stiffness are
    not UFL forms at all: they are element-local triple products assembled in
    scipy by
    :meth:`fem.shells.geometrically_exact_shell_problem.GeometricallyExactShellProblem._bbar_block`,
    so both backends take them pre-assembled as `bbar_blocks` and add them in
    rather than recomputing them.
    """

    class Backend(Enum):
        """Which matrix representation `assemble` produces."""

        SCIPY = "scipy"
        """
        The four stiffness blocks (membrane, bending, shear, drilling) plus
        their sum and the mass, all as scipy CSR.
        """
        PETSC = "petsc"
        """Combined stiffness and mass only, as assembled PETSc AIJ matrices."""

    @dataclass(frozen=True, kw_only=True)
    class ScipySystem:
        """The shell's split stiffness blocks, their sum, and the mass, as scipy CSR."""

        membrane: sp.csr_matrix
        """Membrane stiffness block."""
        bending: sp.csr_matrix
        """Bending stiffness block."""
        shear: sp.csr_matrix
        """Transverse-shear stiffness block."""
        drilling: sp.csr_matrix
        """Drilling stabilization block; all-zero when it is switched off."""
        stiffness: sp.csr_matrix
        """The four blocks above, summed."""
        mass: sp.csr_matrix
        """Mass matrix, with either the full or the projected rotary inertia."""

    @dataclass(frozen=True, kw_only=True)
    class PetscSystem:
        """The shell's combined stiffness and mass, as assembled PETSc AIJ matrices."""

        stiffness: PETSc.Mat
        """Combined stiffness (membrane/bending/shear/drilling summed)."""
        mass: PETSc.Mat
        """Mass matrix, with either the full or the projected rotary inertia."""

    def __init__(
        self,
        forms: ShellForms,
        comm: MPI.Comm,
        backend: "ShellSystemAssembler.Backend",
        bbar_blocks: tuple[sp.csr_matrix, sp.csr_matrix] | None = None,
    ) -> None:
        """
        `forms` are the compiled bilinear forms of the shell configuration to
        assemble. `comm` is the MPI communicator PETSc matrices are built on.
        `backend` selects the matrix representation `assemble` returns.
        `bbar_blocks` are the pre-assembled (membrane, shear) B-bar stiffness
        blocks; required exactly when `forms` came from the B-bar cure
        (`forms.membrane is None`), and unused otherwise.

        :raises ValueError: If `bbar_blocks` is given and `forms` did not come
            from the B-bar cure, or vice versa.
        """
        if (forms.membrane is None) != (bbar_blocks is not None):
            raise ValueError(
                "`bbar_blocks` must be given if and only if `forms` came from "
                f"the B-bar cure, got forms.membrane={forms.membrane} and "
                f"bbar_blocks={bbar_blocks}."
            )
        self._forms = forms
        self._comm = comm
        self._backend = backend
        self._bbar_blocks = bbar_blocks
        self._scipy_system: ShellSystemAssembler.ScipySystem | None = None

    @property
    def backend(self) -> "ShellSystemAssembler.Backend":
        """The matrix representation `assemble` produces."""
        return self._backend

    def assemble(
        self,
    ) -> "ShellSystemAssembler.ScipySystem | ShellSystemAssembler.PetscSystem":
        """
        Assembles the shell's stiffness and mass system in the selected backend.

        The scipy system is cached on first assembly and reused by later calls,
        mirroring the memoization every ARPACK iteration of the small-scale
        solve relies on; the PETSc system is never cached, since its production
        caller (`solve_modes_slepc`) destroys the matrices itself once the
        eigensolve is done.

        :returns: A `ScipySystem` or a `PetscSystem`, per `backend`.
        """
        if self._backend is ShellSystemAssembler.Backend.SCIPY:
            return self._assemble_scipy()
        return self._assemble_petsc()

    def _assemble_scipy(self) -> "ShellSystemAssembler.ScipySystem":
        """
        Assembles the four stiffness blocks and the mass as scipy CSR, caching
        the result on `self`.
        """
        if self._scipy_system is not None:
            return self._scipy_system

        if self._forms.membrane is None:
            membrane, shear = self._bbar_blocks
        else:
            membrane = self._to_scipy(self._forms.membrane)
            shear = self._to_scipy(self._forms.shear)
        bending = self._to_scipy(self._forms.bending)
        drilling = (self._to_scipy(self._forms.drilling)
                    if self._forms.drilling is not None
                    else sp.csr_matrix(membrane.shape))
        mass = self._to_scipy(self._forms.mass)

        self._scipy_system = ShellSystemAssembler.ScipySystem(
            membrane=membrane,
            bending=bending,
            shear=shear,
            drilling=drilling,
            stiffness=membrane + bending + shear + drilling,
            mass=mass,
        )
        return self._scipy_system

    def _assemble_petsc(self) -> "ShellSystemAssembler.PetscSystem":
        """
        Assembles the combined stiffness and the mass natively in PETSc,
        adding the B-bar blocks (assembled in scipy, then wrapped) when the
        combined form does not already carry them.
        """
        K = assemble_matrix(self._forms.stiffness)
        K.assemble()
        if self._forms.membrane is None:
            membrane, shear = self._bbar_blocks
            k_bbar = self._to_petsc((membrane + shear).tocsr())
            K.axpy(1.0, k_bbar,
                   structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            k_bbar.destroy()
        M = assemble_matrix(self._forms.mass)
        M.assemble()
        return ShellSystemAssembler.PetscSystem(stiffness=K, mass=M)

    @staticmethod
    def _to_scipy(form: fem.Form) -> sp.csr_matrix:
        """
        Assembles the compiled bilinear form `form` into a scipy CSR matrix and
        releases the underlying PETSc `Mat`. `form`'s trial and test spaces
        need not match: the B-bar coupling forms are rectangular.

        :returns: The assembled matrix as scipy CSR, of shape (rows, cols)
            taken from the form.
        """
        A = assemble_matrix(form)
        A.assemble()
        ai, aj, av = A.getValuesCSR()
        m, k = A.getSize()  # rows, cols (rectangular for B-bar coupling forms)
        matrix = sp.csr_matrix((av, aj, ai), shape=(m, k)).copy()
        A.destroy()  # release the PETSc Mat; its CSR data is now owned by `matrix`
        return matrix

    def _to_petsc(self, matrix: sp.csr_matrix) -> PETSc.Mat:
        """
        Wraps the square scipy CSR (or CSR-convertible) matrix `matrix` as a
        PETSc AIJ matrix (serial), on `self`'s communicator.

        :returns: The same matrix as an assembled PETSc AIJ `Mat`.
        """
        a = matrix.tocsr()
        out = PETSc.Mat().createAIJ(
            size=a.shape,
            csr=(a.indptr.astype(PETSc.IntType), a.indices.astype(PETSc.IntType),
                 a.data),
            comm=self._comm,
        )
        out.assemble()
        return out
