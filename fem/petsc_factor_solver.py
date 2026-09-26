from enum import Enum

from petsc4py import PETSc


class FactorSolver(Enum):
    """
    The PETSc direct-factorization backends this codebase selects between.

    A member's value is its PETSc `MatSolverType`, the string that crosses into
    `PC.setFactorSolverType` and into `require`. `PC.setFactorSolverType` is a
    pure setter: it stores any string, including a misspelt or unconfigured
    one, without complaint, and the failure only surfaces much later inside
    the factorization as a misleading error. `require` exists to catch that
    up-front (checking, before a factorization starts, that the PETSc build
    in use actually provides the named backend), and lives here because every
    direct factorization in `fem/` (the Crank-Nicolson per-step solve and the
    shift-invert eigensolves on both the heat and the shell side) needs
    exactly the same guard.
    """

    PETSC = "petsc"
    """
    PETSc's own factorization: the one backend that is always available.

    `PETSc.Sys.hasExternalPackage` reports it absent, since it is not an external
    package at all, which is why `require` special-cases it.
    """
    MUMPS = "mumps"
    """The default throughout `fem/`, and the one `fem/environment.yml` ships."""
    SUPERLU_DIST = "superlu_dist"
    """Distributed-memory alternative, where PETSc was configured with it."""
    MKL_PARDISO = "mkl_pardiso"
    """Intel MKL's direct solver, absent from the pinned conda environment."""

    @classmethod
    def require(cls, solver_type: str, purpose: str) -> None:
        """
        Checks up-front that PETSc can actually provide the named direct solver.

        `solver_type` is a `MatSolverType` string (e.g. `FactorSolver.MUMPS.value`)
        rather than a `FactorSolver` because this is the PETSc boundary itself:
        callers hold the enum and cross it here with `.value`, so the check reads
        exactly the string `PC.setFactorSolverType` is about to be given. `purpose`
        is what the factorization is for, used in the error message.

        :raises RuntimeError: If PETSc was not configured with `solver_type`.
        """
        if solver_type == cls.PETSC.value:
            return
        if not PETSc.Sys.hasExternalPackage(solver_type):
            raise RuntimeError(
                f"PETSc was not configured with the direct solver '{solver_type}', "
                f"which is required for {purpose}. Install a PETSc build that "
                f"provides it (the conda environment in `fem/environment.yml` ships "
                f"MUMPS), or select a backend this PETSc has: "
                f"`'{cls.PETSC.value}'` is always available."
            )
