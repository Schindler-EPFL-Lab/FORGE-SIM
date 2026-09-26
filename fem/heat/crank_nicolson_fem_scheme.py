import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from fem.heat.fem_heat_problem import FEMHeatProblem
from fem.petsc_factor_solver import FactorSolver
from rebel_shape import logger


class CrankNicolsonFEMScheme:
    """
    Advances a `FEMHeatProblem` in time with Crank-Nicolson integration.

    Given the problem's assembled mass matrix M and stiffness matrix K, this
    class builds the constant operators

        A = M + 0.5*dt*K,   B = M - 0.5*dt*K

    and the per-step update

        A u^{n+1} = B u^n + dt*F

    where F is the assembled source vector. `step` performs one update,
    reading the problem's current state and committing the result back via
    `FEMHeatProblem.load_state`: this class never touches a `fem.Function`.

    A is symmetric positive definite and constant across steps, so its direct
    Cholesky factorization (PETSc, MUMPS by default) is built once, in
    `__init__`/`from_normalized_time_step`, and reused for every `step`. There
    is no iterative fallback: algebraic multigrid is unreliable on the high
    material contrast of multi-material runs, where it under-converges the
    stiff window DOFs and produces visibly wrong peak temperatures at some
    mesh sizes (see `_build_ksp`).

    The time step can be supplied either as an explicit physical value (`dt`)
    or, via `from_normalized_time_step`, as a fraction of the natural
    relaxation time of the semi-discrete system. The latter runs a SLEPc
    shift-invert eigensolve on the problem's (K, M), the same
    characterization the IGA `CrankNicolsonScheme` uses, so the same
    `dt_over_t_relax` resolves to a near-identical physical step on both
    solvers.
    """

    RELAXATION_FACTOR_SOLVER: FactorSolver = FactorSolver.MUMPS
    """
    PETSc direct-solver backend for the shift-invert relaxation eigensolve's
    Cholesky factorization; there is no fallback. The check fires only where the
    eigensolve is reached (eagerly in `from_normalized_time_step`, lazily on
    first access to `relaxation_time`), and a scheme built with an explicit `dt`
    never triggers it. `FactorSolver.MKL_PARDISO` is not in the pinned conda
    environment.
    """

    LINEAR_FACTOR_SOLVER: FactorSolver = FactorSolver.MUMPS
    """
    PETSc direct-solver backend for the per-step Crank-Nicolson Cholesky
    factorization.

    As above, MUMPS is required and has no fallback. The per-step solve is
    direct-only by design because the SPD LHS factors cleanly and a direct solve
    gives machine-precision energy conservation.
    """

    _RELAXATION_NUM_MODES: int = 5
    """
    Eigenpairs requested from the relaxation eigensolve.

    More than one is required so the zero constant mode can be skipped; the rest
    is cheap headroom (SLEPc routinely converges more than requested at this size
    anyway).
    """

    _RELAXATION_EIGENSOLVE_TOL: float = 1e-8
    """
    Relative tolerance of the relaxation eigensolve.

    `lambda_1` sets the physical step directly (`dt = dt_over_t_relax / lambda_1`),
    so this value, not floating-point arithmetic, bounds how reproducibly a
    `dt_over_t_relax` run resolves its `dt` across environments and library
    versions.
    """

    def __init__(
        self,
        problem: FEMHeatProblem,
        dt: float,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        """
        Initializes the Crank-Nicolson scheme for `problem` with an explicit
        physical time step.

        `problem` is the assembled heat-equation problem this scheme steps.
        `dt` is the time step size in seconds (must be positive), and `comm`
        the MPI communicator for parallel execution.

        :raises ValueError: If `dt` is not a positive real number.
        :raises RuntimeError: If PETSc lacks `LINEAR_FACTOR_SOLVER`, or if the
            Cholesky factorization of the left-hand side cannot be set up.
        """
        if dt <= 0.0:
            raise ValueError(f"`dt` must be a positive real number, got {dt}.")

        self._problem = problem
        self._comm = comm
        self._dt = dt
        self._relaxation_time: float | None = None

        self._M = problem.mass_matrix
        self._K = problem.stiffness_matrix
        self._A, self._B = self._build_cn_operators(self._M, self._K, dt)
        self._F = problem.assemble_source_vector()

        self._rhs = self._F.copy()
        """Right-hand-side work vector, rebuilt from `_B` and `_F` at each `step`."""
        self._tmp = self._F.copy()
        """Scratch vector holding `_B` times the problem's state, reused per `step`."""
        self._u_new = self._F.duplicate()
        """
        Solve buffer `step` writes the new state into, before it is committed
        to `problem`.
        """

        self._steps_taken = 0
        """
        Steps completed so far, i.e. the 0-based index of the next one. Only the
        failure message in `step` reads it, so that a diverged solve names which
        of the several hundred steps it was.
        """

        self._ksp = self._build_ksp(self._A, comm)

    @classmethod
    def from_normalized_time_step(
        cls,
        problem: FEMHeatProblem,
        dt_over_t_relax: float,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> "CrankNicolsonFEMScheme":
        """
        Builds a scheme for `problem` whose physical time step is a fraction of
        the natural relaxation time of the semi-discrete system.

        The physical step is `dt = dt_over_t_relax * T_relax`, where `T_relax`
        is derived from the smallest non-zero generalized eigenvalue of
        `K v = lambda M v` for `problem`'s assembled stiffness K and mass M,
        via a SLEPc shift-invert eigensolve (see `_compute_relaxation_time`).
        This is the same characterization the IGA solver uses, so the same
        `dt_over_t_relax` makes the Crank-Nicolson temporal error
        `~ (dt_over_t_relax)^2` the explicit accuracy knob on both solvers,
        independent of geometry, units, and material heterogeneity.
        `dt_over_t_relax` is the time step normalized by `T_relax` (must be
        positive); `[0.005, 0.05]` is usually a sensible range.

        `problem`'s (K, M) are already assembled by the time this runs, so the
        eigensolve and the Crank-Nicolson operators built by the constructor
        below read the same matrices; nothing here reassembles them.

        :raises ValueError: If `dt_over_t_relax` is not a positive real number.
        :raises ImportError: If slepc4py is not available.
        :raises RuntimeError: If PETSc lacks either factorization backend, or if
            no non-zero eigenvalue can be identified.
        """
        if dt_over_t_relax <= 0.0:
            raise ValueError(
                f"`dt_over_t_relax` must be a positive real number, got "
                f"{dt_over_t_relax}."
            )

        relaxation_time = cls._compute_relaxation_time(
            problem.stiffness_matrix, problem.mass_matrix, comm
        )
        dt = dt_over_t_relax * relaxation_time
        logger.info(
            f"Natural relaxation time T_relax = {relaxation_time:.6g} s. "
            f"Using dt = {dt_over_t_relax:.4g} * T_relax = {dt:.6g} s."
        )

        scheme = cls(problem=problem, dt=dt, comm=comm)
        # Seeds the memo behind `relaxation_time`: the value is already known, and
        # its getter would otherwise run the eigensolve a second time.
        scheme._relaxation_time = relaxation_time
        return scheme

    @staticmethod
    def _build_cn_operators(
        M: PETSc.Mat, K: PETSc.Mat, dt: float
    ) -> tuple[PETSc.Mat, PETSc.Mat]:
        """
        Builds the Crank-Nicolson left- and right-hand-side matrices from M, K and
        the resolved time step:
            LHS: A = M + 0.5*dt*K
            RHS: B = M - 0.5*dt*K

        :returns: A and B, in that order.
        """
        A = M.copy()
        A.axpy(0.5 * dt, K)

        B = M.copy()
        B.axpy(-0.5 * dt, K)
        return A, B

    @classmethod
    def _build_ksp(cls, A: PETSc.Mat, comm: MPI.Comm) -> PETSc.KSP:
        """
        Builds a direct Cholesky solver (MUMPS) for the constant SPD LHS matrix
        A = M + 0.5*dt*K, whose factorization is reused across all time steps.

        The per-step solve is direct-only by design. A is symmetric positive
        definite (`lambda_min(A) >= lambda_min(M) > 0`), so Cholesky needs no
        shift, and a direct factor gives machine-precision solves (energy drift
        ~1e-14). There is deliberately no iterative fallback: algebraic multigrid
        is unreliable on the high material contrast of multi-material runs, where
        it under-converges the stiff window DOFs and produces visibly wrong peak
        temperatures at some mesh sizes. A failed factorization raises loudly
        rather than degrading accuracy silently, so `LINEAR_FACTOR_SOLVER`
        (MUMPS) is a hard requirement.

        :returns: The solver, its factorization of `A` already set up.

        :raises RuntimeError: If PETSc lacks `LINEAR_FACTOR_SOLVER`, or if the
            Cholesky factorization cannot be set up (typically MUMPS workspace
            out-of-memory on very large meshes).
        """
        FactorSolver.require(
            cls.LINEAR_FACTOR_SOLVER.value,
            "the Crank-Nicolson per-step Cholesky solve",
        )

        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A)
        ksp.setType(PETSc.KSP.Type.PREONLY)

        preconditioner = ksp.getPC()
        preconditioner.setType(PETSc.PC.Type.CHOLESKY)
        preconditioner.setFactorSolverType(cls.LINEAR_FACTOR_SOLVER.value)

        try:
            ksp.setUp()
        except PETSc.Error as exc:
            raise RuntimeError(
                "Direct Cholesky factorization of the Crank-Nicolson LHS failed "
                f"(n={A.getSize()[0]}). This is typically a MUMPS workspace "
                "out-of-memory on large meshes; reduce the MPI rank count (the "
                "projector is currently built per rank, so fewer ranks frees "
                "memory) or run on a single rank. An iterative solve is not "
                "substituted, as CG+GAMG is unreliable on high material "
                f"contrast. Original PETSc error: {exc}"
            ) from exc
        return ksp

    @classmethod
    def _compute_relaxation_time(
        cls, K: PETSc.Mat, M: PETSc.Mat, comm: MPI.Comm
    ) -> float:
        r"""
        Computes the natural relaxation time `T_relax = 1 / lambda_1` of the
        semi-discrete system, where `lambda_1` is the smallest non-zero
        generalized eigenvalue of `K v = lambda M v`.

        K is symmetric positive semi-definite: on a closed manifold its kernel
        is the constant (steady) mode, so `lambda_0 = 0` and the slowest decay
        rate is `lambda_1 > 0`; M is symmetric positive definite. The smallest
        non-zero eigenvalue is found with a SLEPc shift-invert eigensolve. The
        spectral transform factorizes `K - sigma*M`, and a small negative
        shift `sigma` keeps that matrix SPD (`K + |sigma|*M`) so a direct
        Cholesky factorization (MUMPS by default; see `RELAXATION_FACTOR_SOLVER`)
        applies. The shift only regularizes the factorization; it is not a claim
        that K is definite, and it mirrors the IGA `compute_relaxation_time`.

        The eigensolve is sized and toleranced by `_RELAXATION_NUM_MODES` and
        `_RELAXATION_EIGENSOLVE_TOL`, and requires `RELAXATION_FACTOR_SOLVER`
        (MUMPS); there is no fallback factorization.

        It takes `K`, the problem's assembled stiffness matrix, and `M`, its
        assembled mass matrix, rather than reading them off an instance because
        `from_normalized_time_step` needs the answer before there is one.
        `comm` is the MPI communicator to run the eigensolve on.

        :returns: The natural relaxation time in seconds.

        :raises ImportError: If slepc4py is not available.
        :raises RuntimeError: If PETSc lacks `RELAXATION_FACTOR_SOLVER`, or if
            no non-zero eigenvalue can be identified.
        """
        try:
            # Deferred: SLEPc is an optional heavy dependency, needed only by this
            # eigensolve. Every other path (an explicit `dt`, a `final_time`)
            # runs on a PETSc-only environment, so importing it at module level
            # would make it a hard requirement of the whole heat pipeline.
            from slepc4py import SLEPc
        except ImportError as exc:
            raise ImportError(
                "SLEPc (slepc4py) is required for normalized `dt_over_t_relax` "
                "time stepping. Install with `conda install -c conda-forge "
                "slepc4py`, or pass an explicit `time_step_size` instead."
            ) from exc

        FactorSolver.require(
            cls.RELAXATION_FACTOR_SOLVER.value,
            "the shift-invert relaxation eigensolve",
        )

        # `Tr(K)/Tr(M)` [1/s] approximates an element-scale rate; a small fraction
        # of it places the shift just below the bottom of the spectrum while
        # keeping `K - sigma*M` well-conditioned and SPD.
        rate_scale = K.getDiagonal().sum() / M.getDiagonal().sum()
        if not np.isfinite(rate_scale) or rate_scale <= 0.0:
            raise RuntimeError(
                "Failed to derive a shift scale from K and M: diagonal trace "
                f"ratio is {rate_scale}. Check that the matrices are well-formed."
            )
        sigma = -1e-6 * rate_scale

        eps = SLEPc.EPS().create(comm)
        eps.setOperators(K, M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setDimensions(nev=cls._RELAXATION_NUM_MODES)
        eps.setTolerances(tol=cls._RELAXATION_EIGENSOLVE_TOL)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        eps.setTarget(sigma)

        # Shift-invert backed by a direct Cholesky factorization of `K - sigma*M`.
        spectral_transform = eps.getST()
        spectral_transform.setType(SLEPc.ST.Type.SINVERT)
        ksp = spectral_transform.getKSP()
        ksp.setType(PETSc.KSP.Type.PREONLY)
        preconditioner = ksp.getPC()
        preconditioner.setType(PETSc.PC.Type.CHOLESKY)
        preconditioner.setFactorSolverType(cls.RELAXATION_FACTOR_SOLVER.value)

        eps.solve()

        num_converged = eps.getConverged()
        if num_converged == 0:
            raise RuntimeError("Relaxation eigsolve converged no eigenpairs.")

        eigenvalues = np.sort(
            [eps.getEigenvalue(i).real for i in range(num_converged)]
        )
        # The constant mode gives `lambda_0 ~ 0`; filter it on a relative cutoff
        # against the largest returned eigenvalue.
        cutoff = max(1e-12, 1e-6 * abs(eigenvalues[-1]))
        nonzero = eigenvalues[eigenvalues > cutoff]
        if nonzero.size == 0:
            raise RuntimeError(
                "Could not identify a non-zero eigenvalue of (K, M) among the "
                f"{num_converged} converged modes (eigenvalues: {eigenvalues})."
            )

        lambda_1 = float(nonzero[0])
        logger.debug(
            f"Relaxation eigsolve: {num_converged} modes, sigma={sigma:.3e}, "
            f"lambda_1={lambda_1:.6e} "
            f"(factor solver: {cls.RELAXATION_FACTOR_SOLVER.value})."
        )

        # Releases the shift-invert factorization (the largest object here by
        # far) and the references the solver holds on K and M, so that a caller
        # destroying those right after this returns is the last owner. Matches
        # `GeometricallyExactShellSolver.solve_modes_slepc`, which tears its own
        # eigensolver down explicitly rather than leaving it to the interpreter.
        eps.destroy()
        return 1.0 / lambda_1

    @property
    def dt(self) -> float:
        """
        Physical time step this scheme was built with, in seconds.

        Read-only: `A = M + 0.5 dt K` and its Cholesky factorization are built from
        it once and never re-derived, so a reassignment would leave the operators
        describing a different step from the one `step` advertises.
        """
        return self._dt

    @property
    def relaxation_time(self) -> float:
        """
        Natural relaxation time of the semi-discrete system in seconds.

        Computed lazily on first access and cached. When the scheme was built
        via `from_normalized_time_step` the value is already populated and no
        eigsolve is triggered.
        """
        if self._relaxation_time is None:
            self._relaxation_time = self._compute_relaxation_time(
                self._K, self._M, self._comm
            )
        return self._relaxation_time

    def scale_source(self, factor: float) -> None:
        """
        Scales the assembled source vector and the underlying source function by
        `factor`.

        The load vector F is linear in the source field, so scaling it is
        equivalent to having assembled with `source *= factor`. Both F (used in
        `step`) and the source function Q (used by
        `FEMHeatProblem.compute_heat_generation_rate` for drift reporting) are
        updated, matching `CrankNicolsonScheme.scale_source` on the IGA side.
        """
        Q = self._problem.material_funcs.source
        self._F.scale(factor)
        Q.x.array[:] *= factor
        Q.x.scatter_forward()

    def step(self) -> None:
        """
        Advances `problem`'s solution by one time step using the Crank-Nicolson
        scheme:
            (M + 0.5*dt*K) u^{n+1} = (M - 0.5*dt*K) u^n + dt*F

        The new state is committed to `problem` via `FEMHeatProblem.load_state`;
        call `problem.compute_total_heat`, `problem.solution`, or
        `problem.snapshot_state` to read it out. Nothing here touches a
        `fem.Function`, by design (see `FEMHeatProblem._state`'s docstring).

        :raises RuntimeError: If the solve did not converge. The factorization is
            validated once in `_build_ksp` and reused unchanged, so a failure here
            is a per-step numerical one (a NaN or Inf entering the right-hand
            side, or MUMPS running out of workspace mid-run) and must not be
            integrated onward as if it were a solution.
        """
        # Compute RHS: B * state + dt * F
        self._B.mult(self._problem.state, self._tmp)
        self._rhs.zeroEntries()
        self._rhs.axpy(1.0, self._tmp)
        self._rhs.axpy(self._dt, self._F)

        # Solve A * u_new = rhs (factorization reused across steps).
        self._ksp.solve(self._rhs, self._u_new)
        reason = self._ksp.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(
                f"Crank-Nicolson step {self._steps_taken} did not converge: PETSc "
                f"KSPConvergedReason {reason} (non-positive reasons are "
                f"divergence). The factorization of the constant left-hand side "
                f"succeeded at construction, so this is a per-step failure: a "
                f"non-finite right-hand side, or MUMPS exhausting its workspace."
            )
        self._steps_taken += 1

        # Commit: problem's state <- u_new.
        self._problem.load_state(self._u_new)
