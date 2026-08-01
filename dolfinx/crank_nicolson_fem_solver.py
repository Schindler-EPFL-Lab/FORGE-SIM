"""
Crank-Nicolson time integration scheme for DOLFINx heat simulations.

This module provides the `CrankNicolsonFEMSolver` class that solves the transient
heat equation on a 2D manifold using the Crank-Nicolson implicit time-stepping
method. It mirrors the IGA solver in
`rebel_shape.isogeometric_analysis.crank_nicolson_scheme` but uses DOLFINx for
FEM discretization, so that FEM and IGA results can be compared one-to-one.

The heat equation on a 2D manifold with material properties is:

    rho c d du/dt - div(k d grad(u)) = f

where:
    - rho is density [kg/m^3]
    - c is specific heat capacity [J/(kg*K)]
    - d is shell thickness [m]
    - k is thermal conductivity [W/(m*K)]
    - u is temperature [K]
    - f is the heat source term [W/m^2]

The Crank-Nicolson scheme discretizes this as:

    (M + 0.5*dt*K) u^{n+1} = (M - 0.5*dt*K) u^n + dt*F

where M is the mass matrix, K is the stiffness matrix, and F is the load vector.
The mass coefficient is rho*c*d, the stiffness coefficient is k*d, and the source
is integrated per unit area (no thickness factor) -- identical to the IGA solver.
"""

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector

from rebel_shape import logger

# PETSc's own factorization is always available; every other `MatSolverType`
# names an external package PETSc must have been configured with.
_BUILTIN_FACTOR_SOLVER = "petsc"

# Eigenpairs requested from the relaxation eigensolve. More than one is required
# so the zero constant mode can be skipped; the rest is cheap headroom (SLEPc
# routinely converges more than requested at this size anyway).
_RELAXATION_NUM_MODES = 5

# Relative tolerance of the relaxation eigensolve. `lambda_1` sets the physical
# step directly (`dt = dt_over_t_relax / lambda_1`), so this value -- not
# floating-point arithmetic -- bounds how reproducibly a `dt_over_t_relax` run
# resolves its `dt` across environments and library versions.
_RELAXATION_EIGENSOLVE_TOL = 1e-8


def _require_factor_solver(solver_type: str, purpose: str) -> None:
    """
    Checks up-front that PETSc can actually provide the named direct solver.

    `PC.setFactorSolverType` is a pure setter: it stores any string, including a
    misspelt or unconfigured one, without complaint, and the failure only
    surfaces much later inside the factorization as a misleading error. This
    turns an unavailable backend into an immediate, accurate message.

    :param solver_type: A PETSc `MatSolverType`, e.g. `"mumps"`.
    :param purpose: What the factorization is for, used in the error message.

    :raises RuntimeError: If PETSc was not configured with `solver_type`.
    """
    if solver_type == _BUILTIN_FACTOR_SOLVER:
        return
    if not PETSc.Sys.hasExternalPackage(solver_type):
        raise RuntimeError(
            f"PETSc was not configured with the direct solver '{solver_type}', "
            f"which is required for {purpose}. Install a PETSc build that "
            f"provides it (the conda environment in `dolfinx/environment.yml` "
            f"ships MUMPS), or point the corresponding "
            f"`CrankNicolsonFEMSolver.*_FACTOR_SOLVER` class constant at a "
            f"backend this PETSc has."
        )


class CrankNicolsonFEMSolver:
    """
    Solves the transient heat equation using Crank-Nicolson time integration.

    This class implements the same numerical scheme as the IGA solver in
    `rebel_shape.isogeometric_analysis.crank_nicolson_scheme`, but using
    DOLFINx for FEM discretization.

    The solver assembles the (time-independent) mass and stiffness matrices once
    during initialization and reuses them for time stepping. The per-step linear
    system is solved with a PETSc direct Cholesky factorization, reused across
    steps since the left-hand side is constant. The factorization requires an
    external direct solver (MUMPS by default); there is no iterative fallback.

    The time step can be supplied either as an explicit physical value (`dt`) or,
    via `from_normalized_time_step`, as a fraction of the natural relaxation time
    of the semi-discrete system. The latter computes `T_relax` from the
    FEM-assembled (K, M) with a SLEPc shift-invert eigensolve -- the same
    characterization the IGA solver uses -- so the same `dt_over_t_relax` resolves
    to a near-identical physical step on both solvers.

    Example usage::

        solver = CrankNicolsonFEMSolver(
            msh=dolfinx_mesh,
            V=function_space,
            u0=initial_temperature,
            material_funcs=material_functions,
            dt=240.0,
        )

        for _ in range(num_steps):
            solver.step()
            current_temp = solver.u_n.x.array.copy()
    """

    RELAXATION_FACTOR_SOLVER = "mumps"
    """PETSc direct-solver backend for the shift-invert relaxation eigensolve's
    Cholesky factorization. MUMPS is **required**, not preferred: there is no
    fallback. Unlike `LINEAR_FACTOR_SOLVER`, the check fires only where the
    eigensolve is actually reached -- eagerly in `from_normalized_time_step`, and
    lazily on first access to `relaxation_time`. A solver built with an explicit
    `dt` never runs the eigensolve, so it constructs and steps quite happily on a
    PETSc lacking this backend. Retarget this at
    another backend PETSc was configured with (e.g. ``"superlu_dist"``) if you
    must; note that ``"mkl_pardiso"`` is *not* in the pinned conda environment."""

    LINEAR_FACTOR_SOLVER = "mumps"
    """PETSc direct-solver backend for the per-step Crank-Nicolson Cholesky
    factorization. As above, MUMPS is **required** and has no fallback. The
    per-step solve is direct-only by design because the SPD LHS factors cleanly
    and a direct solve gives machine-precision energy conservation."""

    def __init__(
        self,
        msh: mesh.Mesh,
        V: fem.FunctionSpace,
        u0: fem.Function,
        material_funcs: dict[str, fem.Function],
        dt: float,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        """
        Initializes the Crank-Nicolson solver with an explicit physical time step.

        :param msh: DOLFINx mesh (2D manifold in 3D)
        :param V: Function space for temperature (typically P1)
        :param u0: Initial temperature function
        :param material_funcs: Dictionary containing material property functions
            with keys: 'conductivity', 'density', 'specific_heat', 'thickness',
            'source'
        :param dt: Time step size in seconds (must be positive)
        :param comm: MPI communicator for parallel execution

        :raises ValueError: If `dt` is not a positive real number.
        """
        if dt <= 0.0:
            raise ValueError("`dt` must be a positive real number.")

        self._init_common(msh, V, u0, material_funcs, comm)
        self.dt = dt
        self._build_cn_system()
        self._setup_solver()

    @classmethod
    def from_normalized_time_step(
        cls,
        msh: mesh.Mesh,
        V: fem.FunctionSpace,
        u0: fem.Function,
        material_funcs: dict[str, fem.Function],
        dt_over_t_relax: float,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> "CrankNicolsonFEMSolver":
        """
        Builds a solver whose physical time step is a fraction of the natural
        relaxation time of the semi-discrete system.

        The physical step is `dt = dt_over_t_relax * T_relax`, where `T_relax`
        is derived from the smallest non-zero generalized eigenvalue of
        `K v = lambda M v` for the FEM-assembled stiffness K and mass M, via a
        SLEPc shift-invert eigensolve (see `_compute_relaxation_time`). This is
        the same characterization the IGA solver uses, so the same
        `dt_over_t_relax` makes the Crank-Nicolson temporal error
        `~ (dt_over_t_relax)^2` the explicit accuracy knob on both solvers,
        independent of geometry, units, and material heterogeneity.

        :param dt_over_t_relax: Time step normalized by `T_relax` (must be
            positive). `[0.005, 0.05]` is usually a sensible range.

        :raises ValueError: If `dt_over_t_relax` is not a positive real number.
        :raises ImportError: If slepc4py is not available.
        """
        if dt_over_t_relax <= 0.0:
            raise ValueError("`dt_over_t_relax` must be a positive real number.")

        instance = cls.__new__(cls)
        instance._init_common(msh, V, u0, material_funcs, comm)

        relaxation_time = instance._compute_relaxation_time()
        instance.dt = dt_over_t_relax * relaxation_time
        instance._relaxation_time = relaxation_time
        logger.info(
            f"Natural relaxation time T_relax = {relaxation_time:.6g} s. "
            f"Using dt = {dt_over_t_relax:.4g} * T_relax = {instance.dt:.6g} s."
        )

        instance._build_cn_system()
        instance._setup_solver()
        return instance

    def _init_common(
        self,
        msh: mesh.Mesh,
        V: fem.FunctionSpace,
        u0: fem.Function,
        material_funcs: dict[str, fem.Function],
        comm: MPI.Comm,
    ) -> None:
        """
        Stores inputs and assembles the time-independent mass and stiffness
        matrices shared by every construction path.
        """
        self.msh = msh
        self.V = V
        self.comm = comm
        self._relaxation_time: float | None = None

        # Function carrying the previous-step temperature. It is used only for
        # form assembly (diagnostics) and output; the actual stepping state lives
        # in a plain PETSc vector (see `_build_cn_system`). It is initialized from
        # u0 here, before any stepping.
        self.u_n = fem.Function(V, name="u_n")
        self.u_n.x.array[:] = u0.x.array

        # Material properties.
        self.k = material_funcs["conductivity"]
        self.rho = material_funcs["density"]
        self.c = material_funcs["specific_heat"]
        self.d = material_funcs["thickness"]
        self.Q = material_funcs["source"]

        self._setup_forms()
        self._assemble_mass_stiffness()

    def _setup_forms(self) -> None:
        """
        Sets up the UFL variational forms for the mass and stiffness matrices and
        the source vector.

        The mass form carries the volumetric heat capacity:
            M = integral( rho c d phi_i phi_j ) dA
        The stiffness form carries the conductivity:
            K = integral( k d grad(phi_i) . grad(phi_j) ) dA
        The source form is per unit area (no thickness factor):
            F = integral( Q phi_i ) dA
        """
        v = ufl.TestFunction(self.V)
        u_tr = ufl.TrialFunction(self.V)

        mass_coeff = self.rho * self.c * self.d
        stiff_coeff = self.k * self.d

        self.a_mass = mass_coeff * ufl.inner(u_tr, v) * ufl.dx
        self.a_stiff = stiff_coeff * ufl.inner(ufl.grad(u_tr), ufl.grad(v)) * ufl.dx
        self.L_source = self.Q * v * ufl.dx

    def _assemble_mass_stiffness(self) -> None:
        """
        Assembles the time-independent mass matrix M and stiffness matrix K.
        """
        self.M = assemble_matrix(fem.form(self.a_mass))
        self.M.assemble()

        self.K = assemble_matrix(fem.form(self.a_stiff))
        self.K.assemble()

    def _build_cn_system(self) -> None:
        """
        Builds the Crank-Nicolson left/right-hand-side matrices and the source
        vector from M, K, and the resolved time step `self.dt`:
            LHS: A = M + 0.5*dt*K
            RHS: B = M - 0.5*dt*K
        The source vector F is assembled once (time-independent source).
        """
        self.A = self.M.copy()
        self.A.axpy(0.5 * self.dt, self.K)

        self.B = self.M.copy()
        self.B.axpy(-0.5 * self.dt, self.K)

        self.F = assemble_vector(fem.form(self.L_source))

        # Work vectors reused across steps.
        self.rhs = self.F.copy()
        self.tmp = self.F.copy()

        # The Crank-Nicolson stepping state is held in plain PETSc vectors, fully
        # decoupled from any fem.Function. Touching a Function's .x (numpy array
        # OR cached petsc_vec wrapper) between steps desynchronizes the wrapper
        # that B.mult/ksp.solve rely on and silently corrupts the time-stepping.
        # Keeping the state in standalone vectors makes snapshots and diagnostics
        # between steps harmless; the Function (self.u_n) is synced from this
        # state only on demand.
        self._state = self.F.duplicate()
        self._u_new = self.F.duplicate()
        self.u_n.x.petsc_vec.copy(self._state)

    def _setup_solver(self) -> None:
        """
        Sets up a direct Cholesky solver (MUMPS) for the constant SPD LHS matrix
        A = M + 0.5*dt*K, whose factorization is reused across all time steps.

        The per-step solve is **direct-only by design**. A is symmetric positive
        definite (`lambda_min(A) >= lambda_min(M) > 0`), so Cholesky needs no
        shift, and a direct factor gives machine-precision solves (energy drift
        ~1e-14). The earlier CG + GAMG fallback has been removed: algebraic
        multigrid is unreliable on the high material contrast of multi-material
        runs -- it silently under-converged the stiff window DOFs and produced
        visibly wrong peak temperatures at some mesh sizes -- so a failed
        factorization now raises loudly rather than degrading accuracy silently.
        `LINEAR_FACTOR_SOLVER` (MUMPS) is therefore a hard requirement.

        :raises RuntimeError: If PETSc lacks `LINEAR_FACTOR_SOLVER`, or if the
            Cholesky factorization cannot be set up (typically MUMPS workspace
            out-of-memory on very large meshes).
        """
        _require_factor_solver(
            self.LINEAR_FACTOR_SOLVER, "the Crank-Nicolson per-step Cholesky solve"
        )

        self.ksp = PETSc.KSP().create(self.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType(PETSc.KSP.Type.PREONLY)

        preconditioner = self.ksp.getPC()
        preconditioner.setType(PETSc.PC.Type.CHOLESKY)
        preconditioner.setFactorSolverType(self.LINEAR_FACTOR_SOLVER)

        try:
            self.ksp.setUp()
        except PETSc.Error as exc:
            raise RuntimeError(
                "Direct Cholesky factorization of the Crank-Nicolson LHS failed "
                f"(n={self.A.getSize()[0]}). This is typically a MUMPS workspace "
                "out-of-memory on large meshes -- reduce the MPI rank count (the "
                "projector is currently built per rank, so fewer ranks frees "
                "memory) or run on a single rank. An iterative solve is not "
                "substituted, as CG+GAMG is unreliable on high material "
                f"contrast. Original PETSc error: {exc}"
            ) from exc

    def _compute_relaxation_time(self) -> float:
        r"""
        Computes the natural relaxation time ``T_relax = 1 / lambda_1`` of the
        semi-discrete system, where ``lambda_1`` is the smallest non-zero
        generalized eigenvalue of ``K v = lambda M v``.

        K is symmetric positive *semi*-definite: on a closed manifold its kernel
        is the constant (steady) mode, so ``lambda_0 = 0`` and the slowest decay
        rate is ``lambda_1 > 0``; M is symmetric positive definite. The smallest
        non-zero eigenvalue is found with a SLEPc shift-invert eigensolve. The
        spectral transform factorizes ``K - sigma*M``, and a small *negative*
        shift ``sigma`` keeps that matrix SPD (``K + |sigma|*M``) so a direct
        Cholesky factorization (MUMPS by default; see ``RELAXATION_FACTOR_SOLVER``)
        applies. The shift only regularizes the factorization -- it is not a claim
        that K is definite -- and mirrors the IGA ``compute_relaxation_time``.

        The eigensolve is sized and toleranced by ``_RELAXATION_NUM_MODES`` and
        ``_RELAXATION_EIGENSOLVE_TOL``, and requires ``RELAXATION_FACTOR_SOLVER``
        (MUMPS) -- there is no fallback factorization.

        :returns: The natural relaxation time in seconds.

        :raises ImportError: If slepc4py is not available.
        :raises RuntimeError: If PETSc lacks ``RELAXATION_FACTOR_SOLVER``, or if
            no non-zero eigenvalue can be identified.
        """
        try:
            from slepc4py import SLEPc
        except ImportError as exc:
            raise ImportError(
                "SLEPc (slepc4py) is required for normalized `dt_over_t_relax` "
                "time stepping. Install with `conda install -c conda-forge "
                "slepc4py`, or pass an explicit `time_step_size` instead."
            ) from exc

        _require_factor_solver(
            self.RELAXATION_FACTOR_SOLVER, "the shift-invert relaxation eigensolve"
        )

        # `Tr(K)/Tr(M)` [1/s] approximates an element-scale rate; a small fraction
        # of it places the shift just below the bottom of the spectrum while
        # keeping `K - sigma*M` well-conditioned and SPD.
        rate_scale = self.K.getDiagonal().sum() / self.M.getDiagonal().sum()
        if not np.isfinite(rate_scale) or rate_scale <= 0.0:
            raise RuntimeError(
                "Failed to derive a shift scale from K and M: diagonal trace "
                f"ratio is {rate_scale}. Check that the matrices are well-formed."
            )
        sigma = -1e-6 * rate_scale

        eps = SLEPc.EPS().create(self.comm)
        eps.setOperators(self.K, self.M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setDimensions(nev=_RELAXATION_NUM_MODES)
        eps.setTolerances(tol=_RELAXATION_EIGENSOLVE_TOL)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        eps.setTarget(sigma)

        # Shift-invert backed by a direct Cholesky factorization of `K - sigma*M`.
        spectral_transform = eps.getST()
        spectral_transform.setType(SLEPc.ST.Type.SINVERT)
        ksp = spectral_transform.getKSP()
        ksp.setType(PETSc.KSP.Type.PREONLY)
        preconditioner = ksp.getPC()
        preconditioner.setType(PETSc.PC.Type.CHOLESKY)
        preconditioner.setFactorSolverType(self.RELAXATION_FACTOR_SOLVER)

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
            f"(factor solver: {self.RELAXATION_FACTOR_SOLVER})."
        )
        return 1.0 / lambda_1

    @property
    def relaxation_time(self) -> float:
        """
        Natural relaxation time of the semi-discrete system in seconds.

        Computed lazily on first access and cached. When the solver was built via
        `from_normalized_time_step` the value is already populated and no
        eigsolve is triggered.
        """
        if self._relaxation_time is None:
            self._relaxation_time = self._compute_relaxation_time()
        return self._relaxation_time

    def scale_source(self, factor: float) -> None:
        """
        Scales the assembled source vector and the underlying source function by
        `factor`.

        The load vector F is linear in the source field, so scaling it is
        equivalent to having assembled with `source *= factor`. Both F (used in
        the time loop) and the source function Q (used by
        `compute_heat_generation_rate` for drift reporting) are updated, matching
        `CrankNicolsonScheme.scale_source` on the IGA side.
        """
        self.F.scale(factor)
        self.Q.x.array[:] *= factor
        self.Q.x.scatter_forward()

    def compute_total_heat(self) -> float:
        """
        Computes the total thermal energy in the system:
            H = integral( rho c d u ) dA

        This matches the energy computation in the IGA solver for comparison.
        ``assemble_scalar`` returns only the rank-local contribution, so the
        partial integrals are reduced across ranks for the global value. The
        stepping state is synced into ``u_n`` first so the form sees it.

        :returns: Total heat in Joules
        """
        self._state.copy(self.u_n.x.petsc_vec)
        self.u_n.x.scatter_forward()
        u_form = self.rho * self.c * self.d * self.u_n * ufl.dx
        local_heat = fem.assemble_scalar(fem.form(u_form))
        return self.comm.allreduce(local_heat, op=MPI.SUM)

    def compute_heat_generation_rate(self) -> float:
        """
        Computes the total heat generation rate from the source term:
            Q_total = integral( Q ) dA

        where Q is power per unit area [W/m^2], so the integral gives total power
        [W]. This matches the IGA formulation where the source is per unit area.
        The rank-local partial integrals are reduced across ranks.

        :returns: Heat generation rate in Watts
        """
        q_form = self.Q * ufl.dx
        local_rate = fem.assemble_scalar(fem.form(q_form))
        return self.comm.allreduce(local_rate, op=MPI.SUM)

    def step(self) -> None:
        """
        Advances the solution by one time step using the Crank-Nicolson scheme:
            (M + 0.5*dt*K) u^{n+1} = (M - 0.5*dt*K) u^n + dt*F

        The new state is stored in the internal PETSc state vector; call
        `compute_total_heat`, `solution`, or `snapshot_state` to read it out.
        Nothing here touches a fem.Function, by design (see `_build_cn_system`).
        """
        # Compute RHS: B * state + dt * F
        self.B.mult(self._state, self.tmp)
        self.rhs.zeroEntries()
        self.rhs.axpy(1.0, self.tmp)
        self.rhs.axpy(self.dt, self.F)

        # Solve A * u_new = rhs (factorization reused across steps).
        self.ksp.solve(self.rhs, self._u_new)

        # Advance: state <- u_new.
        self._u_new.copy(self._state)

    def snapshot_state(self) -> PETSc.Vec:
        """
        Returns a standalone PETSc-vector copy of the current state. Safe to call
        between steps (it does not touch any fem.Function), so callers can collect
        time snapshots during stepping without disturbing the integration. The
        caller owns the returned vector and should ``destroy()`` it when done.
        """
        return self._state.copy()

    def load_state(self, state: PETSc.Vec) -> None:
        """
        Sets the solver state from a snapshot (e.g. to re-evaluate diagnostics or
        write output for a stored time level after the loop).
        """
        state.copy(self._state)

    @property
    def solution(self) -> fem.Function:
        """
        Returns the current temperature as a DOLFINx Function (synced from the
        internal state vector).
        """
        self._state.copy(self.u_n.x.petsc_vec)
        self.u_n.x.scatter_forward()
        return self.u_n
