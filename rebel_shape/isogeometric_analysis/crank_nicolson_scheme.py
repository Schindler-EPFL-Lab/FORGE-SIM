import time
from typing import Optional

import mitsuba as mi
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.sparse import linalg

from rebel_shape import logger
from rebel_shape.isogeometric_analysis.heat_simulation import HeatSimulation
from rebel_shape.isogeometric_analysis.sparse_cholesky import SparseCholeskySolver
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents


def compute_relaxation_time(
    stiffness_matrix: sp.csr_matrix,
    mass_matrix: sp.csr_matrix,
    num_modes: int = 3,
    tol: float = 1e-8,
) -> float:
    r"""
    Computes the natural relaxation time of the semi-discrete heat equation
    on a closed manifold from the smallest non-zero generalized eigenvalue
    of the assembled stiffness and mass matrices.

    For the semi-discrete system `M c_dot + K c = 0` the temperature decays
    as a sum of modes `exp(-lambda_i t) v_i`, where `lambda_i >= 0` solve
    the generalized eigenvalue problem
    ```math
        K v = \lambda M v.
    ```
    On a closed manifold the constant mode is steady, so `lambda_0 = 0`
    and the slowest decay rate is `lambda_1 > 0`. The natural relaxation
    time is `T_relax = 1 / lambda_1`. This characterization automatically
    accounts for heterogeneous material properties, surface geometry, and
    thickness variations; it is what the dimensional estimate
    `L^2 / alpha` is trying to approximate.

    The eigsolve uses ARPACK shift-invert via `scipy.sparse.linalg.eigsh`
    with a small negative shift `sigma < 0`. K is positive semi-definite
    (kernel = constants) and M is positive definite, so the shifted matrix
    `K - sigma * M = K + |sigma| M` is symmetric positive definite. We
    factor it once via `SparseCholeskySolver` and pass the resulting solver
    as `OPinv` to ARPACK, matching the direct-solver path used for the
    Crank-Nicolson LHS in this file, and avoiding the LU factorization
    scipy would otherwise build internally. `sigma` is taken as a small
    fraction of the element-scale rate `Tr(K)/Tr(M)` so it sits just below
    the bottom of the spectrum.

    `num_modes` is the number of eigenvalues to request; >= 3 gives some
    robustness against numerical degeneracy of the constant mode (e.g. when
    the eigsolve returns a tiny positive value where the exact eigenvalue
    is zero).

    :param stiffness_matrix: Global stiffness matrix K (CSR, symmetric
        positive semi-definite).
    :param mass_matrix: Global mass matrix M (CSR, symmetric positive
        definite).
    :param num_modes: Number of eigenvalues to request from ARPACK.
    :param tol: Relative tolerance for the ARPACK eigsolve.
    :returns: The natural relaxation time `T_relax = 1 / lambda_1` in
        seconds.

    :raises ValueError: If `num_modes` is less than 2.
    :raises RuntimeError: If no non-zero eigenvalue can be identified among
        the returned modes.
    """
    if num_modes < 2:
        raise ValueError("`num_modes` must be at least 2.")

    # `Tr(K)/Tr(M)` has units of 1/s and approximates an element-scale
    # relaxation rate. The global rate `1/T_relax` is much smaller, so a
    # small fraction of this scale places `sigma` just below the bottom of
    # the spectrum while keeping `K - sigma*M` well-conditioned.
    rate_scale = stiffness_matrix.diagonal().sum() / mass_matrix.diagonal().sum()
    if not np.isfinite(rate_scale) or rate_scale <= 0.0:
        raise RuntimeError(
            "Failed to derive a shift scale from K and M: diagonal trace "
            f"ratio is {rate_scale}. Check that the matrices are well-formed."
        )
    sigma = -1e-6 * rate_scale

    # Factor K - sigma*M with Cholesky once, then reuse it for every ARPACK
    # iteration.
    shifted = (stiffness_matrix - sigma * mass_matrix).tocsr()
    n_rows = shifted.shape[0]
    chol_solver = SparseCholeskySolver(
        matrix=shifted, layout=SparseCholeskySolver.Layout.CSR
    )

    opinv = linalg.LinearOperator(
        shape=(n_rows, n_rows), matvec=chol_solver.solve, dtype=np.float64
    )

    start_time = time.time()
    eigenvalues = linalg.eigsh(
        A=stiffness_matrix,
        k=num_modes,
        M=mass_matrix,
        sigma=sigma,
        which="LM",
        OPinv=opinv,
        tol=tol,
        return_eigenvectors=False,
    )
    eigsolve_time = time.time() - start_time

    eigenvalues = np.sort(eigenvalues)
    # The constant mode gives `lambda_0 ~ 0`; floating-point noise can yield
    # tiny positive or negative values, so filter on a relative cutoff
    # against the largest returned eigenvalue.
    cutoff = max(1e-12, 1e-6 * abs(eigenvalues[-1]))
    nonzero = eigenvalues[eigenvalues > cutoff]
    if nonzero.size == 0:
        raise RuntimeError(
            f"Could not identify a non-zero eigenvalue of (K, M) within the "
            f"requested {num_modes} modes (eigenvalues: {eigenvalues})."
        )

    lambda_1 = float(nonzero[0])
    logger.debug(
        f"Relaxation eigsolve: {num_modes} modes in {eigsolve_time:.3f} s, "
        f"sigma={sigma:.3e}, lambda_1={lambda_1:.6e}."
    )
    return 1.0 / lambda_1


class CrankNicolsonScheme:
    r"""
    This class solves the transient heat equation on a 2D manifold using the
    Crank-Nicolson time integration scheme.

    The heat equation is given by:
    ```math
        \partial_t \rho(x) c(x) d(x) u(x, t)
        - d(x) \nabla_M \cdot (k(x) \nabla_M u(x, t)) = f(x),
    ```
    where rho(x) is density, c(x) is specific heat, k(x) is thermal conductivity,
    d(x) is thickness, and f(x) is the source term. The Crank-Nicolson method
    is used to discretize the time derivative, leading to an implicit system of
    equations at each time step. The scheme is second-order accurate in time and
    unconditionally stable for linear problems.

    The Crank-Nicolson update is given by solving the following linear system
    ```math
        (M + 0.5 * dt * K) T^(n+1) = (M - 0.5 * dt * K) T^n + dt * F
    ```
    where M is the mass matrix, K is the stiffness matrix, F is the source term vector,
    dt is the time step, and T^n represents the temperature at time step n.

    The resulting system of equations can be solved in one of two ways:
        1. An iterative solver (preconditioned conjugate gradient, default).
        2. A direct solver (sparse Cholesky factorization via
           `SparseCholeskySolver`).
    The choice is controlled by the `use_cholesky` constructor flag. CG is the
    default because the LHS matrix can become quite large (>100k DOFs) and a
    direct factorization is memory-intensive. However, CG can occasionally
    exhibit slow convergence on stiff material configurations, in which case
    the Cholesky path provides a robust direct solve. Because the LHS matrix
    (M + 0.5*dt*K) is symmetric positive definite and constant across time
    steps, a single factorization is reused for every step.

    For the CG path, the lumped mass matrix is used as a preconditioner to
    accelerate convergence by about a factor of 3. The residual tolerance is set
    to 1e-7 by default as this is the machine precision of float32, and
    increasing the tolerance further does not improve accuracy.
    """

    MINIMUM_CG_TOLERANCE = 1e-12
    """Minimum allowed tolerance for the conjugate gradient solver."""

    def __init__(
        self,
        heat_simulation: HeatSimulation,
        time_step_size: float,
        per_step_events: OptimizationEvents,
        use_cholesky: bool = True,
    ) -> None:
        """
        Initializes the Crank-Nicolson scheme solver for the given `heat_simulation`
        problem.

        This constructor assembles all necessary matrices (stiffness, mass, and
        source vectors) and constructs the linear system matrices used in the
        Crank-Nicolson time stepping scheme.

        The `heat_simulation` object contains the discretized heat equation problem
        including material properties and geometry.
        `time_step_size` defines the temporal discretization in seconds and must
        be a positive real number.

        `per_step_events` allows for custom actions to be performed at each
        time step, such as logging or saving intermediate results. Study the
        `OptimizationEvents` class for more details.

        Upon initialization, the left-hand side matrix (M + 0.5*dt*K) and the
        right-hand side matrix (M - 0.5*dt*K) are assembled. When
        `use_cholesky` is False (the default), a lumped-mass preconditioner is
        built for the CG solver. When True, a sparse Cholesky factorization of
        the LHS matrix is precomputed via `SparseCholeskySolver` (double
        precision) and reused at every time step, since the LHS does not
        change.

        :raises ValueError: If `time_step_size` is not a positive real number.
        """
        if time_step_size <= 0.0:
            raise ValueError("`time_step_size` must be a positive real number.")
        self._time_step_size = time_step_size
        """Time step size in seconds for the Crank-Nicolson scheme."""

        self._heat_simulation = heat_simulation
        """Heat simulation class containing geometry and material properties."""
        self._per_step_events = per_step_events
        """Events to be executed at each time step."""
        self._use_cholesky = use_cholesky
        """If True, use a direct sparse Cholesky solver instead of CG."""

        self._stiffness_matrix, self._mass_matrix, self._global_vector = (
            self._assemble_tensors(heat_simulation)
        )

        # Set lazily by `from_normalized_time_step` (or on demand via the
        # `relaxation_time` property), as the eigsolve is only needed when
        # the time step is expressed relative to the natural relaxation
        # time of the discrete system.
        self._relaxation_time: float | None = None

        self._lhs_matrix, self._rhs_matrix = self._setup_system_matrices()

        # Solver-specific auxiliary state. Exactly one of these is populated.
        self._preconditioner: Optional[sp.csr_matrix] = None
        self._cholesky_solver: Optional[SparseCholeskySolver] = None
        if self._use_cholesky:
            self._cholesky_solver = self._build_cholesky_solver()
        else:
            self._preconditioner = self._build_cg_preconditioner()

    @classmethod
    def from_normalized_time_step(
        cls,
        heat_simulation: HeatSimulation,
        dt_over_t_relax: float,
        per_step_events: OptimizationEvents,
        use_cholesky: bool = True,
    ) -> "CrankNicolsonScheme":
        r"""
        Constructs a `CrankNicolsonScheme` with a time step expressed as a
        fraction of the natural relaxation time of the semi-discrete
        system on the given `heat_simulation`.

        The physical time step is `dt = dt_over_t_relax * T_relax`, where
        `T_relax = 1 / lambda_1` and `lambda_1` is the smallest non-zero
        generalized eigenvalue of `K v = lambda M v` for the assembled
        stiffness K and mass M matrices. This couples the time step to the
        actual slowest decay mode of the discretized problem, so the
        Crank-Nicolson temporal error `~ (dt / T_relax)^2` becomes the
        explicit accuracy knob, independent of geometry, units, and
        material heterogeneity. See `compute_relaxation_time` for details
        on the eigsolve.

        For Crank-Nicolson on this class of problems, `dt_over_t_relax` in
        the range `[0.005, 0.05]` is usually a sensible accuracy/runtime
        trade-off (the canonical recommendation is `~0.01`).

        Matrix assembly is performed once and reused for the eigsolve, so
        this classmethod does no extra work compared to calling the
        constructor directly with a precomputed `time_step_size`.

        :param heat_simulation: The heat equation problem to solve.
        :param dt_over_t_relax: Time step normalized by `T_relax`. Must be
            a positive real number.
        :param per_step_events: Callbacks scheduled per time step (see
            `OptimizationEvents`).
        :param use_cholesky: Same semantics as the constructor.

        :raises ValueError: If `dt_over_t_relax` is not a positive real number.
        """
        if dt_over_t_relax <= 0.0:
            raise ValueError("`dt_over_t_relax` must be a positive real number.")

        # Assemble K, M, F up front so the eigsolve and the CN system share
        # the same matrices. `__new__` skips the standard assembly path in
        # `__init__`; the rest of the initialization is performed inline.
        instance = cls.__new__(cls)
        instance._heat_simulation = heat_simulation
        instance._per_step_events = per_step_events
        instance._use_cholesky = use_cholesky

        instance._stiffness_matrix, instance._mass_matrix, instance._global_vector = (
            cls._assemble_tensors(heat_simulation)
        )

        relaxation_time = compute_relaxation_time(
            stiffness_matrix=instance._stiffness_matrix,
            mass_matrix=instance._mass_matrix,
        )
        time_step_size = dt_over_t_relax * relaxation_time
        logger.info(
            f"Natural relaxation time T_relax = {relaxation_time:.6g} s. "
            f"Using dt = {dt_over_t_relax:.4g} * T_relax = {time_step_size:.6g} s."
        )
        instance._relaxation_time = relaxation_time
        instance._time_step_size = time_step_size

        instance._lhs_matrix, instance._rhs_matrix = (
            instance._setup_system_matrices()
        )

        instance._preconditioner = None
        instance._cholesky_solver = None
        if instance._use_cholesky:
            instance._cholesky_solver = instance._build_cholesky_solver()
        else:
            instance._preconditioner = instance._build_cg_preconditioner()

        return instance

    @staticmethod
    def _assemble_tensors(
        heat_simulation: HeatSimulation,
    ) -> tuple[sp.csr_matrix, sp.csr_matrix, npt.NDArray[np.float64]]:
        """
        Assembles the stiffness matrix, mass matrix, and global source vector for the
        heat equation.

        This method constructs the global IGA matrices by first assembling
        local element contributions and then combining them into sparse CSR
        (Compressed Sparse Row) format matrices. The stiffness matrix K represents
        the spatial discretization of the diffusion term, the mass matrix M
        represents the temporal derivative term with material properties, and the
        global vector F represents the source term contributions.

        Assembly times and matrix properties (shape, number of non-zero entries) are
        logged for performance monitoring and debugging purposes.

        :returns: A tuple containing:

            - `stiffness_matrix`: The global stiffness matrix K in CSR sparse format.
            - `mass_matrix`: The global mass matrix M in CSR sparse format.
            - `global_vector`: The global source term vector F as a numpy array.
        """
        # Stiffness matrix assembly
        start_time = time.time()
        local_indices, local_matrices = (
            heat_simulation.assemble_local_stiffness_matrices()
        )
        stiffness_matrix = heat_simulation.assemble_global_matrix(
            local_indices=local_indices, local_matrices=local_matrices
        )
        elapsed_time_stiffness = time.time() - start_time

        # Mass matrix assembly
        start_time = time.time()
        local_indices, local_matrices = (
            heat_simulation.assemble_local_mass_matrices()
        )
        mass_matrix = heat_simulation.assemble_global_matrix(
            local_indices=local_indices, local_matrices=local_matrices
        )
        elapsed_time_mass = time.time() - start_time

        # Global vector assembly (right-hand side)
        start_time = time.time()
        local_indices, local_vectors = heat_simulation.assemble_local_vectors(
            rhs_term=heat_simulation.source_term
        )
        global_vector = heat_simulation.assemble_global_vector(
            local_indices=local_indices, local_vectors=local_vectors
        )
        elapsed_time_vector = time.time() - start_time

        # Logging assembly times and matrix properties
        logger.debug(f"Stiffness matrix assembly time: {elapsed_time_stiffness:.2f} s.")
        logger.info(f"Stiffness matrix shape: {stiffness_matrix.shape}")
        logger.info(f"Number of non-zero entries: {stiffness_matrix.nnz}")
        logger.debug(f"Mass matrix assembly time: {elapsed_time_mass:.3f} s.")
        logger.debug(f"Global vector assembly time: {elapsed_time_vector:.3f} s.")

        return stiffness_matrix, mass_matrix, global_vector

    def _setup_system_matrices(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """
        Constructs the left-hand side and right-hand side matrices for the
        Crank-Nicolson scheme.

        This method builds the system matrices required for the implicit time-stepping
        algorithm. The left-hand side matrix is (M + 0.5*dt*K) and a system solve is
        required at each time step, while the right-hand side matrix is (M - 0.5*dt*K)
        and is applied to the current temperature field.

        :returns: A tuple containing:

            - `lhs_matrix`: The left-hand side system matrix (M + 0.5*dt*K).
            - `rhs_matrix`: The right-hand side system matrix (M - 0.5*dt*K).
        """
        # right-hand side matrix: M - 0.5*dt*K
        system_matrix_low = (
            self._mass_matrix - 0.5 * self._time_step_size * self._stiffness_matrix
        )

        # left-hand side matrix: M + 0.5*dt*K
        system_matrix_high = (
            self._mass_matrix + 0.5 * self._time_step_size * self._stiffness_matrix
        )

        return system_matrix_high, system_matrix_low

    def _build_cg_preconditioner(self) -> sp.csr_matrix:
        """
        Builds the inverse lumped mass matrix used as a preconditioner for the
        preconditioned conjugate gradient solver.

        The lumped mass matrix is obtained by summing rows of the consistent
        mass matrix, resulting in a diagonal matrix that can be efficiently
        inverted; the preconditioning is then a simple element-wise multiplication.
        Since the time step is comparatively small and the mass matrix dominates
        the condition number, this preconditioner accelerates convergence of the
        CG solver by roughly a factor of 3.
        """
        lumped_mass_matrix_diag = self._mass_matrix.sum(axis=1).A1
        return sp.diags(1.0 / lumped_mass_matrix_diag, format="csr", dtype=np.float64)

    def _build_cholesky_solver(self) -> SparseCholeskySolver:
        """
        Precomputes a sparse Cholesky factorization of the LHS matrix
        (M + 0.5*dt*K) in double precision.

        The LHS matrix does not depend on time, so a single factorization is
        reused at every time step.
        """
        return SparseCholeskySolver(
            matrix=self._lhs_matrix.tocsr(),
            layout=SparseCholeskySolver.Layout.CSR,
        )

    def simulate(
        self,
        num_iterations: int,
        cg_residual_tolerance: float = 1e-7,
        drift_reporting_interval: int = 30,
    ) -> npt.NDArray[np.float64]:
        """
        Executes the transient heat simulation using the Crank-Nicolson
        time-stepping scheme.

        This method performs the main time-stepping loop, advancing the temperature
        field forward in time for a specified number of iterations. At each time
        step, the implicit Crank-Nicolson update is applied by solving the linear
        system with either a preconditioned conjugate gradient (CG) solver or
        the precomputed sparse Cholesky factorization, depending on the
        `use_cholesky` flag passed to the constructor.

        `num_iterations` controls how many time steps to simulate, and
        `per_step_callback` is called after each iteration with the current
        iteration number as an argument, allowing for custom actions such as saving
        results or rendering of the temperature field.

        The CG solver tolerance is controlled by `cg_residual_tolerance`, which
        determines the relative residual tolerance for convergence.
        The default value of `1e-7` is suitable for single-precision and should not
        be decreased further as it does not improve accuracy.
        Increasing the tolerance can significantly speed up simulation but the thermal
        drift becomes much more pronounced. This argument is ignored when the
        Cholesky solver is selected.

        The `drift_reporting_interval` specifies how often (in terms of iteration
        count) the total heat drift is computed and logged to monitor energy
        conservation.
        Heat drift is calculated as the difference between the actual total heat and
        the expected heat based on the initial heat plus accumulated heat generation
        from the source term. This follows from a simple energy balance check.

        Performance metrics such as total simulation time and time per iteration are
        logged after the simulation completes. After each iteration, any registered
        events in `per_step_events` are triggered.

        :returns: The final temperature field as a flattened numpy array
            of nodal temperature values.

        :raises ValueError: If `num_iterations` or `drift_reporting_interval`
            are not positive integers, or if `cg_residual_tolerance` is not
            a positive real number.
        """
        if num_iterations < 1:
            raise ValueError("`num_iterations` must be a positive integer.")
        if drift_reporting_interval < 1:
            raise ValueError("`drift_reporting_interval` must be a positive integer.")

        # Small tolerance is can lead to excessive CG iterations without convergence
        if cg_residual_tolerance < self.MINIMUM_CG_TOLERANCE:
            raise ValueError(
                "`cg_residual_tolerance` must be a positive real number greater than "
                f"the minimal tolerance: {self.MINIMUM_CG_TOLERANCE}."
            )

        initial_heat = self._heat_simulation.total_heat
        heat_generation_rate = self._heat_simulation.heat_generation_rate

        logger.info(f"Initial total heat:   {initial_heat} J")
        logger.info(f"Heat generation rate: {heat_generation_rate} W")

        temperature = self._heat_simulation.temperature.numpy().flatten()

        start_time = time.time()
        for it in range(num_iterations):
            # u_{n+1} = (M + 0.5*dt*K)^{-1} @ [(M - 0.5*dt*K) @ u_n + dt * f]
            rhs = (
                self._rhs_matrix @ temperature
                + self._time_step_size * self._global_vector
            )

            solve_start = time.time()
            if self._use_cholesky:
                temperature = self._cholesky_solver.solve(rhs)
                solver_name = "Cholesky"
            else:
                temperature = linalg.cg(
                    A=self._lhs_matrix,
                    b=rhs,
                    x0=temperature,
                    rtol=cg_residual_tolerance,
                    M=self._preconditioner,
                )[0]
                solver_name = "CG"
            logger.debug(
                f"{solver_name} solver time iteration {it}: "
                f"{time.time() - solve_start:.3f} s."
            )

            # Set the internal temperature to the newly computed value
            self._heat_simulation.temperature = mi.TensorXf(temperature.reshape(-1, 1))

            # Report the drift in total heat
            if (it + 1) % drift_reporting_interval == 0 or it == (num_iterations - 1):
                total_heat = self._heat_simulation.total_heat
                expected_heat = (
                    initial_heat
                    + (it + 1) * self._time_step_size * heat_generation_rate
                )

                heat_drift = (total_heat - expected_heat).numpy()
                logger.debug(
                    f"Iteration {it + 1}/{num_iterations} "
                    f"Heat drift: {heat_drift:.6f} J "
                    f"(relative {heat_drift / expected_heat:.3e})"
                )

            # Trigger callbacks assigned to a given time step
            self._per_step_events.trigger_events(iteration=it)

        loop_time = time.time() - start_time
        logger.info(f"Total loop time: {loop_time:.3f} seconds")
        logger.info(f"Time per time step: {loop_time / num_iterations:.4f} seconds")

        return temperature

    @property
    def time_step_size(self) -> float:
        """Physical time step size in seconds for the Crank-Nicolson scheme."""
        return self._time_step_size

    @property
    def relaxation_time(self) -> float:
        """
        Natural relaxation time of the semi-discrete system in seconds.

        Defined as `1 / lambda_1`, where `lambda_1` is the smallest non-zero
        generalized eigenvalue of `K v = lambda M v`. Computed lazily on
        first access via `compute_relaxation_time` and cached. When the
        scheme is built via `from_normalized_time_step` the value is
        already populated and no eigsolve is triggered.
        """
        if self._relaxation_time is None:
            self._relaxation_time = compute_relaxation_time(
                stiffness_matrix=self._stiffness_matrix,
                mass_matrix=self._mass_matrix,
            )
        return self._relaxation_time

    @property
    def normalized_time_step(self) -> float:
        """
        Time step normalized by the natural relaxation time, `dt / T_relax`.

        Equals the `dt_over_t_relax` passed to `from_normalized_time_step`
        when the scheme was constructed that way. For schemes built from a
        physical `time_step_size`, accessing this property triggers the
        relaxation-time eigsolve via the `relaxation_time` property.
        """
        return self._time_step_size / self.relaxation_time

    def scale_source(self, factor: float) -> None:
        """
        Scale the assembled source vector and the underlying source field.

        The assembled vector F is linear in the source field, so scaling
        by `factor` is equivalent to having assembled with
        `source_field *= factor`.  Both `_global_vector` (used in the
        CN time loop) and `HeatSimulation.source_term` (used for
        drift reporting via `heat_generation_rate`) are updated.
        """
        self._global_vector *= factor
        self._heat_simulation.source_term = self._heat_simulation.source_term * factor
