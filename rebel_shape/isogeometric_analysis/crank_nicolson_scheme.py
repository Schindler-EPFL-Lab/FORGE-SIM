import time

import mitsuba as mi
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.sparse import linalg

from rebel_shape import logger
from rebel_shape.isogeometric_analysis.heat_simulation import HeatSimulation
from rebel_shape.scene_optimizers.optimization_events import OptimizationEvents


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
        1. Using a direct solver (e.g., LU decomposition) for small systems.
        2. Using an iterative solver (e.g., conjugate gradient) for larger systems.
    The choice of solver depends on the size of the system and the available memory.

    Here, a preconditioned conjugate gradient solver is used because the system can
    become quite large (>100k DOFs), so a direct solver is too memory-intensive.
    The lumped mass matrix is used as a preconditioner to accelerate convergence
    of the conjugate gradient solver by about a factor of 3. The residual tolerance
    is set to 1e-7 by default as this is the machine precision of float32, and
    increasing the tolerance further does not improve accuracy.
    """

    MINIMUM_CG_TOLERANCE = 1e-12
    """Minimum allowed tolerance for the conjugate gradient solver."""

    def __init__(
        self,
        heat_simulation: HeatSimulation,
        time_step_size: float,
        per_step_events: OptimizationEvents,
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

        Upon initialization, the left-hand side matrix (M + 0.5*dt*K), right-hand
        side matrix (M - 0.5*dt*K), and a preconditioner based on the lumped mass
        matrix are computed and stored for use during the time-stepping iterations.

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

        self._stiffness_matrix, self._mass_matrix, self._global_vector = (
            self._setup_tensors()
        )
        self._lhs_matrix, self._rhs_matrix, self._preconditioner = (
            self._setup_system_matrices()
        )

    def _setup_tensors(
        self,
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
            self._heat_simulation.assemble_local_stiffness_matrices()
        )
        stiffness_matrix = self._heat_simulation.assemble_global_matrix(
            local_indices=local_indices, local_matrices=local_matrices
        )
        elapsed_time_stiffness = time.time() - start_time

        # Mass matrix assembly
        start_time = time.time()
        local_indices, local_matrices = (
            self._heat_simulation.assemble_local_mass_matrices()
        )
        mass_matrix = self._heat_simulation.assemble_global_matrix(
            local_indices=local_indices, local_matrices=local_matrices
        )
        elapsed_time_mass = time.time() - start_time

        # Global vector assembly (right-hand side)
        start_time = time.time()
        local_indices, local_vectors = self._heat_simulation.assemble_local_vectors(
            rhs_term=self._heat_simulation._source_term
        )
        global_vector = self._heat_simulation.assemble_global_vector(
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

    def _setup_system_matrices(
        self,
    ) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
        """
        Constructs the left-hand side and right-hand side matrices for the
        Crank-Nicolson scheme.

        This method builds the system matrices required for the implicit time-stepping
        algorithm. The left-hand side matrix is (M + 0.5*dt*K) and a system solve is
        required at each time step, while the right-hand side matrix is (M - 0.5*dt*K)
        and is applied to the current temperature field.

        Additionally, as the time step is comparatively small, the mass matrix
        dominates the condition numbers, a preconditioner based on the inverse of the
        lumped mass matrix is computed to accelerate the iterative conjugate gradient
        (CG) solver used for solving the linear system at each time step.

        The lumped mass matrix is obtained by summing rows of the consistent mass
        matrix, resulting in a diagonal matrix that can be efficiently inverted, and
        the preconditioning is simply an element-wise multiplication.

        :returns: A tuple containing:

            - `lhs_matrix`: The left-hand side system matrix (M + 0.5*dt*K).
            - `rhs_matrix`: The right-hand side system matrix (M - 0.5*dt*K).
            - `preconditioner`: The inverse lumped mass matrix used as a preconditioner
                for the conjugate gradient solver, stored as a diagonal CSR matrix.
        """
        # right-hand side matrix: M - 0.5*dt*K
        system_matrix_low = (
            self._mass_matrix - 0.5 * self._time_step_size * self._stiffness_matrix
        )

        # left-hand side matrix: M + 0.5*dt*K
        system_matrix_high = (
            self._mass_matrix + 0.5 * self._time_step_size * self._stiffness_matrix
        )

        # Lumped mass matrix for preconditioning
        lumped_mass_matrix_diag = self._mass_matrix.sum(axis=1).A1

        # Pre-conditioner for the conjugate gradient solver
        inv_lumped_mass_matrix = sp.diags(
            1.0 / lumped_mass_matrix_diag, format="csr", dtype=np.float64
        )

        return system_matrix_high, system_matrix_low, inv_lumped_mass_matrix

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
        system using a preconditioned conjugate gradient (CG) solver.

        `num_iterations` controls how many time steps to simulate, and
        `per_step_callback` is called after each iteration with the current
        iteration number as an argument, allowing for custom actions such as saving
        results or rendering of the temperature field.

        The CG solver tolerance is controlled by `cg_residual_tolerance`, which
        determines the relative residual tolerance for convergence.
        The default value of `1e-7` is suitable for single-precision and should not
        be decreased further as it does not improve accuracy.
        Increasing the tolerance can significantly speed up simulation but the thermal
        drift becomes much more pronounced.

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

            cg_start = time.time()
            temperature = linalg.cg(
                A=self._lhs_matrix,
                b=rhs,
                x0=temperature,
                rtol=cg_residual_tolerance,
                M=self._preconditioner,
            )[0]
            logger.debug(
                f"CG solver time iteration {it}: {time.time() - cg_start:.3f} s."
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
