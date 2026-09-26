import time

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.sparse import linalg

from rebel_shape import logger
from rebel_shape.isogeometric_analysis.reissner_mindlin_modal_problem import (
    ReissnerMindlinModalProblem,
)
from rebel_shape.isogeometric_analysis.sparse_cholesky import SparseCholeskySolver


class ModalAnalysisSolver:
    """
    Solves the generalized eigenvalue problem for Reissner-Mindlin modal analysis:

        K * phi = lambda * M * phi

    where K is the shell stiffness matrix and M is the consistent mass matrix.

    The solver uses a Cholesky-based shift-invert strategy: a mass-proportional
    spectral shift `epsilon` is added to K to make it positive definite, then
    a sparse Cholesky factorization (CHOLMOD via `scikit-sparse` when available,
    falling back to `cholespy`) provides the
    shift-invert operator for `scipy.sparse.linalg.eigsh`. The shift is
    subtracted from the eigenvalues after solving, so the eigenvectors and
    eigenvalues are exact (up to the shift being larger than the most negative
    eigenvalue).

    This approach preserves all symmetries of the problem (no DOF pinning),
    which is important for closed manifolds like spheres where the degeneracy
    pattern of the eigenspectrum is physically meaningful.
    """

    def __init__(
        self,
        modal_problem: ReissnerMindlinModalProblem,
    ) -> None:
        """
        Initialize the modal analysis solver and assemble system matrices.

        This constructor creates a solver for the generalized eigenvalue problem
        K*phi = lambda*M*phi, where K is the stiffness matrix and M is the mass
        matrix. Both matrices are assembled from the modal problem during
        initialization.

        :param modal_problem: The Reissner-Mindlin modal problem defining
            the shell geometry, material properties, and assembly routines.
        """
        self._modal_problem = modal_problem
        self._stiffness_matrix, self._mass_matrix = self._assemble_system_matrices()

    def _assemble_system_matrices(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """
        Assemble global stiffness and mass matrices for the modal problem.

        Delegates to the modal problem's `assemble_global_stiffness_matrix`
        and `assemble_global_mass_matrix` methods, which handle both standard
        and batched assembly internally.

        :returns: A tuple containing the global stiffness matrix and global
            mass matrix, both in CSR sparse format.
        """
        start_time = time.time()
        stiffness_matrix = self._modal_problem.assemble_global_stiffness_matrix()
        stiffness_time = time.time() - start_time

        start_time = time.time()
        mass_matrix = self._modal_problem.assemble_global_mass_matrix()
        mass_time = time.time() - start_time

        logger.debug(f"Modal stiffness assembly time: {stiffness_time:.2f} s.")
        logger.debug(f"Modal mass assembly time: {mass_time:.2f} s.")
        logger.info(f"Stiffness matrix shape: {stiffness_matrix.shape}")
        logger.info(f"Mass matrix shape: {mass_matrix.shape}")
        logger.info(f"Stiffness matrix nnz: {stiffness_matrix.nnz}")
        logger.info(f"Mass matrix nnz: {mass_matrix.nnz}")

        return stiffness_matrix, mass_matrix

    def solve(
        self,
        num_modes: int,
        solver_tolerance: float = 1.0e-8,
        max_iterations: int | None = None,
        spectral_shift: float | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Solve the generalized eigenvalue problem and return eigenpairs.

        The solver adds a mass-proportional spectral shift `epsilon` to the
        stiffness matrix to make it positive definite, then uses a sparse
        Cholesky factorization (CHOLMOD) for the shift-invert operator. The
        shift is subtracted from the eigenvalues after solving, so the
        eigenvectors and eigenvalues are exact.

        The eigenvalues are sorted in ascending order.

        :param num_modes: Number of eigenmodes to compute.
        :param solver_tolerance: Convergence tolerance for the iterative
            eigensolver.
        :param max_iterations: Maximum Arnoldi iterations. `None` uses
            scipy default.
        :param spectral_shift: The mass-proportional shift `epsilon` added
            to K to make it positive definite:
            `K_reg = K + epsilon * M`. Must be larger than the magnitude
            of any negative eigenvalue but small relative to the first
            elastic eigenvalue. If `None`, a heuristic based on the
            diagonal norms of K and M is used.

        :returns: A tuple of `(eigenvalues, eigenvectors)` where
            eigenvalues is a sorted array of shape `(num_modes,)` and
            eigenvectors has shape `(n_dofs, num_modes)`.
        """
        if num_modes < 1:
            raise ValueError("`num_modes` must be a positive integer.")
        if solver_tolerance <= 0.0:
            raise ValueError("`solver_tolerance` must be a positive real number.")

        eigenvalues, eigenvectors = self._solve_cholesky(
            stiffness_matrix=self._stiffness_matrix,
            mass_matrix=self._mass_matrix,
            num_modes=num_modes,
            solver_tolerance=solver_tolerance,
            max_iterations=max_iterations,
            spectral_shift=spectral_shift,
        )

        sorted_indices = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        return eigenvalues, eigenvectors

    def _solve_cholesky(
        self,
        stiffness_matrix: sp.csr_matrix,
        mass_matrix: sp.csr_matrix,
        num_modes: int,
        solver_tolerance: float,
        max_iterations: int | None,
        spectral_shift: float | None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Solve the eigenvalue problem using Cholesky-based shift-invert with
        a mass-proportional spectral shift on the full (unpinned) system.

        The regularized problem is:

            (K + epsilon * M) phi = (lambda + epsilon) * M * phi

        Since M-regularization does not change the eigenvectors and shifts
        all eigenvalues uniformly by epsilon, the original eigenvalues are
        recovered by subtracting epsilon after solving. No DOFs are pinned,
        so the full symmetry of the problem is preserved.
        """
        n = stiffness_matrix.shape[0]

        # Determine the spectral shift epsilon
        if spectral_shift is not None:
            epsilon = spectral_shift
        else:
            # Heuristic: use a small fraction of the estimated spectral radius.
            # K_diag_max / M_diag_max approximates the largest eigenvalue;
            # we use 1e-8 of that as the shift, which is sufficient to lift the
            # rigid body and drilling null spaces while being negligible
            # relative to the physical eigenvalues.
            K_diag_norm = np.abs(stiffness_matrix.diagonal()).max()
            M_diag_norm = np.abs(mass_matrix.diagonal()).max()
            epsilon = 1.0e-8 * K_diag_norm / max(M_diag_norm, 1.0e-30)

        K_reg = stiffness_matrix + epsilon * mass_matrix
        logger.info(f"Spectral shift epsilon: {epsilon:.4e}")

        # Cholesky factorization of K_reg (SPD after shift), used as the
        # shift-invert operator for every ARPACK iteration.
        cholesky_solver = SparseCholeskySolver(
            matrix=K_reg, layout=SparseCholeskySolver.Layout.CSC
        )

        K_inv_op = linalg.LinearOperator(
            shape=(n, n),
            matvec=cholesky_solver.solve,
            dtype=np.float64,
        )

        # Solve: K_reg phi = (lambda + epsilon) M phi via shift-invert.
        # `which="LM"` is the only correct choice here: shift-invert makes
        # ARPACK work with the eigenvalues of (K_reg - sigma * M)^-1 M, which
        # are 1 / (lambda - sigma), so the smallest lambda (the modes we want)
        # are the largest in magnitude. `"SM"` would silently return the
        # highest-frequency modes instead.
        logger.info(
            f"Solving eigenvalue problem (Cholesky shift-invert, "
            f"epsilon={epsilon:.4e}): num_modes={num_modes}"
        )

        start_time = time.time()
        eigenvalues_shifted, eigenvectors = linalg.eigsh(
            A=K_reg,
            k=num_modes,
            M=mass_matrix,
            sigma=0.0,
            which="LM",
            tol=solver_tolerance,
            maxiter=max_iterations,
            OPinv=K_inv_op,
        )
        solve_time = time.time() - start_time
        logger.info(f"Eigenvalue solve time: {solve_time:.2f} s.")

        # Subtract the shift to recover original eigenvalues
        eigenvalues = eigenvalues_shifted - epsilon

        return eigenvalues, eigenvectors

    def mass_orthogonality_error(
        self, eigenvectors: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Compute the mass-orthogonality error for a set of eigenvectors.

        For a set of eigenvectors that are solutions to the generalized eigenvalue
        problem K*phi = lambda*M*phi, the eigenvectors should satisfy mass-
        orthogonality: phi_i^T M phi_j = 0 for i != j.

        This method computes the mass product matrix phi^T M phi and returns the
        absolute error from perfect orthogonality (deviation from diagonal).

        :param eigenvectors: A matrix of eigenvectors with shape
            `(n_dofs, n_modes)` where each column is an eigenvector.

        :returns: A symmetric matrix of orthogonality errors where
            off-diagonal elements show the deviation from perfect
            mass-orthogonality. Values should be near zero for accurate
            eigenvectors.
        """
        mass_matrix = self._mass_matrix
        mass_product = eigenvectors.T @ (mass_matrix @ eigenvectors)
        diagonal = np.diag(np.diag(mass_product))
        return np.abs(mass_product - diagonal)
