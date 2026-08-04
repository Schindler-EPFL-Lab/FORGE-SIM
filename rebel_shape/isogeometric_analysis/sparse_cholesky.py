"""
Sparse Cholesky factorization of symmetric positive-definite systems.

The isogeometric solvers share one linear-algebra need: factor a single SPD
matrix and then solve against it many times with different right-hand sides.
The Crank-Nicolson scheme reuses the factorization of `M + 0.5*dt*K` at every
time step, and the modal and relaxation eigensolves reuse the factorization of
their shifted matrix at every ARPACK iteration.

Two backends implement that need, and the faster one is not always installed:

- `scikit-sparse` (CHOLMOD with NESDIS nested-dissection ordering) is
  substantially faster on the FEM/IGA sparsity patterns assembled here and is
  used whenever it is importable.
- `cholespy` is a declared core dependency and is used otherwise, so the
  solvers stay functional on images that ship neither `scikit-sparse` nor the
  system SuiteSparse (the Azure ML images, for instance).

The two backends order the elimination differently, so they agree only to
within floating-point rounding rather than bit for bit.
"""

import time
from enum import Enum

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from rebel_shape import logger


class SparseCholeskySolver:
    """
    A sparse Cholesky factorization that is computed once and applied many
    times.

    The matrix must be symmetric positive definite; neither backend checks
    this, and both will fail or return nonsense otherwise.

    Example usage::

        solver = SparseCholeskySolver(
            matrix=lhs_matrix, layout=SparseCholeskySolver.Layout.CSR
        )
        for right_hand_side in right_hand_sides:
            solution = solver.solve(right_hand_side)
    """

    class Layout(Enum):
        """
        Sparse layout in which the matrix is handed to the `cholespy` backend.

        `cholespy` is given the index buffers of the requested layout and the
        matching `MatrixType`. The choice is explicit rather than defaulted
        because the assembled IGA matrices are symmetric only up to assembly
        rounding, so the CSR and CSC buffers of one matrix are not identical
        and a call site's factorization is only reproducible if its layout is
        preserved. The CHOLMOD backend always receives CSC, which is the
        layout `scikit-sparse` requires.
        """

        CSR = "csr"
        CSC = "csc"

    def __init__(self, matrix: sp.spmatrix, layout: "SparseCholeskySolver.Layout"):
        """
        Factorizes `matrix`, preferring CHOLMOD and falling back to `cholespy`.

        :param matrix: The square symmetric positive-definite matrix to
            factorize, in any scipy sparse format.
        :param layout: The layout handed to the `cholespy` backend; see
            `Layout`.

        :raises ValueError: If `matrix` is not square.
        """
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"`matrix` must be square, got shape {matrix.shape}.")

        start_time = time.time()
        try:
            self._solve_factorized = self._factorize_cholmod(matrix=matrix)
            self._backend = "scikit-sparse (CHOLMOD/NESDIS)"
        except ImportError as error:
            logger.warning(
                f"scikit-sparse unavailable ({error}); falling back to cholespy. "
                "Install scikit-sparse + libsuitesparse-dev for ~30x faster "
                "Cholesky factorization."
            )
            self._solve_factorized = self._factorize_cholespy(
                matrix=matrix, layout=layout
            )
            self._backend = "cholespy"
        factorization_time = time.time() - start_time

        logger.info(
            f"Cholesky factorization time: {factorization_time:.2f} s "
            f"(backend: {self._backend}, n={matrix.shape[0]}, nnz={matrix.nnz})."
        )

    @staticmethod
    def _factorize_cholmod(matrix: sp.spmatrix):
        """
        Factorizes `matrix` with CHOLMOD through `scikit-sparse`, using the
        NESDIS nested-dissection ordering.

        :raises ImportError: If `scikit-sparse` or the underlying SuiteSparse
            library is unavailable, which selects the `cholespy` fallback.
        """
        from sksparse.cholmod import cho_factor

        factor = cho_factor(matrix.tocsc(), order="nesdis")

        def solve(right_hand_side: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return factor.solve(right_hand_side)

        return solve

    @staticmethod
    def _factorize_cholespy(matrix: sp.spmatrix, layout: "SparseCholeskySolver.Layout"):
        """
        Factorizes `matrix` with `cholespy`, which requires int32 index buffers
        and contiguous float64 values.
        """
        from cholespy import CholeskySolverD, MatrixType

        if layout is SparseCholeskySolver.Layout.CSR:
            converted, matrix_type = matrix.tocsr(), MatrixType.CSR
        else:
            converted, matrix_type = matrix.tocsc(), MatrixType.CSC

        solver = CholeskySolverD(
            converted.shape[0],
            np.ascontiguousarray(converted.indptr, dtype=np.int32),
            np.ascontiguousarray(converted.indices, dtype=np.int32),
            np.ascontiguousarray(converted.data, dtype=np.float64),
            matrix_type,
        )

        def solve(right_hand_side: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            solution = np.empty_like(right_hand_side)
            solver.solve(right_hand_side, solution)
            return solution

        return solve

    def solve(
        self, right_hand_side: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Solves the factorized system for the given right-hand side.

        :param right_hand_side: The right-hand side vector, of length equal to
            the matrix dimension. It is flattened and converted to a contiguous
            float64 array, as both backends require.

        :returns: The solution as a flat float64 array.
        """
        return self._solve_factorized(
            np.ascontiguousarray(right_hand_side, dtype=np.float64).ravel()
        )

    @property
    def backend(self) -> str:
        """Human-readable name of the backend that performed the factorization."""
        return self._backend
