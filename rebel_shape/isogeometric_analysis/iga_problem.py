import mitsuba as mi
import numpy as np
import numpy.typing as npt
import scipy.sparse as sps


class IGAProblem:
    """
    Abstract base class for Isogeometric Analysis (IGA) problems.

    Isogeometric Analysis (IGA) is based on a simple idea that the same basis functions
    are used to represent the geometry and the solution to a weak variational problem
    using the Galerkin method. Starting from a Partial Differential Equation (PDE),
    one typically applies the following steps to go from an abstract PDE to a
    Galerkin formulation, and from the Galerkin formulation to an implementation:

        1. Specify the PDE and the domain on which it is solved (Omega). Specify all
           boundary conditions.

        2. Determine the functional space V in which the solution lies (e.g. H1 for
           most second-order PDEs, H2 for the bi-harmonic equation). It is also
           influenced by the choice of boundary conditions.

        3. Multiply the PDE by a test function phi(x), x in Omega, that lives in V
           and integrate over Omega

        4. Apply integration by parts until you are left with operators that are defined
           on the chosen functional space V (e.g. the Laplace operator is not defined
           for H1 functions, but the gradient is, as the H1 Sobolev space requires that
           the function be differentiable at least once, in the weak sense). This
           typically yields something of form:
                a(u, v) = f(v),
           where a is a bounded coercive linear form defined on V and f is a bounded
           continuous linear functional defined on V, u is the solution, and v is the
           test function. These assumptions guarantee the existence of the weak
           solution by the Lax-Milgram theorem.

        5. Choose a finite dimensional subspace of V, denoted Vh, which contains N
           basis functions phi_h^j, and apply the ansatz:
                u(x) = sum_j c_j phi_h^j (x), x in Omega,
           For IGA, the basis functions are just the basis functions of the
           interpolating spline, the same ones that define the geometry.

        6. Substitute the ansatz into the bilinear and linear forms and test against all
           functions phi_h^k that define the space Vh
                a(sum_j c_j phi_h^j(x), phi_h^k(x)) = f(phi_h^k), for all k.
           Since a is a bilinear form we can rewrite this as
                sum_j a(phi_h^j(x), phi_h^k(x)) c_j = f(phi_h^k), for all k.
           From this, it is clear that the left side is actually a matrix vector
           product and that a(phi_h^j(x), phi_h^k(x)) can be represented as a matrix.
           The right side is a vector. If the form is not bilinear, then a procedure
           for non-linear problems has to be used (Picard self-consistent scheme,
           Anderson acceleration, or Newton's method)

        7. Assemble the global system matrix and the global RHS vector and solve the
           system. In IGA, in the same manner as FEM, one chooses phi_k(x) such that
           they are non-zero only on a small number of "elements". In IGA these elements
           are knot-spans. Since most interactions will evaluate to zero, the global
           system matrix will be sparse, with the number of non-zero entries being
           O(N), where N is the number of basis functions. The typical pre-factor is
           ~100 and depends on the polynomial degree of the spline basis functions.
           One starts by assembling local matrices and vectors, one per element, and
           noting which basis functions live there. Then, these are combined into the
           full matrix.

        8. Impose the appropriate boundary conditions (BCs). Dirichlet boundary
           conditions are imposed by "locking" some degrees of freedom to a particular
           value. This entails other removing rows and columns from the matrix and
           rows from the vector. Alternatively, one can zero all the elements in the
           appropriate rows and columns and keep only 1s along the diagonal in the
           matrix and replace the RHS vector values with the fixed ones.
           More complex BCs require additional care (e.g. applying a lifting)

        9. Solve the final system of equations. In 1D and 2D typical problem sizes are
           such that it is possible, and sometimes faster, to perform a direct solve
           of the system using sparse LU or Cholesky factorization. However, in 3D the
           memory footprint of such a procedure would be prohibitively large, so we
           switch to a Krylov subspace solver, e.g. CG (symmetric matrices) or GMRES
           (general sparse matrices). These are iterative solvers, and due to the
           spectral properties of the matrices obtained by the Galerkin method they
           typically perform O(1/h_min) iterations, where h_min is the normalized
           length of the smallest element (e.g. if a 3D cube is subdivided into
           100x100x100 smaller elements, h_min = 0.01). Thus, the overall cost of the
           solve is O(N / h_min)

        10. Validate the solution. This can be done in various ways, some common routes
            include:
              a) Finding a reference solution of the problem in a simple case
              b) Manually setting the solution to a known value and using symbolic
                 differentiation to compute the right-hand side of the PDE.
              c) Ensuring that the boundary conditions are correctly enforced
              d) Visualize the solution to see if it "makes sense"
              3) Perform a convergence test: increase the number of basis functions and
                 plot the solution against the average element size h. The highest
                 resolution is omitted from the graph and kept as a "reference". Take
                 an appropriate norm of the error (e.g. L2 or H1, L_inf) and verify
                 that the rate of error reduction is consistent with the theoretical
                 bounds (e.g. often O(h^{p+1}) for L2 and O(h^{p}) for H1 error,
                 if the PDE has sufficient regularity for the Aubin-Nitsche trick).

    Needless to say, the analysis part has to be done on paper, but from step 7 the
    implementation aspect begins. It is the responsibility of all implementors of
    `IGAProblem` to expose methods that assemble the local matrices and vectors for
    all matrices that are needed (see `HeatSimulation` for an example where one needs
    more than one kind of matrix). This is the reason why this class does not prescribe
    any abstract interfaces - one can need many different matrices and vectors for
    a given problem.

    Unlike the high performance codes, like the FEniCSx project, local matrices are
    assembled fully in memory, and then the global matrix is assembled from these local
    matrices. In standard finite element codes the values are immediately inserted
    into the global sparse matrix as they are computed, which is more memory efficient,
    but requires more complex logic and loops. Since loops are a very expensive
    operation in Python (especially in the context of JIT compilation), this approach
    is avoided in favor of a more memory-intensive one. The memory overhead is 3-5x
    of the size of the global matrix, which is acceptable for most problems.

    The only two methods that are provided are the `assemble_global_matrix` and
    `assemble_global_vector`. Note that the current implementation destroys gradients:
    one cannot perform sensitivity analysis and inverse design (with gradients at least,
    it is still possible to use gradient-free schemes, when those are applicable).
    This could be circumvented by using pytorch for matrix construction and wrapping it
    into a drjit compatible layer. The details of this are left to future work.

    A point has to be made about singe vs double precision: almost all mathematical
    codes use double precision, mostly because solvers require well conditioned systems
    compared to the smallest relative error, and this requirement does not hold true
    for large systems that originate from FEM and IGA analysis - the condition number
    can easily exceed 10^5, comparable to ~10^7 "resolution" of float32 numbers.

    However, since the optimization loop and computer graphics are typically done in
    float32, this is kept for constructing local matrices and vectors, while the
    global matrix and vector are assembled from local matrices and vectors in float64.
    The accuracy of the obtained solution will still be bound by the precision of
    float32, but the solvers will be able to use float64 precision to avoid numerical
    issues.

    When assembling local and global matrices and vectors, one typically has to
    evaluate an integral that cannot be computed analytically, so numerical quadrature
    has to be used. The highest accuracy is achieved by using the Gauss-Legendre
    quadrature rule, which exact up to polynomial degree 2p-1, where p is the
    number of quadrature points and has a exponential convergence rate with respect
    to the number of quadrature points. However, unlike for FEM meshes where it is
    possible to integrate exactly, in IGA the curvature computation inevitably
    introduces non-polynomial terms. When combined with the float32 representation
    of the local matrices and vectors, it is clear that a small number of quadrature
    points are sufficient - typically the same order as the polynomial degree of the
    spline basis functions.
    """

    @staticmethod
    def assemble_global_matrix(
        local_indices: npt.NDArray[np.uint32], local_matrices: "mi.TensorXf"
    ) -> sps.csr_matrix:
        """
        Assembles the global matrix for the IGA problem for the provided local matrices.

        `local_indices` is a tensor of shape (n_elements, n_dofs_per_element), where
        n_elements is the total number of elements in the spline (number of knot spans),
        and n_dofs_per_element (number of active basis functions per span). For more
        complex splines (like T-splines), the number of dofs per element is not fixed,
        so some changes would be needed to support this kind of element. The indices
        need not be a Mitsuba tensor to preserve

        `local_matrices` are a Mitsuba tensor of shape (n_elements, n_dofs_per_element,
        n_dofs_per_element). The same remarks from above hold for this argument as
        well.

        :returns: A sparse matrix in the CSR format representing the global matrix.
        :raises ValueError: If the input tensors do not have the expected shapes.
        """
        if local_indices.ndim != 2 or local_matrices.ndim != 3:
            raise ValueError(
                "`local_indices` must be a 2D tensor and `local_matrices` a 3D tensor."
            )
        if local_indices.shape[0] != local_matrices.shape[0]:
            raise ValueError(
                "`indices and `local_matrices` must have the same number of elements."
            )
        if local_indices.shape[1] != local_matrices.shape[1]:
            raise ValueError(
                "`local_indices` and `local_matrices` must have the same number of "
                "degrees of freedom per element."
            )
        if local_matrices.shape[2] != local_matrices.shape[1]:
            raise ValueError(
                "`local_matrices` must be square matrices, i.e. the number of dofs per "
                "element must match the number of dofs per element."
            )

        # Get dimensions
        n_elements, n_dofs = local_indices.shape

        # Create row and column indices by repeating each index
        # Each local index needs to be paired with every other local index
        rows = np.repeat(local_indices[:, :, np.newaxis], n_dofs, axis=2)
        cols = np.repeat(local_indices[:, np.newaxis, :], n_dofs, axis=1)

        # Flatten the arrays for sparse matrix construction
        row_indices = rows.flatten()
        col_indices = cols.flatten()
        values = np.array(local_matrices.numpy().flatten(), dtype=np.float64)

        # Create the sparse matrix, summing duplicate entries
        n_global_dofs = np.max(local_indices) + 1  # Assuming 0-based indexing
        global_matrix = sps.coo_matrix(
            (values, (row_indices, col_indices)),
            shape=(n_global_dofs, n_global_dofs),
            dtype=np.float64,
        )
        # Convert to CSR format (efficient for operations)
        global_matrix = global_matrix.tocsr()

        return global_matrix

    @staticmethod
    def assemble_global_vector(
        local_indices: npt.NDArray[np.uint32], local_vectors: "mi.TensorXf"
    ) -> npt.NDArray[np.float64]:
        """
        Assembles the global vector for the IGA problem for the provided local vectors.

        `local_indices` is a tensor of shape (n_elements, n_dofs_per_element), where
        n_elements is the total number of elements in the spline (number of knot spans),
        and n_dofs_per_element (number of active basis functions per span). For more
        complex splines (like T-splines), the number of dofs per element is not fixed,
        so some changes would be needed to support this kind of element. The indices
        need not be a Mitsuba tensor to preserve.

        `local_vectors` are a Mitsuba tensor of shape (n_elements, n_dofs_per_element),
        representing the local contributions to the global vector. The same remarks
        from above hold for this argument as well.

        :returns: A 1D numpy array representing the global vector.
        :raises ValueError: If the input tensors do not have the expected shapes.
        """
        if local_indices.ndim != 2 or local_vectors.ndim != 2:
            raise ValueError(
                "`local_indices` must be a 2D tensor and `local_vectors` a 2D tensor."
            )
        if local_indices.shape[0] != local_vectors.shape[0]:
            raise ValueError(
                "`local_indices` and `local_vectors` must have the same number of "
                "elements."
            )
        if local_indices.shape[1] != local_vectors.shape[1]:
            raise ValueError(
                "`local_indices` and `local_vectors` must have the same number of "
                "degrees of freedom per element."
            )

        # Flatten the arrays for sparse vector construction
        row_indices = np.array(local_indices).flatten()
        values = local_vectors.numpy().flatten()

        # Create the global vector, summing duplicate entries
        n_global_dofs = np.max(local_indices) + 1  # Assuming 0-based indexing
        global_vector = np.zeros(n_global_dofs, dtype=np.float64)

        # Accumulate values to the global vector
        np.add.at(global_vector, row_indices, values)

        return global_vector
