from collections.abc import Callable

import drjit as dr
import mitsuba as mi
import numpy as np
import scipy.sparse as sp

from forge_sim.isogeometric_analysis.galerkin_projector import GalerkinProjector
from forge_sim.splines.iga_spline_patch import IGASplinePatch


class GalerkinProjectorSmooth(GalerkinProjector):
    """
    A class for projecting functions onto IGA spline basis using the L2 norm.

    This class implements Galerkin projection methods to approximate discontinuous
    functions using spline basis functions. To avoid oscillations (spline equivalent
    of the Runge phenomenon), it uses a regularized L2 projection method that
    incorporates a Laplacian term to smooth the approximation.

    Conceptually, we want u_h(xi) ~= f(xi) for all parametric coordinates xi in the
    parametric domain of the spline surface with an additional regularization term
    that penalizes rapid variations in the solution. u_h is the approximation of f in
    the space defined by basis functions phi_j, so we have

        u_h(xi) = sum_j u_j phi_j(xi).

    Drawing inspiration from the weak formulation of PDEs, we can add a small diffusion
    term to the left-hand side of the equation. To arrive at a Galerkin formulation,
    multiply both sides by a test function that lies in the same space, a basis
    function phi_k, and integrate over the parametric domain (denoted as uv):

        - int_uv lambda Laplacian(u_h(xi)) phi_k(xi) dA + int_uv u_h(xi) phi_k(xi) dA
        = int_uv f(xi) phi_k(xi) dA.

    Lambda is a small positive constant that controls the strength of the diffusion.
    Note that this is different to the standard IGA weak formulation, because we are
    integrating over the parametric domain, not the physical domain. This simplifies
    many aspects of the implementation, as we do not need to compute the Jacobian
    or the metric tensor for the physical domain. Identifying the matrices and vectors
    in the Galerkin projection, we have:

        A_jk = int_uv (d(phi_j(xi))/d xi dot d(phi_k(xi))/d xi) dA,
        M_jk = int_uv phi_j(xi) phi_k(xi) dA,
        f_k = int_uv f(xi) phi_k(xi) dA,

        (lambda * A + M) u = f,

    where A is the global stiffness matrix, M is the global mass matrix, u is the
    vector of coefficients for the spline basis functions, and f is the global
    right-hand side vector obtained by integrating the target function against the
    basis functions.

    Solving this system of equations gives us the coefficients that represent the
    best approximation of the target function in the L2 norm with an additional
    regularization term that penalizes rapid variations in the solution. An
    appropriately chosen lambda does not excessively increase the L2 error
    compared to the standard L2 projection, while preventing oscillations in the
    approximation. This is particularly useful for discontinuous functions.

    The implementation follows the standard Galerkin method:
    1. Assemble local mass (and stiffness) matrices for each knot span
    2. Assemble local right-hand side vectors by integrating the target function
       against the basis functions
    3. Assemble global matrices and vectors
    4. Solve the resulting linear system using a conjugate gradient solver.
    """

    def __init__(self, spline_surface: IGASplinePatch):
        """
        Initialize the Galerkin projector with a spline surface.
        """
        super().__init__(spline_surface)

    def _assemble_local_stiffness_matrices(self) -> tuple["mi.TensorXu", "mi.TensorXf"]:
        """
        Assembles the local stiffness matrices for the Galerkin projection problem on
        the spline surface.

        This method computes the local stiffness matrices for each knot span of the
        spline surface. The stiffness matrix entries are computed using the formula:

            A_jk^(e) ~= area_e sum_q w_q (d(phi_j(xi_q))/d xi dot d(phi_k(xi_q))/d xi),

        where A_jk^(e) is the local stiffness matrix (entry), w_q are the Gauss-Legendre
        quadrature weights, xi_q are the quadrature points in the parametric domain,
        phi_j and phi_k are the spline basis functions.

        Time complexity: O(n_knot_spans * n_dofs^2 * n_quadrature_points^2),
        where n_dofs is the number of degrees of freedom per knot span. With the
        default quadrature order (degree + 1) and n_dofs ~ (degree + 1)^2, this
        becomes O(n_knot_spans * degree^6), but the dominant term is interpolation
        which is O(n_knot_spans * degree^4).

        :returns: A tuple that consists of

            - dofs_per_span: A tensor of shape (n_knot_spans, n_dofs) that contains
              the indices of the degrees of freedom for each knot span.
            - local_matrices: A tensor of shape (n_knot_spans, n_dofs, n_dofs) that
              contains the local stiffness matrices for each knot span.
        """
        interpolating_spline = self._spline.interpolating_spline

        knot_span_handler = self.KnotSpanHandler(
            patch=self._spline, integration_order=self._spline.degree + 1
        )
        num_dofs = interpolating_spline.basis_functions_per_span

        local_matrices = dr.zeros(
            dtype=mi.TensorXf,
            shape=(knot_span_handler.num_knot_spans, num_dofs, num_dofs),
        )

        for parametric_coords, weight in knot_span_handler.gl_coords_and_weights():
            basis_function_derivative = (
                interpolating_spline.evaluate_basis_functions_first_derivative(
                    parametric_coords=parametric_coords
                )
            )

            for i in range(num_dofs):
                dot_product = dr.sum(
                    value=(
                        basis_function_derivative[:, :, :]
                        * basis_function_derivative[:, i, :][:, None, :]
                    ),
                    axis=2,
                )
                values = dot_product * weight[:, None]

                self._scatter_values_to_local_matrices(
                    local_matrices=local_matrices,
                    values=values,
                    num_knot_spans=knot_span_handler.num_knot_spans,
                    num_dofs=num_dofs,
                    row=i,
                )

        dofs_per_span = interpolating_spline.active_basis_function_indices(
            parametric_coords=knot_span_handler.midpoint_parametric_coordinates
        )

        return dofs_per_span, local_matrices

    def galerkin_project_l2(
        self,
        rhs_term_function: Callable[["mi.TensorXf"], "mi.TensorXf"],
        smoothing_strength: float = 0.5,
    ) -> "mi.TensorXf":
        """
        Projects the `rhs_term_function` onto the spline surface using smooth L2
        Galerkin projection.

        This method finds the best approximation of the target function in the L2 norm
        with an additional regularization term that penalizes rapid variations in the
        solution. This is done by adding a diffusion-like term to the system that we
        aim to solve

            -lambda * Laplacian(u) + u = f,

        Carrying out the multiplication by the basis functions, integrating over the
        parametric domain and introducing the global matrices, we obtain the final
        system of equations to solve:

            (lambda * A + M) u = f,

        where A is the global stiffness matrix, M is the global mass matrix, u is the
        vector of coefficients for the spline basis functions, f is the global
        right-hand side vector, and lambda is the smoothing strength. Taking into
        account the scaling of the stiffness matrix (gradients scale as 1/h, where h is
        the knot span size), lambda can be expressed as

            lambda = smoothing_strength * h^2,

        which ensures that the regularization term is not excessively large compared
        to the mass matrix. The scale factor h is defined as the maximum knot span
        size, and is thus a conservative estimate - it is assumed that when this
        function is called the stability of the interpolation is much more important
        than purely minimizing the L2 error.

        The `smoothing_strength` parameter controls how strongly the Laplacian term
        regularizes the solution. A higher value results in smoother approximations,
        with 0.5 being the smallest value that prevents oscillations, i.e. removes
        the spline equivalent of the Runge phenomenon. This assumes that all knot
        spans have the same size.

        if the function that is approximated is smooth, then this method should not be
        used as it will significantly reduce the accuracy of the approximation. While
        the usual accuracy is O(h^(p+1)), where h is the largest knot span size and p is
        the degree of the spline, the accuracy of this method is at best O(h^2) due to
        the magnitude of the regularization term being proportional to h^2.

        For functions with jumps, the L2 error estimate deteriorates to just O(h^0.5),
        so the added regularization term does destroy the approximation in the same
        way it does for smooth functions - it only increases the L2 error by a factor
        ~2, depending on the smoothing strength and the function being approximated.
        The major advantage is that it prevents oscillations which are inevitably
        present in the non-regularized L2 projection and are typically not desired in
        practical applications.

        Note that the method bears some resemblance to the H1 projection, which
        minimizes the H1 norm of the error, that is the L2 norm of the function plus
        the L2 norm of the gradient:

        ||f - u_h||_{H1}^2 = int_M (f(x) - u_h(x))^2 dA + int_M ||grad(f - u_h)||^2 dA.

        However, since we wish to compute with discontinuous functions, their gradient
        is not well-defined in the strong sense, so we cannot compute the H1 norm
        directly.

        Time complexity: O(n_knot_spans * degree^4) for matrix assembly plus
        O(n_dofs^1.5) for the CG solver, where n_dofs is the total number of degrees
        of freedom, proportional to n_knot_spans * (degree + 1)^2.

        :returns: A tensor of shape (n_dofs, 1) containing the coefficients of the
            spline basis functions that represent the H1 projection of the target
            function onto the spline surface.
        """
        local_indices, local_matrices = self._assemble_local_mass_matrices()
        global_mass_matrix = self.assemble_global_matrix(
            local_indices=local_indices, local_matrices=local_matrices
        )

        local_indices, local_matrices = self._assemble_local_stiffness_matrices()
        global_stiffness_matrix = self.assemble_global_matrix(
            local_indices=local_indices, local_matrices=local_matrices
        )

        knot_span_handler = self.KnotSpanHandler(
            patch=self._spline, integration_order=self._spline.degree + 1
        )

        # Smallest length of the knot spans in the u and v directions
        scale_factor = smoothing_strength * knot_span_handler.max_span_size**2

        global_matrix = scale_factor * global_stiffness_matrix + global_mass_matrix

        local_indices, local_vectors = self._assemble_local_vectors(
            rhs_term_function=rhs_term_function
        )
        global_vector = self.assemble_global_vector(
            local_indices=local_indices, local_vectors=local_vectors
        )

        # Compute pre-conditioner for the conjugate gradient solver
        lumped_mass_matrix_diag = global_matrix.sum(axis=1).A1
        inv_lumped_mass_matrix = sp.diags(
            1.0 / lumped_mass_matrix_diag, format="csr", dtype=np.float64
        )

        # Solve the linear system A * u = b
        solution = sp.linalg.cg(
            A=global_matrix,
            b=global_vector,
            rtol=self.SOLVER_TOL,
            M=inv_lumped_mass_matrix,
        )[0]

        return mi.TensorXf(solution)[:, None]
