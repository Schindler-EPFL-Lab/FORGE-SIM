import drjit as dr
import mitsuba as mi

from forge_sim.isogeometric_analysis.iga_manifold_problem import IGAManifoldProblem
from forge_sim.splines.iga_spline_patch import IGASplinePatch


class ManifoldPoissonProblem(IGAManifoldProblem):
    """
    Solves the Poisson equation on a 2D spline patch surface M using IGA.

    For more details about the details of the derivation and the implementation,
    see "Isogeometric analysis for second order  partial differential equations on
    surfaces" by L. Dedè, A. Quarteroni (2012)

    Following the 10 steps from the `IGAProblem` class, we obtain the following:
    1. Problem statement:
        -Delta_M u = f, on M
        u = 0,          on boundary of M,
    where Delta_M is the Laplace-Beltrami operator on the manifold M, u is the unknown
    solution, and f is a right-hand side (the source term).

    2. The solution u lives in H_0^1(M) and is approximated by a finite-dimensional
    subspace V_h spanned by the spline basis functions.

    3. Weak Formulation: Find u in H_0^1(M) such that for all v in H_0^1(M) it holds
        int_M (grad_M u) dot (grad_M v) dA = int_M f v dA

    4. Galerkin Method: Approximate u as:
        u_h(x) = sum_j c_j phi_j(x),
    where phi_j are the spline basis functions.

    5. Matrix Formulation: The discrete system is
        sum_j a(phi_j, phi_k) c_j = f(phi_k),   for all k,
        a(phi_j, phi_k) = int_M (grad_M phi_j) dot (grad_M phi_k) dA
        f(phi_k) = int_M f phi_k dA

    6. Local Assembly: For each element (knot span), local matrices and vectors are
       assembled. Instead of integrating over the manifold M, we integrate over each
       knot span e in the parametric domain (u, v), using the inverse mapping. Since the
       integration cannot be done analytically, we use Gauss-Legendre quadrature to
       approximate the integrals. The local stiffness matrix and load vector are
       computed as

        A_jk^(e) ~= area_e sum_q w_q (
            d(phi_j(xi_q))/d xi dot (G^{-1}(xi_q) d(phi_(xi_q))/d xi)
        ) sqrt(det(G(xi_q))),

        F_j^(e) ~= area_e sum_q w_q (f(xi_q) phi_j(xi_q)) sqrt(det(G(xi_q)))

       where A_jk^(e) is the local stiffness matrix (entry), F_j^(e) is the local load
       vector (entry), w_q are the Gauss-Legendre quadrature weights, xi_q are the
       quadrature points in the parametric domain, and G is the metric tensor at the
       quadrature point.

    7. Global Assembly: Local matrices and vectors are assembled into the global
       system using the methods from IGAProblem.

    8. Boundary Conditions are not imposed directly in this class and should be
       handled externally. Since this problem is more of a showcase and validation of
       the IGA framework, Dirichlet boundary conditions are assumed to be employed, but
       have to be enforced externally.

    9. Solution: Solve the linear system using the CG solver as the system matrix is
       symmetric and positive definite. The solution is the coefficients c_j of the
       spline basis functions, which can be used to reconstruct the solution u_h(x) on
       the manifold M.

    10. Validation: The sine function is an eigenfunction of the Laplace-Beltrami
        operator on a flat surface, so we can use it to validate the implementation
        in this simplified case. This is why the `project_function` method is
        implemented for interpolating splines, to project the sine function onto the
        spline basis functions. Alternatively, one manufactures the solution and then
        uses symbolic differentiation to compute the right-hand side term f.

    The two main methods of interest are `assemble_local_matrices` and
    `assemble_local_vectors`. The first one assembles the local stiffness matrices,
    while the second assembles the local load vectors. This is done as described in
    the 6th step above.
    """

    def __init__(self, spline_surface: IGASplinePatch):
        """
        Initializes the Poisson problem with a given spline surface.

        The geometry of the spline patch is used to compute the local stiffness matrices
        and local right hand side vectors.

        The problem that is solved is the Poisson equation on the spline surface:
            -Delta_M u = f, on `spline_surface`,
            u = 0,          on boundary of `spline_surface`.
        """
        self._spline = spline_surface

    def assemble_local_matrices(self) -> tuple["mi.TensorXu", "mi.TensorXf"]:
        """
        Assembles the local stiffness matrices for the Poisson problem on the spline
        surface.

        This method computes the local stiffness matrices for each knot span of the
        spline surface using Gauss-Legendre quadrature. The following formula is applied
        to compute the local stiffness matrix for each knot span:

            A_jk^(e) ~= area_e sum_q w_q (
                d(phi_j(xi_q))/d xi dot (G^{-1}(xi_q) d(phi_k(xi_q))/d xi)
            ) sqrt(det(G(xi_q))),

        where A_jk^(e) is the local stiffness matrix (entry), w_q are the Gauss-Legendre
        quadrature weights, xi_q are the quadrature points in the parametric domain,
        phi_j and phi_k are the spline basis functions, and G is the metric tensor at
        the quadrature point.

        The current implementation is designed to have a small memory footprint as one
        never exceeds the O(n_knot_spans * n_dofs^2) memory complexity, meaning that
        the return value takes up most of the memory footprint.

        The time complexity is O(n_knot_spans * n_dofs^2 * n_quadrature_points^2),
        where n_knot_spans is the number of knot spans, n_dofs is the number of
        degrees of freedom per knot span, and n_quadrature_points is the number of
        Gauss-Legendre quadrature points used for integration in one direction (u or
        v). The number of quadrature points is chosen to be equal to the spline
        degree as this is the minimum number of points that achieves errors of less
        than 1e-7 (higher number of points does not improve accuracy at all).

        Since n_dofs ~ (degree + 1)^2, the time complexity can be approximated as
        O(n_knot_spans * degree^6). The most expensive component of the
        computation is evaluating the basis functions and this is slightly cheaper,
        O(n_knot_spans * degree^4).

        :returns: A tuple that consists of

            - dofs_per_span: A tensor of shape (n_knot_spans, n_dofs) that contains
              the indices of the degrees of freedom for each knot span.
            - local_matrices: A tensor of shape (n_knot_spans, n_dofs, n_dofs) that
              contains the local stiffness matrices for each knot span.
        """
        interp_spline = self._spline.interpolating_spline

        knot_span_handler = self.KnotSpanHandler(
            patch=self._spline, integration_order=self._spline.degree
        )
        num_dofs = interp_spline.basis_functions_per_span

        local_matrices = dr.zeros(
            dtype=mi.TensorXf,
            shape=(knot_span_handler.num_knot_spans, num_dofs, num_dofs),
        )

        for parametric_coords, weight in knot_span_handler.gl_coords_and_weights():
            basis_function_derivative = (
                interp_spline.evaluate_basis_functions_first_derivative(
                    parametric_coords=parametric_coords
                )
            )

            jacobian = self._spline.evaluate_jacobian(parametric_coords)
            metric = self._spline.metric_tensor(jacobian)
            inv_metric, det_metric = self._spline.inv_metric_tensor(metric)

            transformed_basis_derivative = self._transform_basis_derivative(
                basis_derivative=basis_function_derivative, inv_metric=inv_metric
            )

            for i in range(num_dofs):
                dot_product = dr.sum(
                    value=(
                        transformed_basis_derivative[:, :, :]
                        * basis_function_derivative[:, i, :][:, None, :]
                    ),
                    axis=2,
                )
                values = dot_product * (dr.sqrt(det_metric) * weight)[:, None]

                self._scatter_values_to_local_matrices(
                    local_matrices=local_matrices,
                    values=values,
                    num_knot_spans=knot_span_handler.num_knot_spans,
                    num_dofs=num_dofs,
                    row=i,
                )

        dofs_per_span = interp_spline.active_basis_function_indices(
            parametric_coords=knot_span_handler.midpoint_parametric_coordinates
        )

        return dofs_per_span, local_matrices

    def assemble_local_vectors(
        self, rhs_term: "mi.TensorXf"
    ) -> tuple["mi.TensorXu", "mi.TensorXf"]:
        """
        Assembles the local right-hand side vectors for the Poisson problem on the
        spline surface.

        The `rhs_term` is a tensor of shape (n_basis_functions, 1) that represents
        the right-hand side term of the Poisson equation. Each entry has to correspond
        to a basis function of the spline surface.

        The method computes the local right-hand side vectors for each knot span
        of the spline surface using Gauss-Legendre quadrature. The following formula is
        applied to compute the local right-hand side vector for each knot span:

            F_j^(e) ~= area_e sum_q w_q (f(xi_q) phi_j(xi_q)) sqrt(det(G(xi_q)),

        where F_j^(e) is the local right-hand side vector (entry), w_q are the
        Gauss-Legendre quadrature weights, xi_q are the quadrature points in the
        parametric domain, phi_j are the spline basis functions, and G is the metric
        tensor at the quadrature point.

        The current implementation is designed to have a small memory footprint as one
        never exceeds the O(n_knot_spans * n_dofs) memory complexity, meaning that
        the return value takes up most of the memory footprint.

        The time complexity is O(n_knot_spans * n_dofs * n_quadrature_points^2),
        where n_knot_spans is the number of knot spans, n_dofs is the number of
        degrees of freedom per knot span, and n_quadrature_points is the number of
        Gauss-Legendre quadrature points used for integration in one direction (u or
        v). The number of quadrature points is chosen to be equal to the spline
        degree as this is the minimum number of points that achieves errors of less
        than 1e-7 (higher number of points does not improve accuracy at all).

        Since n_dofs ~ (degree + 1)^2, the time complexity can be approximated as
        O(n_knot_spans * degree^4). The most expensive component of the
        computation is evaluating the basis functions, and it has the same time
        complexity, O(n_knot_spans * degree^4). Thus, local matrix and vector assembly
        should have comparable performance, with matrix assembly being slightly
        more expensive (but not as much as simple asymptotic analysis suggests).

        :returns: A tuple that consists of

            - dofs_per_span: A tensor of shape (n_knot_spans, n_dofs) that contains
              the indices of the degrees of freedom for each knot span.
            - local_vectors: A tensor of shape (n_knot_spans, n_dofs, 1) that contains
              the local right-hand side vectors for each knot span.

        :raises ValueError: If `rhs_term` is not a 2D tensor of shape
            (n_basis_functions, 1).
        """
        if rhs_term.ndim != 2 or rhs_term.shape[1] != 1:
            raise ValueError(
                "rhs_term must be a 2D tensor with shape (n_basis_functions, 1). "
                f"Got shape {rhs_term.shape} instead."
            )
        if rhs_term.shape[0] != self._spline.interpolating_spline.num_basis_functions:
            raise ValueError(
                f"rhs_term must have shape ("
                f"{self._spline.interpolating_spline.num_basis_functions}, 1), "
            )

        interp_spline = self._spline.interpolating_spline

        knot_span_handler = self.KnotSpanHandler(
            patch=self._spline, integration_order=self._spline.degree
        )
        num_dofs = interp_spline.basis_functions_per_span

        local_vectors = dr.zeros(
            dtype=mi.TensorXf, shape=(knot_span_handler.num_knot_spans, num_dofs)
        )

        for parametric_coords, weight in knot_span_handler.gl_coords_and_weights():
            basis_functions = interp_spline.evaluate_basis_functions(
                parametric_coords=parametric_coords
            )

            jacobian = self._spline.evaluate_jacobian(parametric_coords)
            metric = self._spline.metric_tensor(jacobian)
            _, det_metric = self._spline.inv_metric_tensor(metric)

            rhs_value = interp_spline.interpolate(
                values=rhs_term, parametric_coords=parametric_coords
            )

            product = basis_functions * rhs_value
            local_vectors += product * (weight * dr.sqrt(det_metric))[:, None]

        dofs_per_span = interp_spline.active_basis_function_indices(
            parametric_coords=knot_span_handler.midpoint_parametric_coordinates
        )

        return dofs_per_span, local_vectors

    @property
    def spline(self) -> IGASplinePatch:
        """
        The spline surface associated with this problem.
        """
        return self._spline
