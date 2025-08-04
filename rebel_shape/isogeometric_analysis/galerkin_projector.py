from collections.abc import Callable

import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import scipy.sparse as sp

from rebel_shape.isogeometric_analysis.iga_manifold_problem import IGAManifoldProblem
from rebel_shape.splines.iga_spline_patch import IGASplinePatch


class GalerkinProjector(IGAManifoldProblem):
    """
    A class for projecting functions onto IGA spline basis using the L2 norm.

    This class implements Galerkin projection methods to approximate arbitrary functions
    using spline basis functions in the L2 norm.

    Conceptually, we want u_h(xi) ~= f(xi) for all parametric coordinates xi in the
    parametric domain of the spline surface. u_h is the approximation of f in the
    space defined by basis functions phi_j, so we have

        u_h(xi) = sum_j u_j phi_j(xi).

    Drawing inspiration from the weak formulation of PDEs, we can multiply both sides
    by a test function that lies in the same space, a basis function phi_k,
    and integrate over the parametric domain (denoted as uv):

        int_uv u_h(xi) phi_k(xi) dA = int_uv f(xi) phi_k(xi) dA.

    Note that this is different to the standard IGA weak formulation, because we are
    integrating over the parametric domain, not the physical domain. This simplifies
    many aspects of the implementation, as we do not need to compute the Jacobian
    or the metric tensor for the physical domain. Identifying the matrices and vectors
    in the Galerkin projection, we have:

        M_jk = int_uv phi_j(xi) phi_k(xi) dA,
        f_k = int_uv f(xi) phi_k(xi) dA,
        M u = f,

    where M is the global mass matrix, u is the vector of coefficients for the spline
    basis functions, and f is the global right-hand side vector obtained by integrating
    the target function against the basis functions.

    Solving the linear system M u = f gives us the coefficients u_j that
    represent the best approximation of the target function f in the spline basis
    in the L2 norm.

    The implementation follows the standard Galerkin method:
    1. Assemble local mass (and stiffness) matrices for each knot span
    2. Assemble local right-hand side vectors by integrating the target function
       against the basis functions
    3. Assemble global matrices and vectors
    4. Solve the resulting linear system using a conjugate gradient solver
    """

    SOLVER_TOL: float = 1e-9
    """Conjugate gradient solver residual tolerance for linear system solve."""

    def __init__(self, spline_surface: IGASplinePatch):
        """
        Initialize the Galerkin projector with a spline surface.
        """
        self._spline = spline_surface

    def _assemble_local_mass_matrices(self) -> tuple["mi.TensorXu", "mi.TensorXf"]:
        """
        Assembles the local mass matrices for the Galerkin projection problem on the
        spline surface.

        This method computes the local mass matrices for each knot span of the spline
        surface. The mass matrix entries are computed using the formula:

            M_jk^(e) ~= area_e sum_q w_q (phi_j(xi_q) phi_k(xi_q)),

        where M_jk^(e) is the local mass matrix entry for element e, w_q are the
        Gauss-Legendre quadrature weights, xi_q are the quadrature points in the
        parametric domain, area_e is the area of the element's knot span in the
        parametric domain, phi_j and phi_k are the spline basis functions.

        Time complexity: O(n_knot_spans * n_dofs^2 * n_quadrature_points^2),
        where n_dofs is the number of degrees of freedom per knot span. With the
        default quadrature order (degree + 1) and n_dofs ~ (degree + 1)^2, this
        becomes O(n_knot_spans * degree^6), but the dominant term is interpolation
        which is O(n_knot_spans * degree^4).

        :returns: A tuple that consists of

            - dofs_per_span: A tensor of shape (n_knot_spans, n_dofs) that contains
              the indices of the degrees of freedom for each knot span.
            - local_matrices: A tensor of shape (n_knot_spans, n_dofs, n_dofs) that
              contains the local mass matrices for each knot span.
        """
        interpolating_spline = self._spline.interpolating_spline

        knot_span_handler = self.KnotSpanHandler(
            patch=self._spline, integration_order=self._spline.degree + 1
        )
        num_dofs = self._spline.interpolating_spline.basis_functions_per_span

        local_matrices = dr.zeros(
            dtype=mi.TensorXf,
            shape=(knot_span_handler.num_knot_spans, num_dofs, num_dofs),
        )

        for parametric_coords, weight in knot_span_handler.gl_coords_and_weights():
            basis_functions = interpolating_spline.evaluate_basis_functions(
                parametric_coords=parametric_coords
            )

            for i in range(num_dofs):
                values = (
                    basis_functions[:, :] * (basis_functions[:, i] * weight)[:, None]
                )

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

    def _assemble_local_vectors(
        self, rhs_term_function: Callable[["mi.TensorXf"], "mi.TensorXf"]
    ) -> tuple["mi.TensorXu", "mi.TensorXf"]:
        """
        Assembles the local vectors for the Galerkin projection problem on the spline
        surface.

        This method computes the local vectors for each knot span of the spline surface
        by integrating the target function against the spline basis functions. The
        local vector entries are computed using the formula:

            F_j^(e) ~= area_e sum_q w_q (f(xi_q) phi_j(xi_q)),

        where F_j^(e) is a local vector entry for element e, w_q are the Gauss-Legendre
        quadrature weights, xi_q are the quadrature points in the parametric domain,
        area_e is the area of the element's knot span in the parametric domain,
        f(xi_q) is the `rhs_term_function` evaluated at the quadrature point,
        and phi_j are the spline basis functions.

        `rhs_term_function` is a callable that takes parametric coordinates of
        shape (n_points, 2) and returns the target function values of shape
        (n_points, 1) evaluated at those coordinates.

        Time complexity: O(n_knot_spans * n_dofs * n_quadrature_points^2),
        where n_dofs is the number of degrees of freedom per knot span. With the
        default quadrature order (degree + 1) and n_dofs ~ (degree + 1)^2, this
        becomes O(n_knot_spans * degree^4).

        :returns: A tuple that consists of

            - dofs_per_span: A tensor of shape (n_knot_spans, n_dofs) that contains
              the indices of the degrees of freedom for each knot span.
            - local_vectors: A tensor of shape (n_knot_spans, n_dofs) that contains
              the local vectors for each knot span.
        """
        interpolating_spline = self._spline.interpolating_spline

        knot_span_handler = self.KnotSpanHandler(
            patch=self._spline, integration_order=self._spline.degree + 1
        )
        num_dofs = interpolating_spline.basis_functions_per_span

        local_vectors = dr.zeros(
            dtype=mi.TensorXf, shape=(knot_span_handler.num_knot_spans, num_dofs)
        )

        for parametric_coords, weight in knot_span_handler.gl_coords_and_weights():
            basis_functions = interpolating_spline.evaluate_basis_functions(
                parametric_coords=parametric_coords
            )

            product = basis_functions * rhs_term_function(parametric_coords)
            local_vectors += product * weight[:, None]

        dofs_per_span = interpolating_spline.active_basis_function_indices(
            parametric_coords=knot_span_handler.midpoint_parametric_coordinates
        )

        return dofs_per_span, local_vectors

    def compute_l2_error(
        self,
        projected_values: "mi.TensorXf",
        rhs_term_function: Callable[["mi.TensorXf"], "mi.TensorXf"],
    ) -> "mi.TensorXd":
        """
        Computes the L2 error of the Galerkin projection.

        This method evaluates the L2 error between the target function and its
        spline approximation by computing the integral of the squared difference
        over the spline surface:

            ||f - u_h||_{L2} = sqrt(int_uv (f(x) - u_h(x))^2 dA).

        The error is computed using Gauss-Legendre quadrature for numerical integration.
        Since the squared error will have order (O(degree + 1)^2), the quadrature order
        must be set to at least 2 * degree + 1 to ensure accurate integration.

        Time complexity: O(n_knot_spans * degree^4),

        `rhs_term_function` is a callable that takes parametric coordinates of
            shape (n_points, 2) and returns the target function values of shape
            (n_points, 1) evaluated at those coordinates.

        :returns: A Mitsuba tensor of shape (1,) containing the L2 error of the
            Galerkin projection (double precision), the root-mean-square error between
            the spline basis approximation and the target function.
        """
        knot_span_handler = self.KnotSpanHandler(
            patch=self._spline, integration_order=2 * self._spline.degree + 1
        )

        l2_error = mi.TensorXd(0.0)

        for parametric_coords, weight in knot_span_handler.gl_coords_and_weights():
            interpolated = self._spline.interpolating_spline.interpolate(
                values=projected_values, parametric_coords=parametric_coords
            )
            function_values = rhs_term_function(parametric_coords)

            difference = interpolated - function_values
            squared_difference = difference * difference

            l2_error += dr.sum(mi.TensorXd(squared_difference * weight[:, None]))

        return dr.sqrt(l2_error)

    def galerkin_project_l2(
        self, rhs_term_function: Callable[["mi.TensorXf"], "mi.TensorXf"]
    ) -> "mi.TensorXf":
        """
        Projects the target function onto the spline surface using L2 Galerkin
        projection.

        This method finds the best approximation of the target function in the L2 norm
        by solving the linear system:

            M u = f,

        where M is the global mass matrix assembled from local mass matrices, u is the
        vector of coefficients for the spline basis functions, and f is the global
        right-hand side vector obtained by integrating the target function against
        the basis functions.

        The L2 projection minimizes the squared L2 norm of the error between the
        target function and its spline approximation:

            ||f - u_h||_{L2}^2 = int_M (f(x) - u_h(x))^2 dA,

        where u_h(x) = sum_j u_j phi_j(x) is the spline approximation.

        The linear system is solved using the conjugate gradient method with a
        relative tolerance of 1e-10, which is efficient for the symmetric positive
        definite mass matrix.

        Time complexity: O(n_knot_spans * degree^6) for matrix assembly plus
        O(n_dofs^1.5) for the CG solver, where n_dofs is the total number of degrees
        of freedom.

        `rhs_term_function` A callable that takes parametric coordinates of
        shape (n_points, 2) and returns the target function values of shape
        (n_points, 1) evaluated at those coordinates.

        :returns: A tensor of shape (num_basis_functions, 1) containing the
            coefficients of the spline basis functions that represent the L2
            projection of the target function onto the spline surface.
        """
        local_indices, local_matrices = self._assemble_local_mass_matrices()
        global_matrix = self.assemble_global_matrix(
            local_indices=local_indices, local_matrices=local_matrices
        )

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

    def plot_projected_values(
        self,
        projected_values: "mi.TensorXf",
        rhs_term_function: Callable[["mi.TensorXf"], "mi.TensorXf"],
        samples_per_direction: int = 101,
        show_difference: bool = True,
    ) -> None:
        """
        Visualizes the results of Galerkin projection for debugging and analysis.

        This method creates side-by-side 3D surface plots comparing the projected spline
        approximation with the original target function. It samples both functions on a
        regular grid over the parametric domain of the underlying spline and displays
        them as surface plots for visual comparison.

        The visualization helps assess the quality of the Galerkin projection
        approximation and overall behavior of the projected solution. It is primarily
        intended for debugging and light visualization purposes.

        `projected_values` is a tensor of shape (n_dofs, 1) containing the coefficients
        of the spline basis functions representing the projected function. These are
        typically the output of `galerkin_project_l2` or `galerkin_project_l2_smooth`.

        `rhs_term_function` is a callable that takes parametric coordinates of
        shape (n_points, 2) and returns the target function values of shape
        (n_points, 1). This should be the same function used for the projection.

        `samples_per_direction` is the number of sample points along each parametric
        direction (u and v). If `show_difference` is enabled, a third subplot showing
        the point-wise difference between the projected and target functions is also
        created.

        :raises ValueError: If samples_per_direction is less than 2.
        """
        if samples_per_direction < 2:
            raise ValueError("samples_per_direction must be at least 2")

        if (
            projected_values.shape[0]
            != self._spline.interpolating_spline.num_basis_functions
        ):
            raise ValueError(
                "projected_values must have shape (num_basis_functions, 1), "
                f"got {projected_values.shape}"
            )
        if len(projected_values.shape) != 2 or projected_values.shape[1] != 1:
            raise ValueError(
                "projected_values must be a 2D tensor with shape "
                f"(num_basis_functions, 1), got {projected_values.shape}"
            )

        # Generate sampling grid over parametric domain
        (u_min, u_max), (v_min, v_max) = self._spline.parametric_domain_size
        u = np.linspace(u_min, u_max, samples_per_direction)
        v = np.linspace(v_min, v_max, samples_per_direction)

        uu, vv = np.meshgrid(u, v)
        sample_points = mi.TensorXf(np.column_stack([uu.flatten(), vv.flatten()]))

        # Evaluate projected and target functions at sample points
        interpolated_values = self._spline.interpolating_spline.interpolate(
            values=projected_values, parametric_coords=sample_points
        )
        expected_values = rhs_term_function(sample_points)

        # Reshape to 2D matrices for surface plotting
        interpolated_values_mat = interpolated_values.numpy().reshape(
            samples_per_direction, samples_per_direction
        )
        expected_values_mat = expected_values.numpy().reshape(
            samples_per_direction, samples_per_direction
        )

        # Create plots
        num_plots = 3 if show_difference else 2
        fig = plt.figure(figsize=(5 * num_plots, 4))

        # Plot projected function
        ax1 = fig.add_subplot(1, num_plots, 1, projection="3d")
        ax1.plot_surface(uu, vv, interpolated_values_mat, cmap="viridis")
        ax1.set_title("Projected Function")
        ax1.set_xlabel("u")
        ax1.set_ylabel("v")
        ax1.set_zlabel("Value")

        # Plot target function
        ax2 = fig.add_subplot(1, num_plots, 2, projection="3d")
        ax2.plot_surface(uu, vv, expected_values_mat, cmap="plasma")
        ax2.set_title("Target Function")
        ax2.set_xlabel("u")
        ax2.set_ylabel("v")
        ax2.set_zlabel("Value")

        # Plot difference if requested
        if show_difference:
            difference_mat = interpolated_values_mat - expected_values_mat

            ax3 = fig.add_subplot(1, num_plots, 3, projection="3d")
            ax3.plot_surface(uu, vv, difference_mat, cmap="RdBu_r")
            ax3.set_title("Difference (Projected - Target)")
            ax3.set_xlabel("u")
            ax3.set_ylabel("v")
            ax3.set_zlabel("Error")

        plt.tight_layout()
        plt.show()
