import drjit as dr
import mitsuba as mi

from rebel_shape.isogeometric_analysis.iga_manifold_problem import IGAManifoldProblem
from rebel_shape.meshing.composite_iga_spline_surface import CompositeIGASplineSurface


class HeatSimulation(IGAManifoldProblem):
    """
    Solves the simple heat equation on a composite spline surface with varying
    material properties.

    For more details about the details of the derivation and the implementation,
    see "Isogeometric analysis for second order  partial differential equations on
    surfaces" by L. Dedè, A. Quarteroni (2012)

    Following the 10 steps from the `IGAProblem` class, we obtain the following:

    1. The problem that is solved is the heat equation on the spline surface:
        partial_t rho(x) c(x) d(x) u(x, t) - d(x) div_M (k(x) grad_M u(x, t)) = f(x, t),
    As the surface is closed, there are no boundary conditions. rho(x) is the
    density, c(x) is the specific heat, k(x) is the conductivity, d(x) is the
    thickness of the surface, and f(x, t) is the source term per unit area.
    The solution u(x, t) is the temperature at point x on the surface at time t.

    2. The solution u lives in H^1(M) and is approximated by a finite-dimensional
    subspace V_h spanned by the spline basis functions. While a Poisson problem on a
    closed manifold does not have a unique solution, requiring additional constraints
    (like integral u(x) = 0), the solution of the heat equation is unique due to the
    unique initial condition u(x, 0) = u_0(x) and the fact that the evolution is
    governed by linear operators.

    3. Weak Formulation: Find u in H^1(M) such that for all v in H^1(M) it holds that
        ```math
        int_M (v rho c d partial_t u) - int_M d (grad_M u) dot(k grad_M v) dA
        = int_M f v dA
        ```

    4. Galerkin Method: Approximate u with phi_j spline basis functions:
        u_h(x, t) = sum_j c_j(t) phi_j(x),

    5. Matrix Formulation: The semi-discrete problem can be written as
        sum_j (partial_t c_j m(phi_j, phi_k)) + sum_j (c_j a(phi_j, phi_k)) = f(phi_k),

        m(phi_j, phi_k) = int_M rho c d phi_j phi_k dA
        a(phi_j, phi_k) = int_M d (grad_M phi_j) dot (k grad_M phi_k) dA
        f(phi_k) = int_M f phi_k dA

    5.5. A fully discrete problem is obtained by using the Crank-Nicolson scheme. The
    scheme is effectively a trapezoidal rule for time integration. So, integrating both
    sides of the equation over the time interval [t_n, t_{n+1}] we obtain:
        sum_j ((c_j^{n+1} - c_j^n) m(phi_j, phi_k))
        + 0.5 * dt * sum_j ((c_j^{n+1} + c_j^n) a(phi_j, phi_k))
        = 0.5 * dt * (f(phi_k)^n + f(phi_k)^{n+1}),

    Using matrix notation this can be simplified to:
        (M + 0.5 dt A) c^{n+1} = (M - 0.5 dt A) c^n + 0.5 dt (f^n + f^{n+1}),

    Note that computing the solution at the next time step requires solving a system
    of linear equations. This can either be done using a direct solver or an iterative
    one (e.g. conjugate gradient). For the 2D problem either will do. CG converges
    quickly as a good initial guess is provided by the previous time step and one
    can easily design a good pre-conditioner - the inverted lumped mass matrix.

    6. Local Assembly: For each element (knot span), local matrices and vectors are
    assembled. Instead of integrating over the manifold M, we integrate over each
    knot span e in the parametric domain (u, v), using the inverse mapping. Since the
    integration cannot be done analytically, we use Gauss-Legendre quadrature to
    approximate the integrals. The local stiffness matrix and load vector are
    computed as

        A_jk^(e) ~= area_e sum_q w_q (d(xi_q) k(xi_q)
            partial_xi phi_j(xi_q) dot (G^{-1}(xi_q) partial_xi phi_(xi_q))
        ) sqrt(det(G(xi_q))),

        M_jk^(e) ~= area_e sum_q w_q (
            rho(xi_q) c(xi_q) d(xi_q) phi_j(xi_q) phi_k(xi_q)
        ) sqrt(det(G(xi_q))),

        F_j^(e) ~= area_e sum_q w_q (f(xi_q) phi_j(xi_q)) sqrt(det(G(xi_q)))

        total_heat = area_e sum_j sum_q w_q (
            rho(xi_q) c(xi_q) d(xi_q) (c_j phi_j(xi_q)) sqrt(det(G(xi_q)))
        ),

    where A_jk^(e) is the local stiffness matrix (entry), M_jk^(e) is the local mass
    matrix (entry), F_j^(e) is the local load vector (entry), w_q are the
    Gauss-Legendre quadrature weights, xi_q are the quadrature points in the parametric
    domain, area_e is the area of the element's knot span in the parametric domain,
    phi_j and phi_k are the spline basis functions, G is the metric tensor at the
    quadrature point, and c_j are the solution coefficients for the spline basis
    functions. This has to be applied to each patch separately and then all the
    local matrices and vectors are concatenated.

    7. Global Assembly: Local matrices and vectors are assembled into the global
    system using the methods from IGAProblem.

    8. There are no boundary conditions as the surface is closed.

    9. Solution: Repeatedly solve the linear system of equations obtained in step 5.5
    for each time step until the desired final time is reached. Note that since the
    Crank-Nicolson scheme is used, the method is unconditionally stable for all choices
    of dt > 0. However, the choice of dt affects the accuracy of the solution as the
    scheme is second order accurate in time O(dt^2). The characteristic timescale T
    of the problem is T ~ L^2 / alpha, alpha = k / (rho c) is the thermal diffusivity.
    Thus, the surface area of the problem and the diffusion coefficient introduce a
    natural timescale for the evolution. dt ~ 0.01 T is typically a reasonable choice.

    10. Validation: Since the stiffness matrix and the rhs vector are assembled in the
    same way as for the Poisson problem (minus some constants), they are implicitly
    tested by the tests for the Poisson problem. Additionally, one can check that the
    energy is conserved over time, i.e. the total heat in the system does not change
    as the Crank-Nicolson scheme guarantees this property.
    """

    def __init__(
        self,
        composite_iga_spline: CompositeIGASplineSurface,
        conductivity: "mi.TensorXf",
        density: "mi.TensorXf",
        specific_heat: "mi.TensorXf",
        thickness: "mi.TensorXf",
        source_term: "mi.TensorXf",
        integration_order: int | None = None,
    ) -> None:
        """
        Initialize the heat simulation problem with a composite spline surface and
        material properties.

        The problem is the heat equation on the `composite_iga_spline` surface:
        partial_t rho(x) c(x) d(x) u(x, t) - d(x) div_M (k(x) grad_M u(x, t)) = f(x, t),

        where rho(x) is the `density`, c(x) is the `specific_heat`, k(x) is the
        `conductivity`, d(x) is the `thickness` of the surface, and f(x, t) is the
        `source_term` per unit area. All of them are scalar fields, so they must have
        a shape of (n_global_dofs, 1), where n_global_dofs is the number of global
        degrees of freedom in the composite spline surface.

        All patches in the composite spline surface must have the same degree,
        so that all local matrices have the same shape and can be concatenated
        into a single tensor.

        `integration_order` is the order of the Gauss-Legendre quadrature used for
        numerical integration, along both the u and v directions. If not specified,
        it defaults to the spline_surface_degree + 1, which is the minimum number of
        points needed to achieve an accuracy better than 1e-7.

        :raises ValueError: If the shapes of the material properties are not 2D tensors
            with shape (n_global_dofs, 1), where n_global_dofs is the number of
            global degrees of freedom in the composite spline surface, or if
            the degree of the patches in the composite spline surface is not the same.
        """
        n_dofs = composite_iga_spline.global_field_values.shape[0]

        for field in [conductivity, density, specific_heat, thickness, source_term]:
            if len(field.shape) != 2 or field.shape[1] != 1 or field.shape[0] != n_dofs:
                raise ValueError(
                    "Properties must be 2D tensors shaped (n_global_dofs, 1). "
                    f"Got shape {field.shape}."
                )

        degree = composite_iga_spline.degree
        for patch in composite_iga_spline:
            if patch.degree != degree:
                raise ValueError(
                    "All patches in the composite spline surface must have the same "
                    f"degree. The global surface has degree {degree}, but "
                    f"patch {patch.name} has degree {patch.degree}."
                )

        if integration_order is None:
            integration_order = composite_iga_spline.degree + 1

        self._composite_iga_spline = composite_iga_spline

        self._conductivity = conductivity
        self._density = density
        self._specific_heat = specific_heat
        self._thickness = thickness
        self._source_term = source_term

        self._integration_order = integration_order

    def assemble_local_stiffness_matrices(self) -> tuple["mi.TensorXu", "mi.TensorXf"]:
        """
        Assembles the local stiffness matrices for the Heat simulation problem on the
        spline surface defined by multiple patches.

        This method computes the local stiffness matrices for each knot span of each
        patch that constitutes the composite spline surface. The following formula is
        applied to compute the local stiffness matrix for each knot span:

            A_jk^(e) ~= area_e sum_q w_q (d(xi_q) k(xi_q)
                partial_xi phi_j(xi_q) dot (G^{-1}(xi_q) partial_xi phi_(xi_q))
            ) sqrt(det(G(xi_q))),

        where A_jk^(e) is the local stiffness matrix for the element e, w_q are the
        Gauss-Legendre quadrature weights, xi_q are the quadrature points in the
        parametric domain, area_e is the area of the element's knot span in the
        parametric domain, phi_j and phi_k are the spline basis functions, G is the
        metric tensor at the quadrature point, k(xi_q) is the conductivity at the
        quadrature point, and d(xi_q) is the thickness at the quadrature point.
        This is done for each patch.

        When the matrices of each span are assembled, they are concatenated into a
        single tensor that contains all the local stiffness matrices for the entire
        composite spline surface. This assumes that all patches have the same degree;
        otherwise they could not be concatenated.

        The current implementation is designed to have a small memory footprint as one
        never exceeds the O(n_knot_spans * n_dofs) memory complexity, meaning that
        the return value takes up most of the memory footprint.

        Time complexity: O(n_patches * n_knot_spans * n_dofs^2 * n_quadrature_points^2),
        where n_dofs is the number of degrees of freedom per knot span. Plugging in the
        default quadrature order (degree + 1), and that n_dofs ~ (degree + 1)^2,
        we obtain O(n_patches * n_knot_spans * degree^6).
        Reality is a little more optimistic as this dominant component has a small
        constant and the term basis function evaluation term dominates execution time.
        It scales as O(n_patches * n_knot_spans * degree^4).

        :returns: A tuple that consists of

            - dofs_per_span: A tensor of shape (sum_p n_knot_spans_p, n_dofs) that
              contains the indices of the degrees of freedom for each knot span.
            - local_matrices: A tensor of shape (sum_p n_knot_spans_p, n_dofs, n_dofs)
              that contains the local mass matrices for each knot span,

            where p is the patch index and sum_p n_knot_spans_p is the total number of
            knot spans across all patches.
        """
        patch_local_matrices: list[mi.TensorXf] = []
        patch_local_indices: list[mi.TensorXu] = []

        for patch, mapping in self._composite_iga_spline.patches_with_mapping:
            interp_spline = patch.interpolating_spline

            knot_span_handler = self.KnotSpanHandler(
                patch=patch, integration_order=self._integration_order
            )
            num_dofs = interp_spline.basis_functions_per_span

            local_matrices = dr.zeros(
                dtype=mi.TensorXf,
                shape=(knot_span_handler.num_knot_spans, num_dofs, num_dofs),
            )

            patch_conductivity = self._conductivity[dr.ravel(mi.TensorXu(mapping))]
            patch_thickness = self._thickness[dr.ravel(mi.TensorXu(mapping))]

            for (
                param_coords,
                scaled_weight,
            ) in knot_span_handler.gl_coords_and_weights():
                basis_function_ders = (
                    interp_spline.evaluate_basis_functions_first_derivative(
                        parametric_coords=param_coords
                    )
                )

                conductivity = interp_spline.interpolate(
                    values=patch_conductivity, parametric_coords=param_coords
                )
                thickness = interp_spline.interpolate(
                    values=patch_thickness, parametric_coords=param_coords
                )

                jacobian = patch.evaluate_jacobian(param_coords)
                metric = patch.metric_tensor(jacobian)
                inv_metric, det_metric = patch.inv_metric_tensor(metric)

                transformed_basis_function_ders = self._transform_basis_derivative(
                    basis_derivative=basis_function_ders, inv_metric=inv_metric
                )

                for i in range(num_dofs):
                    dot_product = dr.sum(
                        value=(
                            transformed_basis_function_ders[:, :, :]
                            * basis_function_ders[:, i, :][:, None, :]
                        ),
                        axis=2,
                    )
                    values = (dot_product * conductivity * thickness) * (
                        dr.sqrt(det_metric) * scaled_weight
                    )[:, None]

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

            patch_local_matrices.append(local_matrices)
            patch_local_indices.append(dofs_per_span)

        return self._concatenate_patch_local_matrices(
            composite_iga_spline=self._composite_iga_spline,
            local_indices_list=patch_local_indices,
            local_matrices_list=patch_local_matrices,
        )

    def assemble_local_mass_matrices(self) -> tuple["mi.TensorXu", "mi.TensorXf"]:
        """
        Assembles the local mass matrices for the Heat simulation problem on the
        spline surface defined by multiple patches.

        This method computes the local mass matrices for each knot span of each patch
        that constitutes the composite spline surface. The following formula is
        applied to compute the local mass matrix for each knot span:

            M_jk^(e) ~= area_e sum_q w_q (
                rho(xi_q) c(xi_q) d(xi_q) phi_j(xi_q) phi_k(xi_q)
            ) sqrt(det(G(xi_q))),

        where M_jk^(e) is the local mass matrix for the element e, w_q are the
        Gauss-Legendre quadrature weights, xi_q are the quadrature points in the
        parametric domain, area_e is the area of the element's knot span in the
        parametric domain, phi_j and phi_k are the spline basis functions, G is the
        metric tensor at the quadrature point, rho(xi_q) is the density at the
        quadrature point, c(xi_q) is the specific heat at the quadrature point, and
        d(xi_q) is the thickness at the quadrature point. This is done for each
        patch.

        When the matrices of each span are assembled, they are concatenated into a
        single tensor that contains all the local mass matrices for the entire
        composite spline surface. This assumes that all patches have the same degree;
        otherwise they could not be concatenated.

        The current implementation is designed to have a small memory footprint as one
        never exceeds the O(n_knot_spans * n_dofs) memory complexity, meaning that
        the return value takes up most of the memory footprint.

        Time complexity: O(n_patches * n_knot_spans * n_dofs^2 * n_quadrature_points^2),
        where n_dofs is the number of degrees of freedom per knot span. Plugging in the
        default quadrature order (degree + 1), and that n_dofs ~ (degree + 1)^2,
        we obtain O(n_patches * n_knot_spans * degree^6).
        Reality is a little more optimistic as this dominant component has a small
        constant and the term basis function evaluation term dominates execution time.
        It scales as O(n_patches * n_knot_spans * degree^4).

        :returns: A tuple that consists of

            - dofs_per_span: A tensor of shape (sum_p n_knot_spans_p, n_dofs) that
              contains the indices of the degrees of freedom for each knot span.
            - local_matrices: A tensor of shape (sum_p n_knot_spans_p, n_dofs, n_dofs)
              that contains the local mass matrices for each knot span,

            where p is the patch index and sum_p n_knot_spans_p is the total number of
            knot spans across all patches.
        """
        patch_local_indices: list[mi.TensorXu] = []
        patch_local_matrices: list[mi.TensorXf] = []

        for patch, mapping in self._composite_iga_spline.patches_with_mapping:
            interp_spline = patch.interpolating_spline

            knot_span_handler = self.KnotSpanHandler(
                patch=patch, integration_order=self._integration_order
            )
            num_dofs = interp_spline.basis_functions_per_span

            local_matrices = dr.zeros(
                dtype=mi.TensorXf,
                shape=(knot_span_handler.num_knot_spans, num_dofs, num_dofs),
            )

            patch_density = self._density[dr.ravel(mi.TensorXu(mapping))]
            patch_specific_heat = self._specific_heat[dr.ravel(mi.TensorXu(mapping))]
            patch_thickness = self._thickness[dr.ravel(mi.TensorXu(mapping))]

            for param_coords, weight in knot_span_handler.gl_coords_and_weights():
                basis_functions = interp_spline.evaluate_basis_functions(
                    parametric_coords=param_coords
                )

                density = interp_spline.interpolate(
                    values=patch_density, parametric_coords=param_coords
                )
                heat_capacity = interp_spline.interpolate(
                    values=patch_specific_heat, parametric_coords=param_coords
                )
                thickness = interp_spline.interpolate(
                    values=patch_thickness, parametric_coords=param_coords
                )

                jacobian = patch.evaluate_jacobian(param_coords)
                metric = patch.metric_tensor(jacobian)
                _, det_metric = patch.inv_metric_tensor(metric)

                for i in range(num_dofs):
                    values = (
                        basis_functions[:, :]
                        * basis_functions[:, i][:, None]
                        * density
                        * heat_capacity
                        * thickness
                        * (dr.sqrt(det_metric) * weight)[:, None]
                    )

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

            patch_local_matrices.append(local_matrices)
            patch_local_indices.append(dofs_per_span)

        return self._concatenate_patch_local_matrices(
            composite_iga_spline=self._composite_iga_spline,
            local_indices_list=patch_local_indices,
            local_matrices_list=patch_local_matrices,
        )

    def assemble_local_vectors(
        self, rhs_term: "mi.TensorXf"
    ) -> tuple["mi.TensorXu", "mi.TensorXf"]:
        """
        Assembles the local vectors for the Heat simulation problem on the spline
        surface defined by multiple patches.

        The `rhs_term` is a tensor of shape (n_basis_functions, 1) that represents
        the right-hand side term of the Poisson equation. Each entry has to correspond
        to a basis function of the spline surface.

        The method computes the local vectors for each knot span of each patch
        that constitutes the composite spline surface. The following formula is
        applied to compute the local vector for each knot span:

            F_j^(e) ~= area_e sum_q w_q (f(xi_q) phi_j(xi_q)) sqrt(det(G(xi_q))),

        where F_j^(e) is the local vector for the element e, w_q are the Gauss-Legendre
        quadrature weights, xi_q are the quadrature points in the parametric domain,
        area_e is the area of the element's knot span in the parametric domain,
        phi_j are the spline basis functions, G is the metric tensor at the quadrature
        point, and f(xi_q) is the right-hand side term at the quadrature point.
        This is done for each patch.

        When the vectors of each span are assembled, they are concatenated into a
        single tensor that contains all the local vectors for the entire composite
        spline surface. This assumes that all patches have the same degree;
        otherwise they could not be concatenated.

        The current implementation is designed to have a small memory footprint as one
        never exceeds the O(n_knot_spans * n_dofs) memory complexity, meaning that
        the return value takes up most of the memory footprint.

        Time complexity: O(n_patches * n_knot_spans * n_dofs * n_quadrature_points^2),
        where n_dofs is the number of degrees of freedom per knot span. Plugging in the
        default quadrature order (degree + 1), and that n_dofs ~ (degree + 1)^2,
        we obtain O(n_patches * n_knot_spans * degree^4). Most of the time is spent
        computing the basis functions, which scales with the same order. This means
        that in practice vector assembly is faster than matrix assembly, but not by
        a large margin (not degree^2 as one might expect from pure asymptotic analysis).

        :returns: A tuple that consists of

            - dofs_per_span: A tensor of shape (sum_p n_knot_spans_p, n_dofs) that
                contains the indices of the degrees of freedom for each knot span.
            - local_vectors: A tensor of shape (sum_p n_knot_spans_p, n_dofs) that
                contains the local vectors for each knot span,

            where p is the patch index and sum_p n_knot_spans_p is the total number of
            knot spans across all patches.

        :raises ValueError: If the `rhs_term` is not a 2D tensor with shape
            (n_basis_functions, 1), where n_basis_functions is the number of basis
            functions in the composite spline surface (same as number of global
            control points).
        """
        if rhs_term.ndim != 2 or rhs_term.shape[1] != 1:
            raise ValueError(
                "rhs_term must be a 2D tensor with shape (n_basis_functions, 1). "
                f"Got shape {rhs_term.shape} instead."
            )
        if rhs_term.shape[0] != self._composite_iga_spline.global_field_values.shape[0]:
            raise ValueError(
                f"rhs_term must have shape ("
                f"{self._composite_iga_spline.global_field_values.shape[0]}, 1). "
            )

        patch_local_indices: list["mi.TensorXu"] = []
        patch_local_vectors: list["mi.TensorXf"] = []

        for patch, mapping in self._composite_iga_spline.patches_with_mapping:
            interp_spline = patch.interpolating_spline

            knot_span_handler = self.KnotSpanHandler(
                patch=patch, integration_order=self._integration_order
            )
            num_dofs = interp_spline.basis_functions_per_span

            local_vectors = dr.zeros(
                dtype=mi.TensorXf, shape=(knot_span_handler.num_knot_spans, num_dofs)
            )

            local_rhs_term = rhs_term[dr.ravel(mi.TensorXu(mapping))]

            for param_coords, weight in knot_span_handler.gl_coords_and_weights():
                basis_functions = interp_spline.evaluate_basis_functions(
                    parametric_coords=param_coords
                )

                jacobian = patch.evaluate_jacobian(param_coords)
                metric = patch.metric_tensor(jacobian)
                _, det_metric = patch.inv_metric_tensor(metric)

                rhs_value = interp_spline.interpolate(
                    values=local_rhs_term, parametric_coords=param_coords
                )

                product = basis_functions * rhs_value
                local_vectors += product * (weight * dr.sqrt(det_metric))[:, None]

            dofs_per_span = interp_spline.active_basis_function_indices(
                parametric_coords=knot_span_handler.midpoint_parametric_coordinates
            )

            patch_local_vectors.append(local_vectors)
            patch_local_indices.append(dofs_per_span)

        return self._concatenate_patch_local_vectors(
            composite_iga_spline=self._composite_iga_spline,
            local_indices_list=patch_local_indices,
            local_vectors_list=patch_local_vectors,
        )

    @property
    def total_heat(self) -> "mi.TensorXd":
        """
        Computes the total heat in the system by integrating the temperature field
        over the entire spline surface.

        The total heat is computed using the Gauss-Legendre quadrature over the
        parametric domain of each patch in the composite spline surface. The formula
        used for the integration is:

            total_heat = sum_p sum_e area_e sum_q w_q (
                rho_p(xi_q) c_p(xi_q) d_p(xi_q) u_h(xi_q) sqrt(det(G_p(xi_q)))
            ),

        where p is the patch index, e is the knot span index, w_q are the Gauss-Legendre
        quadrature weights, xi_q are the quadrature points in the parametric domain,
        area_e is the area of the element's knot span in the parametric domain,
        rho_p(xi_q) is the density at the quadrature point of patch p, c_p(xi_q) is the
        specific heat at the quadrature point of patch p, d_p(xi_q) is the thickness
        at the quadrature point of patch p, u_h(xi_q) is the temperature at the
        quadrature point of patch p, and G_p(xi_q) is the metric tensor at the
        quadrature point of patch p.

        The expression effectively iterates over all patches and knot spans, evaluating
        the material properties and temperature at the predetermined quadrature points
        and accumulates the resulting heat contributions. This is done in double
        precision to offset accumulation errors.

        Time complexity: O(n_patches * n_knot_spans * n_dofs * n_quadrature_points^2),
        where n_dofs is the number of degrees of freedom per knot span. This n_dofs
        term appears because the temperature field and material properties have to be
        evaluated at each quadrature point, which requires evaluating the basis
        functions, which there are n_dofs of. Plugging in the default quadrature
        order (degree + 1), and that n_dofs ~ (degree + 1)^2, we obtain
        O(n_patches * n_knot_spans * degree^4), so roughly the same as for local
        vector assembly and comparable to local matrix assembly.

        :returns: A double precision tensor of shape (1,) that contains the total
            heat in the system.
        """
        # Use double precision to offset accumulation errors
        total_heat = mi.TensorXd(0.0)

        for patch, mapping in self._composite_iga_spline.patches_with_mapping:
            interp_spline = patch.interpolating_spline

            knot_span_handler = self.KnotSpanHandler(
                patch=patch, integration_order=self._integration_order
            )

            patch_temperature = self.temperature[dr.ravel(mi.TensorXu(mapping))]
            patch_density = self._density[dr.ravel(mi.TensorXu(mapping))]
            patch_specific_heat = self._specific_heat[dr.ravel(mi.TensorXu(mapping))]
            patch_thickness = self._thickness[dr.ravel(mi.TensorXu(mapping))]

            for param_coords, weight in knot_span_handler.gl_coords_and_weights():
                temperature = interp_spline.interpolate(
                    values=patch_temperature, parametric_coords=param_coords
                )
                density = interp_spline.interpolate(
                    values=patch_density, parametric_coords=param_coords
                )
                specific_heat = interp_spline.interpolate(
                    values=patch_specific_heat, parametric_coords=param_coords
                )
                thickness = interp_spline.interpolate(
                    values=patch_thickness, parametric_coords=param_coords
                )

                jacobian = patch.evaluate_jacobian(param_coords)
                metric = patch.metric_tensor(jacobian)
                _, det_metric = patch.inv_metric_tensor(metric)

                integrand = temperature * density * specific_heat * thickness
                total_heat += dr.sum(
                    mi.TensorXd(integrand * (weight * dr.sqrt(det_metric))[:, None])
                )

        return total_heat

    @property
    def heat_generation_rate(self) -> "mi.TensorXd":
        r"""
        Computes the total heat generation rate in the system by integrating the
        source term over the entire spline surface.

        The total heat generation rate is computed using the Gauss-Legendre
        quadrature over the parametric domain of each patch in the composite spline
        surface. The formula used for the integration is:
        ```math
            total_heat_generation_rate = \sum_p \sum_e A_e \sum_q w_q (
                f_p(xi_q) \sqrt(\det(G_p(xi_q)))
            ),
        ```
        where p is the patch index, e is the knot span index, w_q are the Gauss-Legendre
        quadrature weights, xi_q are the quadrature points in the parametric domain,
        area_e is the area of the element's knot span in the parametric domain,
        f_p(xi_q) is the source term at the quadrature point of patch p
        and G_p(xi_q) is the metric tensor at the quadrature point of patch p.

        The expression effectively iterates over all patches and knot spans, evaluating
        the source term at the predetermined quadrature points and accumulates the
        resulting contributions. This is done in double precision to offset
        accumulation errors.

        Time complexity: O(n_patches * n_knot_spans * n_quadrature_points^2), where
        n_quadrature_points is the number of quadrature points per knot span. Plugging
        in the default quadrature order (degree + 1), we obtain
        O(n_patches * n_knot_spans * degree^2), which is faster than local vector
        assembly and much faster than local matrix assembly.

        :returns: A double precision tensor of shape (1,) that contains the total
            heat generation rate in the system.
        """
        # Use double precision to offset accumulation errors
        total_heat_generation_rate = mi.TensorXd(0.0)

        for patch, mapping in self._composite_iga_spline.patches_with_mapping:
            interp_spline = patch.interpolating_spline

            knot_span_handler = self.KnotSpanHandler(
                patch=patch, integration_order=self._integration_order
            )

            local_source_term = self._source_term[dr.ravel(mi.TensorXu(mapping))]

            for param_coords, weight in knot_span_handler.gl_coords_and_weights():
                source_term = interp_spline.interpolate(
                    values=local_source_term, parametric_coords=param_coords
                )

                jacobian = patch.evaluate_jacobian(param_coords)
                metric = patch.metric_tensor(jacobian)
                _, det_metric = patch.inv_metric_tensor(metric)

                integrand = source_term
                total_heat_generation_rate += dr.sum(
                    mi.TensorXd(integrand * (weight * dr.sqrt(det_metric))[:, None])
                )

        return total_heat_generation_rate

    @property
    def spline(self) -> CompositeIGASplineSurface:
        """
        The composite spline surface associated with this heat simulation.
        """
        return self._composite_iga_spline

    @property
    def temperature(self) -> "mi.TensorXf":
        """
        Returns the temperature field of the composite spline surface, shaped as (V,).
        """
        return self._composite_iga_spline.global_field_values

    @temperature.setter
    def temperature(self, value: "mi.TensorXf") -> None:
        """
        Sets the temperature field of the composite spline surface to the given `value`.

        The value must be a 2D tensor of shape (n_global_dofs, 1), where n_global_dofs
        is the number of global degrees of freedom in the composite spline surface.
        This means it must have the same shape as the current temperature field.

        :raises ValueError: If the shape of the provided temperature `value` does not
            match the shape of the current global field values.
        """
        self._composite_iga_spline.global_field_values = value
