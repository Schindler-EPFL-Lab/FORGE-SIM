from collections.abc import Callable

import drjit as dr
import mitsuba as mi
import numpy as np

from rebel_shape.splines.bspline import BSpline
from rebel_shape.splines.interpolating_spline import InterpolatingSpline


class BivariateBSpline(InterpolatingSpline):
    """
    Represents a tensor product B-Spline surface made up of two `BSpline` instances.

    B-splines form a tensor product basis, i.e. they consist of a number of univariate
    B-splines (implemented in `BSpline`). For surfaces, this means that the basis
    functions used to interpolate the values can be expressed as:

        B_k(u, v) = M_j(v) N_i(u),  k = n_basis_functions_u * j + i,

    where B_k are the bivariate basis functions, N are the univariate B-Spline basis
    functions along u, M are the univariate B-spline basis functions along v, k is the
    global basis function index, i the basis function index along the u direction,
    and j the function index along the v direction.

    Each of the two univariate B-Spline has its knot vector and order; the orders of the
    two splines can be different. In practice, this is typically not wanted, unless one
    direction is much smaller than the other, e.g. a thin narrow rectangular slip.

    For more details about the B-splines themselves, see the `BSpline` class.
    """

    PROJECTION_TOL = 1.192092896e-07 * 2
    """
    Tolerance for projection of points onto the spline curve. It is twice the
    machine epsilon for float32, which is the smallest difference between two
    representable numbers. Factor 2 is added for safety. This value should be as small
    as it can be to keep the projection stable without introducing large errors.
    """

    def __init__(self, u_spline: BSpline, v_spline: BSpline) -> None:
        """
        Initializes the Bivariate B-spline surface with two univariate B-splines.

        `u_spline` and `v_spline` are the B-splines that define the surface in the
        u and v directions, respectively. Each carries their own knot vector and
        degree.
        """
        self._u_spline = u_spline
        self._v_spline = v_spline

    def validate_parametric_coords(self, parametric_coords: "mi.TensorXf") -> None:
        """
        Check if the parametric coordinates are valid for the B-spline surface.

        According to the contract of `CoordinateSpline`, the `parametric_coords` must
        be a 2D tensor of shape (n_points, 2). The values must be within the range of
        the knot vectors defined by the `u_spline` and `v_spline`.

        :raises ValueError: If the provided `parametric_coords` is not a 2D tensor,
            if it does not have exactly two columns, or if the values are outside the
            range of the knot vectors.
        """
        if len(parametric_coords.shape) != 2:
            raise ValueError("`parametric_coords` must be a 2D tensor!")
        if parametric_coords.shape[1] != 2:
            raise ValueError(
                "`parametric_coords` must be a 2D tensor of shape (n_points, 2)."
            )

        if (
            dr.min(parametric_coords[:, 0]) < self._u_spline.knots[0]
            or dr.max(parametric_coords[:, 0]) > self._u_spline.knots[-1]
        ):
            raise ValueError(
                f"first component of `parametric_coords` must be in the range of the "
                f"knot vector: [{self._u_spline.knots[0]}, {self._u_spline.knots[-1]}]"
            )
        if (
            dr.min(parametric_coords[:, 1]) < self._v_spline.knots[0]
            or dr.max(parametric_coords[:, 1]) > self._v_spline.knots[-1]
        ):
            raise ValueError(
                f"second component of `parametric_coords` must be in the range of the "
                f"knot vector: [{self._v_spline.knots[0]}, {self._v_spline.knots[-1]}]"
            )

    def interpolate(
        self,
        values: "mi.TensorXf",
        parametric_coords: "mi.TensorXf",
    ) -> "mi.TensorXf":
        """
        Compute the B-spline curve coordinates at the given `parametric_coords`.

        `parametric_coords` is a 2D tensor of shape (n_points, 2) where n_points
        is the number of points to evaluate, and the second dimension is 2 because
        a surface has two parametric directions (u, v).

        The curve is evaluated at parametric points with the formula:
            I_d(u, v) = sum_{i=0 to n_u} sum_{j=0 to n_v}
                        N_{i,p_u}(u) * M_{j,p_v}(v) * I_{i,j,d},
        where:
            - d is the coordinate index (0, 1, or 2)
            - P_{i,j,d} is the d-th coordinate of the control point P_{i,j}
            - N_{i,p_u}(u) is the i-th B-spline basis function in the u direction
            - M_{j,p_v}(v) is the j-th B-spline basis function in the v direction

        Only a subset of basis functions are non-zero at a given point (u, v).
        Specifically, the u and v parameters are mapped to the appropriate knot spans
        and each has `degree + 1` active basis functions. The tensor product nature
        thus means that the there are a total of `(degree_u + 1) * (degree_v + 1)`
        non-zero basis functions at a given point (u, v).

        :returns: A Mitsuba tensor of shape (n_points, n_dimensions) representing the
            interpolated field values at the specified `parametric_coords`.

        :raises ValueError: If `parametric_coords` is not a 2D tensor (n_points, 1),
            if the values in `parametric_coords` are outside the knot vector range,
            or if the shape of `values` does not match the number of basis functions
            defined by the B-spline surface.
        """
        self.validate_interpolating_values(values)
        self.validate_parametric_coords(parametric_coords)

        u_span_data = self._u_spline.compute_span_data(parametric_coords[:, 0])
        v_span_data = self._v_spline.compute_span_data(parametric_coords[:, 1])

        u_basis_funcs = self._u_spline.evaluate_basis_functions_internal(
            parametric_span_data=u_span_data, degree=self._u_spline.degree
        )
        v_basis_funcs = self._v_spline.evaluate_basis_functions_internal(
            parametric_span_data=v_span_data, degree=self._v_spline.degree
        )

        result = dr.zeros(
            dtype=mi.TensorXf, shape=(parametric_coords.shape[0], values.shape[1])
        )

        for j in range(self._v_spline.degree + 1):
            basis_index_v = v_span_data.global_basis_function_index(local_index=j)

            for k in range(self._u_spline.degree + 1):
                basis_index_u = u_span_data.global_basis_function_index(local_index=k)

                cp_index = dr.ravel(
                    basis_index_v * self._u_spline.num_basis_functions + basis_index_u
                )
                result += (v_basis_funcs[:, j] * u_basis_funcs[:, k])[:, None] * values[
                    cp_index, :
                ]

        return result

    def evaluate_basis_functions(
        self, parametric_coords: "mi.TensorXf"
    ) -> "mi.TensorXf":
        """
        Evaluates the basis functions of the B-spline surface at the given
        `parametric_coords`.

        `parametric_coords` is a 2D tensor of shape (n_points, 2) where n_points
        is the number of points to evaluate, and the second dimension is 2 because
        a surface has two parametric directions (u, v).

        Since the bivariate B-spline surface is a tensor product of two univariate
        B-splines, the basis functions are evaluated by first computing the
        basis functions for the u and v directions separately, and then combining
        them to form the bivariate basis functions.

        The formula is simply:
            B_k(u, v) = M_j(v) N_i(u), k = n_basis_functions_u * j + i,

        :returns: A Mitsuba tensor of shape (n_points, n_basis_functions) where each row
                  corresponds to the evaluated basis functions at the respective
                  `parametric_coords`.
        :raises ValueError: If `parametric_coords` is not a 2D tensor of shape
            (n_points, 2).
        """
        self.validate_parametric_coords(parametric_coords)

        u_span_data = self._u_spline.compute_span_data(parametric_coords[:, 0])
        v_span_data = self._v_spline.compute_span_data(parametric_coords[:, 1])

        u_basis_funcs = self._u_spline.evaluate_basis_functions_internal(
            parametric_span_data=u_span_data, degree=self._u_spline.degree
        )
        v_basis_funcs = self._v_spline.evaluate_basis_functions_internal(
            parametric_span_data=v_span_data, degree=self._v_spline.degree
        )

        result = dr.zeros(
            dtype=mi.TensorXf,
            shape=(parametric_coords.shape[0], self.basis_functions_per_span),
        )

        for j in range(self._v_spline.degree + 1):
            for i in range(self._u_spline.degree + 1):
                k = j * (self._u_spline.degree + 1) + i

                result[:, k] = dr.ravel(v_basis_funcs[:, j] * u_basis_funcs[:, i])

        return result

    def evaluate_basis_functions_first_derivative(
        self, parametric_coords: "mi.TensorXf"
    ) -> "mi.TensorXf":
        """
        Evaluate the first derivatives of the basis functions of the B-spline surface
        at the given `parametric_coords`.

        `parametric_coords` is a 2D tensor of shape (n_points, 2) where n_points
        is the number of points to evaluate, and the second dimension is 2 because
        a surface has two parametric directions (u, v).

        Since the bivariate B-spline surface is a tensor product of two univariate
        B-splines, the first derivatives are computed by evaluating the basis functions
        and their derivatives in both directions separately, and then combining
        them to obtain two tensors of first derivatives along the u and v directions.

        The formula is:
            partial_u B_k(u, v) = M_j(v) N_i'(u),
            partial_v B_k(u, v) = M_j'(v) N_i(u),
            k = j * n_basis_functions_u + i,    (denoted as `cp_index` in the code)

        :returns: A Mitsuba tensor of shape (n_points, n_basis_functions, 2) where
            each row corresponds to the evaluated first derivatives of the basis
            functions at the respective `parametric_coords`.

        :raises ValueError: If `parametric_coords` is not a 2D tensor of shape
            (n_points, 2).
        """
        self.validate_parametric_coords(parametric_coords)

        u_span_data = self._u_spline.compute_span_data(parametric_coords[:, 0])
        v_span_data = self._v_spline.compute_span_data(parametric_coords[:, 1])

        u_basis_func_values_and_first_derivatives = (
            self._u_spline.evaluate_basis_function_derivatives(
                parametric_span_data=u_span_data, derivative_order=1
            )
        )
        v_basis_func_values_and_first_derivatives = (
            self._v_spline.evaluate_basis_function_derivatives(
                parametric_span_data=v_span_data, derivative_order=1
            )
        )

        result = dr.zeros(
            dtype=mi.TensorXf,
            shape=(parametric_coords.shape[0], self.basis_functions_per_span, 2),
        )

        for j in range(self._v_spline.basis_functions_per_span):
            for i in range(self._u_spline.basis_functions_per_span):
                cp_index = j * self._u_spline.basis_functions_per_span + i

                result[:, cp_index, 0] = dr.ravel(
                    u_basis_func_values_and_first_derivatives[:, 1, i]
                    * v_basis_func_values_and_first_derivatives[:, 0, j]
                )
                result[:, cp_index, 1] = dr.ravel(
                    u_basis_func_values_and_first_derivatives[:, 0, i]
                    * v_basis_func_values_and_first_derivatives[:, 1, j]
                )

        return result

    def active_basis_function_indices(
        self, parametric_coords: "mi.TensorXf"
    ) -> "mi.TensorXu":
        """
        Returns the indices of the active basis functions at the specified
        `parametric_coords`.

        `parametric_coords` is a 2D tensor of shape (n_points, 2) where n_points
        is the number of points to evaluate, and the second dimension is 2 because
        a surface has two parametric directions (u, v).

        The first step in evaluating the basis functions at a given parametric point
        is determining which basis functions are non-zero at that point. This is done
        by determining which knot spans contain the parametric coordinates and then
        returning the indices of the basis functions that are active in those spans.
        This is done for both the u and v directions separately, and then the final
        index is pieced together as
            k = n_basis_functions_u * j + i,
        - k is the global basis function index,
        - j is the index of the basis function in the v direction,
        - i is the index of the basis function in the u direction.

        These indices can be used interchangeably as indices of control points as
        each basis function corresponds to a control point in the B-spline.

        :returns: A tensor of shape (n_points, basis_functions_per_span) containing the
            indices of the active basis functions at the given `parametric_coords`.

        :raises ValueError: If `parametric_coords` is not a 2D tensor of shape
            (n_points, 2).
        """
        self.validate_parametric_coords(parametric_coords)

        n_points = parametric_coords.shape[0]
        u_coords = dr.reshape(
            dtype=mi.TensorXf, value=parametric_coords[:, 0], shape=n_points
        )
        v_coords = dr.reshape(
            dtype=mi.TensorXf, value=parametric_coords[:, 1], shape=n_points
        )

        u_span_data = self._u_spline.compute_span_data(u_coords)
        v_span_data = self._v_spline.compute_span_data(v_coords)

        result = dr.zeros(
            dtype=mi.TensorXu,
            shape=(n_points, (self._u_spline.degree + 1) * (self._v_spline.degree + 1)),
        )

        for j in range(self._v_spline.degree + 1):
            basis_index_v = v_span_data.global_basis_function_index(local_index=j)

            for k in range(self._u_spline.degree + 1):
                basis_index_u = u_span_data.global_basis_function_index(local_index=k)

                cp_index = j * (self._u_spline.degree + 1) + k
                result[:, cp_index] = dr.ravel(
                    basis_index_v * self._u_spline.num_basis_functions + basis_index_u
                )

        return result

    def insert_knot_u(self, knot_position: float) -> None:
        """
        Insert a knot into the u-direction B-spline at position `knot_position`.

        Calls the `insert_knot` method of the underlying `BSpline` instance accessible
        through the `u_spline` property.

        :raises ValueError: If `knot_position` is outside the range of the knot vector,
            if the multiplicity of internal knots is greater than `degree`, or if the
            multiplicity of the first or the last knot is greater than `degree + 1`,
        """
        self._u_spline.insert_knot(knot_position=knot_position)

    def insert_knot_v(self, knot_position: float) -> None:
        """
        Insert a knot into the u-direction B-spline at position `knot_position`.

        Calls the `insert_knot` method of the underlying `BSpline` instance accessible
        through the `u_spline` property.

        :raises ValueError: If `knot_position` is outside the range of the knot vector,
            if the multiplicity of internal knots is greater than `degree`, or if the
            multiplicity of the first or the last knot is greater than `degree + 1`,
        """
        self._v_spline.insert_knot(knot_position=knot_position)

    def insert_knot_and_update_values_u(
        self, knot_position: float, values_list: list["mi.TensorXf"]
    ) -> list["mi.TensorXf"]:
        """
        Insert a knot into the B-spline along the u direction at `knot_position`
        and update all values in `values_list` according to the new basis functions.

        Knot insertion is performed by calling the `insert_knot_u` method.

        When inserting knots, it is typically necessary to also update the values
        that the spline interpolates, as the number of basis functions increases
        with the number of knot spans. The `values_list` parameter is a list of
        Mitsuba tensors of shape (n_basis_functions, n_dimensions_k) where
        n_dimensions_k is the number of dimensions of the k-th values tensor in the
        list.

        This is performed in a similar manner to the univariate B-spline (see
        `BSpline.insert_knot_and_update_values`), but the basis has to be reshaped to
        (n_basis_functions_v,  n_basis_functions_u, n_dimensions_k) to allow for
        inserting new values along the u direction. Thus, there are
        (n_basis_functions_v,  n_basis_functions_u + 1, n_dimensions_k) values
        after the knot insertion. The new values are then flattened.

        :returns: A list of Mitsuba tensors of shape (n_basis_functions, n_dimensions_k)
            where `n_basis_functions` is the number of basis functions after knot
            insertion and `n_dimensions_k` is the number of components of k-th element.

        :raises ValueError: If `knot_position` is outside the range of the knot vector,
            if the multiplicity of internal knots is greater than `degree`, or if the
            B-spline does not have an open knot vector.
        """
        if not self.has_open_knot_vectors:
            raise ValueError(
                "Knot insertion is only supported for B-spline surfaces with open "
                "knot vectors."
            )

        span_data_u = self._u_spline.compute_span_data(mi.TensorXf([knot_position]))
        position_u = int(span_data_u.knot_spans.numpy().astype(np.uint32)[0])
        p = self._u_spline.degree

        original_knot_vector_u = self._u_spline.knots.copy()
        self._u_spline.insert_knot(knot_position=knot_position)

        new_values_list: list["mi.TensorXf"] = []

        for values in values_list:
            dofs_per_cp = values.shape[1]

            values = values.numpy().reshape(
                self._v_spline.num_basis_functions,
                self._u_spline.num_basis_functions - 1,  # Got increased by 1 at insert
                dofs_per_cp,
            )
            new_values = np.zeros(
                (
                    self._v_spline.num_basis_functions,
                    self._u_spline.num_basis_functions,
                    dofs_per_cp,
                ),
                dtype=np.float64,
            )
            new_values[:, : position_u - p + 1, :] = values[:, : position_u - p + 1, :]
            new_values[:, position_u + 1 :, :] = values[:, position_u:, :]

            idx = np.arange(position_u - p + 1, position_u + 1)
            alphas = (knot_position - original_knot_vector_u[idx]) / (
                original_knot_vector_u[idx + p] - original_knot_vector_u[idx]
            )
            new_values[:, idx, :] = (
                alphas[None, :, None] * values[:, idx, :]
                + (1 - alphas[None, :, None]) * values[:, idx - 1, :]
            )

            new_values_list.append(mi.TensorXf(new_values.reshape(-1, dofs_per_cp)))

        return new_values_list

    def insert_knot_and_update_values_v(
        self, knot_position: float, values_list: list["mi.TensorXf"]
    ) -> list["mi.TensorXf"]:
        """
        Insert a knot into the B-spline along the v direction at `knot_position`
        and update all values in `values_list` according to the new basis functions.

        Knot insertion is performed by calling the `insert_knot_v` method.

        When inserting knots, it is typically necessary to also update the values
        that the spline interpolates, as the number of basis functions increases
        with the number of knot spans. The `values_list` parameter is a list of
        Mitsuba tensors of shape (n_basis_functions, n_dimensions_k) where
        n_dimensions_k is the number of dimensions of the k-th values tensor in the
        list.

        This is performed in a similar manner to the univariate B-spline (see
        `BSpline.insert_knot_and_update_values`), but the basis has to be reshaped to
        (n_basis_functions_u,  n_basis_functions_v, n_dimensions_k) to allow for
        inserting new values along the v direction. Thus, there are
        (n_basis_functions_u + 1,  n_basis_functions_v, n_dimensions_k) values
        after the knot insertion. The new values are then flattened.

        :returns: A list of Mitsuba tensors of shape (n_basis_functions, n_dimensions_k)
            where `n_basis_functions` is the number of basis functions after knot
            insertion and `n_dimensions_k` is the number of components of k-th element.

        :raises ValueError: If `knot_position` is outside the range of the knot vector,
            if the multiplicity of internal knots is greater than `degree`, or if the
            B-spline does not have an open knot vector.
        """
        if not self.has_open_knot_vectors:
            raise ValueError(
                "Knot insertion is only supported for B-spline surfaces with open "
                "knot vectors."
            )

        span_data_v = self._v_spline.compute_span_data(mi.TensorXf([knot_position]))
        position_v = int(span_data_v.knot_spans.numpy().astype(np.uint32)[0])
        q = self._v_spline.degree

        original_knot_vector_v = self._v_spline.knots.copy()
        self._v_spline.insert_knot(knot_position=knot_position)

        new_values_list: list["mi.TensorXf"] = []

        for values in values_list:
            dofs_per_cp = values.shape[1]

            values = values.numpy().reshape(
                self._v_spline.num_basis_functions - 1,  # Got increased by 1 at insert
                self._u_spline.num_basis_functions,
                dofs_per_cp,
            )
            new_values = np.zeros(
                (
                    self._v_spline.num_basis_functions,
                    self._u_spline.num_basis_functions,
                    dofs_per_cp,
                ),
                dtype=np.float64,
            )
            new_values[: position_v - q + 1, :, :] = values[: position_v - q + 1, :, :]
            new_values[position_v + 1 :, :, :] = values[position_v:, :, :]

            idx = np.arange(position_v - q + 1, position_v + 1)
            alphas = (knot_position - original_knot_vector_v[idx]) / (
                original_knot_vector_v[idx + q] - original_knot_vector_v[idx]
            )
            new_values[idx, :, :] = (
                alphas[:, None, None] * values[idx, :, :]
                + (1 - alphas[:, None, None]) * values[idx - 1, :, :]
            )

            new_values_list.append(mi.TensorXf(new_values.reshape(-1, dofs_per_cp)))

        return new_values_list

    def h_refinement(
        self, factor: int, values_list: list["mi.TensorXf"]
    ) -> list["mi.TensorXf"]:
        """
        Increases the resolution of the B-spline by `factor` by subdividing each knot
        span into `factor` equal parts and inserting new knots at these positions.

        This is one of the 3 main methods to increase the accuracy of spline based
        representations: h-refinement, p-refinement, and k-refinement.
        h-refinement is the process of increasing the number of knot spans in
        the parametric space by means of knot insertion.

        When inserting knots, it is typically necessary to also update the values
        that the spline interpolates, as the number of basis functions increases
        with the number of knot spans. The `values_list` parameter is a list of
        Mitsuba tensors of shape (n_basis_functions, n_dimensions_k) where
        n_dimensions_k is the number of dimensions of the k-th values tensor in the
        list.

        Note that this operation is quite expensive overall, as at each iteration all
        control points are updated once, leading to a total of O((n * factor)^2 * n)
        operations, where n is the larger resolution of the two univariate B-splines.
        Still, the operation is typically performed only once or a few times during
        the entire runtime of the application, so it is not a bottleneck.

        :returns: A list of Mitsuba tensors of shape (n_basis_functions, n_dimensions_k)
            where `n_basis_functions` is the number of basis functions after refinement.

        :raises ValueError: If `factor` is not a positive integer, or if the spline
            does not have an open knot vector.
        """
        if factor < 1:
            raise ValueError("`factor` must be a positive integer.")

        for knot_start, knot_end in self._u_spline.knot_spans.numpy():
            for i in range(1, factor):
                new_knot = knot_start + i * (knot_end - knot_start) / factor
                values_list = self.insert_knot_and_update_values_u(
                    values_list=values_list, knot_position=new_knot
                )

        for knot_start, knot_end in self._v_spline.knot_spans.numpy():
            for i in range(1, factor):
                new_knot = knot_start + i * (knot_end - knot_start) / factor
                values_list = self.insert_knot_and_update_values_v(
                    values_list=values_list, knot_position=new_knot
                )

        return values_list

    def project_function(
        self, func: Callable[["mi.TensorXf"], "mi.TensorXf"]
    ) -> "mi.TensorXf":
        """
        Projects the given function `func` onto the bivariate B-spline basis functions.

        In most practical applications, the data is not available in the form of
        values at the control points of the spline, but rather as a function defined
        over the parametric space of the spline.

        This function can be arbitrary, so long as it takes as inputs a Mitsuba
        tensor of shape (n_points, n_params) and returns a Mitsuba tensor of shape
        (n_points, 1), where `n_points` is the number of points at which the function
        is evaluated and `n_params` is the number of parametric dimensions.

        This class assumes that the function has the same smoothness as the B-spline
        basis functions, i.e., it is at least `degree` times continuously
        differentiable. Furthermore, it assumes that sampling the function at
        predefined points yields a reconstruction in terms of Lagrange interpolating
        basis functions, which can then be extracted in a one-to-one mapping
        to the spline control points.

        For more details about the method see the paper: "Lagrange extraction and
        projection for NURBS basis functions: A direct link between isogeometric and
        standard nodal finite element formulations" by (D. Schillinger et al., 2016).

        Note that the sample points need to be chosen very carefully and have to
        account for machine precision errors to always land in the correct knot
        spans, since sampling is done at the edges of the knot spans. This edge
        sampling also means that the function is sampled multiple times at
        the edges of the knot spans, and naively accumulating the function values
        at these points would lead to incorrect control point values. To account for
        this, the control point values are divided by the number of times they were
        sampled.

        The order of accuracy of this reconstruction should be O(h^{p+1}), where h is
        the distance between the sample points and p is the degree of the B-spline basis
        functions. The accuracy is limited by how well Lagrange interpolation
        approximates the function on small intervals.

        :returns: A tensor of shape (n_control_points, 1) representing
            the projected function values at the control points of the spline patch.
        """
        p = self._u_spline.degree
        q = self._v_spline.degree
        entries = (p + 1) * (q + 1)

        knot_spans = self.knot_spans

        # Compute how small of a perturbation can be used while still making sure that
        # each sampled point does not coincide with the position of a knot.
        interval_length_u = dr.max(knot_spans[:, 0, 1]) - dr.min(knot_spans[:, 0, 0])
        interval_length_v = dr.max(knot_spans[:, 1, 1]) - dr.min(knot_spans[:, 1, 0])

        lengths_u = knot_spans[:, 0, 1] - knot_spans[:, 0, 0]
        lengths_v = knot_spans[:, 1, 1] - knot_spans[:, 1, 0]

        average_u = 0.5 * (knot_spans[:, 0, 0] + knot_spans[:, 0, 1])
        average_v = 0.5 * (knot_spans[:, 1, 0] + knot_spans[:, 1, 1])

        smallest_span_length_u = dr.min(lengths_u)
        smallest_span_length_v = dr.min(lengths_v)

        epsilon_u = interval_length_u / smallest_span_length_u * self.PROJECTION_TOL * p
        epsilon_v = interval_length_v / smallest_span_length_v * self.PROJECTION_TOL * q

        samples_u = 2.0 * (1 - epsilon_u) * np.arange(p + 1) / p - (1.0 - epsilon_u)
        samples_v = 2.0 * (1 - epsilon_v) * np.arange(q + 1) / q - (1.0 - epsilon_v)

        cp_values = dr.zeros(dtype=mi.TensorXf, shape=(self.num_basis_functions, 1))
        parametric_coords = dr.empty(dtype=mi.TensorXf, shape=(knot_spans.shape[0], 2))

        parametric_coords[:, 0] = dr.ravel(average_u)
        parametric_coords[:, 1] = dr.ravel(average_v)

        dofs_per_span = self.active_basis_function_indices(
            parametric_coords=parametric_coords
        )

        transformation_matrices = dr.zeros(
            dtype=mi.TensorXf, shape=(knot_spans.shape[0], entries, entries)
        )
        function_values = dr.zeros(
            dtype=mi.TensorXf, shape=(knot_spans.shape[0], entries)
        )

        for j, pos_v in enumerate(samples_v):
            for i, pos_u in enumerate(samples_u):
                real_pos_u = average_u + 0.5 * lengths_u * pos_u
                real_pos_v = average_v + 0.5 * lengths_v * pos_v

                parametric_coords[:, 0] = dr.ravel(real_pos_u)
                parametric_coords[:, 1] = dr.ravel(real_pos_v)

                basis_functions = self.evaluate_basis_functions(
                    parametric_coords=parametric_coords
                )
                func_val = func(parametric_coords)

                k = j * (p + 1) + i

                transformation_matrices[:, k, :] = dr.ravel(basis_functions)
                function_values[:, k] = dr.ravel(func_val)

        product = np.einsum(
            "ijk,ik->ij",
            np.linalg.inv(transformation_matrices.numpy().astype(np.float64)),
            function_values.numpy(),
        )
        dr.scatter_add(
            target=cp_values.array,
            value=dr.ravel(mi.TensorXf(product)),
            index=dr.ravel(dofs_per_span),
            mode=dr.ReduceMode.Direct,
        )

        # Divide by the number of times each basis function was sampled
        ones = dr.ones(dtype=mi.TensorXf, shape=product.shape)
        repeat = dr.zeros(dtype=mi.TensorXf, shape=(self.num_basis_functions, 1))
        dr.scatter_add(
            target=repeat.array,
            value=dr.ravel(ones),
            index=dr.ravel(dofs_per_span),
            mode=dr.ReduceMode.Direct,
        )

        cp_values /= repeat

        return cp_values

    @property
    def num_basis_functions(self) -> int:
        """
        Total number of basis functions used to define the B-spline surface.

        The total number of basis functions is the product of the number of basis
        functions in the u and v direction univariate B-splines.
        """
        return self._u_spline.num_basis_functions * self._v_spline.num_basis_functions

    @property
    def basis_functions_per_span(self) -> int:
        """
        Returns the number of basis functions per knot span in the B-spline surface.

        The number of basis functions per knot span is given by the product of the
        number of basis functions in the u and v directions, which is equal to
        (degree_u + 1) * (degree_v + 1).

        :returns: An integer representing the number of basis functions per knot span.
        """
        return (self._u_spline.degree + 1) * (self._v_spline.degree + 1)

    @property
    def degree(self) -> int:
        """
        Polynomial degree of the B-spline surface.

        Returns the higher degree of the two univariate B-splines that form the surface.
        """
        return max(self._u_spline.degree, self._v_spline.degree)

    @property
    def u_spline(self) -> BSpline:
        """
        The univariate B-spline in the u direction.
        """
        return self._u_spline

    @property
    def v_spline(self) -> BSpline:
        """
        The univariateB-spline in the v direction.
        """
        return self._v_spline

    @property
    def knot_spans(self) -> "mi.TensorXf":
        """
        Knot spans of the B-spline.

        Since the bivariate B-spline is a tensor product of two univariate B-splines,
        this property returns a 3D tensor of shape (n_spans, 2, 2) where each span is
        represented by a pair of start and end knots in the u and v directions.

        :returns: A tensor of shape (n_spans, 2, 2) where n_spans is the number of
            knot spans in the u and v directions combined. Each span is represented
            by a pair of start and end knots in the u and v directions, first one
            corresponds to the u direction, second one to the v direction.
        """
        u_knot_spans = self._u_spline.knot_spans.numpy()
        v_knot_spans = self._v_spline.knot_spans.numpy()

        num_u_spans = u_knot_spans.shape[0]
        num_v_spans = v_knot_spans.shape[0]

        result = np.zeros((num_u_spans * num_v_spans, 2, 2))

        # u dimension (first row)
        result[:, 0, 0] = np.tile(u_knot_spans[:, 0], num_v_spans)  # v start
        result[:, 0, 1] = np.tile(u_knot_spans[:, 1], num_v_spans)  # v end

        # v dimension (second row)
        result[:, 1, 0] = np.repeat(v_knot_spans[:, 0], num_u_spans)  # u start
        result[:, 1, 1] = np.repeat(v_knot_spans[:, 1], num_u_spans)  # u end

        return mi.TensorXf(result)

    @property
    def has_open_knot_vectors(self) -> bool:
        """
        Check if the B-spline surface has open knot vectors in both u and v directions.

        An open knot vector means that the first and last knots are repeated
        `degree + 1` times, which is a common property for B-splines to ensure that the
        curve starts and ends at the first and last control points.
        """
        return (
            self._u_spline.has_open_knot_vector and self._v_spline.has_open_knot_vector
        )
