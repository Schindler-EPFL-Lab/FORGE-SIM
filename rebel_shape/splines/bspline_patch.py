from typing import Any

import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import numpy.typing as npt

from rebel_shape.meshing.patch import Patch
from rebel_shape.splines.bivariate_bspline import BivariateBSpline
from rebel_shape.splines.bspline import BSpline
from rebel_shape.splines.spline_patch import SplinePatch


class BSplinePatch(SplinePatch):
    r"""
    Mitsuba compatible implementation of a B-Spline patch.

    A B-Spline patch uses the tensor product of basis functions from two abstract
    `BSpline` objects weighted by a set of `control_points` to define a surface in
    3D space.

    In general, a B-Spline patch can be defined as a function of two parameters (u, v):
    ```math
        S(u, v) = \sum_{i=0}^{n_u} \sum_{j=0}^{n_v} N_i(u) M_j(v) P_ij
    ```
    where:
        - S(u, v) is the surface point (3D value) at (u, v)
        - $N_i(u)$ are the basis functions in the u direction
        - $M_j(v)$ are the basis functions in the v direction
        - $P_ij$ are the control points
    The tensor product name comes from the fact that basis functions in the u and v
    directions are independent. In general, most basis functions evaluate to zero:
    only `u.degree + 1` * `v.degree + 1` basis functions are non-zero at a given
    point (u, v).

    The number of control points is dictated by the number of basis functions in the
    two `BSpline` objects (product of these two numbers). Control points are best
    visualized as a grid of points in 3D space.
    """

    def __init__(
        self,
        name: str,
        u_spline: BSpline,
        v_spline: BSpline,
        control_points: "mi.TensorXf",
    ) -> None:
        """
        Initialize a B-spline surface named `patch_name` with the given abstract
        B-Splines (`u_spline` and`v_spline`), and `control_points`.

        The number of provided control points must be compatible with the knot vectors
        from provided `u_spline` and `v_spline`: product of the number of basis
        functions in each direction must match the number of control points.

        The control points are treated as if they are provide a grid-like
        structure; the first dimension
        (think of it as x, columns) corresponds to the `u_spline` and the second
        dimension (think of it as y, rows) corresponds to the `v_spline`.

        :raises ValueError: If `control_points` is not a 2D tensor, if `control_points`
            does not have the correct shape, if the number of control points does not
            match the number of control points derived from the `u_spline` and
            `v_spline` knot vectors.
        """
        self._name = name

        self._spline = BivariateBSpline(u_spline=u_spline, v_spline=v_spline)

        # Checks the shape of the control points
        super().__init__(control_points=control_points)

        self._quads = self._build_quads()

    def __getstate__(self) -> dict[str, Any]:
        """
        Exports the B-spline surface to a serializable state.

        `_control_points` and `_quads` are not serializable, so they are explicitly
        converted to numpy arrays for serialization.

        :returns: A dictionary representing the state of the object, ready for
                  serialization with `jsonpickle`.
        """
        st = self.__dict__.copy()
        # replace the problematic attribute with its array form
        st["_control_points"] = self.control_points.numpy()
        st["_quads"] = self.quads.numpy()
        return st

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restore the B-spline surface from a serialized state.

        `state` is a dictionary that contains the serialized state of the object
        including values of `control_points and `quads` which are converted back to
        Mitsuba tensors.
        """
        control_points = state.pop("_control_points")
        quads = state.pop("_quads")

        self.__dict__.update(state)

        self._control_points = mi.TensorXf(control_points)
        self._quads = mi.TensorXu(quads)

    def _build_quads(self) -> "mi.TensorXu":
        """
        Build the quads that make up the B-spline patch.

        The quads are defined by the control points in a grid-like structure.
        Each quad is defined by four control points, and the indices are
        calculated based on the number of control points in each direction.

        Control points follow a simple pattern:

            n_u(n_v - 1)  ...  ... n_u * n_v - 1
                :          :         :
               n_u --- n_u + 1 ... 2 * n_u - 1
                |          |         |
                0 -------- 1 ... ... n_u - 1

        :returns: A Mitsuba array of quad indices.
        """
        num_quads = (self._num_cps_u - 1) * (self._num_cps_v - 1)

        # Create arrays for all i and j indices
        i_indices = dr.ravel(dr.arange(dtype=mi.TensorXu, size=self._num_cps_u - 1))
        j_indices = dr.ravel(dr.arange(dtype=mi.TensorXu, size=self._num_cps_v - 1))

        # Create meshgrid of indices
        j_grid, i_grid = dr.meshgrid(j_indices, i_indices, indexing="ij")
        i_grid = dr.ravel(i_grid)
        j_grid = dr.ravel(j_grid)

        # Calculate base indices for all quads at once
        base_indices = j_grid * self._num_cps_u + i_grid

        # Create quad array
        quad_indices = dr.zeros(dtype=mi.TensorXu, shape=(num_quads, 4))

        quad_indices[:, 0] = base_indices
        quad_indices[:, 1] = base_indices + 1
        quad_indices[:, 2] = base_indices + self._num_cps_u + 1
        quad_indices[:, 3] = base_indices + self._num_cps_u

        return quad_indices

    def evaluate_coordinates(self, parametric_coords: "mi.TensorXf") -> "mi.TensorXf":
        r"""
        Compute the B-spline patch coordinates at the given `parametric_coords`.

        `parametric_coords` is a 2D tensor of shape (n_points, 2) where n_points
        is the number of points to evaluate, and the second dimension is 2 because
        a surface has two parametric directions (u, v).

        The curve is evaluated at parametric points with the formula:
        ```math
            S_d(u, v) = \sum_{i=0}^{n_u} \sum_{j=0}^{n_v}
                        N_{i,p_u}(u) M_{j,p_v}(v) P_{i,j,d},
        ```
        where:
            - d is the coordinate index (0, 1, or 2)
            - $P_{i,j,d}$ is the d-th coordinate of the control point P_{i,j}
            - $N_{i,p_u}(u)$ is the i-th B-spline basis function in the u direction
            - $M_{j,p_v}(v)$ is the j-th B-spline basis function in the v direction

        There are a total of `(degree_u + 1) * (degree_v + 1)` non-zero basis
        functions at a given point (u, v).

        :returns: A tensor of shape (n_points, 3) representing the 3D points
                  corresponding to the given `parametric_coords`.

        :raises ValueError: If `parametric_coords` is not a 2D tensor (n_points, 2), or
            if the values in `parametric_coords` are outside the knot vector range.
        """
        return self._spline.interpolate(
            parametric_coords=parametric_coords, values=self._control_points
        )

    @property
    def interpolating_spline(self) -> BivariateBSpline:
        """
        Returns an instance of the underling `BivariateBSpline` that defines the
        abstract B-spline basis functions used to evaluate the surface.
        """
        return self._spline

    def evaluate_first_derivatives(
        self, parametric_coords: "mi.TensorXf"
    ) -> tuple["mi.TensorXf", "mi.TensorXf"]:
        r"""
        Compute the partial derivatives of the B-spline surface at the given
        `parametric_coordinates`.

        `parametric_coords` is a 2D tensor of shape (n_points, 2) where n_points
        is the number of points to evaluate, and the second dimension is 2 because
        a surface has two parametric directions (u, v).

        The derivatives are computed by taking a derivative of the formula used in
        evaluation, and they follow the same logic as the implementation from
        `BSplineCurve.evaluate_first_derivative`. The formula for the u derivative is:
        ```math
            d_u C_d(u, v) = \sum_{i=0}^{n_u} \sum_{j=0}^{n_v}
                            M_{j,p_v}(v) N_{i,p_u}(u)' P_{i,j,d}
        ```
        where:
            - d is the coordinate index (0, 1, or 2)
            - $P_{i,j,d}$ is the d-th coordinate of the control point P_{i,j}
            - $N_{i,p_u}(u)$ is the i-th B-spline basis function in the u direction
            - $M_{j,p_v}(v)$ is the j-th B-spline basis function in the v direction

        The algorithm proceeds as in `evaluate_coordinates`; the u component uses
        the derivative of the basis function in the u direction, while the v component
        uses the regular basis functions in the v direction. The weighted difference
        of control points yields the final derivative. Analogous procedure is used for
        the v derivative, but the roles of u and v are swapped.

        :returns: A tuple of tensors (du, dv) where:
                 - du: derivative with respect to u parameter, shape (n_points, 3)
                 - dv: derivative with respect to v parameter, shape (n_points, 3)

        :raises ValueError: If `parametric_coords` is not a 2D tensor (n_points, 2), or
            if the values in `parametric_coords` are outside the knot vector range.
        """
        self._spline.validate_parametric_coords(parametric_coords)

        u_span_data = self._spline.u_spline.compute_span_data(parametric_coords[:, 0])
        v_span_data = self._spline.v_spline.compute_span_data(parametric_coords[:, 1])

        # Get basis functions and their derivatives
        u_basis_func_ders = self._spline.u_spline.evaluate_basis_function_derivatives(
            parametric_span_data=u_span_data, derivative_order=1
        )
        v_basis_funcs_ders = self._spline.v_spline.evaluate_basis_function_derivatives(
            parametric_span_data=v_span_data, derivative_order=1
        )

        # Initialize result tensors for derivatives
        du = dr.zeros(dtype=mi.TensorXf, shape=(parametric_coords.shape[0], 3))
        dv = dr.zeros(dtype=mi.TensorXf, shape=(parametric_coords.shape[0], 3))

        for j, basis_index_v in enumerate(v_span_data.basis_function_indices):
            for i, basis_index_u in enumerate(u_span_data.basis_function_indices):
                cp_index = dr.ravel(basis_index_v * self._num_cps_u + basis_index_u)

                du += (u_basis_func_ders[:, 1, i] * v_basis_funcs_ders[:, 0, j])[
                    :, None
                ] * self._control_points[cp_index]

                dv += (u_basis_func_ders[:, 0, i] * v_basis_funcs_ders[:, 1, j])[
                    :, None
                ] * self._control_points[cp_index]

        return du, dv

    def evaluate_jacobian(self, parametric_coords: "mi.TensorXf") -> "mi.TensorXf":
        """
        Compute the Jacobian of the B-spline surface at the given `parametric_coords`.

        `parametric_coords` is a 2D tensor of shape (n_points, 2) where n_points
        is the number of points to evaluate, and the second dimension is 2 because
        a surface has two parametric directions (u, v).

        The Jacobian is a tensor that describes how the surface changes with respect
        to the parametric coordinates. It is useful for computing derivatives and
        understanding the local geometry of the surface.

        The Jacobian is computed as a 3D tensor of shape (n_points, 3, 2) where n_points
        is the number of points to evaluate, and the second dimension is 3 because
        a surface has three spatial dimensions (x, y, z), and the third dimension is 2
        because a surface has two parametric directions (u, v).

        :returns: A tensor of shape (n_points, 3, 2) representing the Jacobian at
                  the specified parametric coordinates.

        :raises ValueError: If `parametric_coords` is not a 2D tensor (n_points, 2), or
            if the values in `parametric_coords` are outside the knot vector range.
        """
        du, dv = self.evaluate_first_derivatives(parametric_coords=parametric_coords)
        jacobian = dr.empty(dtype=mi.TensorXf, shape=(parametric_coords.shape[0], 3, 2))
        jacobian[:, :, 0] = dr.ravel(du)
        jacobian[:, :, 1] = dr.ravel(dv)

        return jacobian

    def evaluate_second_derivatives(
        self, parametric_coords: "mi.TensorXf"
    ) -> tuple["mi.TensorXf", "mi.TensorXf", "mi.TensorXf"]:
        r"""
        Compute the second derivatives of the B-spline surface at the given
        `parametric_coords`.

        `parametric_coords` is a 2D tensor of shape (n_points, 2) where n_points
        is the number of points to evaluate, and the second dimension is 2 because
        a surface has two parametric directions (u, v).

        The algorithm uses the exact derivatives of the BSplines. The formulas for the
        uu, uv, vv derivatives are:
        ```math
            d_{uu} C_d(u, v) = \sum_{i=0}^{n_u} \sum_{j=0}^{n_v}
                            M_{j,p_v}(v) N_{i,p_u}(u)'' P_{i,j,d} \\
            d_{uv} C_d(u, v) = \sum_{i=0}^{n_u} \sum_{j=0}^{n_v}
                            M_{j,p_v}(v)' N_{i,p_u}(u)' P_{i,j,d} \\
            d_{vv} C_d(u, v) = \sum_{i=0}^{n_u} \sum_{j=0}^{n_v}
                            M_{j,p_v}(v)' N_{i,p_u}(u)' P_{i,j,d}
        ```
        where:
            - d is the coordinate index (0, 1, or 2)
            - $P_{i,j,d}$ is the d-th coordinate of the control point $P_{i,j}$
            - $N_{i,p_u}(u)$ is the i-th B-spline basis function in the u direction
            - $M_{j,p_v}(v)$ is the j-th B-spline basis function in the v direction

        :returns: A tuple of tensors (duu, duv, dvv) where:
                 - duu: the second derivative d()/du^2, shape (n_points, 3)
                 - duv: the second derivative d()/dudv, shape (n_points, 3)
                 - dvv: the second derivative d()/dv^2, shape (n_points, 3)

        :raises ValueError: If `parametric_coords` is not a 2D tensor (n_points, 2), or
            if the values in `parametric_coords` are outside the knot vector range.
        """
        self._spline.validate_parametric_coords(parametric_coords)

        u_span_data = self._spline.u_spline.compute_span_data(parametric_coords[:, 0])
        v_span_data = self._spline.v_spline.compute_span_data(parametric_coords[:, 1])

        u_basis_func_ders = self._spline.u_spline.evaluate_basis_function_derivatives(
            parametric_span_data=u_span_data, derivative_order=2
        )
        v_basis_func_ders = self._spline.v_spline.evaluate_basis_function_derivatives(
            parametric_span_data=v_span_data, derivative_order=2
        )

        duu = dr.zeros(dtype=mi.TensorXf, shape=(parametric_coords.shape[0], 3))
        duv = dr.zeros(dtype=mi.TensorXf, shape=(parametric_coords.shape[0], 3))
        dvv = dr.zeros(dtype=mi.TensorXf, shape=(parametric_coords.shape[0], 3))

        for j, basis_index_v in enumerate(v_span_data.basis_function_indices):
            for i, basis_index_u in enumerate(u_span_data.basis_function_indices):
                cp_index = dr.ravel(basis_index_v * self._num_cps_u + basis_index_u)

                duu += (u_basis_func_ders[:, 2, i] * v_basis_func_ders[:, 0, j])[
                    :, None
                ] * self._control_points[cp_index]

                duv += (u_basis_func_ders[:, 1, i] * v_basis_func_ders[:, 1, j])[
                    :, None
                ] * self._control_points[cp_index]

                dvv += (u_basis_func_ders[:, 0, i] * v_basis_func_ders[:, 2, j])[
                    :, None
                ] * self._control_points[cp_index]

        return duu, duv, dvv

    def evaluate_hessian(self, parametric_coords: "mi.TensorXf") -> "mi.TensorXf":
        r"""
        Compute the Hessian of the B-spline surface at the given `parametric_coords`.

        `parametric_coords` is a 2D tensor of shape (n_points, 2) where n_points
        is the number of points to evaluate, and the second dimension is 2 because
        a surface has two parametric directions (u, v).

        The Hessian is a tensor that describes the local curvature of a function of
        many variables. It is useful for understanding the local geometry of the
        surface.

        The Hessian is computed as a 4D tensor of shape (n_points, 3, 2, 2)
        where n_points is the number of points to evaluate, and the second dimension is
        3 because a surface has three spatial dimensions (x, y, z), and the third and
        forth dimensions are 2 because a surface has two parametric directions (u, v).

        :returns: A tensor of shape (n_points, 3, 2, 2) representing the Hessian at
          the specified parametric coordinates. The tensor element at
          [point_idx, coord_idx] is a 2x2 matrix of second derivatives:
          ```math
            \begin{bmatrix}
            \frac{\partial^2(\dots)}{\partial u^2} & \frac{\partial^2(\dots)}
            {\partial u \partial v} \\
            \frac{\partial^2(\dots)}{\partial u \partial v} & \frac{\partial^2(\dots)}
            {\partial u^2}
            \end{bmatrix}
          ```

          where derivatives are taken with respect to the coordinate specified
          by coord_idx (0, 1, 2 for x, y, and z, respectively), at the parametric
          coordinate indexed by point_idx.

        :raises ValueError: If `parametric_coords` is not a 2D tensor (n_points, 2), or
            if the values in `parametric_coords` are outside the knot vector range.
        """
        duu, duv, dvv = self.evaluate_second_derivatives(parametric_coords)

        hessian = dr.empty(
            dtype=mi.TensorXf, shape=(parametric_coords.shape[0], 3, 2, 2)
        )
        hessian[:, :, 0, 0] = dr.ravel(duu)
        hessian[:, :, 0, 1] = dr.ravel(duv)
        hessian[:, :, 1, 0] = dr.ravel(duv)
        hessian[:, :, 1, 1] = dr.ravel(dvv)

        return hessian

    def evaluate_normals(self, parametric_coords: "mi.TensorXf") -> "mi.TensorXf":
        """
        Compute the surface normals at the given `parametric_coords`.

        `parametric_coords` is a 2D tensor of shape (n_points, 2) where n_points
        is the number of points to evaluate, and the second dimension is 2 because
        a surface has two parametric directions (u, v).

        The normal is calculated as the cross-product of the partial derivatives
        with respect to u and v parameters. If the magnitude of the resulting product
        is too small, normalization is not performed to avoid NaNs.

        :returns: A tensor of shape (n_points, 3) representing the surface normals
                  at the given parametric coordinates.

        :raises ValueError: If `parametric_coords` is not a 2D tensor (n_points, 2), or
            if the values in `parametric_coords` are outside the knot vector range.
        """
        du, dv = self.evaluate_first_derivatives(parametric_coords=parametric_coords)
        n_points = du.shape[0]

        normals = dr.empty(dtype=mi.TensorXf, shape=(n_points, 3))
        normals[:, 0] = dr.ravel(du[:, 1] * dv[:, 2] - du[:, 2] * dv[:, 1])
        normals[:, 1] = dr.ravel(du[:, 2] * dv[:, 0] - du[:, 0] * dv[:, 2])
        normals[:, 2] = dr.ravel(du[:, 0] * dv[:, 1] - du[:, 1] * dv[:, 0])

        magnitudes = dr.safe_sqrt(dr.sum(normals * normals, axis=1))[:, None]

        safe_magnitudes = dr.select(
            magnitudes > BSpline.ZERO_TOL,
            magnitudes,
            dr.full(dtype=mi.TensorXf, value=1.0, shape=(n_points, 1)),
        )

        return normals / safe_magnitudes

    @property
    def control_points_grid(self) -> "mi.TensorXf":
        """
        Returns the control points as a grid of shape (num_cps_v, num_cps_u, 3).
        """
        return dr.reshape(
            dtype=mi.TensorXf,
            value=self._control_points,
            shape=(self._num_cps_v, self._num_cps_u, 3),
        )

    def insert_knot_u(self, knot_position: float) -> None:
        """
        Insert a knot into the u direction BSpline of the underlying bivariate
        B-spline at the specified `knot_position`.

        Calls the `BivariateBSpline.insert_knot_and_update_values_u` method to insert
        the knot and update the control points accordingly.

        :raises ValueError: If `knot_position` is outside the range of the knot vector,
            if the multiplicity of internal knots is greater than `degree`, or if the
            B-spline does not have an open knot vector.
        """
        self._control_points = self._spline.insert_knot_and_update_values_u(
            knot_position=knot_position, values_list=[self._control_points]
        )[0]
        self._quads = self._build_quads()

    def insert_knot_v(self, knot_position: float) -> None:
        """
        Insert a knot into the v direction BSpline of the underlying bivariate
        B-spline at the specified `knot_position`.

        Calls the `BivariateBSpline.insert_knot_and_update_values_v` method to insert
        the knot and update the control points accordingly.

        :raises ValueError: If `knot_position` is outside the range of the knot vector,
            if the multiplicity of internal knots is greater than `degree`, or if the
            B-spline does not have an open knot vector.
        """
        self._control_points = self._spline.insert_knot_and_update_values_v(
            knot_position=knot_position, values_list=[self._control_points]
        )[0]
        self._quads = self._build_quads()

    def h_refinement(self, factor: int = 2) -> None:
        """
        Refine the B-spline patch in both u and v directions by a factor of `factor`.

        This method inserts knots uniformly across the u and v directions, effectively
        increasing the number of control points and basis functions in both directions.

        Calls `BivariateBSpline.h_refinement` method to perform the refinement.

        :raises ValueError: If factor is not a positive integer, or if the
            underlying interpolating spline does not have an open knot vector.
        """
        self._control_points = self._spline.h_refinement(
            factor=factor, values_list=[self._control_points]
        )[0]
        self._quads = self._build_quads()

    @property
    def _num_cps_u(self):
        """
        If control points were converted into a grid, this property returns the
        number of control points in the u direction (i.e., the number of columns in
        the control point grid).
        """
        return self._spline.u_spline.num_basis_functions

    @property
    def _num_cps_v(self):
        """
        If control points were converted into a grid, this property returns the
        number of control points in the v direction (i.e., the number of rows in
        the control point grid).
        """
        return self._spline.v_spline.num_basis_functions

    @property
    def parametric_domain_size(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Size of the parametric domain for the B-spline patch.

        The B-spline surface is a tensor product of two B-spline curves, so the
        parametric domain size is defined by the knot vector bounds of both
        `u_spline` and `v_spline`.

        :returns: Tuple of tuples representing the parametric domain size:
                  ((u_min, u_max), (v_min, v_max)).
        """
        return (
            (
                float(self._spline.u_spline.knots[0]),
                float(self._spline.u_spline.knots[-1]),
            ),
            (
                float(self._spline.v_spline.knots[0]),
                float(self._spline.v_spline.knots[-1]),
            ),
        )

    @property
    def quads(self) -> "mi.TensorXu":
        """
        Returns the indices of the quads that make up the B-spline patch.

        The quads are defined by the control points in a grid-like structure.
        Each quad is defined by four control points, and the indices are
        calculated based on the number of control points in each direction.

        :returns: A Mitsuba array of quad indices shaped (n_quads, 4)
        """
        return self._quads

    def get_node_indices_for(self, side: Patch.Side) -> "mi.ArrayXu":
        """
        Returns the indices of the control points that belong to the specified `side`
        of the B-spline surface patch.

        For a visual representation of how the control points are selected, check the
        `_build_quads` method.

        :returns: A Mitsuba array of control point indices that belong to the
                  specified side.
        """
        if side == Patch.Side.LEFT:
            return dr.ravel(
                self._num_cps_u * dr.arange(dtype=mi.TensorXu, size=self._num_cps_v)
            )
        if side == Patch.Side.RIGHT:
            return dr.ravel(
                self._num_cps_u * dr.arange(dtype=mi.TensorXu, size=self._num_cps_v)
                + (self._num_cps_u - 1)
            )
        if side == Patch.Side.TOP:
            return dr.ravel(
                dr.arange(dtype=mi.TensorXu, size=self._num_cps_u)
                + self._num_cps_u * (self._num_cps_v - 1)
            )
        if side == Patch.Side.BOTTOM:
            return dr.ravel(dr.arange(dtype=mi.TensorXu, size=self._num_cps_u))

    @property
    def nodes(self) -> "mi.TensorXf":
        """
        When dealing with B-spline patches, the nodes are the control points.
        """
        return self._control_points

    @property
    def name(self) -> str:
        """
        Name of the B-spline surface patch.
        """
        return self._name

    @classmethod
    def unit_square(
        cls,
        patch_name: str,
        degree: int = 3,
        u_subdivisions: int = 1,
        v_subdivisions: int = 1,
    ) -> "BSplinePatch":
        """
        Creates a new spline patch with the shape of the unit square in the XY plane
        with the specified `degree` and the number of subdivisions in the u and v
        directions (`u_subdivisions` and `v_subdivisions`).

        The unit square is defined by the four corner vertices:

                            y ^
                              |
            (-0.5, 0.5, 0) ---|--- (0.5, 0.5, 0)
                      |       |       |
                      |       |       |
                ----------------------------> x
                      |       |       |
                      |       |       |
            (-0.5,-0.5, 0) ---|--- (0.5,-0.5, 0)
                              |

        The unit square patch has `u_subdivisions + degree` x `v_subdivisions + degree`
        control points, where `degree` is the degree of the B-spline basis functions
        used to define the patch and `u_subdivisions` and `v_subdivisions` are the
        number of knot spans in the u and v directions, respectively. The knots
        form an evenly spaced grid in the XY plane.

        The control point positions are should not be uniformly spaced as this does not
        represent well their influence on the surface. The correct distribution is
        obtained by starting with a Bezier patch, where the optimal arrangement is
        indeed a grid, and then inserting knots to create the distribution of points
        that preserves the uniform control point influence of the Bezier patch.

        :returns: A new instance of `BSplinePatch` representing the unit square patch.
        :raises ValueError: If `degree` is less than 1, or if `u_subdivisions` or
            `v_subdivisions` is less than 1.
        """
        if degree < 1:
            raise ValueError("`degree` must be at least 1.")
        if u_subdivisions < 1 or v_subdivisions < 1:
            raise ValueError(
                "`u_subdivisions` and `v_subdivisions` must be at least 1."
            )

        start_knots = [0.0 for _ in range(degree + 1)]
        end_knots = [1.0 for _ in range(degree + 1)]

        u_spline = BSpline(knot_vector=start_knots + end_knots, degree=degree)
        v_spline = BSpline(knot_vector=start_knots + end_knots, degree=degree)

        x_coords = np.linspace(-0.5, 0.5, degree + 1)
        y_coords = np.linspace(-0.5, 0.5, degree + 1)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing="xy")

        control_points_np = np.stack(
            arrays=[x_grid.ravel(), y_grid.ravel(), np.zeros_like(x_grid.ravel())],
            axis=1,
        )

        patch = cls(
            name=patch_name,
            u_spline=u_spline,
            v_spline=v_spline,
            control_points=mi.TensorXf(control_points_np),
        )

        for i in range(u_subdivisions - 1):
            patch.insert_knot_u(knot_position=(i + 1) / u_subdivisions)

        for j in range(v_subdivisions - 1):
            patch.insert_knot_v(knot_position=(j + 1) / v_subdivisions)

        return patch

    def visualize(
        self,
        num_points_u: int = 25,
        num_points_v: int = 25,
        num_normals_u: int = 4,
        num_normals_v: int = 4,
    ) -> None:
        """
        Visualize the B-spline surface in 3D using matplotlib.

        `num_points_u` and`num_points_v` define the number of points to evaluate in the
        u and v directions, respectively, forming a grid of points. Similarly,
        `num_normals_u` and `num_normals_v` define the number of points to evaluate
        for the normal and tangent vectors.

        Correctness of splines cannot be easily verified by hand, so a visualization
        is provided to help with debugging and create a visual representation of the
        surface.

        :returns: Tuple of matplotlib plot objects (figure, axes)
        :raises ValueError: If `num_points_u` or `num_points_v` is less than 2, or if
            `num_normals_u` or `num_normals_v` is negative.
        """
        if num_points_u < 2 or num_points_v < 2:
            raise ValueError(
                "`num_points_u` and `num_points_v` must be greater than 1."
            )
        if num_normals_u < 0 or num_normals_v < 0:
            raise ValueError(
                "`num_normals_u` and `num_normals_v` must be non-negative."
            )

        (u_min, u_max), (v_min, v_max) = self.parametric_domain_size

        u_params = dr.linspace(
            dtype=mi.TensorXf, start=u_min, stop=u_max, num=num_points_u + 1
        )
        v_params = dr.linspace(
            dtype=mi.TensorXf, start=v_min, stop=v_max, num=num_points_v + 1
        )

        u_grid, v_grid = dr.meshgrid(
            dr.ravel(u_params), dr.ravel(v_params), indexing="xy"
        )

        u_flat = dr.ravel(u_grid)
        v_flat = dr.ravel(v_grid)

        uv_params = dr.zeros(dtype=mi.TensorXf, shape=(len(u_flat), 2))
        uv_params[:, 0] = u_flat
        uv_params[:, 1] = v_flat

        # Evaluate the surface at the grid points
        points = self.evaluate_coordinates(uv_params)
        points_np = points.numpy().reshape(num_points_v + 1, num_points_u + 1, 3)
        control_points_np = self._control_points.numpy()

        # Create figure and axis
        fig = plt.figure(figsize=(10, 8), dpi=150)
        ax = fig.add_subplot(111, projection="3d")

        # Plot the surface
        surf = ax.plot_surface(
            points_np[:, :, 0],
            points_np[:, :, 1],
            points_np[:, :, 2],
            cmap="viridis",
            alpha=0.7,
            linewidth=0,
            antialiased=True,
            label="B-Spline Surface",
        )

        # Plot control points
        ax.scatter(
            control_points_np[:, 0],
            control_points_np[:, 1],
            control_points_np[:, 2],
            color="r",
            s=20,
            label="Control Points",
        )

        # Plot control net (reshaping control points to grid)
        cp_grid = control_points_np.reshape(self._num_cps_v, self._num_cps_u, 3)

        # Plot u-lines and v-lines
        for i in range(self._num_cps_v):
            ax.plot(
                cp_grid[i, :, 0],
                cp_grid[i, :, 1],
                cp_grid[i, :, 2],
                "r-",
                linewidth=0.5,
                alpha=0.5,
            )

        for j in range(self._num_cps_u):
            ax.plot(
                cp_grid[:, j, 0],
                cp_grid[:, j, 1],
                cp_grid[:, j, 2],
                "r-",
                linewidth=0.5,
                alpha=0.5,
            )

        # Display normal vectors if requested
        if num_normals_u > 0 and num_normals_v > 0:
            # Compute parameters for normal points (evenly spaced)
            u_normal_params = dr.linspace(
                dtype=mi.TensorXf, start=u_min, stop=u_max, num=num_normals_u + 1
            )
            v_normal_params = dr.linspace(
                dtype=mi.TensorXf, start=v_min, stop=v_max, num=num_normals_v + 1
            )
            u_normal_grid, v_normal_grid = dr.meshgrid(
                dr.ravel(u_normal_params), dr.ravel(v_normal_params), indexing="xy"
            )
            u_normal_flat = dr.ravel(u_normal_grid)
            v_normal_flat = dr.ravel(v_normal_grid)

            uv_normal_params = dr.zeros(
                dtype=mi.TensorXf, shape=(len(u_normal_flat), 2)
            )
            uv_normal_params[:, 0] = u_normal_flat
            uv_normal_params[:, 1] = v_normal_flat

            # Evaluate points and normals at these parameters
            normal_points = self.evaluate_coordinates(uv_normal_params).numpy()
            normal_vectors = self.evaluate_normals(uv_normal_params).numpy()

            # Get tangent vectors (partial derivatives)
            du_vectors, dv_vectors = self.evaluate_first_derivatives(uv_normal_params)
            du_vectors = du_vectors.numpy()
            dv_vectors = dv_vectors.numpy()

            du_norm = np.sqrt(np.sum(du_vectors**2, axis=1)).reshape(-1, 1)
            dv_norm = np.sqrt(np.sum(dv_vectors**2, axis=1)).reshape(-1, 1)

            du_vectors_normalized = du_vectors / du_norm
            dv_vectors_normalized = dv_vectors / dv_norm

            vector_scale = 0.2

            def plot_vector(
                ax: plt.Axes,
                start: npt.NDArray[np.float32],
                end: npt.NDArray[np.float32],
                color: str,
            ) -> None:
                """
                Small utility function that draws a vector in the current `ax` instance.

                The vector goes from `start` to `end` and is painted with `color`.
                """
                ax.quiver(
                    start[0],
                    start[1],
                    start[2],
                    end[0] - start[0],
                    end[1] - start[1],
                    end[2] - start[2],
                    color=color,
                    arrow_length_ratio=0.2,
                )

            # Plot the normal vectors
            for i in range(len(normal_points)):
                start_point = normal_points[i]

                # Normal vector (green)
                end_point = start_point + vector_scale * normal_vectors[i]
                plot_vector(ax, start_point, end_point, "g")

                # U tangent vector (blue)
                end_point_u = start_point + vector_scale * du_vectors_normalized[i]
                plot_vector(ax, start_point, end_point_u, "b")

                # V tangent vector (red)
                end_point_v = start_point + vector_scale * dv_vectors_normalized[i]
                plot_vector(ax, start_point, end_point_v, "r")

        # Add a color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        x_min, x_max = points_np[:, :, 0].min(), points_np[:, :, 0].max()
        y_min, y_max = points_np[:, :, 1].min(), points_np[:, :, 1].max()
        z_min, z_max = points_np[:, :, 2].min(), points_np[:, :, 2].max()

        padding = max(0.2, 0.1 * max(x_max - x_min, y_max - y_min, z_max - z_min))

        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_zlim(z_min - padding, z_max + padding)

        # Set labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(
            f"B-Spline Surface (u_degree: {self._spline.u_spline.degree}, "
            f"v_degree: {self._spline.v_spline.degree}, "
            f"control points: {self.control_points.shape[0]})"
        )

        # Add a legend (need a proxy for the surface)
        proxy = plt.Rectangle((0, 0), 1, 1)
        handles = [
            proxy,
            plt.Line2D([0], [0], linestyle="-", color="r", alpha=0.5),
            plt.Line2D([0], [0], marker="o", color="r", linestyle="None"),
        ]
        labels = ["B-Spline Surface", "Control Net", "Control Points"]
        ax.legend(handles, labels, loc="upper right")

        plt.tight_layout()
        plt.show()
