import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi

from rebel_shape.splines.bspline import BSpline
from rebel_shape.splines.coordinate_spline import CoordinateSpline


class BSplineCurve(CoordinateSpline):
    r"""
    Mitsuba compatible explicit B-Spline curve in 3D space. A B-Spline curve is
    defined by a univariate B-Spline `spline` and `control_points`.

    The abstract B-Spline is used to provide the basis functions to interpolate
    the 3D coordinates of the curve. For more details about the B-Spline
    itself, see the `BSpline` class.

    While the `knot_vector` describes the curve in an "abstract" space, the curve is
    defined in 3D space by `control_points`. Control points are typically not
    located on the curve and constitute a polygonal control network. A curve meets
    them only if that control point is associated with a knot of multiplicity
    equal to the degree of the curve. Since open knot vectors are typically used,
    the first and last control points are the start and the end points of the curve.

    Each control point is assigned to a basis function, and the curve is defined by
    the linear combination of these basis functions. Thus, the number of control points
    has to be equal to the number of basis functions, which depends on the knot
    vector and the degree of the spline.

    A general B-Spline curve is defined as a function of one parametric variable (u):
    ```math
        C_d(u) = \sum_{i=0}^{n_u} N_{i,p}(u) P_{i,d}
    ```
    where:
        - C(u) is the curve point (3D value) at (u)
        - $N_{i,p} (u)$ are the B-Spline basis functions of order p
        - $P_{i,d}$ are the control points (d is the coordinate, x, y, or z)
    In general most of the basis functions are zero, and only `degree + 1` from an
    appropriate knot span are non-zero.

    For more details, see "The NURBS book" by Piegl and Tiller, and "Differential
    Rendering of Parametric Geometry" (2023) by Worchel and Alexa. Another resource:
    https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-derv.html
    contains a high level overview of the formulas.
    """

    def __init__(self, spline: BSpline, control_points: "mi.TensorXf") -> None:
        """
        Initialize a B-Spline curve with the given `spline`, and `control_points`.

        The number of provided control points must be compatible with the knot vector
        from the provided `spline`: number of basis functions in `spline` must be equal
        to the number of control points.

        :raises ValueError: If `control_points` is not a 2D tensor, if `control_points`
            does not have the correct shape, if the number of control points does not
            match the number of control points derived from the `knot_vector`.
        """
        self._spline = spline

        super().__init__(control_points=control_points)

    def evaluate_coordinates(self, parametric_coords: "mi.TensorXf") -> "mi.TensorXf":
        r"""
        Compute the B-spline curve coordinates at the given parametric coordinates.

        `parametric_coords` is a 2D tensor of shape (n_points, 1) where n_points
        is the number of points to evaluate, and the second dimension is 1 because
        a curve has a single parametric direction (u).

        The interpolated point on the curve is computed using the B-spline basis
        functions weighted by the control points:
        ```math
            C_d(u) = \sum_{i=0}^{n_u} N_{i,p}(u) P_{i,d}
        ```
        where d is the coordinate (x, y, or z), and n is the number of control points.

        Most B-Spline basis functions N_{i,p}(u) are zero, and only `degree + 1` from
        an appropriate knot span are non-zero. They are first extracted from the knot
        span, then the Cox-de Boor algorithm is used to compute the value at the
        specified parametric coordinate, and finally the position is computed by
        weighting the control points with the basis functions.

        :returns: A tensor of shape (n_points, 3) representing the 3D points
                  corresponding to the given `parametric_coords`.

        :raises ValueError: If `parametric_coords` is not a 2D tensor (n_points, 1), or
            if the values in `parametric_coords` are outside the knot vector range.
        """
        return self._spline.interpolate(
            values=self._control_points, parametric_coords=parametric_coords
        )

    def evaluate_first_derivatives(
        self, parametric_coords: "mi.TensorXf"
    ) -> "mi.TensorXf":
        r"""
        Compute the derivative of the B-spline curve at the given `parametric_coords`.

        `parametric_coords` is a 2D tensor of shape (n_points, 1) where n_points
        is the number of points to evaluate, and the second dimension is 1 because
        a curve has a single parametric direction (u).

        The derivative of the B-spline curve can be computed in one of two ways:
        either by evaluating the derivative of the B-spline basis functions of a lower
        order and then weighting them by control point differences (as presented in
        https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-derv.html
        ), or by directly computing the derivatives of the B-spline basis functions.

        Since the latter is much simpler, without reducing performance (at least not
        noticeably), we use the latter approach here. Thus, the procedure is the same as
        for `evaluate_coordinates`.

        The interpolated derivative vector on the curve is computed using the
        first derivatives of B-spline basis functions weighted by the control points:
        ```math
            C_d(u)' = \sum_{i=0}^{n_u} N_{i,p}(u)' P_{i,d}
        ```
        where d is the coordinate (x, y, or z), and n is the number of control points.

        :returns: A tensor of shape (n_points, 3) representing the derivative vectors
                  at the specified `parametric_coords`.

        :raises ValueError: If `parametric_coords` is not a 2D tensor (n_points, 1), or
            if the values in `parametric_coords` are outside the knot vector range.
        """
        self._spline.validate_parametric_coords(parametric_coords)

        n_points = parametric_coords.shape[0]
        span_data = self._spline.compute_span_data(
            parametric_coords=parametric_coords[:, 0]
        )
        basis_funcs = self._spline.evaluate_basis_function_derivatives(
            parametric_span_data=span_data, derivative_order=1
        )[:, 1, :]

        # Compute point on curve
        result = dr.zeros(dtype=mi.TensorXf, shape=(n_points, 3))
        values = self._control_points

        for i, basis_func_index in enumerate(span_data.basis_function_indices):
            result += basis_funcs[:, i][:, None] * values[basis_func_index]

        return result

    def evaluate_second_derivatives(
        self, parametric_coords: "mi.TensorXf"
    ) -> "mi.TensorXf":
        r"""
        Compute the second derivative of the B-spline curve at the given
        `parametric_coords`.

        `parametric_coords` is a 2D tensor of shape (n_points, 1) where n_points
        is the number of points to evaluate, and the second dimension is 1 because
        a curve has a single parametric direction (u).

        The interpolated derivative vector on the curve is computed using the
        first derivatives of B-spline basis functions weighted by the control points:
        ```math
            C_d(u)'' = \sum_{i=0}^{n_u} N_{i,p}(u)'' P_{i,d}
        ```
        where d is the coordinate (x, y, or z), and n is the number of control points.

        :returns: A tensor of shape (n_points, 3) representing the second derivative
                  vectors at the specified `parametric_coords`.

        :raises ValueError: If `parametric_coords` is not a 2D tensor (n_points, 1), or
            if the values in `parametric_coords` are outside the knot vector range.
        """
        self._spline.validate_parametric_coords(parametric_coords)

        n_points = parametric_coords.shape[0]
        span_data = self._spline.compute_span_data(
            parametric_coords=parametric_coords[:, 0]
        )
        basis_function_derivatives = self._spline.evaluate_basis_function_derivatives(
            parametric_span_data=span_data, derivative_order=2
        )[:, 2, :]

        # Compute point on curve
        result = dr.zeros(dtype=mi.TensorXf, shape=(n_points, 3))
        values = self._control_points

        for i, basis_func_index in enumerate(span_data.basis_function_indices):
            result += (
                basis_function_derivatives[:, i][:, None] * values[basis_func_index]
            )

        return result

    def evaluate_tangents(self, parametric_coords: "mi.TensorXf") -> "mi.TensorXf":
        """
        Compute the unit tangent vectors of the B-spline curve at the given
        `parametric_coords`.

        The tangent vector is the normalized first derivative of the curve.

        :returns: A tensor of shape (n_points, 3) representing the unit tangent vectors
                  at the specified `parametric_coords`.

        :raises ValueError: If `parametric_coords` is not a 2D tensor (n_points, 1), or
            if the values in `parametric_coords` are outside the knot vector range.
        """
        derivatives = self.evaluate_first_derivatives(
            parametric_coords=parametric_coords
        )
        n_points = derivatives.shape[0]

        magnitudes = dr.safe_sqrt(dr.sum(derivatives * derivatives, axis=1))[:, None]
        safe_magnitudes = dr.select(
            magnitudes > BSpline.ZERO_TOL,
            magnitudes,
            dr.full(dtype=mi.TensorXf, value=1.0, shape=(n_points, 1)),
        )

        return derivatives / safe_magnitudes

    def insert_knot(self, knot: float) -> None:
        """
        Insert a knot into the underlying B-spline at the specified `knot_position`.

        Calls the `BSpline.insert_knot_and_update_values` method to insert the knot
        and update the control points accordingly.

        :raises ValueError: If `knot_position` is outside the range of the knot vector,
            if the multiplicity of internal knots is greater than `degree`, or if the
            B-spline does not have an open knot vector.
        """
        self._control_points = self._spline.insert_knot_and_update_values(
            knot_position=knot, values_list=[self._control_points]
        )[0]

    def h_refinement(self, factor: int = 2) -> None:
        """
        Refine the B-spline curve by inserting knots uniformly across the
        parametric direction (u).

        This method inserts knots uniformly across the u and v directions, effectively
        increasing the number of control points and basis functions in both directions.

        Calls `BSpline.h_refinement` method to perform the refinement.

        :raises ValueError: If factor is not a positive integer, or if the
            underlying interpolating spline does not have an open knot vector.
        """
        self._control_points = self._spline.h_refinement(
            factor=factor, values_list=[self._control_points]
        )[0]

    @property
    def interpolating_spline(self) -> BSpline:
        """
        An instance of the `InterpolatingSpline` that this coordinate spline uses
        for interpolation.

        This property is used to access the underlying interpolating spline that
        provides the interpolation functionality for the coordinate spline.

        :returns: An instance of `InterpolatingSpline`.
        """
        return self._spline

    @property
    def parametric_domain_size(self) -> tuple[float, float]:
        """
        Size of the parametric domain for the B-spline curve.

        The B-spline curve is defined in a 1D parametric space (u), so the size
        is represented as a tuple that contains the smallest and largest
        values of the knot vector.
        """
        return float(self._spline.knots[0]), float(self._spline.knots[-1])

    def visualize(self, num_points: int = 100, num_tangents: int = 10) -> None:
        """
        Visualize the B-spline curve in 3D using matplotlib.

        Correctness of splines cannot be easily verified by hand, so a visualization
        is provided to help with debugging and create a visual representation of the
        spline.

        :raises ValueError: If `num_points` is less than 2 or if
            `num_tangents` is negative.
        """
        if num_points < 2:
            raise ValueError("`num_points` must be greater than 1.")
        if num_tangents < 0:
            raise ValueError("`num_tangents` must be non-negative.")

        u_min, u_max = float(self._spline.knots[0]), float(self._spline.knots[-1])

        params = dr.linspace(
            dtype=mi.TensorXf, start=u_min, stop=u_max, num=num_points + 1
        )
        params = dr.reshape(dtype=mi.TensorXf, value=params, shape=(num_points + 1, 1))

        points = self.evaluate_coordinates(params)
        points_np = points.numpy()

        # Get control points for visualization
        control_points_np = self._control_points.numpy()

        # Create figure and axis
        fig = plt.figure(figsize=(8, 8), dpi=150)
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            points_np[:, 0],
            points_np[:, 1],
            points_np[:, 2],
            "b-",
            linewidth=2,
            label="B-Spline Curve",
        )

        ax.plot(
            control_points_np[:, 0],
            control_points_np[:, 1],
            control_points_np[:, 2],
            "r--",
            linewidth=1,
            label="Control Polygon",
        )
        ax.scatter(
            control_points_np[:, 0],
            control_points_np[:, 1],
            control_points_np[:, 2],
            color="r",
            s=50,
            label="Control Points",
        )

        # Display tangent vectors if requested
        if num_tangents > 0:
            # Compute parameters for tangent points (evenly spaced)
            tangent_params = dr.linspace(
                dtype=mi.TensorXf, start=u_min, stop=u_max, num=num_tangents + 1
            )
            tangent_params = dr.reshape(
                dtype=mi.TensorXf, value=tangent_params, shape=(num_tangents + 1, 1)
            )

            # Evaluate points and tangents at these parameters
            tangent_points = self.evaluate_coordinates(tangent_params).numpy()
            tangent_vectors = self.evaluate_tangents(tangent_params).numpy()

            # Scale the tangent vectors for visualization
            scale = 0.3

            # Plot the tangent vectors
            for i in range(num_tangents):
                start_point = tangent_points[i]
                end_point = start_point + scale * tangent_vectors[i]

                ax.quiver(
                    start_point[0],
                    start_point[1],
                    start_point[2],
                    end_point[0] - start_point[0],
                    end_point[1] - start_point[1],
                    end_point[2] - start_point[2],
                    color="g",
                    arrow_length_ratio=0.05,
                )

        # Set labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(
            f"B-Spline Curve (degree: {self.degree}, "
            f"control points: {self.control_points.shape[0]})"
        )
        ax.legend()

        plt.tight_layout()
        plt.show()
