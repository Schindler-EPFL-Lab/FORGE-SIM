import logging
import math
from collections.abc import Callable

import drjit as dr
import mitsuba as mi
import numpy as np
import numpy.typing as npt

from rebel_shape.splines.interpolating_spline import InterpolatingSpline
from rebel_shape.splines.parametric_span_data import ParametricSpanData


class BSpline(InterpolatingSpline):
    """
    Mitsuba compatible implementation of a B-Spline in abstract space.
    It is defined by the `degree`, and `knot_vector`.

    The degree of the spline determines the polynomial degree of the basis functions.
    The degree of a polynomial ax^2 + bx + c is 2, while the order of the polynomial
    is 3. The maximum smoothness of a B-Spline is equal to the polynomial degree minus
    one. E.g., a linear B-Spline (degree 1) is continuous, while a quadratic B-Spline
    (degree 2) is continuous and has continuous first derivatives.

    The `knot_vector` describes how parametric space is divided into intervals. Knots
    have a certain multiplicity, indicating how many times they are repeated in
    the knot vector. If a knot is repeated, the overall smoothness of the curve is
    reduced by one degree for each repetition. Intuitively, if the B-Spline is
    used to interpolate a curve, knots of higher multiplicity create a denser basis
    around the knot, forcing the curve to pass close to the appropriate control points.

    A continuous curve cannot have a knot of multiplicity greater than the degree of
    the curve. The exception are the first and last knots, which can be repeated to
    create an open knot vector. An open knot vector is used to create interpolating
    B-Spline curves, where the first and last control points are also the endpoints of
    the curve. The first and last knots are repeated `degree` + 1 times, and the
    edges of the parametric interval would correspond to the first and las control
    points.

    The number of control points depends on the `knot_vector`; if the `knot_vector` is
    open, the number of control points is equal to the number of unique knots in the
    `knot_vector` + the `degree` of the curve. On the other end, if each knot is
    repeated `degree` times, we get a series of Bézier curves, the number of control
    points being roughly `num_unique_knots * (degree + 1)`.

    The values at a particular parametric point are evaluated using the Cox-de Boor
    algorithm (a generalization of de Casteljau's algorithm for Bézier curves). For
    more details, see "The NURBS book" by Piegl and Tiller, and "Differential
    Rendering of Parametric Geometry" (2023) by Worchel and Alexa.
    """

    KNOT_TOL = 1.0e-12
    """Determines when two knots are considered equal (absolute tolerance float64)."""
    ZERO_TOL = 1.0e-6
    """
    Determines when a denominator is considered zero in the Cox-de Boor algorithm. 
    The lower accuracy reflects the use of float32 numbers in the algorithm.
    """
    PROJECTION_TOL = 1.192092896e-07 * 2
    """
    Tolerance for projection of points onto the spline curve. It is twice the
    machine epsilon for float32, which is the smallest difference between two
    representable numbers. Factor 2 is added for safety. This value should be as small
    as it can be to keep the projection stable without introducing large errors.
    """

    def __init__(self, knot_vector: list[float], degree: int) -> None:
        """
        Initialize a B-Spline curve with the given `knot_vector`, `degree`, and
        `control_points`.

        The number of provided control points must be compatible with the provided
        `knot_vector` and `degree`. The entries of the `knot_vector` must be in
        ascending order, and the first and last knots must have the same multiplicity.
        All other knots can have multiplicities up to the degree of the curve.

        It is not advised to use B-splines of degree greater than 9, as they result in
        excessive compilation time and slow down spline evaluation.

        :raises ValueError: If `degree` is not a positive number, if `knot_vector`
            has less than two entries, if the `knot_vector` elements are not in
            ascending order, if the multiplicity of internal knots is greater than
            `degree`, if the multiplicity of the first or last knot is greater than
            `degree + 1`, or if the first and last knots do not have the same
            multiplicity.
        """
        if degree < 1:
            raise ValueError("`degree` must be a positive integer.")
        if degree > 9:
            logging.warning(
                "High degree B-splines result in excessive compilation "
                "time and slow down evaluation. "
            )

        self._degree = degree
        self._knots = np.array(knot_vector, dtype=np.float64)

        self._num_elements, self._unique_knots, self._knot_multiplicities = (
            BSpline.process_knot_vector(knot_vector=knot_vector, degree=self._degree)
        )

    @staticmethod
    def process_knot_vector(
        knot_vector: list[float], degree: int
    ) -> tuple[int, npt.NDArray[np.float64], npt.NDArray[np.uint32]]:
        """
        Process the knot vector and return the number of elements, unique knots,
        and knot multiplicities.

        The `knot_vector` is checked for validity, ensuring that it has at least two
        entries, that the elements are in ascending order, and that the multiplicities
        of the knots are within the allowed limits based on the `degree` of the spline.

        :return: A tuple containing:
                 - `num_elements`: Number of unique knots
                 - `unique_knots`: Array of unique knots
                 - `knot_multiplicities`: Array of knot multiplicities

        :raises ValueError: If `knot_vector` has less than two entries,
            if the `knot_vector` elements are not in ascending order, if the
            multiplicity of internal knots is greater than `degree`, if the
            multiplicity of the first or last knot is greater than `degree + 1`,
            or if the first and last knots do not have the same multiplicity.
        """
        if len(knot_vector) < 2:
            raise ValueError(
                "The knot vector must contain at least two knots to define a B-spline."
            )
        for i in range(len(knot_vector) - 1):
            if knot_vector[i + 1] < knot_vector[i]:
                raise ValueError("Knot positions must be provided in ascending order!")

        num_elements = 0
        for i in range(1, len(knot_vector)):
            if not math.isclose(
                knot_vector[i],
                knot_vector[i - 1],
                abs_tol=BSpline.KNOT_TOL,
                rel_tol=0.0,
            ):
                num_elements += 1

        unique_knots = np.zeros(num_elements + 1, dtype=np.float64)
        knot_multiplicities = np.zeros(num_elements + 1, dtype=np.uint32)

        counter = -1
        last_knot: float | None = None
        for i in range(0, len(knot_vector)):
            if last_knot is None or (
                not math.isclose(
                    knot_vector[i],
                    last_knot,
                    abs_tol=BSpline.KNOT_TOL,
                    rel_tol=0.0,
                )
            ):
                counter += 1
                unique_knots[counter] = knot_vector[i]

            last_knot = float(knot_vector[i])
            knot_multiplicities[counter] += 1

        if knot_multiplicities[0] != knot_multiplicities[-1]:
            raise ValueError(
                "Knots on the edges of the interval must have the same multiplicity, "
                "i.e. they have to be repeated the same number of times"
            )

        for i, multiplicity in enumerate(knot_multiplicities):
            if i != 0 and i != (len(knot_multiplicities) - 1) and multiplicity > degree:
                raise ValueError(
                    f"Multiplicity of knot {unique_knots[i]} is {multiplicity}, "
                    f"which is greater than the degree of the curve ({degree}). "
                )
            elif (
                i == 0 or i == len(knot_multiplicities) - 1
            ) and multiplicity > degree + 1:
                raise ValueError(
                    f"Multiplicity of edge knot {unique_knots[i]} is {multiplicity}, "
                    f"which is greater than the degree of the curve + 1 ({degree + 1})."
                )

        return num_elements, unique_knots, knot_multiplicities

    def get_knot(self, index: int) -> float:
        """
        Get a knot with a possibly out-of-range `index`.

        If `index` is out of range, ghost knots are conjured by looking at the other
        end of the vector. This functionality is useful to avoid introducing a
        special-case for the first and last unique knots when evaluating the spline.

        Assumes that the first and last unique knots have the same multiplicity.

        :return: The knot at the specified index.
        """
        if index < 0:
            index += len(self._knots) - int(self._knot_multiplicities[-1])
            return float(self._knots[0] - (self._knots[-1] - self._knots[index]))

        if index >= len(self._knots):
            index -= len(self._knots) - int(self._knot_multiplicities[0])
            return float(self._knots[-1] + (self._knots[index] - self._knots[0]))

        return float(self._knots[index])

    def compute_extended_knot_vector(self, degree: int) -> npt.NDArray[np.float64]:
        """
        Compute the extended knot vector with "ghost knots".

        Ghost knots are used to evaluate the spline in the Cox-de Boor algorithm by
        avoiding special cases for the first and last unique knots. For flexibility,
        they can be computed for any `degree` lower than or equal to the degree of the
        spline.

        :return: A numpy array of ghost knots and knots. The length of the array is
                 `len(self._knots) + 2 * (degree + 1)`.
        """
        extended_knots: list[float] = []
        num_ghosts = degree + 1

        for i in range(-num_ghosts, len(self._knots) + num_ghosts):
            extended_knots += [self.get_knot(i)]

        return np.array(extended_knots)

    def compute_span_data(self, parametric_coords: "mi.TensorXf") -> ParametricSpanData:
        """
        Get the knot span (element) indices for all given parametric coordinates.

        The binary search algorithm is used to find the which knot span each
        parametric coordinate belongs to. `parametric_coords` here is a 1D Tensor!
        This is a necessary step in the Cox-de Boor algorithm and also for isogemetric
        analysis.

        :return: A `ParametricSpanData` object containing the provided parametric
            coordinates and computed knot spans.
        :raises ValueError: If `parametric_coords` is not a 1D tensor.
        """
        if len(parametric_coords.shape) != 1:
            raise ValueError("`parametric_coords` must be a 1D tensor!")

        n_knots = len(self._knots)
        n_points = parametric_coords.shape[0]
        knots = mi.TensorXf(self._knots)

        start = dr.full(dtype=mi.TensorXu, value=0, shape=n_points)
        end = dr.full(dtype=mi.TensorXu, value=n_knots - 1, shape=n_points)
        iterations = dr.log2i(n_knots - 1) + 1

        # Binary search to find the knot span
        for _ in range(iterations):
            middle = (start + end) >> 1

            condition = parametric_coords < knots[dr.ravel(middle)]
            start = dr.select(condition, start, dr.minimum(middle + 1, end))
            end = dr.select(condition, middle, end)

        knot_spans = dr.clip(
            value=start - 1,
            min=self._knot_multiplicities[0] - 1,
            max=n_knots - self._knot_multiplicities[0] - 1,
        )

        expected_control_points = len(self._knots) - self._knot_multiplicities[0]

        return ParametricSpanData(
            parametric_coords=dr.ravel(parametric_coords),
            knot_spans=dr.ravel(knot_spans),
            num_basis_functions=int(expected_control_points),
            spline_degree=self._degree,
        )

    def evaluate_basis_functions_internal(
        self,
        parametric_span_data: ParametricSpanData,
        degree: int | None = None,
    ) -> "mi.TensorXf":
        """
        Evaluate the B-Spline basis functions at the given parametric coordinates.

        This is the Cox-de Boor algorithm for B-Spline basis function evaluation.
        It is a recursive algorithm that starts with the zeroth degree basis functions
        and builds up to the desired degree. To avoid control logic at the edges
        of the knot vector, ghost knots are introduced, creating an extended knot
        vector. For reference, the implementation is based on Algorithm A2.2 from
        "The NURBS Book" (by Piegl and Tiller).

        The `parametric_span_data` contains the parametric coordinates and the knot
        spans for the given parametric coordinates. Each parametric coordinate is
        assigned `degree` + 1 basis function values evaluated at the corresponding knot
        span and parametric coordinate.

        The `degree` parameter is introduced to allow flexibility in the evaluation
        of the basis functions (e.g., a derivative requires one degree lower than the
        original spline). If `degree` is not provided, the degree of the spline is used.

        :returns: A tensor of shape (n_points, `degree` + 1) representing the basis
                  function values at the given parametric coordinates.

        :raises ValueError: If `degree` is not a non-negative integer, or if `degree`
            is greater than the degree of the spline.
        """
        if degree is None:
            degree = self._degree

        if degree < 0:
            raise ValueError("`degree` must be a positive integer.")
        if degree > self._degree:
            raise ValueError(
                f"`degree` must be less than or equal to the degree of the spline "
                f"({self._degree})."
            )

        coords = mi.TensorXf(parametric_span_data.parametric_coords)
        n_points = coords.shape[0]

        basis_funcs = dr.zeros(dtype=mi.TensorXf, shape=(n_points, degree + 1))
        left = dr.zeros(dtype=mi.TensorXf, shape=(n_points, degree + 1))
        right = dr.zeros(dtype=mi.TensorXf, shape=(n_points, degree + 1))

        extended_knot_spans = dr.ravel(parametric_span_data.knot_spans + degree + 1)
        extended_knots = mi.TensorXf(self.compute_extended_knot_vector(degree=degree))

        # Basis function values at degree 0 are just indicator functions
        basis_funcs[:, 0] = 1.0

        for j in range(1, degree + 1):
            left[:, j] = dr.ravel(coords - extended_knots[extended_knot_spans + 1 - j])
            right[:, j] = dr.ravel(coords - extended_knots[extended_knot_spans + j])

            saved = dr.zeros(dtype=mi.TensorXf, shape=n_points)

            for r in range(j):
                denominator = left[:, j - r] - right[:, r + 1]
                temp = dr.select(
                    denominator > BSpline.ZERO_TOL,
                    basis_funcs[:, r] / denominator,
                    0.0,
                )
                basis_funcs[:, r] = dr.ravel(saved - right[:, r + 1] * temp)
                saved = left[:, j - r] * temp

            basis_funcs[:, j] = dr.ravel(saved)

        return basis_funcs

    def evaluate_basis_function_derivatives(
        self,
        parametric_span_data: ParametricSpanData,
        derivative_order: int = 1,
    ) -> "mi.TensorXf":
        """
        Evaluate the B-spline basis functions and their derivatives up to order
        `derivative_order` at the given `parametric_span_data`.

        This method implements a generalized version of the Cox-de Boor algorithm
        that computes all derivatives of the B-spline basis functions at the given
        parametric coordinates.

        The algorithm was copied almost line for line from "The NURBS Book" - Algorithm
        A2.3. At a high level, it computes the basis function for all splines up to
        the degree of the spline, keeping all intermediate results in a 3D tensor
        of shape (n_points, `p` + 1, `p` + 1), where `p` is the degree of the spline.
        Then, it exploits the recursive nature of how the B-spline basis functions are
        defined to compute the derivatives using divided differences.

        :returns: Tensor of shape (n_points, `derivative_order` + 1, `p` + 1) where
                  element (i, k, j) is N_{i−p+j,p}^{(k)}(u_i).

        :raises ValueError: If `derivative_order` is not a positive integer, or if
            `derivative_order` is greater than the degree of the spline.
        """
        p = self._degree
        """Alias for the degree of the spline to shorten the code."""

        if derivative_order < 0 or derivative_order > p:
            raise ValueError(f"`derivative_order` must be in [0, {p}].")

        coords = mi.TensorXf(parametric_span_data.parametric_coords)
        num_points = coords.shape[0]

        extended_knot_spans = dr.ravel(parametric_span_data.knot_spans + p + 1)
        extended_knots = mi.TensorXf(self.compute_extended_knot_vector(degree=p))

        basis_table = dr.zeros(dtype=mi.TensorXf, shape=(num_points, p + 1, p + 1))
        left = dr.zeros(dtype=mi.TensorXf, shape=(num_points, p + 1))
        right = dr.zeros(dtype=mi.TensorXf, shape=(num_points, p + 1))

        derivatives = dr.zeros(
            dtype=mi.TensorXf, shape=(num_points, derivative_order + 1, p + 1)
        )
        # a two-row buffer used in divided differences
        dd_buffer = dr.zeros(dtype=mi.TensorXf, shape=(num_points, 2, p + 1))

        basis_table[:, 0, 0] = 1.0

        for j in range(1, p + 1):
            left[:, j] = dr.ravel(coords - extended_knots[extended_knot_spans + 1 - j])
            right[:, j] = dr.ravel(extended_knots[extended_knot_spans + j] - coords)

            saved = dr.zeros(dtype=mi.TensorXf, shape=num_points)
            for r in range(j):
                basis_table[:, j, r] = dr.ravel(left[:, j - r] + right[:, r + 1])
                temp = basis_table[:, r, j - 1] / basis_table[:, j, r]

                basis_table[:, r, j] = dr.ravel(saved + right[:, r + 1] * temp)
                saved = left[:, j - r] * temp

            basis_table[:, j, j] = dr.ravel(saved)

        derivatives[:, 0, :] = dr.ravel(basis_table[:, :, p])

        # Loop over all basis functions that are in the knot span
        for r in range(0, p + 1):
            dd_buffer[:, 0, 0] = 1.0
            s1, s2 = 0, 1

            # Divided differences loop for derivatives
            for k in range(1, derivative_order + 1):
                derivative = dr.zeros(dtype=mi.TensorXf, shape=num_points)

                rk = r - k
                pk = p - k

                if r >= k:
                    dd_buffer[:, s2, 0] = dr.ravel(
                        dd_buffer[:, s1, 0] / basis_table[:, pk + 1, rk]
                    )
                    derivative = dd_buffer[:, s2, 0] * basis_table[:, rk, pk]

                j1 = 1 if rk >= -1 else -rk
                j2 = k - 1 if r - 1 <= pk else p - r

                for j in range(j1, j2 + 1):
                    inv = 1.0 / basis_table[:, pk + 1, rk + j]
                    dd_buffer[:, s2, j] = dr.ravel(
                        (dd_buffer[:, s1, j] - dd_buffer[:, s1, j - 1]) * inv
                    )
                    derivative += dd_buffer[:, s2, j] * basis_table[:, rk + j, pk]

                if r <= pk:
                    dd_buffer[:, s2, k] = dr.ravel(
                        -dd_buffer[:, s1, k - 1] / basis_table[:, pk + 1, r]
                    )
                    derivative += dd_buffer[:, s2, k] * basis_table[:, r, pk]

                derivatives[:, k, r] = dr.ravel(derivative)
                s1, s2 = s2, s1

        fact = p
        for k in range(1, derivative_order + 1):
            derivatives[:, k, :] = dr.ravel(derivatives[:, k, :] * fact)
            fact *= p - k

        return derivatives

    def evaluate_basis_functions(
        self, parametric_coords: "mi.TensorXf"
    ) -> "mi.TensorXf":
        """
        Evaluate the B-spline basis functions at the given `parametric_coords`.

        `parametric_coords` is a 2D tensor of shape (n_points, 1) where n_points
        is the number of points to evaluate, and the second dimension is 1 because
        a curve has a single parametric direction (u).

        This method internally computes the knot spans for the given `parametric_coords`
        and proceeds to call `evaluate_basis_functions_internal` to compute the
        basis functions using the Cox-de Boor algorithm.

        :returns: A tensor of shape (n_points, degree + 1) representing the values
                  of the basis functions at the given `parametric_coords`.

        :raises ValueError: If `parametric_coords` is not a 2D tensor (n_points, 1), or
            if the values in `parametric_coords` are outside the knot vector range.
        """
        self.validate_parametric_coords(parametric_coords)
        span_data = self.compute_span_data(parametric_coords[:, 0])

        return self.evaluate_basis_functions_internal(
            parametric_span_data=span_data,
            degree=self._degree,
        )

    def evaluate_basis_functions_first_derivative(
        self, parametric_coords: "mi.TensorXf"
    ) -> "mi.TensorXf":
        """
        Evaluate the first derivative of the B-spline basis functions at the given
        `parametric_coords`.

        `parametric_coords` is a 2D tensor of shape (n_points, 1) where n_points
        is the number of points to evaluate, and the second dimension is 1 because
        a curve has a single parametric direction (u).

        This method internally computes the knot spans for the given `parametric_coords`
        and proceeds to call `evaluate_basis_function_derivatives` with
        `derivative_order=1`, keeping just the first derivatives of the basis
        functions.

        :returns: A tensor of shape (n_points, degree + 1) representing the first
                  derivatives of the basis functions at the given `parametric_coords`.

        :raises ValueError: If `parametric_coords` is not a 2D tensor (n_points, 1), or
            if the values in `parametric_coords` are outside the knot vector range.
        """
        self.validate_parametric_coords(parametric_coords)
        span_data = self.compute_span_data(parametric_coords[:, 0])

        return self.evaluate_basis_function_derivatives(
            parametric_span_data=span_data,
            derivative_order=1,
        )[:, 1, :]

    def validate_parametric_coords(self, parametric_coords: "mi.TensorXf") -> None:
        """
        Check if the provided `parametric_coords` are valid.

        According to the contract of `CoordinateSpline`, the `parametric_coords` must
        be a 2D tensor of shape (n_points, 1). The values must be within the range of
        the knot vector defined in the B-Spline.

        :raises ValueError: If `parametric_coords` is not a 2D tensor (n_points, 1), or
            if the values in `parametric_coords` are outside the knot vector range.
        """
        if len(parametric_coords.shape) != 2:
            raise ValueError("`parametric_coords` must be a 2D tensor!")
        if parametric_coords.shape[1] != 1:
            raise ValueError(
                "`parametric_coords` must be a 2D tensor of shape (n_points, 1)."
            )
        if (
            dr.min(parametric_coords) < self.knots[0]
            or dr.max(parametric_coords) > self.knots[-1]
        ):
            raise ValueError(
                f"`parametric_coords` values must be within the knot vector range: "
                f"[{self.knots[0]}, {self.knots[-1]}]."
            )

    def interpolate(
        self, values: "mi.TensorXf", parametric_coords: "mi.TensorXf"
    ) -> "mi.TensorXf":
        """
        Compute the B-spline interpolation of `values` at the specified
        `parametric_coords`.

        `parametric_coords` is a 2D tensor of shape (n_points, 1) where n_points
        is the number of points to evaluate, and the second dimension is 1 because
        a curve has a single parametric direction (u).

        The interpolated value is computed using the B-spline basis functions
        weighted by the provided
            I_{d}(u) = sum(i = 0 to n_basis) N_{i}(u) * V_{i, d}
        where:
            - V{i, d} is the value at the i-th basis function and d-th dimension
            - N_{i}(u) is the i-th B-spline basis function evaluated at u
            - I_d(u) is the d-th component of the interpolated values at coordinate u

        Most B-Spline basis functions N_{i,p} are zero for a given parametric
        coordinate; only `degree + 1` from an appropriate knot span are non-zero.
        They are first extracted from the knot span, then the Cox-de Boor algorithm
        is used to compute the basis function values at the specified parametric
        coordinate, and finally the value is computed as a weighted sum of the
        values.

        :returns: A tensor of shape (n_points, 3) representing the 3D points
                  corresponding to the given `parametric_coords`.

        :raises ValueError: If `parametric_coords` is not a 2D tensor (n_points, 1),
            if the values in `parametric_coords` are outside the knot vector range,
            or if `values` is not a 2D tensor of shape(n_basis_functions, dimensions),
            where `n_basis_functions` is the number of basis functions in the B-spline.
        """
        self.validate_interpolating_values(values)
        self.validate_parametric_coords(parametric_coords)

        span_data = self.compute_span_data(parametric_coords=parametric_coords[:, 0])
        basis_funcs = self.evaluate_basis_functions_internal(
            parametric_span_data=span_data
        )

        result = dr.zeros(
            dtype=mi.TensorXf, shape=(parametric_coords.shape[0], values.shape[1])
        )

        for j in range(self.basis_functions_per_span):
            basis_func_index = span_data.global_basis_function_index(local_index=j)
            result += basis_funcs[:, j][:, None] * values[dr.ravel(basis_func_index)]

        return result

    def active_basis_function_indices(
        self, parametric_coords: "mi.TensorXf"
    ) -> "mi.TensorXu":
        """
        Returns the indices of the active basis functions at the specified
        `parametric_coords`.

        `parametric_coords` is a 2D tensor of shape (n_points, 1) where n_points
        is the number of points to evaluate, and the second dimension is 1 because
        a curve has a single parametric direction (u).

        The first step in evaluating the basis functions at a given parametric point
        is determining which basis functions are non-zero at that point. This is done
        by determining which knot spans contain the parametric coordinates and then
        returning the indices of the basis functions that are active in those spans.

        These indices can be used interchangeably as indices of control points as
        each basis function corresponds to a control point in the B-spline.

        :returns: A tensor of shape (n_points, basis_functions_per_span) containing the
            indices of the active basis functions at the given `parametric_coords`.

        :raises ValueError: If `parametric_coords` is not a 2D tensor (n_points, 1), or
            if the values in `parametric_coords` are outside the knot vector range.
        """
        self.validate_parametric_coords(parametric_coords)

        span_data = self.compute_span_data(parametric_coords=parametric_coords[:, 0])

        result = dr.zeros(
            dtype=mi.TensorXu, shape=(parametric_coords.shape[0], self._degree + 1)
        )

        for j in range(self.basis_functions_per_span):
            basis_func_index = span_data.global_basis_function_index(local_index=j)
            result[:, j] = dr.ravel(basis_func_index)

        return result

    def insert_knot(self, knot_position: float) -> None:
        """
        Insert a knot into the B-spline at position `knot_position`.

        Inserts a knot to position `knot_position` in the knot vector. First, the
        position is checked to be within the range of the knot vector. Next, the
        position is found using binary search, and the knot is inserted at the
        appropriate position.

        If the new knot is close to the first or the last knot (within `KNOT_TOL`),
        a second knot is inserted at the other end of the knot vector to maintain
        the same multiplicity for the first and last unique knots. This is necessary
        to ensure the validity of the B-spline.

        Finally, updates the internal knot vector, unique knots,
        and knot multiplicities.

        :raises ValueError: If `knot_position` is outside the range of the knot vector,
            if the multiplicity of internal knots is greater than `degree`, or if the
            multiplicity of the first or the last knot is greater than `degree + 1`.
        """
        if knot_position < self._knots[0] or knot_position > self._knots[-1]:
            raise ValueError(
                f"Knot must be within the range of the knot vector: "
                f"[{self._knots[0]}, {self._knots[-1]}]."
            )

        # Find insertion position
        position = np.searchsorted(self._knots, knot_position, side="right")

        new_knots = np.insert(self._knots, position, knot_position)

        # If the new knot matches the first or last unique knot, it has to be be
        # inserted to the other side as well to maintain same multiplicity.
        if position == self._knot_multiplicities[0] and math.isclose(
            self._unique_knots[0], knot_position, abs_tol=self.KNOT_TOL, rel_tol=0.0
        ):
            new_knots = np.append(new_knots, self._unique_knots[-1])
        if position == len(self._knots) and math.isclose(
            self._unique_knots[-1], knot_position, abs_tol=self.KNOT_TOL, rel_tol=0.0
        ):
            new_knots = np.insert(
                new_knots, self._knot_multiplicities[0], self._unique_knots[0]
            )

        self._knots = new_knots

        # Recompute the unique knots and multiplicities
        self._num_elements, self._unique_knots, self._knot_multiplicities = (
            BSpline.process_knot_vector(
                knot_vector=self._knots.tolist(), degree=self._degree
            )
        )

    def insert_knot_and_update_values(
        self, knot_position: float, values_list: list["mi.TensorXf"]
    ) -> list["mi.TensorXf"]:
        """
        Insert a knot at `knot_position` and update the `values_list` that is
        interpolated by the B-spline.

        Knot insertion is performed by calling the `insert_knot` method.

        When inserting knots, it is typically necessary to also update the values
        that the spline interpolates, as the number of basis functions increases
        with the number of knot spans. The `values_list` parameter is a list of
        Mitsuba tensors of shape (n_basis_functions, n_dimensions_k) where
        n_dimensions_k is the number of dimensions of the k-th values tensor in the
        list.

        For more information about how the process is derived, a useful reference is
        https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/NURBS-knot-insert.html

        :returns: A list of Mitsuba tensors of shape (n_basis_functions, n_dimensions_k)
            where `n_basis_functions` is the number of basis functions after knot
            insertion and `n_dimensions_k` is the number of components of k-th element.

        :raises ValueError: If `knot_position` is outside the range of the knot vector,
            if the multiplicity of internal knots is greater than `degree`, or if the
            B-spline does not have an open knot vector.
        """
        if not self.has_open_knot_vector:
            raise ValueError(
                "Value updates are only supproted for open knot vectors "
                "with the first and last knots having multiplicity degree + 1."
            )

        for values in values_list:
            self.validate_interpolating_values(values)

        span_data = self.compute_span_data(mi.TensorXf([knot_position]))
        knot_span = int(span_data.knot_spans.numpy().astype(np.uint32)[0])
        p = self.degree

        original_knot_vector = self._knots.copy()

        self.insert_knot(knot_position=knot_position)

        new_values_list: list["mi.TensorXf"] = []

        for values in values_list:
            values = values.numpy().astype(np.float64)
            # Number of basis functions increased after knot insertion
            new_values = np.zeros(
                shape=(self.num_basis_functions, values.shape[1]), dtype=np.float64
            )

            new_values[: knot_span - p + 1, :] = values[: knot_span - p + 1, :]
            new_values[knot_span + 1 :, :] = values[knot_span:, :]

            idx = np.arange(knot_span - p + 1, knot_span + 1)
            alphas = (knot_position - original_knot_vector[idx]) / (
                original_knot_vector[idx + p] - original_knot_vector[idx]
            )
            new_values[idx, :] = (
                alphas[:, None] * values[idx, :]
                + (1 - alphas[:, None]) * values[idx - 1, :]
            )
            new_values_list.append(mi.TensorXf(new_values))

        return new_values_list

    def h_refinement(
        self, factor: int, values_list: list["mi.TensorXf"]
    ) -> list["mi.TensorXf"]:
        """
        Increases the resolution of the bivariate B-spline by `factor` by subdividing
        each knot span of both underlying `BSpline` instances into `factor` equal parts
        and inserting new knots at these positions.

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

        :returns: A list of Mitsuba tensors of shape (n_basis_functions, n_dimensions_k)
            where `n_basis_functions` is the number of basis functions after refinement.

        :raises ValueError: If `factor` is not a positive integer, or if the spline
            does not have an open knot vector.
        """
        if factor < 1:
            raise ValueError("`factor` must be a positive integer.")

        for knot_start, knot_end in self.knot_spans.numpy():
            for i in range(1, factor):
                new_knot = knot_start + i * (knot_end - knot_start) / factor
                values_list = self.insert_knot_and_update_values(
                    knot_position=new_knot, values_list=values_list
                )

        return values_list

    def project_function(
        self, func: Callable[["mi.TensorXf"], "mi.TensorXf"]
    ) -> "mi.TensorXf":
        """
        Projects the given function `func` onto the B-spline basis functions.

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
        p = self.degree
        knot_spans = self.knot_spans

        # Compute how small of a perturbation can be used while still making sure that
        # each sampled point does not coincide with the position of a knot.
        lengths = knot_spans[:, 1] - knot_spans[:, 0]
        average = 0.5 * (knot_spans[:, 0] + knot_spans[:, 1])

        interval_length = dr.max(knot_spans[:, 1]) - dr.min(knot_spans[:, 0])
        smallest_span_length = dr.min(lengths)
        epsilon = interval_length / smallest_span_length * self.PROJECTION_TOL

        samples = 2.0 * (1 - epsilon) * np.arange(p + 1) / p - (1.0 - epsilon)

        cp_values = dr.zeros(dtype=mi.TensorXf, shape=(self.num_basis_functions, 1))
        parametric_coords = average[:, None]

        dofs_per_span = self.active_basis_function_indices(
            parametric_coords=parametric_coords
        )

        transformation_matrices = dr.zeros(
            dtype=mi.TensorXf, shape=(knot_spans.shape[0], p + 1, p + 1)
        )
        function_values = dr.zeros(
            dtype=mi.TensorXf, shape=(knot_spans.shape[0], p + 1)
        )

        for i, pos_u in enumerate(samples):
            real_pos_u = average + 0.5 * lengths * pos_u
            parametric_coords[:, 0] = dr.ravel(real_pos_u)

            basis_functions = self.evaluate_basis_functions(
                parametric_coords=parametric_coords
            )
            func_val = func(parametric_coords)

            transformation_matrices[:, i, :] = dr.ravel(basis_functions)
            function_values[:, i] = dr.ravel(func_val)

        product = np.einsum(
            "sij,sj->si",
            np.linalg.inv(transformation_matrices.numpy().astype(np.float64)),
            function_values.numpy(),
        )
        dr.scatter_add(
            target=cp_values.array,
            value=dr.ravel(mi.TensorXf(product)),
            index=dr.ravel(dofs_per_span),
        )

        # Divide by the number of times each basis function was sampled
        ones = dr.ones(dtype=mi.TensorXf, shape=product.shape)
        repeat = dr.zeros(dtype=mi.TensorXf, shape=(self.num_basis_functions, 1))
        dr.scatter_add(
            target=repeat.array, value=dr.ravel(ones), index=dr.ravel(dofs_per_span)
        )

        cp_values /= repeat

        return cp_values

    @property
    def degree(self) -> int:
        """Degree of the B-spline."""
        return self._degree

    @property
    def num_basis_functions(self) -> int:
        """
        The number active basis functions in the B-spline.

        This is closely related to the knot vector. If the knot vector is open,
        which is very common in practice, the number of basis functions is equal to
        the number of unique knots in the knot vector plus the degree of the spline.
        """
        return int(len(self._knots) - self._knot_multiplicities[0])

    @property
    def basis_functions_per_span(self) -> int:
        """
        The number of basis functions per knot span.

        This is equal to the degree of the spline plus one, as each knot span
        contributes `degree + 1` basis functions.
        """
        return self._degree + 1

    @property
    def knots(self) -> npt.NDArray[np.float64]:
        """
        Knot vector of the B-spline.

        The knot vector is a numpy array of shape (n_knots,) and it contains the
        information used to define the B-spline curve in parametric space.
        """
        return self._knots

    @property
    def knot_spans(self) -> "mi.TensorXf":
        """
        Knot spans of the B-spline.

        Knots spans are formed by two consecutive unique knots, and they
        represent the intervals in which the B-spline basis functions are defined.

        :returns: A numpy array of shape (n_spans, 2) where each row contains the start
            and end of a knot span. The number of spans is equal to the number of
            unique knots minus one.
        """
        return mi.TensorXf(
            np.stack(arrays=[self._unique_knots[:-1], self._unique_knots[1:]], axis=1)
        )

    @property
    def has_open_knot_vector(self) -> bool:
        """
        Check if the B-spline has an open knot vector.

        An open knot vector is defined as having the first and last knots with
        multiplicity equal to `degree + 1`.
        """
        return (
            self._knot_multiplicities[0] == self._degree + 1
            and self._knot_multiplicities[-1] == self._degree + 1
        )
