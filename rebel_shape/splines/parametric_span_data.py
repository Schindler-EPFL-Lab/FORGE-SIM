import drjit as dr
import mitsuba as mi


class ParametricSpanData:
    """
    A class that handles the data needed to evaluate the basis functions for a
    given set of `parametric_coords`.

    Each parametric coordinate is mapped to an appropriate `knot_span`, an index that
    designates which at which knot the knot span starts. The `knot_span` is al
    """

    def __init__(
        self,
        parametric_coords: "mi.ArrayXf",
        knot_spans: "mi.ArrayXu",
        num_basis_functions: int,
        spline_degree: int,
    ) -> None:
        """
        `parametric_coords` is a 1D array of coordinates in the parametric domain. It
        is assumed that they were checked to be inside the range of the domain.

        `knot_spans` is a 1D array of indices that designate the index of the knot
        at which the knot span starts. It is closely related to the global index of the
        basis function that is active in the knot span.

        Since the largest value in `knot_spans` must be smaller than
        `num_basis_function` that the abstract spline supports (example: `BSpline`).
        Finally, `degree` is the polynomial degree of the spline basis functions.

        :raises ValueError: If the largest index in `knot_spans` is larger than
            `num_control_points`.
        """
        if len(parametric_coords.shape) != 1:
            raise ValueError("`parametric_coords` must be a 1D tensor!")
        if len(knot_spans.shape) != 1:
            raise ValueError("`knot_spans` must be a 1D tensor!")
        if parametric_coords.shape[0] != knot_spans.shape[0]:
            raise ValueError(
                "`parametric_coords` and `knot_spans` must have the same length!"
            )

        self._parametric_coords = parametric_coords

        if dr.max(knot_spans) >= num_basis_functions:
            raise ValueError(
                "Largest `knot_span_index` must be smaller than `num_control_points`."
            )
        self._knot_spans = knot_spans

        self._num_control_points = num_basis_functions
        self._degree = spline_degree

    def global_basis_function_index(self, local_index: int = 0) -> "mi.ArrayXu":
        """
        Returns the index of the basis function with index `shift` in the
        appropriate knot span.

        Each spline has `degree + 1` active basis functions in a knot span. The
        `shift` parameter is used to select which basis function is "active" in the
        knot span. Therefore, the `shift` parameter must be in the range [0, degree].

        If the spline is periodic, wraparound effects are also considered. E.g.,
        the first knot span in a periodic spline has a basis function that is typically
        considered to be the last one in the previous knot span.

        The index of the basis function is usually equivalent to the index of the
        control point that influences the parametric coordinate. There is a distinction
        between local indices (the index of the basis function in the knot span) and
        global indices (the index of the basis function in the entire spline).

        :return: A 1D array (ravelled) of indices that designate the index of the
            spline basis function / control point index.
        :raises ValueError: If `local_index` is not in the range [0, degree].
        """
        if local_index < 0 or local_index > self._degree:
            raise ValueError(
                f"There are {self._degree + 1} active basis functions in a knot span."
                f"`local_index` must be in interval [0, {self._degree}]."
            )
        knot_spans_int = dr.ravel(mi.TensorXi(self._knot_spans))

        wrapped_spans = dr.select(
            knot_spans_int - self._degree + local_index >= 0,
            knot_spans_int - self._degree + local_index,
            knot_spans_int + self._num_control_points - self._degree + local_index,
        )

        return dr.ravel(mi.TensorXu(wrapped_spans))

    @property
    def parametric_coords(self) -> "mi.TensorXf":
        return self._parametric_coords

    @property
    def knot_spans(self) -> "mi.ArrayXu":
        return self._knot_spans
