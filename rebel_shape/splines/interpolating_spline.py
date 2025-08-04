import abc
from collections.abc import Callable

import mitsuba as mi


class InterpolatingSpline(abc.ABC):
    """
    Basic interface that is implemented by all interpolating splines and reflects their
    fundamental property - interpolation.

    An interpolating spline is defined by a set of basis functions that exist in
    parametric space. The dimensionality of the parametric space depends on the
    dimension of the values that the spline interpolates. For example, a curve is a
    1D object, and thus an interpolating spline for a curve requires one parametric
    direction. The length of the interval on which spline basis functions are
    defined is finite.

    In general, an interpolating spline needs to be able to perform only one operation:
    interpolate the provided values V, which there are as many as the number of
    basis functions, at the specified parametric coordinates u.

    The general interpolating spline equation can thus be written as follows:
        I_{d}(u) = sum(i = 0 to n_basis) B_{i}(u) * V_{i,d}
    where:
        - V is a tensor of shape (n_basis, n_dimensions) representing the values
        - n_basis is the number of basis functions
        - n_dimensions is the number of dimensions of the values
        - d is an index representing the specific dimension of the values (e.g. x, y, z)
        - B_{i}(u) is the i-th basis function evaluated at u
        - I_d(u) is the d-th component of the interpolated values at
          the parametric coordinate u

    Thus, the interface requires implementing the `interpolate` method, which performs
    the operations specified above, and the `num_basis_functions` property, which
    returns the number of basis functions define by the spline.

    Additionally, since the spline is intended to be used in Isogemetric Analysis,
    the basis functions and their derivatives ought to be provided for tensor assembly.
    For this purpose, information about `knot_spans`, `basis_functions_per_span`,
    `active_basis_function_indices`, `evaluate_basis_functions`,
    `evaluate_basis_functions_first_derivative`, and `project_function` are also
    required.

    For more details about parametric space and basic spline properties, study the
    `BSpline` class that is a concrete implementation of this interface.
    """

    @abc.abstractmethod
    def interpolate(
        self, values: "mi.TensorXf", parametric_coords: "mi.TensorXf"
    ) -> "mi.TensorXf":
        """
        Interpolates the provided `values` at the specified `parametric_coords`.

        The `values` is a tensor of shape (n_basis_functions, n_dimensions),
        where `n_basis_functions` is the number of basis functions used by the pline,
        and `n_dimensions` is the number of dimensions of the values being interpolated.

        `parametric_coords` is a 2D tensor of shape (n_points, n_params) where n_points
        is effectively the batch size, while n_params denotes how many parametric
        directions there are. For example, curves have 1 parametric value,
        surfaces have 2, etc.

        All implementations should in general be expressed as:
            I_d(u) = sum_{i=0 to n_basis} B_i(u) * V_{i, d}
        where:
            - V{i, d} is the value at the i-th basis function and d-th dimension
            - B_{i}(u) is the i-th basis function evaluated at u
            - I_d(u) is the d-th component of the interpolated values at coordinate u

        Typically, only a small subset of basis functions are non-zero at a given
        point in parametric space so the cost of evaluating the spline is proportional
        to the number of non-zero basis functions at that point. For example, for
        polynomial basis functions of degree p in D parametric dimensions, the cost of
        interpolating a single point is O(p^D).

        :returns: A Mitsuba tensor of shape (n_points, n_dimensions) representing the
            interpolated field values at the specified `parametric_coords`.

        :raises ValueError: If the shape of `values` does not match the number of basis
            functions or if the shape of `parametric_coords` does not match the
            expected number of parametric dimensions, of if `parametric_coords` values
            are outside the range of the knot spans defined by the spline.

        :raises NotImplementedError: If a child class does not implement this method.
        """
        raise NotImplementedError(
            "Classes that inherit from `InterpolatingSpline` need to implement the "
            "`interpolate` method."
        )

    @abc.abstractmethod
    def evaluate_basis_functions(
        self, parametric_coords: "mi.TensorXf"
    ) -> "mi.TensorXf":
        """
        Evaluates the basis functions at the specified `parametric_coords`.

        This is a necessary step when performing interpolation, as it determines how
        heavily each basis function contributes to the final interpolated value at
        the specified parametric coordinates.

        Each type of splines has a dedicated algorithm for performing this operation,
        e.g. Bézier curves use the de Casteljau's algorithm, while B-splines use
        the Cox-de Boor recursion formula.

        Furthermore, this method is used to assemble the local matrices for
        isogeometric analysis problems, where the basis functions are evaluated at
        specific points in parametric space to compute the local contributions to the
        global matrix.

        :returns: A Mitsuba tensor of shape (n_points, n_basis_functions) representing
            the values of the basis functions at the specified `parametric_coords`.

        :raises ValueError: If the shape of `parametric_coords` does not match the
            expected number of parametric dimensions or if the values are out of bounds
            of the parametric space defined by the spline.

        :raises NotImplementedError: If a child class does not implement this method.
        """
        raise NotImplementedError(
            "Classes that inherit from `InterpolatingSpline` need to implement the "
            "`evaluate_basis_functions` method."
        )

    @abc.abstractmethod
    def evaluate_basis_functions_first_derivative(
        self, parametric_coords: "mi.TensorXf"
    ) -> "mi.TensorXf":
        """
        Evaluates the basis functions derivative at the specified `parametric_coords`.

        Like `evaluate_basis_functions`, this method is used to compute the values of
        basis function derivatives at specific points in parametric space. These can
        then be used to compute the first derivatives of the interpolated values.

        For each spline type it is also possible to create dedicated algorithms to
        perform derivative evaluation, e.g. for B-splines, the first derivative
        can be computed using degree minus one basis functions, but significantly
        complicating the implementation. Thus, it is recommended to use this method
        and then simply interpolate values as in the `interpolate` method.

        Additionally, this method is used to assemble the local stiffness matrices
        for isogeometric analysis problems, where the basis function derivatives are
        evaluated at specific points in parametric space to compute the local
        contributions to the global stiffness matrix.

        :returns: A Mitsuba tensor of shape (n_points, n_basis_functions, n_directions)
            representing the derivatives of the basis functions at the specified
            `parametric_coords` with respect to all parametric directions.

        :raises ValueError: If the shape of `parametric_coords` does not match the
            expected number of parametric dimensions or if the values are out of bounds
            of the parametric space defined by the spline.

        :raises NotImplementedError: If a child class does not implement this method.
        """
        raise NotImplementedError(
            "Classes that inherit from `InterpolatingSpline` need to implement the "
            "`evaluate_basis_functions` method."
        )

    @abc.abstractmethod
    def active_basis_function_indices(
        self, parametric_coords: "mi.TensorXf"
    ) -> "mi.TensorXu":
        """
        Returns the indices of the active basis functions at the specified
        `parametric_coords`.

        The first step in evaluating the basis functions at a given parametric point
        is determining which basis functions are non-zero at that point. Typically,
        this is done by checking which knot spans contain the parametric coordinates
        and then returning the indices of the basis functions that are active in those
        spans, while this notion might be different for some spline types.

        This method is also crucial for isogeometric analysis problems, where it is
        necessary to know at which indices in the global matrix the local
        contributions should be added. The indices returned by this method are used to
        construct the local matrices and vectors that are then assembled into the
        global matrix and vector.

        :returns: A Mitsuba tensor of shape (n_points, n_active_basis_functions)
            containing the indices of the active basis functions at the specified
            `parametric_coords`.

        :raises ValueError: If the shape of `parametric_coords` does not match the
            expected number of parametric dimensions or if the values are out of bounds
            of the parametric space defined by the spline.

        :raises NotImplementedError: If a child class does not implement this method.
        """
        raise NotImplementedError(
            "Classes that inherit from `InterpolatingSpline` need to implement the "
            "`active_basis_function_indices` method."
        )

    @abc.abstractmethod
    def h_refinement(
        self, factor: int, values_list: list["mi.TensorXf"]
    ) -> list["mi.TensorXf"]:
        """
        Increases the resolution of the spline by `factor` by reducing the side of
        knot spans in the parametric space.

        This is one of the 3 main methods to increase the accuracy of spline based
        representations: h-refinement, p-refinement, and k-refinement.
        h-refinement is the process of increasing the number of knot spans in
        the parametric space by means of knot insertion. Depending on the spline type,
        this can either be a global (B-Splines, NURBS) or a local operation (T-Splines).

        When inserting knots, it is typically necessary to also update the values
        that the spline interpolates, as the number of basis functions increases
        with the number of knot spans. The `values_list` parameter is a list of
        Mitsuba tensors of shape (n_basis_functions, n_dimensions_k) where
        n_dimensions_k is the number of dimensions of the k-th values tensor in the
        list.

        :returns: A list of Mitsuba tensors of shape (n_basis_functions, n_dimensions_k)
            where `n_basis_functions` is the number of basis functions after refinement.

        :raises ValueError: If `factor` is not a positive integer or if each tensor in
            `values_list` does not have shape (n_basis_functions, n_dimensions_k),
            where `n_basis_functions` is the number of basis functions before refinement
            and `n_dimensions_k` is the number of dimensions of the k-th values tensor.

        :raises NotImplementedError: If a child class does not implement this method.
        """
        raise NotImplementedError(
            "Classes that inherit from `InterpolatingSpline` need to implement the "
            "`h_refinement` method."
        )

    @abc.abstractmethod
    def project_function(
        self, func: Callable[["mi.TensorXf"], "mi.TensorXf"]
    ) -> "mi.TensorXf":
        """
        Projects the given function `func` onto the spline basis functions.

        In most practical applications, the data is not available in the form of
        values at the control points of the spline, but rather as a function defined
        over the parametric space of the spline.

        This function can be arbitrary, so long as it takes as inputs a Mitsuba
        tensor of shape (n_points, n_params) and returns a Mitsuba tensor of shape
        (n_points, 1), where `n_points` is the number of points at which the function
        is evaluated and `n_params` is the number of parametric dimensions.

        Each implementor can set constraints on the function and the accuracy of the
        retrieved spline representation, e.g. by requiring that the function is
        as least as smooth as the spline basis functions, which can be used to
        obtain a unique solution for the spline control points. Other times it might
        introduce a least-squares problem, where the function is sampled at more points
        than the number of control points, and the goal is to find the best fit
        spline representation of the function.

        While `func` has to return only a single component, this is not a big issue
        since one can simply call this method on each component of the
        function separately.

        :returns: A tensor of shape (n_control_points, 1) representing
            the projected function values at the control points of the spline patch.

        :raises NotImplementedError: If a child class does not implement this method.
        """
        raise NotImplementedError(
            "Classes inheriting from `SplinePatch` need to implement the "
            "`project_function` method."
        )

    @property
    @abc.abstractmethod
    def degree(self) -> int:
        """
        Degree of polynomial basis functions used to define the spline.

        For tensor product splines, e.g. B-Spline surfaces, this is the larger degree
        of the two spline degrees used to define the basis functions of the surface.

        :returns: Degree of the spline
        :raises NotImplementedError: If a child class does not implement this property.
        """
        raise NotImplementedError(
            "Classes that inherit from `CoordinateSpline` need to implement the "
            "`degree` property."
        )

    @property
    @abc.abstractmethod
    def num_basis_functions(self) -> int:
        """
        Returns the number of basis functions used by the spline.

        This is the number of basis functions that the spline uses to interpolate
        the values and dictates the shape of the `values` tensor passed to the
        `interpolate` method. It is expected to be a positive integer.

        :returns: The number of basis functions used by the spline.
        :raises NotImplementedError: If a child class does not implement this property.
        """
        raise NotImplementedError(
            "Classes that inherit from `InterpolatingSpline` need to implement the "
            "`num_basis_functions` property."
        )

    @property
    @abc.abstractmethod
    def basis_functions_per_span(self) -> int:
        """
        Returns the number of basis functions per knot span in the B-spline surface.

        The number of basis functions per knot span is given by the product of the
        number of basis functions in the u and v directions, which is equal to
        (degree_u + 1) * (degree_v + 1).

        :returns: An integer representing the number of basis functions per knot span.
        """
        raise NotImplementedError(
            "Classes that inherit from `InterpolatingSpline` need to implement the "
            "`functions_per_span` property."
        )

    @property
    @abc.abstractmethod
    def knot_spans(self) -> "mi.TensorXf":
        """
        Returns the knot spans of the spline in parametric space.

        The knot spans are a tensor that defines the intervals in which the basis
        functions are defined. For example, for a B-spline curve, this would be a 2D
        tensor where the first dimension has the number of knot spans and the
        second one is a tuple of the start and end of each knot span.

        For more complex splines, such as T-splines, the essence still holds true, but
        the domain is no longer necessarily rectangular in the parametric space
        (in some special cases). However, since it is unlikely that this will be
        implemented in the near future, we will keep the interface simple and
        assume that the knot spans are always rectangular in the parametric space.

        :returns: A Mitsuba tensor representing the knot spans of the spline in the
            shape of (n_spans, n_params, 2), where `n_spans` is the number of knot
            spans, `n_params` is the number of parametric dimensions, and the last
            dimension contains the start and end of each knot span.

        :raises NotImplementedError: If a child class does not implement this property.
        """
        raise NotImplementedError(
            "Classes that inherit from `InterpolatingSpline` need to implement the "
            "`knot_spans` property."
        )

    def validate_interpolating_values(self, values: "mi.TensorXf") -> None:
        """
        Verifies that the provided `values` tensor contiains the correct number of
        elements for the spline's basis functions.

        The `values` must be a 2D tensor with the same number of elements as the
        number of basis functions in the spline. This is necessary for interpolation
        to work correctly and under all circumstances.

        :raises ValueError: If `values` is not a 2D tensor or if its length does not
            match the number of active basis functions in the spline.
        """
        if len(values.shape) != 2:
            raise ValueError("`values` must be a 2D tensor!")
        if values.shape[0] != self.num_basis_functions:
            raise ValueError(
                f"`values` must have {self.num_basis_functions} elements, "
                f"but has {values.shape[0]}."
            )
