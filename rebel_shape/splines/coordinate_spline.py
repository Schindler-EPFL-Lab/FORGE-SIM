import abc

import mitsuba as mi

from rebel_shape.splines.interpolating_spline import InterpolatingSpline


class CoordinateSpline(abc.ABC):
    """
    Abstract base class for explicit splines in coordinate space.

    Aims to expose the minimal interface for splines to be used in the context of
    spline-based optimization, while other functionality related to specific
    spline classes should be exposed by the child classes.

    It represents an explicit spline that can be evaluated in 3D space. In simple
    terms, coordinate splines use an underlying interpolating spline to interpolate
    coordinates in space, typically referred to as control points. Note that some
    splines might have more than 3 coordinates per control point, such as NURBS,
    which have an additional weight coordinate.

    The fundamental function is `evaluate_coordinates` and it returns the coordinates
    of a point in 3D space for a specified point in parametric space.
    Two other essential properties are also defined: `degree` and `control_points`.
    The only method that is required to be implemented by child classes is the
    `interpolating_spline` property, which returns an instance of the
    `InterpolatingSpline` that this explicit spline uses for interpolation.

    The constructor is not meant to be used directly, but rather serves to facilitate
    control point initialization and getter/setter functionality for child classes.
    It is expected that child classes will implement their own initialization logic
    and may require additional parameters in their constructors.
    """

    def __init__(self, control_points: "mi.TensorXf") -> None:
        """
        Initialize the CoordinateSpline with control points.

        The constructor is basic and only initializes the control points - insufficient
        for a fully functional spline, but serves as a base for child classes to
        implement their own initialization logic.

        :raises ValueError: If `control_points` is not a 2D tensor or does not have
            the shape (n_cp, 3), where n_cp is the number of control points.
        """
        if len(control_points.shape) != 2:
            raise ValueError("`control_points` must be a 2D tensor!")
        if control_points.shape[1] != 3:
            raise ValueError("`control_points` must be a 2D tensor of shape (n_cp, 3).")

        if control_points.shape[0] != self.interpolating_spline.num_basis_functions:
            raise ValueError(
                f"Number of control points ({control_points.shape[0]}) "
                f"must match the number of control points specified by the "
                f"associated interpolating spline: "
                f"({self.interpolating_spline.num_basis_functions})."
            )

        self._control_points = control_points

    @property
    @abc.abstractmethod
    def interpolating_spline(self) -> InterpolatingSpline:
        """
        An instance of the `InterpolatingSpline` that this coordinate spline uses
        for interpolation.

        This property is used to access the underlying interpolating spline that
        provides the interpolation functionality for the coordinate spline.

        :returns: An instance of `InterpolatingSpline`.
        :raises NotImplementedError: If a child class does not implement this property.
        """
        raise NotImplementedError(
            "Classes that inherit from `CoordinateSpline` need to implement the "
            "`interpolating_spline` property."
        )

    def evaluate_coordinates(self, parametric_coords: "mi.TensorXf") -> "mi.TensorXf":
        """
        Evaluate the spline 3D coordinates at multiple specified parametric points.

        `parametric_coords` is a 2D tensor of shape (n_points, n_params) where n_points
        is effectively the batch size, while n_params denotes how many parametric
        directions there are. For example, curves have 1 parametric value,
        surfaces have 2, etc.

        :returns: A tensors of shape (n_points, 3) representing a series of 3D points
                  that correspond to `parametric_coords`.
        """
        self.interpolating_spline.interpolate(
            values=self.control_points, parametric_coords=parametric_coords
        )

    @property
    def degree(self) -> int:
        """
        Degree of polynomial basis functions used to define the spline.

        For tensor product splines, e.g. B-Spline surfaces, this is the larger degree
        of the two spline degrees used to define the basis functions of the surface.

        :returns: Degree of the spline
        """
        return self.interpolating_spline.degree

    @property
    def control_points(self) -> "mi.TensorXf":
        """
        Control points of the spline.

        The control points define the shape of the spline in 3D space. For a given
        parametric coordinate, the appropriate basis functions are selected, then
        evaluated, and the control points are weighted by the basis functions to
        determine the final point in 3D space.

        :returns: Control points of the spline with shape (n_control_points, 3).
        """
        return self._control_points

    @control_points.setter
    def control_points(self, control_points: "mi.TensorXf") -> None:
        """
        Set the control points of the spline.

        The new control points must have the same shape as the original control points.

        :raises ValueError: If the new `control_points` do not have the same shape
            as the old ones.
        """
        if control_points.shape != self._control_points.shape:
            raise ValueError(
                f"New `control_points` must be of shape {self._control_points.shape}."
            )

        self._control_points = control_points
