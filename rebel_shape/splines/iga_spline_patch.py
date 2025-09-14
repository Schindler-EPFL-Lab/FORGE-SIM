from typing import Any

import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi

from rebel_shape.meshing.patch import Patch
from rebel_shape.splines.bspline_patch import BSplinePatch
from rebel_shape.splines.interpolating_spline import InterpolatingSpline
from rebel_shape.splines.spline_patch import SplinePatch


class IGASplinePatch(SplinePatch):
    """
    A class that represents a `SplinePatch` that can be used in isogeometric analysis
    (IGA) applications.

    It wraps a `SplinePatch` and provides methods to evaluate the field values at
    the specified parametric coordinates. Its intended use is to optimize the
    properties attached to the control points of the spline patch, such as
    temperature.

    The reason for the wrapper style behavior is to avoid shifting the responsibility
    of keeping track of field values to classes that are responsible for shape
    optimization of surfaces (e.g. BSplinePatch). Thus, the `IGASplinePatch` class
    receives a concrete implementation of a `SplinePatch` while also implementing the
    `SplinePatch` interface. This allows it to be used in the same way as a
    `SplinePatch`, keeping compatibility with existing code that expects `SplinePatch`
    instances.

    Since he field is often obtained from images, a utility method to project the field
    onto a texture isIGASplinePatch(IGA)  provided, which evaluates the spline patch
    with a specified resolution and returns the evaluated field values as a texture.
    """

    def __init__(self, spline_patch: BSplinePatch, field_values: "mi.TensorXf") -> None:  # pyright: ignore[reportMissingSuperCall]
        """
        Initializes the IGA spline patch with a `SplinePatch` and `field_values`.

        The spline patch is a properly initialized concrete implementation of
        the `SplinePatch` interface, such as `BSplinePatch`, and the `field_values`
        are a tensor of shape (n_control_points, dofs_per_cp), where n_control_points
        is the number of control points and dofs_per_cp is the number of degrees of
        freedom per control point.
        For now, the type is annotated as `BSplinePatch` as this is the only
        concrete implementation of `SplinePatch` available at the time of writing.

        Note that the number of control points coincides with the number of basis
        functions of the underlying interpolating spline, and the same holds true for
        the values associated with the control points.

        This constructor does not call the `super().__init__(control_points)` method
        as it would add a copy of the control points to this instance, which is not
        desired; it introduces state which should not be duplicated. Since that is the
        only functionality of the `CoordinateSpline` constructor (at the time of
        writing), it is not necessary to call it here.

        :raises ValueError: If `field_values` is not a 2D tensor or does not have the
            shape (n_control_points, dofs_per_cp)
        """
        self._spline_patch = spline_patch

        if len(field_values.shape) != 2:
            raise ValueError(
                "The control points of the B-spline patch must be a 2D tensor of shape "
                + "(n_cp, dofs_per_cp)."
            )
        if field_values.shape[0] != spline_patch.control_points.shape[0]:
            raise ValueError(
                "The number of rows in `field_values` must match the number of control "
                + "points."
            )
        self._field_values = field_values

    def evaluate_coordinates(self, parametric_coords: "mi.TensorXf") -> "mi.TensorXf":
        """
        Computes the coordinates of the spline patch at the given `parametric_coords`.

        `parametric_coords` is a 2D tensor of shape (n_points, 2) where n_points
        is the number of points to evaluate, and the second dimension is 2 because
        a surface has two parametric directions (u, v).

        Interpolation is performed using the underlying concrete implementation of the
        `SplinePatch` interface, which typically calls the `interpolate` method of
        the associated `InterpolatingSpline`.

        :returns: A tensor of shape (n_points, 3) representing the 3D coordinates
            of the spline patch at the specified `parametric_coords`.

        :raises ValueError: If `parametric_coords` is not a 2D tensor (n_points, 2),
            or if the values in `parametric_coords` are outside the parametric domain
            of the spline patch.
        """
        return self._spline_patch.evaluate_coordinates(parametric_coords)

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
        return self._spline_patch.evaluate_normals(parametric_coords)

    def evaluate_jacobian(self, parametric_coords: "mi.TensorXf") -> "mi.TensorXf":
        """
        Computes the Jacobian of the spline patch at the given `parametric_coords`.

        `parametric_coords` is a 2D tensor of shape (n_points, 2) where n_points
        is the number of points to evaluate, and the second dimension is 2 because
        a surface has two parametric directions (u, v).

        The Jacobian is a tensor that describes how the surface changes with respect
        to the parametric coordinates. It is useful for computing derivatives and
        understanding the local geometry of the surface.

        :returns: A tensor of shape (n_points, 3, 2) representing the Jacobian at
                  the specified parametric coordinates.

        :raises ValueError: If `parametric_coords` is not a 2D tensor (n_points, 2),
            or if the values in `parametric_coords` are outside the knot vector range.
        """
        return self._spline_patch.evaluate_jacobian(parametric_coords)

    def evaluate_hessian(self, parametric_coords: "mi.TensorXf") -> "mi.TensorXf":
        """
        Computes the Hessian of the spline patch at the given `parametric_coords`.

        `parametric_coords` is a 2D tensor of shape (n_points, 2) where n_points
        is the number of points to evaluate, and the second dimension is 2 because
        a surface has two parametric directions (u, v).

        The Hessian is a rank-3 tensor that describes how the surface curvature changes
        with respect to the parametric coordinates. It is useful for computing
        principal curvatures and understanding the local geometry of the surface.

        :returns: A tensor of shape (n_points, 3, 2, 2) representing the Hessian at
                  the specified parametric coordinates.

        :raises ValueError: If `parametric_coords` is not a 2D tensor (n_points, 2),
            or if the values in `parametric_coords` are outside the knot vector range.
        """
        return self._spline_patch.evaluate_hessian(parametric_coords)

    def evaluate_field(self, parametric_coords: "mi.TensorXf") -> "mi.TensorXf":
        """
        Computes the interpolated field values at the specified `parametric_coords`.

        `parametric_coords` is a 2D tensor of shape (n_points, 2) where n_points
        is the number of points to evaluate, and the second dimension is 2 because
        a surface has two parametric directions (u, v).

        Similarly to how the `evaluate_coordinates` method interpolates the coordinates
        of the control points of the spline patch, this method uses the underlying
        `InterpolatingSpline` to interpolate the field values at the specified
        `parametric_coords`.

        :returns: A tensor of shape (n_points, dofs_per_cp) representing the field
            values at the specified `parametric_coords`.

        :raises ValueError: If `parametric_coords` is not a 2D tensor (n_points, 2),
            if the values in `parametric_coords` are outside the parametric domain
            of the spline patch, or if the shape of `field_values` does not match the
            number of basis functions defined by the underlying `InterpolatingSpline`.
        """
        return self._spline_patch.interpolating_spline.interpolate(
            values=self._field_values, parametric_coords=parametric_coords
        )

    def __getstate__(self) -> dict[str, Any]:
        """
        Exports the IGA spline patch for serialization.

        `_field_values` are converted to a numpy array as they are Mitsuba tensors,
        and thus are not directly serializable with `jsonpickle`. The rest of the
        state is copied from the wrapped instance's `__dict__`, triggering its own
        serialization logic.

        :returns: A dictionary representing the state of the object, ready for
                  serialization with `jsonpickle`.
        """
        state = self.__dict__.copy()

        state["_field_values"] = self._field_values.numpy()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restores the IGA spline patch from a serialized state.

        `state` is a dictionary that contains the serialized state of the object
        including values of `control_points and `quads` which are converted back to
        Mitsuba tensors.
        """
        field_values = state.pop("_field_values")

        self.__dict__.update(state)
        self._field_values = mi.TensorXf(field_values)

    @property
    def interpolating_spline(self) -> InterpolatingSpline:
        """
        An instance of the `InterpolatingSpline` associated with this IGA spline patch.

        This property is used to access the underlying interpolating spline that
        provides the interpolation functionality for both the coordinates and
        the values at the control points of the spline patch.
        """
        return self._spline_patch.interpolating_spline

    @property
    def control_points(self) -> "mi.TensorXf":
        """
        Control points of the wrapped spline patch.

        The control points define the surface in 3D space by interpolating the
        coordinates of the control points using the basis functions defined by the
        underlying `InterpolatingSpline`.
        """
        return self._spline_patch.control_points

    @control_points.setter
    def control_points(self, control_points: "mi.TensorXf") -> None:
        """
        Updates the control points of the wrapped spline patch.

        `control_points` is a tensor of shape (n_control_points, 3) where
        n_control_points is the number of control points.

        :raises ValueError: If the shape of `control_points` does not match the
            shape of the current control points
        """
        self._spline_patch.control_points = control_points

    @property
    def control_points_grid(self) -> "mi.TensorXf":
        """
        Returns the control points of the spline patch as a grid.

        The control points are reshaped into a grid-like structure based on the
        number of control points in each direction (u and v). This is useful for
        visualizing the control net of the spline surface.

        :returns: A tensor of shape (n_cps_v, n_cps_u, 3) representing the control
                  points in a grid format.
        """
        return self._spline_patch.control_points_grid

    @property
    def parametric_domain_size(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Size of the parametric domain for the B-spline curve.

        The B-spline surface is a tensor product of two B-spline curves, so the
        parametric domain size is defined by the knot vector bounds of both
        `u_spline` and `v_spline`.

        :returns: Tuple of tuples representing the parametric domain size:
                  ((u_min, u_max), (v_min, v_max)).
        """
        return self._spline_patch.parametric_domain_size

    def side_knot_spans(self, side: Patch.Side) -> "mi.TensorXf":
        """
        Returns the knot spans along the specified `side` of the patch.

        This method is important for validating connections between patches, as
        they must have the same knot vector along shared sides. It is also used
        for highly accurate numerical integration along the edges of the patch.

        :returns: A Mitsuba tensor of knot spans along the specified side.
        """
        return self._spline_patch.side_knot_spans(side)

    def side_parametric_coords(
        self, side: Patch.Side, side_coords: "mi.Float", flip: bool = False
    ) -> "mi.TensorXf":
        """
        Returns the parametric coordinates along the specified `side` of the patch.

        `side_coords` is a 1D tensor containing the parametric coordinates along
        the specified side. The values in `side_coords` should be within the range
        defined by the knot vector of the corresponding parametric direction.

        `flip` indicates whether the order of the coordinates should be reversed.
        This is useful when computing with shared edges between patches to ensure
        consistent orientation.

        :returns: A 2D tensor of shape (n_points, 2) representing the parametric
                  coordinates along the specified side.
        :raises ValueError: If `side_coords` is not a 1D tensor or if the values
            in `side_coords` are outside the parametric domain of the spline patch.
        """
        return self._spline_patch.side_parametric_coords(side, side_coords, flip)

    @property
    def quads(self) -> "mi.TensorXu":
        """
        Returns the indices of the quads that make up the spline patch.

        The quads are defined by the control points in a grid-like structure.
        Each quad is defined by four control points, and the indices are
        calculated based on the number of control points in each direction.

        :returns: A Mitsuba array of quad indices shaped (n_quads, 4)
        """
        return self._spline_patch.quads

    def get_node_indices_for(self, side: Patch.Side) -> "mi.ArrayXu":
        """
        Returns the node indices of the specified side of the spline patch.

        `side` is an enum that specifies which side of the patch to query.

        :returns: A Mitsuba array of vertex indices that belong to the specified side.
        """
        return self._spline_patch.get_node_indices_for(side)

    @property
    def nodes(self) -> "mi.TensorXf":
        """
        For all spline patches, the nodes are the control points.
        """
        return self._spline_patch.nodes

    @property
    def name(self) -> str:
        """
        Name of the spline patch.
        """
        return self._spline_patch.name

    def h_refinement(self, factor: int = 2) -> None:
        """
        Increase the resolution of the patch in the u and v directions by `factor`.

        This method refines the patch by increasing the number of knot spans in both
        parametric directions by the specified factor. Both the control points and the
        attached field values are updated to account for the increased number of
        basis functions.

        Calls the `h_refinement` method of the underlying `InterpolatingSpline`.

        :raises ValueError: If `factor` is not a positive integer.
        """
        if factor < 1:
            raise ValueError("`factor` must be a positive integer.")

        new_values = self.interpolating_spline.h_refinement(
            factor=factor, values_list=[self.control_points, self._field_values]
        )
        self._spline_patch._control_points = new_values[0]
        self._field_values = new_values[1]

    @property
    def field_values(self) -> "mi.TensorXf":
        """
        Returns the field values of the spline patch.

        The field values are a tensor of shape (n_control_points, dofs_per_cp), where
        n_control_points is the number of control points and dofs_per_cp is the number
        of degrees of freedom per control point. They are used to interpolate the IGA
        field values at the specified parametric coordinates.
        """
        return self._field_values

    @field_values.setter
    def field_values(self, field_values: "mi.TensorXf") -> None:
        """
        Updates the field values of the wrapped spline patch.

        `field_values` is a tensor of shape (n_control_points, dofs_per_cp) where
        n_control_points is the number of control points and dofs_per_cp is the number
        of degrees of freedom per control point.

        :raises ValueError: If the new `field_values` do not have the same shape
            as the current ones.
        """
        if field_values.shape != self._field_values.shape:
            raise ValueError(
                "New `field_values` must have the same shape as the current ones, "
                f"expected {self._field_values.shape}, but got {field_values.shape}."
            )
        self._field_values = field_values

    def evaluate_field_texture(self, width: int, height: int) -> "mi.TensorXf":
        """
        Evaluates the spline patch for texture mapping.

        The simplest way of visualizing the field values of the spline patch is to
        apply it as a texture to a triangulated surface (supported by Mitsuba).

        This method evaluates the field values at a grid of parametric coordinates
            [step_w / 2, width - step_w / 2] x [step_h / 2, height - step_h / 2],
        where `step_w` and `step_h` are the "pixel sizes" in the u and v directions.
        This ensures that the center of each pixel is evaluated and that the
        linear interpolation of the texture is smooth.

        Note that taking this single sample captures the value of the spline field
        values only accuracy O(step^2). This is typically sufficient, as linear
        interpolation is also a second-order approximation, but this under-integration
        must be kept in mind if it ever does become an issue. Gauss-Legendre quadrature
        can be used to improve the accuracy if necessary, but unfortunately Mitsuba
        does not support bicubic interpolation of textures, which limits the
        high-order accuracy.

        Computations with a typical spline patch have an error of O(h^{p+1}), where h
        is the size of the largest knot span and p is the degree of the polynomial basis
        functions used to define the spline. Thus, to achieve comparable error,
        assuming that there are n knot spans and degree 3 splines are used, the minimum
        resolution should be n^2 along both u and v directions.
        For typical values of n (8 - 32), this means that effectively no error is
        introduced at this step for sufficiently high resolutions.

        :returns: A tensor of shape (width, height, dofs_per_cp) representing the
            evaluated field values for texture mapping.
        """
        (u_min, u_max), (v_min, v_max) = self.parametric_domain_size
        step_u = (u_max - u_min) / width
        step_v = (v_max - v_min) / height

        width_positions = dr.ravel(
            dr.linspace(
                dtype=mi.TensorXf,
                start=u_min + 0.5 * step_u,
                stop=u_max - 0.5 * step_u,
                num=width,
            )
        )
        height_positions = dr.ravel(
            dr.linspace(
                dtype=mi.TensorXf,
                start=v_min + 0.5 * step_v,
                stop=v_max - 0.5 * step_v,
                num=height,
            )
        )

        u_coords, v_coords = dr.meshgrid(
            width_positions, height_positions, indexing="xy"
        )
        parametric_coords = dr.empty(dtype=mi.TensorXf, shape=(width * height, 2))
        parametric_coords[:, 0] = u_coords
        parametric_coords[:, 1] = v_coords

        evaluated_texture = self.evaluate_field(parametric_coords)
        evaluated_texture = dr.reshape(
            dtype=mi.TensorXf,
            value=dr.ravel(evaluated_texture),
            shape=(width, height, self._field_values.shape[1]),
        )

        return evaluated_texture

    def visualize(
        self,
        num_points_u: int = 32,
        num_points_v: int = 32,
        visualize_field: bool = True,
        visualize_control_grid: bool = False,
    ) -> None:
        """
        Visualize the IGA spline patch as a 3D surface.

        `num_points_u` and`num_points_v` define the number of points to evaluate in the
        u and v directions, respectively, forming a grid of points.

        Correctness of splines cannot be easily verified by hand, so a visualization
        is provided to help with debugging and create a visual representation of the
        surface.

        If `visualize_field` is True, the field values will be used to color the
        surface. Only a single channel field is supported for visualization, and it will
        be visualized using the 'turbo' colormap.

        If `visualize_control_grid` is True, the control points and the control grid
        will be overlaid on the surface. The control points are shown as red dots, and
        the control grid is shown as red lines connecting the control points.

        :raises ValueError: If `num_points_u` or `num_points_v` is less than 2, or if
            the field values have more than one channel when `visualize_field` is True.
        """
        if num_points_u < 2 or num_points_v < 2:
            raise ValueError(
                "`num_points_u` and `num_points_v` must be greater than 1."
            )
        if visualize_field and self._field_values.shape[1] > 1:
            raise ValueError(
                f"Field visualization only supports 1 channel, but got "
                f"{self._field_values.shape[1]} channels."
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

        uv_params = dr.zeros(
            dtype=mi.TensorXf, shape=((num_points_u + 1) * (num_points_v + 1), 2)
        )
        uv_params[:, 0] = dr.ravel(u_grid)
        uv_params[:, 1] = dr.ravel(v_grid)

        # Evaluate points and control grid
        points = self.evaluate_coordinates(uv_params)
        points_np = points.numpy().reshape(num_points_v + 1, num_points_u + 1, 3)
        control_points_np = self.control_points.numpy()

        fig = plt.figure(figsize=(10, 8), dpi=150)
        ax = fig.add_subplot(111, projection="3d")

        # Default plotting params
        facecolors = None
        cmap = None
        clim = None  # for colorbar limits

        if visualize_field:
            field_values = self.evaluate_field(uv_params)
            field_np = field_values.numpy()

            raw = field_np.reshape(num_points_v + 1, num_points_u + 1)
            raw_averaged = 0.25 * (
                raw[:-1, :-1] + raw[1:, :-1] + raw[:-1, 1:] + raw[1:, 1:]
            )

            field_min, field_max = raw_averaged.min(), raw_averaged.max()
            normed = (raw_averaged - field_min) / (field_max - field_min + 1e-12)
            cmap = plt.get_cmap("turbo")
            # rgba shape: (nv, nu, 4)
            facecolors = cmap(normed)
            clim = (field_min, field_max)

        _ = ax.plot_surface(
            points_np[:, :, 0],
            points_np[:, :, 1],
            points_np[:, :, 2],
            facecolors=facecolors,
            cmap=None if facecolors is not None else "viridis",
            shade=False,
            linewidth=0,
            antialiased=True,
        )

        # Add colorbar for single-channel field
        if visualize_field and facecolors is not None and cmap is not None:
            assert clim is not None
            sm = plt.cm.ScalarMappable(
                cmap=cmap,
                norm=plt.Normalize(vmin=clim[0], vmax=clim[1]),
            )
            sm.set_array([])
            fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5)

        # Overlay control grid if requested
        if visualize_control_grid:
            ax.scatter(
                control_points_np[:, 0],
                control_points_np[:, 1],
                control_points_np[:, 2],
                color="r",
                s=10,
                label="Control Points",
            )

            cp_grid = self.control_points_grid
            for i in range(cp_grid.shape[0]):
                ax.plot(
                    cp_grid[i, :, 0],
                    cp_grid[i, :, 1],
                    cp_grid[i, :, 2],
                    "r-",
                    linewidth=0.5,
                    alpha=0.5,
                )
            for j in range(cp_grid.shape[1]):
                ax.plot(
                    cp_grid[:, j, 0],
                    cp_grid[:, j, 1],
                    cp_grid[:, j, 2],
                    "r-",
                    linewidth=0.5,
                    alpha=0.5,
                )

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

        # Get spline information from the underlying interpolating spline
        title = f"Visualization of {self.name} "

        if visualize_field:
            title += "\nScalar Field Visualization"
        ax.set_title(title)

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
