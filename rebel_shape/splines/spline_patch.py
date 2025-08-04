import abc
from typing import Any

import drjit as dr
import mitsuba as mi

from rebel_shape.meshing.patch import Patch
from rebel_shape.meshing.rectangular_mesh import RectangularMesh
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from rebel_shape.splines.coordinate_spline import CoordinateSpline


class SplinePatch(CoordinateSpline, Patch):
    """
    Abstract base class for spline patches in 3D space.

    Extends the `CoordinateSpline` class to represent a patch defined by two
    parametric directions (u and v). To correctly optimize the patch, a separate
    `evaluate_normals` method is provided to compute the normals at the given
    parametric coordinates.

    It also inherits from `Patch`, allowing it to be used in composite spline
    surfaces and other mesh structures.
    """

    @property
    @abc.abstractmethod
    def control_points_grid(self) -> "mi.TensorXf":
        """
        A 3D tensor of shape (n_control_points_v, n_control_points_u, 3) representing
        the control points of the spline patch in 3D space.

        :returns: A tensor containing the control points of the spline patch.
        :raises NotImplementedError: If a child class does not implement this property.
        """
        raise NotImplementedError(
            "Classes inheriting from `SplinePatch` need to implement the "
            "`control_points_grid` property."
        )

    @abc.abstractmethod
    def evaluate_normals(self, parametric_coords: "mi.TensorXf") -> "mi.TensorXf":
        """
        Evaluate the normals of the spline patch at the given parametric coordinates.

        A 2D tensor of shape (n_points, 2) representing the parametric coordinates
        at which the normals are evaluated.

        :returns: A tensor of shape (n_points, 3) representing the normal vectors
                  at the specified parametric coordinates.
        :raises NotImplementedError: If a child class does not implement this property.
        """
        raise NotImplementedError(
            "Classes inheriting from `SplinePatch` need to implement the "
            "`evaluate_normals` method."
        )

    @abc.abstractmethod
    def evaluate_jacobian(self, parametric_coords: "mi.TensorXf") -> "mi.TensorXf":
        """
        Evaluate the Jacobian of the spline patch at the given `parametric_coords`.

        The Jacobian is a tensor that describes how the surface changes with respect
        to the parametric coordinates. It is useful for computing derivatives and
        understanding the local geometry of the surface.

        :returns: A tensor of shape (n_points, 3, 2) representing the Jacobian at
                  the specified parametric coordinates.
        """
        raise NotImplementedError(
            "Classes inheriting from `SplinePatch` need to implement the "
            "`evaluate_jacobian` method."
        )

    @staticmethod
    def metric_tensor(jacobian: "mi.TensorXf") -> "mi.TensorXf":
        """
        Compute the metric tensor from the `jacobian`.

        The Jacobian is a tensor of shape (n_jacobians, 3, 2) where each slice
        represents the partial derivatives of the surface with respect to the
        parametric coordinates (u, v). It was evaluated at the given parametric
        coordinates.

        The metric tensor is derived from the Jacobian as J^T @ J and is used to
        compute lengths, angles, and areas on the surface. From the definition, it
        follows that it is a symmetric 2x2 matrix for each point on the patch.

        :returns: A tensor of shape (n_jacobians, 2, 2) representing the metric tensor
            from the provided `jacobian` tensor.

        :raises ValueError: If the shape of `jacobian` is not (n_jacobians, 3, 2).
        """
        if len(jacobian.shape) != 3 or jacobian.shape[1] != 3 or jacobian.shape[2] != 2:
            raise ValueError(
                "Jacobian must be a 3D tensor of shape (n_jacobians, 3, 2)."
            )

        metric = dr.empty(dtype=mi.TensorXf, shape=(jacobian.shape[0], 2, 2))

        metric[:, 0, 0] = dr.ravel(dr.sum(jacobian[:, :, 0] ** 2, axis=1))
        metric[:, 1, 1] = dr.ravel(dr.sum(jacobian[:, :, 1] ** 2, axis=1))

        metric[:, 0, 1] = dr.ravel(
            dr.sum(jacobian[:, :, 0] * jacobian[:, :, 1], axis=1)
        )
        metric[:, 1, 0] = dr.ravel(metric[:, 0, 1])

        return metric

    @staticmethod
    def inv_metric_tensor(metric: "mi.TensorXf") -> tuple["mi.TensorXf", "mi.TensorXf"]:
        """
        Compute the inverse of the metric tensor.

        The `metric` tensor is a 2D tensor of shape (n_points, 2, 2), expected to be
        computed by the `metric_tensor` method. It represents the local geometry of the
        spline patch at the given parametric coordinates.

        This method computes the inverse of the metric tensor and returns it along with
        its determinant as it is a useful intermediate result. The inverse is computed
        using the formula for the inverse of a 2x2 matrix, noting that the 2x2
        metric tensor is symmetric.

        :returns: A tuple of two Mitsuba tensors:

            - `inv_metric`: A tensor of shape (n_points, 2, 2) representing
                the inverse of the metric tensor.
            - `det_metric`: A tensor of shape (n_points,) representing the
                determinant of the metric tensor at each point. This is a useful
                intermediate value that is necessary for computing the inverse.

        :raises ValueError: If the shape of `metric` is not (n_points, 2, 2).
        """
        if len(metric.shape) != 3 or metric.shape[1] != 2 or metric.shape[2] != 2:
            raise ValueError(
                "Metric tensor must be a 3D tensor of shape (n_points, 2, 2)."
            )

        det_metric = (
            metric[:, 0, 0] * metric[:, 1, 1] - metric[:, 0, 1] * metric[:, 1, 0]
        )
        inv_metric = dr.empty(dtype=mi.TensorXf, shape=metric.shape)

        inv_metric[:, 0, 0] = dr.ravel(metric[:, 1, 1] / det_metric)
        inv_metric[:, 1, 1] = dr.ravel(metric[:, 0, 0] / det_metric)
        inv_metric[:, 0, 1] = dr.ravel(-metric[:, 0, 1] / det_metric)
        inv_metric[:, 1, 0] = dr.ravel(-metric[:, 1, 0] / det_metric)

        return inv_metric, det_metric

    @property
    @abc.abstractmethod
    def parametric_domain_size(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        A tuple representing the parametric domain size of the spline patch.

        Since it represents a 2D surface, it returns a pair of tuples:
        (min_knot, max_knot) for u and v directions, respectively. Each tuple contains
        the minimum and maximum values of the parametric coordinates in that direction.

        :returns: Tuple of tuples representing the parametric domain size:
                  ((u_min, u_max), (v_min, v_max)).
        :raises NotImplementedError: If a child class does not implement this property.
        """
        raise NotImplementedError(
            "Classes inheriting from `Spline` need to implement the "
            "`parametric_domain_size` property."
        )

    @abc.abstractmethod
    def __getstate__(self) -> dict[str, Any]:
        """
        Exports the state of the Patch for serialization.

        All Mitsuba tensors are converted to numpy arrays or lists to ensure
        compatibility with `jsonpickle` serialization.

        :returns: A dictionary representing the state of the object, ready for
                  serialization with `jsonpickle`.
        :raises NotImplementedError: If a child class does not implement this method.
        """
        raise NotImplementedError(
            "Classes inheriting from `SplinePatch` need to implement the "
            "`__getstate__` method."
        )

    @abc.abstractmethod
    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restore the B-spline surface from a serialized state.

        `state` is a dictionary that contains the serialized state of the object
        including values of `control_points and `quads` which are converted back to
        Mitsuba tensors.

        :raises NotImplementedError: If a child class does not implement this method.
        """
        raise NotImplementedError(
            "Classes inheriting from `SplinePatch` need to implement the "
            "`__setstate__` method."
        )

    @abc.abstractmethod
    def h_refinement(self, factor: int = 2) -> None:
        """
        Increase the resolution of the patch in the u and v directions by `factor`.

        This method should be implemented by subclasses to refine the patch
        by increasing the number of knot spans in both parametric directions by
        the specified factor. The refinement process has to update the control points
        to maintain the spline's shape and continuity.

        :raises ValueError: If `factor` is not a positive integer.
        :raises NotImplementedError: If a child class does not implement this method.
        """
        raise NotImplementedError(
            "Classes inheriting from `SplinePatch` need to implement the "
            "`h_refinement` method."
        )

    @property
    @abc.abstractmethod
    def quads(self) -> "mi.TensorXu":
        """
        Returns the indices of the quads that make up the spline patch.

        This method should be implemented by subclasses to provide the indices
        of the quads in the tessellated mesh representation of the spline patch.

        :returns: A tensor of shape (n_quads, 4) where each row contains the indices
                  of the vertices that form a quad face of the spline patch.
        :raises NotImplementedError: If a child class does not implement this property.
        """
        raise NotImplementedError(
            "Classes inheriting from `SplinePatch` need to implement the "
            "`quads` property."
        )

    def texcoords_to_parametric_coords(self, texcoords: "mi.TensorXf") -> "mi.TensorXf":
        """
        Utility method to convert texture coordinates of a mesh to coordinates
        in the parametric domain of the spline patch.

        The spline patch is typically tessellated into a grid of points and is
        evaluated at texcoords scaled to cover the entire parametric domain.

        One can also use this method to convert parametric sampling from range
        [0, 1] to the actual parametric coordinates of the spline patch.

        :returns: A 2D tensor of shape (n_points, 2) representing the parametric
                  coordinates that correspond to
        """
        if len(texcoords.shape) != 2 or texcoords.shape[1] != 2:
            raise ValueError(
                "Texture coordinates must be a 2D tensor of shape (n_points, 2)."
            )
        if not dr.all((texcoords >= 0) & (texcoords <= 1)):
            raise ValueError("Texture coordinates must be in the range [0, 1]. ")

        (u_min, u_max), (v_min, v_max) = self.parametric_domain_size

        parametric_coords = dr.zeros(dtype=mi.TensorXf, shape=texcoords.shape)
        parametric_coords[:, 0] = dr.ravel(u_min + (u_max - u_min) * texcoords[:, 0])
        parametric_coords[:, 1] = dr.ravel(v_min + (v_max - v_min) * texcoords[:, 1])

        return parametric_coords

    def tessellate(
        self, u_subdivisions: int, v_subdivisions: int, bsdf: DiffuseBSDF | None = None
    ) -> RectangularMesh:
        """
        Tessellate the spline surface into a Mitsuba mesh.

        This method creates a regular grid of points in the parametric domain
        of resolution (`u_subdivisions`, `v_subdivisions`). It evaluates the spline
        surface at these points, and constructs a triangulated mesh with vertex normals
        that correspond to the surface normals at those points.

        The number of triangles directly influences how accurately the spline
        surface is represented, so care should be taken to choose values that are
        high enough to capture the surface's detail without being excessively high
        and causing performance issues.

        Texture coordinates are also generated for each vertex, normalized to the range
        [0, 1]. There is a one-to-one mapping between the texture coordinates and the
        parametric coordinates based on `parametric_domain_size`.

        If `bsdf` is `None`, a default `DiffuseBSDF` will be used.

        :returns: `RectangularMesh` object representing the tessellated spline surface.
        :raises ValueError: If the number of subdivisions is less than 1
        """
        # Subdivisions checked in `unit_square` method of RectangularMesh.
        mesh = RectangularMesh.unit_square(
            mesh_name=self.name + "_tessellated",
            width_subdivisions=u_subdivisions,
            height_subdivisions=v_subdivisions,
            bsdf=bsdf,
        )

        parametric_coords = self.texcoords_to_parametric_coords(mesh.texcoords)
        vertex_positions = self.evaluate_coordinates(parametric_coords)
        vertex_normals = self.evaluate_normals(parametric_coords)

        mesh.vertices = vertex_positions
        mesh.vertex_normals = vertex_normals

        return mesh
