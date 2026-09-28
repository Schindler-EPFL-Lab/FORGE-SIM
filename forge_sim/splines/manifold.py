import abc

import drjit as dr
import mitsuba as mi


class Manifold(abc.ABC):
    @abc.abstractmethod
    def evaluate_positions(self, parametric_coords: "mi.TensorXf") -> "mi.TensorXf":
        """
        Evaluate a position on the manifold at the given `parametric_coords`.

        A 2D tensor of shape (n_points, 2) representing the parametric coordinates
        at which the positions are evaluated.

        :returns: A tensor of shape (n_points, 3) representing the positions
                  at the specified parametric coordinates.
        :raises NotImplementedError: If a child class does not implement this property.
        """
        raise NotImplementedError(
            "Classes inheriting from `Manifold` need to implement the "
            + "`evaluate_positions` method."
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
            "Classes inheriting from `Manifold` need to implement the "
            + "`evaluate_normals` method."
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
            "Classes inheriting from `Manifold` need to implement the "
            + "`evaluate_jacobian` method."
        )

    @abc.abstractmethod
    def evaluate_hessian(self, parametric_coords: "mi.TensorXf") -> "mi.TensorXf":
        """
        Evaluate the Hessian of the spline patch at the given `parametric_coords`.

        The Hessian is a tensor that describes how the curvature of the surface changes
        with respect to the parametric coordinates. It is useful for computing
        the second fundamental form and understanding the local geometry of the surface.

        :returns: A tensor of shape (n_points, 3, 2, 2) representing the Hessian at
                  the specified parametric coordinates.
        """
        raise NotImplementedError(
            "Classes inheriting from `Manifold` need to implement the "
            + "`evaluate_hessian` method."
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
        min_det = float(dr.min(dr.abs(det_metric)).numpy())
        if min_det < 1.0e-30:
            raise ValueError(
                f"Degenerate metric tensor detected: min |det(g)| = {min_det:.4e}. "
                "Check the surface geometry for collapsed or degenerate elements."
            )
        inv_metric = dr.empty(dtype=mi.TensorXf, shape=metric.shape)

        inv_metric[:, 0, 0] = dr.ravel(metric[:, 1, 1] / det_metric)
        inv_metric[:, 1, 1] = dr.ravel(metric[:, 0, 0] / det_metric)
        inv_metric[:, 0, 1] = dr.ravel(-metric[:, 0, 1] / det_metric)
        inv_metric[:, 1, 0] = dr.ravel(-metric[:, 1, 0] / det_metric)

        return inv_metric, det_metric

    @staticmethod
    def second_fundamental_form(
        hessian: "mi.TensorXf", normal: "mi.TensorXf"
    ) -> "mi.TensorXf":
        """
        Compute the second fundamental form of the spline patch.
        The second fundamental form defines extrinsic invariants of the surface,
        namely the principal curvatures, which represent the minimum and maximum
        curvatures of the surface.
        At a parametric point (u, v), the second fundamental form is a 2x2 matrix:
        [[L, M],
         [M, N]]
        where
        - L = second partial derivative f_uu(u, v) dotted with n
        - M = mixed partial derivative f_uv(u, v) dotted with n
        - N = second partial derivative f_vv(u, v) dotted with n
        - n is the unit normal vector at (u, v)
        - f: R^2 -> R^3 is the parametric surface

        :returns: Tensor of shape (n_points, 2, 2) representing the
            second fundamental form.

        :raises ValueError: If the shape of `hessian` is not (n_points, 3, 2, 2) or
            if the shape of `normal` is not (n_points, 3).
        """
        if (
            len(hessian.shape) != 4
            or hessian.shape[1] != 3
            or hessian.shape[2] != 2
            or hessian.shape[3] != 2
        ):
            raise ValueError(
                "Hessian must be a 4D tensor of shape (n_points, 3, 2, 2)."
            )
        if (
            len(normal.shape) != 2
            or normal.shape[1] != 3
            or normal.shape[0] != hessian.shape[0]
        ):
            raise ValueError("Normal must be a 2D tensor of shape (n_points, 3).")

        n_points = hessian.shape[0]
        second_form = dr.empty(dtype=mi.TensorXf, shape=(n_points, 2, 2))

        for i in range(2):
            for j in range(2):
                second_form[:, i, j] = dr.ravel(
                    dr.sum(hessian[:, :, i, j] * normal, axis=1)
                )

        return second_form

    @staticmethod
    def principal_curvatures(
        inv_metric_tensor: "mi.TensorXf", second_form: "mi.TensorXf"
    ) -> tuple["mi.TensorXf", "mi.TensorXf"]:
        """
        Compute the principal curvatures at each point on the spline patch based on
        inverse metric tensor `inv_metric_tensor` (the inverse of the first
        fundamental form) and the second form `second_form`.

        Principal curvatures at a given point of a surface are the maximum and minimum
        values of the curvature, measuring how the surface bends in different
        directions at that point.

        :returns
            tuple[mi.TensorXf, mi.TensorXf]: Two tensors of shape (n_points,),
            containing the maximum and minimum principal curvatures at each point.
        :raises ValueError:
            if `inv_metric_tensor` is not of shape (n_points, 2, 2) or
            if `second_form` is not of shape (n_points, 2, 2) or does not match the
            number of points in `inv_metric_tensor`.
        """

        if (
            len(inv_metric_tensor.shape) != 3
            or inv_metric_tensor.shape[1] != 2
            or inv_metric_tensor.shape[2] != 2
        ):
            raise ValueError(
                "Inverse metric tensor must be a 3D tensor of shape (n_points, 2, 2)."
            )
        if (
            len(second_form.shape) != 3
            or second_form.shape[1] != 2
            or second_form.shape[2] != 2
            or second_form.shape[0] != inv_metric_tensor.shape[0]
        ):
            raise ValueError(
                "Second fundamental form must be a 3D tensor of shape (n_points, 2, 2)."
            )

        n_points = inv_metric_tensor.shape[0]
        # Manual matrix multiplication: S = inv_metric_tensor @ second_form
        shape_operator = dr.empty(dtype=mi.TensorXf, shape=(n_points, 2, 2))

        for i in range(2):
            for j in range(2):
                shape_operator[:, i, j] = dr.ravel(
                    inv_metric_tensor[:, i, 0] * second_form[:, 0, j]
                    + inv_metric_tensor[:, i, 1] * second_form[:, 1, j]
                )

        # Manual eigenvalue calculation for 2x2 symmetric matrix
        a = shape_operator[:, 0, 0]
        b = shape_operator[:, 0, 1]
        c = shape_operator[:, 1, 0]
        d = shape_operator[:, 1, 1]

        trace = a + d
        det = a * d - b * c
        temp = dr.safe_sqrt(0.25 * trace * trace - det)

        eig1 = 0.5 * trace + temp
        eig2 = 0.5 * trace - temp

        return eig1, eig2

    @staticmethod
    def normal_derivative(
        jacobian: "mi.TensorXf", hessian: "mi.TensorXf", normal: "mi.TensorXf"
    ) -> "mi.TensorXf":
        """
        Compute the partial derivatives of the unit surface normal with respect
        to the parametric coordinates, given precomputed geometric quantities.

        The unit normal is computed as
            n = w / ||w||,   w = f_u × f_v,
        with f_u and f_v being the Jacobian columns. Using this normalization, the
        derivatives follow
            n_u = (w_u - n (n · w_u)) / ||w||,
            n_v = (w_v - n (n · w_v)) / ||w||,
        where
            w_u = f_uu x f_v + f_u x f_uv,
            w_v = f_uv x f_v + f_u x f_vv.

        :returns: A tensor of shape (n_points, 3, 2) representing the normal derivative
            at each point on the surface.

        :raises ValueError: If the shapes of `jacobian`, `hessian`, or `normal` do not
            match the expected shapes.
        """
        if len(jacobian.shape) != 3 or jacobian.shape[1] != 3 or jacobian.shape[2] != 2:
            raise ValueError("Jacobian must be a 3D tensor of shape (n_points, 3, 2).")

        if (
            len(hessian.shape) != 4
            or hessian.shape[1] != 3
            or hessian.shape[2] != 2
            or hessian.shape[3] != 2
        ):
            raise ValueError(
                "Hessian must be a 4D tensor of shape (n_points, 3, 2, 2)."
            )
        if (
            len(normal.shape) != 2
            or normal.shape[1] != 3
            or normal.shape[0] != jacobian.shape[0]
        ):
            raise ValueError("Normal must be a 2D tensor of shape (n_points, 3).")

        def cross(a: "mi.TensorXf", b: "mi.TensorXf") -> "mi.TensorXf":
            out = dr.empty(dtype=mi.TensorXf, shape=a.shape)
            out[:, 0] = dr.ravel(a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1])
            out[:, 1] = dr.ravel(a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2])
            out[:, 2] = dr.ravel(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
            return out

        du = jacobian[:, :, 0]
        dv = jacobian[:, :, 1]

        w = cross(du, dv)
        magnitude = dr.safe_sqrt(dr.sum(w * w, axis=1))
        safe_magnitude = dr.select(
            magnitude > 1.0e-6,
            magnitude,
            dr.full(dtype=mi.TensorXf, value=1.0, shape=magnitude.shape),
        )
        inv_safe_magnitude = 1.0 / safe_magnitude

        w_u = cross(hessian[:, :, 0, 0], dv) + cross(du, hessian[:, :, 0, 1])
        w_v = cross(hessian[:, :, 0, 1], dv) + cross(du, hessian[:, :, 1, 1])

        proj_u = dr.sum(normal * w_u, axis=1)
        proj_v = dr.sum(normal * w_v, axis=1)

        normal_deriv = dr.empty(dtype=mi.TensorXf, shape=(jacobian.shape[0], 3, 2))

        normal_deriv[:, :, 0] = dr.ravel(
            inv_safe_magnitude[:, None] * (w_u - normal * proj_u[:, None])
        )
        normal_deriv[:, :, 1] = dr.ravel(
            inv_safe_magnitude[:, None] * (w_v - normal * proj_v[:, None])
        )

        return normal_deriv
