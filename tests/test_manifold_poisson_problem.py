import unittest

import drjit as dr
import mitsuba as mi
import numpy as np
import numpy.typing as npt
from scipy.sparse import linalg

from rebel_shape.isogeometric_analysis.manifold_poisson_problem import (
    ManifoldPoissonProblem,
)
from rebel_shape.meshing.patch import Patch
from rebel_shape.splines.bspline import BSpline
from rebel_shape.splines.bspline_patch import BSplinePatch
from rebel_shape.splines.iga_spline_patch import IGASplinePatch


def compute_and_print_symbolic_forcing_term() -> None:
    """
    Computes and prints the forcing term that has the solution
        u_hat(u, v) = sin(pi * u) * sin(pi * v)
    for a degree 2 Bezier patch with control points such that the middle control point
    has coordinates (0.5, 0.5, 4.0).

    The forcing term is computed symbolically using SymPy and is expected to be used
    in the ManifoldPoissonProblem for testing purposes. Note that this function
    is not part of the unit tests because it takes over 30 seconds to evaluate.
    Nonetheless, it is provided as a reference to understand how the very long forcing
    term from `test_poisson_manifold` was derived.
    """
    import sympy as sp

    # Define symbols
    u, v = sp.symbols("u v", real=True)

    # Geometry when the middle control point of a degree 2 bezier patch has coordinate
    # (0.5, 0.5, 4.0)
    z = 16 * u * (1 - u) * v * (1 - v)
    zu = sp.diff(z, u)
    zv = sp.diff(z, v)

    # Metric components
    g11 = 1 + zu**2
    g22 = 1 + zv**2
    g12 = zu * zv
    g = g11 * g22 - g12**2

    # Inverse metric
    g_inv_11 = g22 / g
    g_inv_12 = -g12 / g
    g_inv_22 = g11 / g

    # Manufactured solution u_hat
    u_hat = sp.sin(sp.pi * u) * sp.sin(sp.pi * v)
    u_hat_u = sp.diff(u_hat, u)
    u_hat_v = sp.diff(u_hat, v)

    # Flux components q^1, q^2
    sqrt_g = sp.sqrt(g)
    q1 = sqrt_g * (g_inv_11 * u_hat_u + g_inv_12 * u_hat_v)
    q2 = sqrt_g * (g_inv_12 * u_hat_u + g_inv_22 * u_hat_v)

    # Compute the right-hand side f = -Δ_M u_hat
    f_expr = -1 / sqrt_g * (sp.diff(q1, u) + sp.diff(q2, v))
    f_simplified = sp.simplify(f_expr)

    print(f_simplified)


class TestManifoldPoissonProblem(unittest.TestCase):
    def setUp(self) -> None:
        mi.set_variant("llvm_ad_rgb")

        self.bezier_2_u_knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        self.bezier_2_v_knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        self.bezier_2_control_points = mi.TensorXf(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.5, 0.0],
                [1.0, 0.5, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        self.bezier_2_u_spline = BSpline(knot_vector=self.bezier_2_u_knots, degree=2)
        self.bezier_2_v_spline = BSpline(knot_vector=self.bezier_2_v_knots, degree=2)
        self.bezier_2_surface = BSplinePatch(
            name="bezier_2_surface",
            u_spline=self.bezier_2_u_spline,
            v_spline=self.bezier_2_v_spline,
            control_points=self.bezier_2_control_points,
        )

        self.bezier_2_field_values = dr.zeros(
            dtype=mi.TensorXf,
            shape=(self.bezier_2_surface.interpolating_spline.num_basis_functions, 1),
        )
        self.bezier_2_iga_surface = IGASplinePatch(
            spline_patch=self.bezier_2_surface, field_values=self.bezier_2_field_values
        )

    def test_poisson_constant(self):
        """
        Tests the Poisson problem on a flat surface with a constant forcing term.

        While this problem can be solved analytically, there remains a Fourier series
        representation of the solution that is not trivial to compute. Therefore, we
        use a successive over-relaxation (SOR) red-black method to solve the
        problem with finite differences and compare the result against the
        IGA solution.
        """
        poisson = ManifoldPoissonProblem(spline_surface=self.bezier_2_iga_surface)
        poisson._spline.h_refinement(10)

        constant_term = dr.ones(
            dtype=mi.TensorXf,
            shape=(poisson._spline.interpolating_spline.num_basis_functions, 1),
        )
        local_indices, local_matrices = poisson.assemble_local_matrices()
        global_matrix = poisson.assemble_global_matrix(
            local_indices=local_indices, local_matrices=local_matrices
        )

        local_indices, local_vectors = poisson.assemble_local_vectors(
            rhs_term=constant_term
        )
        global_vector = poisson.assemble_global_vector(
            local_indices=local_indices, local_vectors=local_vectors
        )

        left_edge = poisson._spline.get_node_indices_for(side=Patch.Side.LEFT)
        right_edge = poisson._spline.get_node_indices_for(side=Patch.Side.RIGHT)
        bottom_edge = poisson._spline.get_node_indices_for(side=Patch.Side.BOTTOM)
        top_edge = poisson._spline.get_node_indices_for(side=Patch.Side.TOP)

        boundary_indices = np.concatenate(
            [
                left_edge.numpy(),
                right_edge.numpy(),
                bottom_edge.numpy(),
                top_edge.numpy(),
            ]
        )
        boundary_indices = np.unique(boundary_indices)

        all_dofs = np.arange(global_vector.shape[0], dtype=np.int32)
        active_dofs = np.setdiff1d(all_dofs, boundary_indices)

        solution = linalg.cg(
            A=global_matrix[active_dofs, :][:, active_dofs],
            b=global_vector[active_dofs],
        )

        full_solution = np.zeros(global_vector.shape, dtype=np.float64)
        full_solution[active_dofs] = solution[0]

        reference_solution = self._poisson_fd_sor_red_black()  # (65, 65)

        np.testing.assert_allclose(
            np.max(full_solution), np.max(reference_solution), rtol=3e-4
        )

    def test_poisson_sine(self):
        """
        Tests the Poisson problem on a flat surface wih a forcing term that is the
        eigenfunction of the Laplace operator on the surface. This problem has an
        analytical solution that is easy to compare against to verify the
        implementation.
        """
        poisson = ManifoldPoissonProblem(spline_surface=self.bezier_2_iga_surface)
        poisson.spline.h_refinement(16)

        def forcing_term_flat(parametric_coords: "mi.TensorXf") -> "mi.TensorXf":
            return dr.sin(dr.pi * parametric_coords[:, 0]) * dr.sin(
                dr.pi * parametric_coords[:, 1]
            )

        values = poisson.spline.interpolating_spline.project_function(
            func=forcing_term_flat
        )

        local_indices, local_matrices = poisson.assemble_local_matrices()
        global_matrix = poisson.assemble_global_matrix(
            local_indices=local_indices, local_matrices=local_matrices
        )

        local_indices, local_vectors = poisson.assemble_local_vectors(rhs_term=values)
        global_vector = poisson.assemble_global_vector(
            local_indices=local_indices, local_vectors=local_vectors
        )

        left_edge = poisson.spline.get_node_indices_for(side=Patch.Side.LEFT)
        right_edge = poisson.spline.get_node_indices_for(side=Patch.Side.RIGHT)
        bottom_edge = poisson.spline.get_node_indices_for(side=Patch.Side.BOTTOM)
        top_edge = poisson.spline.get_node_indices_for(side=Patch.Side.TOP)

        boundary_indices = np.concatenate(
            [
                left_edge.numpy(),
                right_edge.numpy(),
                bottom_edge.numpy(),
                top_edge.numpy(),
            ]
        )
        boundary_indices = np.unique(boundary_indices)

        all_dofs = np.arange(global_vector.shape[0], dtype=np.int32)
        active_dofs = np.setdiff1d(all_dofs, boundary_indices)

        solution = linalg.cg(
            A=global_matrix[active_dofs, :][:, active_dofs],
            b=global_vector[active_dofs],
        )

        full_solution = np.zeros(global_vector.shape, dtype=np.float64)
        full_solution[active_dofs] = solution[0]

        poisson._spline.field_values = mi.TensorXf(full_solution[:, None])

        # The forcing term is an eigenfunction of the Laplacian, so the exact solution
        # has the same form scaled by 1 / (2 * pi^2).
        def func(coords: "mi.TensorXf") -> "mi.TensorXf":
            return forcing_term_flat(coords) / (2.0 * dr.pi**2)

        sol = poisson._spline.interpolating_spline.project_function(func)

        np.testing.assert_allclose(np.max(full_solution), np.max(sol), rtol=1e-5)

    def test_poisson_manifold(self):
        """
        Tests the ManifoldPoissonProblem on a degree 2 Bezier patch with a middle
        control point (0.5, 0.5, 4.0), making sure that the curvature is accounted for
        correctly in the assembly of the linear system.

        The forcing term is designed such that the solution of the problem is
        u_hat(u, v) = sin(pi * u) * sin(pi * v).
        This was done by calling the function `compute_and_print_symbolic_forcing_term`
        and copying its output into the `forcing_term` function below. Calling the
        function inside of tests would be too slow, as it takes over 30 seconds
        to evaluate the symbolic expression.
        """
        updated_cps = self.bezier_2_control_points
        updated_cps[4, 2] = 4.0  # Modify a control point to extrude the surface
        self.bezier_2_iga_surface.control_points = updated_cps

        poisson = ManifoldPoissonProblem(spline_surface=self.bezier_2_iga_surface)
        poisson._spline.h_refinement(12)

        values = poisson.spline.interpolating_spline.project_function(
            func=self._forcing_term_manifold
        )

        # Tensor assembly
        local_indices, local_matrices = poisson.assemble_local_matrices()
        global_matrix = poisson.assemble_global_matrix(
            local_indices=local_indices, local_matrices=local_matrices
        )
        local_indices, local_vectors = poisson.assemble_local_vectors(rhs_term=values)
        global_vector = poisson.assemble_global_vector(
            local_indices=local_indices, local_vectors=local_vectors
        )

        # Boundary condition setup
        left_edge = poisson._spline.get_node_indices_for(side=Patch.Side.LEFT)
        right_edge = poisson._spline.get_node_indices_for(side=Patch.Side.RIGHT)
        bottom_edge = poisson._spline.get_node_indices_for(side=Patch.Side.BOTTOM)
        top_edge = poisson._spline.get_node_indices_for(side=Patch.Side.TOP)

        boundary_indices = np.concatenate(
            [
                left_edge.numpy(),
                right_edge.numpy(),
                bottom_edge.numpy(),
                top_edge.numpy(),
            ]
        )
        boundary_indices = np.unique(boundary_indices)

        all_dofs = np.arange(global_vector.shape[0], dtype=np.int32)
        active_dofs = np.setdiff1d(all_dofs, boundary_indices)

        # Solve linear system
        solution = linalg.cg(
            A=global_matrix[active_dofs, :][:, active_dofs],
            b=global_vector[active_dofs],
            rtol=1e-6,
        )

        # Process solution
        full_solution = np.zeros(global_vector.shape, dtype=np.float64)
        full_solution[active_dofs] = solution[0]
        poisson.spline.field_values = mi.TensorXf(full_solution.reshape(-1, 1))

        def func(parametric_coords: "mi.TensorXf") -> "mi.TensorXf":
            return dr.sin(dr.pi * parametric_coords[:, 0]) * dr.sin(
                dr.pi * parametric_coords[:, 1]
            )

        # Validate solution against expected analytical solution
        expected_sol = poisson.spline.interpolating_spline.project_function(func=func)
        np.testing.assert_allclose(
            np.max(full_solution), np.max(expected_sol), rtol=1e-3
        )

    @staticmethod
    def _poisson_fd_sor_red_black(
        n: int = 65, max_iter: int = 500, tol: float = 1e-5
    ) -> npt.NDArray[np.float64]:
        """
        Finite difference solution of the Poisson equation on a flat surface
        [0, 1] x [0, 1] with a forcing term f = 1 using the successive
        over-relaxation (SOR) method with red-black ordering and optimal
        relaxation parameter.

        This kind of high-performance style implementation is used to obtain a
        reasonably accurate solution in a short amount of time, which is useful in
        the time constrained testing environment.

        The finite difference five point stencil approximation of the Laplacian yields:
            u_{i, j}^{k+1} = (1 - omega) u_{i, j}^{k} + omega * 0.25 * (
                u_{i+1, j}^{k} + u_{i-1, j}^{k} + u_{i, j+1}^{k} + u_{i, j-1}^{k}
                + h^2 * f_{i, j}
            )
        For numerical stability reasons, the grid has to be colored in a red-black
        pattern, where the red points are updated first, followed by the black points.
        So there is actually an intermediate state after the red points are updated,
        which is not indicated in the formula above.

        The iterations stop when the maximum change in the solution is less than the
        specified tolerance `tol` or when the maximum number of iterations `max_iter`
        is reached. Typically, it takes about 2 * n iterations to converge to a
        tolerance of 1e-5.

        :returns: A numpy array of shape (n, n) containing the solution u.
        """
        u = np.zeros((n, n))
        f = np.ones((n, n))
        omega = 2.0 / (1.0 + np.sin(np.pi / (n + 1)))
        h2 = 1.0 / (n - 1) ** 2

        for _ in range(max_iter):
            u_old = u.copy()
            # Update red points
            u[1:-1:2, 1:-1:2] = (1 - omega) * u[1:-1:2, 1:-1:2] + omega * 0.25 * (
                u[2::2, 1:-1:2]
                + u[:-2:2, 1:-1:2]
                + u[1:-1:2, 2::2]
                + u[1:-1:2, :-2:2]
                + h2 * f[1:-1:2, 1:-1:2]
            )
            u[2:-1:2, 2:-1:2] = (1 - omega) * u[2:-1:2, 2:-1:2] + omega * 0.25 * (
                u[3::2, 2:-1:2]
                + u[1:-2:2, 2:-1:2]
                + u[2:-1:2, 3::2]
                + u[2:-1:2, 1:-2:2]
                + h2 * f[2:-1:2, 2:-1:2]
            )

            # Update black points similarly
            u[1:-1:2, 2:-1:2] = (1 - omega) * u[1:-1:2, 2:-1:2] + omega * 0.25 * (
                u[2::2, 2:-1:2]
                + u[:-2:2, 2:-1:2]
                + u[1:-1:2, 3::2]
                + u[1:-1:2, 1:-2:2]
                + h2 * f[1:-1:2, 2:-1:2]
            )
            u[2:-1:2, 1:-1:2] = (1 - omega) * u[2:-1:2, 1:-1:2] + omega * 0.25 * (
                u[3::2, 1:-1:2]
                + u[1:-2:2, 1:-1:2]
                + u[2:-1:2, 2::2]
                + u[2:-1:2, :-2:2]
                + h2 * f[2:-1:2, 1:-1:2]
            )

            if np.linalg.norm(u - u_old, ord=np.inf) < tol:
                break

        return u

    @staticmethod
    def _forcing_term_manifold(parametric_coords: "mi.TensorXf") -> "mi.TensorXf":
        """
        The forcing term that makes the solution of the Poisson problem equal to
            u_hat(u, v) = sin(pi * u) * sin(pi * v)
        for a degree 2 Bezier patch with control points such that the middle control
        point has coordinates (0.5, 0.5, 4.0). This results in a parametric surface
            z(u, v) = 4 * u * (1 - u) * 4 * v * (1 - v).

        `parametric_coords` are a tensor of shape (n, 2) where the first column
        corresponds to the u parameter and the second column corresponds to the v
        parameter of the surface.

        The forcing term is derived by analytically computing the manifold Laplacian
        applied to the solution u_hat. This computation is too long to be carried out
        by hand so it was computed with SymPy and copied into this function. The
        function that generates the term is `compute_and_print_symbolic_forcing_term`.

        :returns: A tensor of shape (n, 1) containing the forcing term evaluated at
            the `parametric_coords`.
        """
        u = parametric_coords[:, 0]
        v = parametric_coords[:, 1]

        return (
            dr.pi
            * (
                (
                    -65536
                    * u**2
                    * v**2
                    * (1 - 2 * u) ** 2
                    * (1 - u) ** 2
                    * (1 - 2 * v) ** 2
                    * (1 - v) ** 2
                    + (256 * u**2 * (1 - u) ** 2 * (1 - 2 * v) ** 2 + 1)
                    * (256 * v**2 * (1 - 2 * u) ** 2 * (1 - v) ** 2 + 1)
                )
                * (
                    256
                    * dr.pi
                    * u
                    * v
                    * (1 - 2 * u)
                    * (1 - u)
                    * (1 - 2 * v)
                    * (1 - v)
                    * dr.cos(dr.pi * u)
                    * dr.cos(dr.pi * v)
                    - 512
                    * u
                    * v
                    * (1 - 2 * u)
                    * (1 - u)
                    * (1 - v)
                    * dr.sin(dr.pi * v)
                    * dr.cos(dr.pi * u)
                    + 256
                    * u
                    * (1 - u)
                    * (1 - 2 * v)
                    * (u * v - u * (1 - v) - v * (1 - u) + (1 - u) * (1 - v))
                    * dr.sin(dr.pi * v)
                    * dr.cos(dr.pi * u)
                    - 512
                    * v
                    * (1 - 2 * u)
                    * (1 - v)
                    * (u * v - u * (1 - v) - v * (1 - u) + (1 - u) * (1 - v))
                    * dr.sin(dr.pi * u)
                    * dr.cos(dr.pi * v)
                    + dr.pi
                    * (256 * v**2 * (1 - 2 * u) ** 2 * (1 - v) ** 2 + 1)
                    * dr.sin(dr.pi * u)
                    * dr.sin(dr.pi * v)
                )
                + (
                    -65536
                    * u**2
                    * v**2
                    * (1 - 2 * u) ** 2
                    * (1 - u) ** 2
                    * (1 - 2 * v) ** 2
                    * (1 - v) ** 2
                    + (256 * u**2 * (1 - u) ** 2 * (1 - 2 * v) ** 2 + 1)
                    * (256 * v**2 * (1 - 2 * u) ** 2 * (1 - v) ** 2 + 1)
                )
                * (
                    256
                    * dr.pi
                    * u
                    * v
                    * (1 - 2 * u)
                    * (1 - u)
                    * (1 - 2 * v)
                    * (1 - v)
                    * dr.cos(dr.pi * u)
                    * dr.cos(dr.pi * v)
                    - 512
                    * u
                    * v
                    * (1 - u)
                    * (1 - 2 * v)
                    * (1 - v)
                    * dr.sin(dr.pi * u)
                    * dr.cos(dr.pi * v)
                    - 512
                    * u
                    * (1 - u)
                    * (1 - 2 * v)
                    * (u * v - u * (1 - v) - v * (1 - u) + (1 - u) * (1 - v))
                    * dr.sin(dr.pi * v)
                    * dr.cos(dr.pi * u)
                    + 256
                    * v
                    * (1 - 2 * u)
                    * (1 - v)
                    * (u * v - u * (1 - v) - v * (1 - u) + (1 - u) * (1 - v))
                    * dr.sin(dr.pi * u)
                    * dr.cos(dr.pi * v)
                    + dr.pi
                    * (256 * u**2 * (1 - u) ** 2 * (1 - 2 * v) ** 2 + 1)
                    * dr.sin(dr.pi * u)
                    * dr.sin(dr.pi * v)
                )
                + (
                    256
                    * u
                    * v
                    * (1 - 2 * u)
                    * (1 - u)
                    * (1 - 2 * v)
                    * (1 - v)
                    * dr.sin(dr.pi * u)
                    * dr.cos(dr.pi * v)
                    + (-256 * u**2 * (1 - u) ** 2 * (1 - 2 * v) ** 2 - 1)
                    * dr.sin(dr.pi * v)
                    * dr.cos(dr.pi * u)
                )
                * (
                    u**2
                    * v**2
                    * (1 - u) ** 2
                    * (1 - 2 * v) ** 2
                    * (1 - v) ** 2
                    * (524288 * u - 262144)
                    + 131072
                    * u
                    * v**2
                    * (1 - 2 * u) ** 2
                    * (1 - u)
                    * (1 - 2 * v)
                    * (1 - v) ** 2
                    * (u * v - u * (1 - v) - v * (1 - u) + (1 - u) * (1 - v))
                    - 512
                    * u
                    * (1 - u)
                    * (1 - 2 * v)
                    * (256 * v**2 * (1 - 2 * u) ** 2 * (1 - v) ** 2 + 1)
                    * (u * v - u * (1 - v) - v * (1 - u) + (1 - u) * (1 - v))
                    + 1024
                    * v**2
                    * (1 - 2 * u)
                    * (1 - v) ** 2
                    * (256 * u**2 * (1 - u) ** 2 * (1 - 2 * v) ** 2 + 1)
                )
                + (
                    256
                    * u
                    * v
                    * (1 - 2 * u)
                    * (1 - u)
                    * (1 - 2 * v)
                    * (1 - v)
                    * dr.sin(dr.pi * u)
                    * dr.cos(dr.pi * v)
                    + (-256 * u**2 * (1 - u) ** 2 * (1 - 2 * v) ** 2 - 1)
                    * dr.sin(dr.pi * v)
                    * dr.cos(dr.pi * u)
                )
                * (
                    u**2
                    * v**2
                    * (1 - u) ** 2
                    * (131072 - 262144 * u)
                    * (v - 1) ** 2
                    * (2 * v - 1) ** 2
                    + 65536
                    * u
                    * v**2
                    * (1 - u)
                    * (1 - v) ** 2
                    * (2 * u - 1) ** 2
                    * (2 * v - 1)
                    * (u * v + u * (v - 1) + v * (u - 1) + (u - 1) * (v - 1))
                    - 256
                    * u
                    * (1 - u)
                    * (2 * v - 1)
                    * (256 * v**2 * (1 - v) ** 2 * (2 * u - 1) ** 2 + 1)
                    * (u * v + u * (v - 1) + v * (u - 1) + (u - 1) * (v - 1))
                    + 512
                    * v**2
                    * (2 * u - 1)
                    * (v - 1) ** 2
                    * (256 * u**2 * (1 - u) ** 2 * (2 * v - 1) ** 2 + 1)
                )
                + (
                    256
                    * u
                    * v
                    * (1 - 2 * u)
                    * (1 - u)
                    * (1 - 2 * v)
                    * (1 - v)
                    * dr.sin(dr.pi * v)
                    * dr.cos(dr.pi * u)
                    + (-256 * v**2 * (1 - 2 * u) ** 2 * (1 - v) ** 2 - 1)
                    * dr.sin(dr.pi * u)
                    * dr.cos(dr.pi * v)
                )
                * (
                    u**2
                    * v**2
                    * (1 - 2 * u) ** 2
                    * (1 - u) ** 2
                    * (1 - v) ** 2
                    * (524288 * v - 262144)
                    + 131072
                    * u**2
                    * v
                    * (1 - 2 * u)
                    * (1 - u) ** 2
                    * (1 - 2 * v) ** 2
                    * (1 - v)
                    * (u * v - u * (1 - v) - v * (1 - u) + (1 - u) * (1 - v))
                    + 1024
                    * u**2
                    * (1 - u) ** 2
                    * (1 - 2 * v)
                    * (256 * v**2 * (1 - 2 * u) ** 2 * (1 - v) ** 2 + 1)
                    - 512
                    * v
                    * (1 - 2 * u)
                    * (1 - v)
                    * (256 * u**2 * (1 - u) ** 2 * (1 - 2 * v) ** 2 + 1)
                    * (u * v - u * (1 - v) - v * (1 - u) + (1 - u) * (1 - v))
                )
                + (
                    256
                    * u
                    * v
                    * (1 - 2 * u)
                    * (1 - u)
                    * (1 - 2 * v)
                    * (1 - v)
                    * dr.sin(dr.pi * v)
                    * dr.cos(dr.pi * u)
                    + (-256 * v**2 * (1 - 2 * u) ** 2 * (1 - v) ** 2 - 1)
                    * dr.sin(dr.pi * u)
                    * dr.cos(dr.pi * v)
                )
                * (
                    u**2
                    * v**2
                    * (1 - v) ** 2
                    * (131072 - 262144 * v)
                    * (u - 1) ** 2
                    * (2 * u - 1) ** 2
                    + 65536
                    * u**2
                    * v
                    * (1 - u) ** 2
                    * (1 - v)
                    * (2 * u - 1)
                    * (2 * v - 1) ** 2
                    * (u * v + u * (v - 1) + v * (u - 1) + (u - 1) * (v - 1))
                    + 512
                    * u**2
                    * (u - 1) ** 2
                    * (2 * v - 1)
                    * (256 * v**2 * (1 - v) ** 2 * (2 * u - 1) ** 2 + 1)
                    - 256
                    * v
                    * (1 - v)
                    * (2 * u - 1)
                    * (256 * u**2 * (1 - u) ** 2 * (2 * v - 1) ** 2 + 1)
                    * (u * v + u * (v - 1) + v * (u - 1) + (u - 1) * (v - 1))
                )
            )
            / (
                -65536
                * u**2
                * v**2
                * (1 - 2 * u) ** 2
                * (1 - u) ** 2
                * (1 - 2 * v) ** 2
                * (1 - v) ** 2
                + (256 * u**2 * (1 - u) ** 2 * (1 - 2 * v) ** 2 + 1)
                * (256 * v**2 * (1 - 2 * u) ** 2 * (1 - v) ** 2 + 1)
            )
            ** 2
        )
