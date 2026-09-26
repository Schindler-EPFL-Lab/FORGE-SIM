import unittest
import warnings

import numpy as np
import scipy.sparse as sp
import ufl
from _fem_fixtures import (
    STEEL_SHELL,
    relative_asymmetry,
    sphere_mesh,
    steel_sphere_problem,
    steel_sphere_solver,
)
from dolfinx import fem

from fem.shells.geometrically_exact_shell_problem import GeometricallyExactShellProblem
from fem.shells.shell_element import ShellElement
from fem.shells.shell_system_assembler import ShellSystemAssembler


class TestGeometricallyExactShellProblem(unittest.TestCase):
    """
    `GeometricallyExactShellProblem` construction and its B-bar internals.

    Covers the per-point surface geometry (normal choice, and the warning when
    the analytic sphere normal is paired with a flat mesh), the order-aware
    PSRI reduced quadrature degree, `ShellElement.label()` (the element choice
    the problem is built from), and the B-bar projection machinery
    (`_tangential_tensor`, `_block_diag_inverse`, cached `bbar_blocks`) that
    makes B-bar's symmetric, rigid-body-null stiffness possible.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.solver = steel_sphere_solver(
            0.8, 2, locking_cure=ShellElement.LockingCure.BBAR
        )

    def test_element_label_reports_the_active_scheme(self) -> None:
        """`label()` goes into the run log; it must name the cure that is
        actually in force, alongside the element degrees and the normal."""
        self.assertEqual(
            ShellElement(degree=3, locking_cure=ShellElement.LockingCure.BBAR).label(),
            "P3/P3+Bbar+geometric",
        )
        self.assertEqual(
            ShellElement(degree=2, locking_cure=ShellElement.LockingCure.PSRI).label(),
            "P2/P2+PSRI+geometric",
        )
        self.assertEqual(
            ShellElement(degree=2, locking_cure=ShellElement.LockingCure.FULL).label(),
            "P2/P2+full+geometric",
        )

    def test_reduced_quadrature_degree_is_order_aware(self) -> None:
        """PSRI's reduced degree defaults to 2*(p-1) (degree 2 hourglasses at
        P3, per the class docstring), and an explicit setting wins."""
        for degree in (2, 3):
            with self.subTest(degree=degree):
                problem = GeometricallyExactShellProblem(
                    sphere_mesh(0.8, 2),
                    element=ShellElement(
                        degree=degree, locking_cure=ShellElement.LockingCure.PSRI
                    ),
                    **STEEL_SHELL,
                )
                self.assertEqual(problem.reduced_quad, 2 * (degree - 1))
                self.assertEqual(problem.quad_degree, 2 * degree + 2)
        problem = GeometricallyExactShellProblem(
            sphere_mesh(0.8, 2),
            element=ShellElement(
                degree=2, locking_cure=ShellElement.LockingCure.PSRI, reduced_quad=1
            ),
            **STEEL_SHELL,
        )
        self.assertEqual(problem.reduced_quad, 1)

    def test_membrane_material_operator_is_self_adjoint_on_the_reduced_space(
        self,
    ) -> None:
        """
        The precondition the paper's symmetry argument rests on, checked where it
        is actually needed: on the B-bar reduced space `W`.

        `G = <h C_ps(Q), R>` is the material-weighted reduced-space mass, and the
        projected stiffness `Pi^T G Pi` is symmetric if and only if `G` is.
        `C_ps(e) = lambda_ps tr(e) P + 2 mu e` is self-adjoint exactly on
        tangential tensors: the antisymmetric part of the pairing reduces to
        `lambda_ps (tr(q) <P, r> - tr(r) <P, q>)`, and `<P, e> == tr(e)` holds iff
        `e` is tangential. `W` is therefore carried as the three components of a
        symmetric tangential tensor in the local frame, not as unconstrained 3x3
        tensors: the latter is a strictly larger space containing directions on
        which `C_ps` is not self-adjoint, and it left `G` asymmetric by 0.21-0.27
        relative and the assembled block by ~1e-3.

        Measured now: 5.7e-17 (P2, h = 0.8), 8.1e-17 (P2, h = 0.5), 1.2e-16 (P3),
        i.e. round-off. The bound is 1e-12, four orders above that and nine below
        what the unconstrained space gave.
        """
        g = ShellSystemAssembler._to_scipy(
            self.solver._problem._forms.bbar_membrane.material_mass
        )
        self.assertLess(relative_asymmetry(g), 1.0e-12)

    def test_membrane_reduced_space_is_the_symmetric_tangential_tensors(
        self,
    ) -> None:
        """
        The space has three components per point, not nine: a symmetric
        tangential tensor has exactly three independent entries in a tangent
        frame. Checked as a dimension count against the shear space, which is a
        genuine three-vector on the same `DG_{p-1}` scalar basis, so the two must
        have identical dimension.

        This is what stops the space silently widening back to unconstrained 3x3
        tensors, which is the only way the symmetry above can be lost.
        """
        problem = self.solver._problem
        membrane = ShellSystemAssembler._to_scipy(
            problem._forms.bbar_membrane.reduced_mass
        )
        shear = ShellSystemAssembler._to_scipy(problem._forms.bbar_shear.reduced_mass)
        self.assertEqual(membrane.shape, shear.shape)

        n_cells = problem.msh.topology.index_map(problem.msh.topology.dim).size_local
        scalar_dofs_per_cell = problem.el.degree * (problem.el.degree + 1) // 2
        self.assertEqual(membrane.shape[0], 3 * scalar_dofs_per_cell * n_cells)

    def test_reduced_space_tensors_are_symmetric_and_tangential(self) -> None:
        """
        The reconstruction `_tangential_tensor` is what makes the space what it
        claims to be, so it is checked directly rather than only through its
        consequences: for arbitrary components the rebuilt tensor `E` must
        satisfy `E == E^T` and `E n == 0` pointwise.

        Evaluated by projecting `|E - E^T|^2` and `|E n|^2` onto DG0 and taking
        the largest cell value, normalised by `|E|^2`. Bound 1e-20 on a squared
        relative quantity, i.e. 1e-10 on the quantity itself; measured ~1e-32.
        """
        problem = self.solver._problem
        components = ufl.as_vector([1.0, -2.0, 3.0])
        tensor = problem._tangential_tensor(components)
        norm = ufl.inner(tensor, tensor)
        measures = {
            "asymmetry": ufl.inner(tensor - ufl.transpose(tensor),
                                   tensor - ufl.transpose(tensor)) / norm,
            "normal_component": ufl.dot(ufl.dot(tensor, problem.n),
                                        ufl.dot(tensor, problem.n)) / norm,
        }
        dg0 = fem.functionspace(problem.msh, ("DG", 0))
        for name, measure in measures.items():
            with self.subTest(quantity=name):
                projected = fem.Function(dg0)
                projected.interpolate(
                    fem.Expression(measure, dg0.element.interpolation_points())
                )
                self.assertLess(float(np.abs(projected.x.array).max()), 1.0e-20)

    def test_bbar_blocks_are_cached(self) -> None:
        """The blocks cost a sparse triple product each and are used by both
        solve paths, so they are computed once."""
        first = self.solver._problem.bbar_blocks()
        self.assertIs(first, self.solver._problem.bbar_blocks())

    def test_inverts_the_reduced_space_mass_matrix(self) -> None:
        """Round-off bound 1e-10: these are small (3x3-per-node) well-conditioned
        blocks scaled by cell area, so the residual is ~1e-16 in practice."""
        problem = self.solver._problem
        mass_form = problem._forms.bbar_shear.reduced_mass
        mass = ShellSystemAssembler._to_scipy(mass_form)
        inverse = problem._block_diag_inverse(mass)
        self.assertIsNotNone(inverse)
        residual = (inverse @ mass) - sp.identity(mass.shape[0], format="csr")
        self.assertLess(abs(residual).max(), 1.0e-10)

    def test_returns_none_for_a_non_block_diagonal_matrix(self) -> None:
        """The guard that keeps the caller's fallback honest: a matrix with an
        entry outside the per-cell blocks must be rejected, not inverted."""
        problem = self.solver._problem
        mass_form = problem._forms.bbar_shear.reduced_mass
        mass = ShellSystemAssembler._to_scipy(mass_form).tolil()
        mass[0, mass.shape[1] - 1] = 1.0  # couples the first and last cells
        self.assertIsNone(problem._block_diag_inverse(mass.tocsr()))

    def test_returns_none_when_the_size_is_not_a_cell_multiple(self) -> None:
        """The other structural guard: a matrix whose dimension is not divisible
        by the cell count cannot be the expected uniform-block-size layout."""
        problem = self.solver._problem
        mass_form = problem._forms.bbar_shear.reduced_mass
        mass = ShellSystemAssembler._to_scipy(mass_form)
        trimmed = mass[:-1, :-1].tocsr()
        self.assertIsNone(problem._block_diag_inverse(trimmed))

    def test_cell_size_matches_the_longest_triangle_edge(self) -> None:
        """`_cell_size` feeds both the PSRI weight `(t/h)^2` and the drilling
        stiffness `E t^3 / h^2`, so an off-by-a-factor here would silently
        mis-weight two energy terms. Recomputed here from the vertices."""
        problem = steel_sphere_problem(
            0.8, 2, locking_cure=ShellElement.LockingCure.BBAR
        )
        dofmap = np.asarray(problem.msh.geometry.dofmap)
        vertices = problem.msh.geometry.x[dofmap[:, :3]]
        expected = np.max(
            [
                np.linalg.norm(vertices[:, a] - vertices[:, b], axis=1)
                for a, b in ((0, 1), (1, 2), (0, 2))
            ],
            axis=0,
        )
        space = problem.h_cell.function_space
        np.testing.assert_allclose(
            problem.h_cell.x.array[space.dofmap.list[:, 0]], expected, rtol=1.0e-14
        )

    def test_analytic_normal_on_a_flat_mesh_warns(self) -> None:
        """The constructor's guard: an analytic normal is inconsistent with a
        flat element's own geometry, which corrupts the rigid-body modes."""
        with self.assertWarns(RuntimeWarning):
            GeometricallyExactShellProblem(
                sphere_mesh(0.8, 1),
                element=ShellElement(
                    normal=ShellElement.NormalKind.ANALYTIC_SPHERE
                ),
                **STEEL_SHELL,
            )

    def test_geometric_normal_on_a_flat_mesh_does_not_warn(self) -> None:
        """The default is safe at every geometry order, so it must stay quiet."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            GeometricallyExactShellProblem(
                sphere_mesh(0.8, 1), element=ShellElement(), **STEEL_SHELL
            )
