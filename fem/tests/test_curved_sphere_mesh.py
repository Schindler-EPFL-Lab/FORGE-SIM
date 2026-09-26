import unittest

import _fem_fixtures  # noqa: F401
import numpy as np

from fem.shells.curved_sphere_mesh import make_curved_sphere_mesh


class TestCurvedSphereMesh(unittest.TestCase):
    """`make_curved_sphere_mesh` is the shared gmsh sphere builder: production
    code, and the geometry every other shell test in this package stands on."""

    def test_order_two_nodes_lie_on_the_sphere(self) -> None:
        """
        The point of an isoparametric mesh: every geometry node, mid-edge nodes
        included, sits on the exact surface. gmsh places them by projecting onto
        the OCC sphere, so the residual is its own geometric tolerance: 1e-9
        relative is generous against the measured ~1e-15 and would still catch
        a mesh whose curved nodes were interpolated rather than projected (those
        sit ~h^2/8 ~ 3e-2 off a radius-1 sphere).
        """
        for radius in (1.0, 2.5):
            with self.subTest(radius=radius):
                msh = make_curved_sphere_mesh(radius, 0.8, order=2)
                radii = np.linalg.norm(msh.geometry.x, axis=1)
                np.testing.assert_allclose(radii, radius, rtol=1.0e-9)

    def test_order_one_has_three_nodes_per_cell_and_order_two_has_six(self) -> None:
        """Order 1 gives P1 triangles, order 2 gives P2 (three vertices plus
        three mid-edge nodes): the layout `_cell_size` relies on when it takes
        the first three geometry nodes as the vertices."""
        for order, nodes_per_cell in ((1, 3), (2, 6)):
            with self.subTest(order=order):
                msh = make_curved_sphere_mesh(1.0, 0.8, order)
                self.assertEqual(msh.geometry.dofmap.shape[1], nodes_per_cell)

    def test_repeated_calls_succeed(self) -> None:
        """gmsh is a process-global singleton; the builder must leave it clean so
        a suite can build many meshes (and so a leaked session elsewhere in the
        process does not poison it)."""
        first = make_curved_sphere_mesh(1.0, 0.9, order=1)
        second = make_curved_sphere_mesh(1.0, 0.9, order=1)
        np.testing.assert_allclose(first.geometry.x, second.geometry.x)
