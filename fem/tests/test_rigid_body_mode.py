import unittest

import numpy as np
from _fem_fixtures import RIGID_BODY_LABELS

from fem.shells.rigid_body_mode import RigidBodyMode


class TestRigidBodyMode(unittest.TestCase):
    """
    `RigidBodyMode.standard_basis()` and the per-mode `rotation` and
    `displacement` it builds: translation then rotation about x, y, z, in that
    fixed order, since
    `GeometricallyExactShellModalSolver.rigid_body_energies()` pairs each with
    its Rayleigh quotient positionally, and the drilling-objectivity tests
    (`test_shell_system_assembler.py`) rebuild the same six fields by hand.
    """

    # Three points, one per column (the `(gdim, n)` layout `displacement`
    # expects).
    POINTS = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    def test_standard_basis_returns_the_six_modes_in_order(self) -> None:
        """Translation then rotation about x, y, z, each carrying the
        corresponding standard-basis vector as its axis."""
        basis = RigidBodyMode.standard_basis()
        self.assertEqual(tuple(mode.label for mode in basis), RIGID_BODY_LABELS)
        self.assertEqual(
            [mode.kind for mode in basis],
            [RigidBodyMode.Kind.TRANSLATION] * 3 + [RigidBodyMode.Kind.ROTATION] * 3,
        )
        expected_axes = np.tile(np.eye(3), (2, 1))
        np.testing.assert_array_equal(
            np.array([mode.axis for mode in basis]), expected_axes
        )

    def test_rotation_property_is_zero_for_a_translation_mode(self) -> None:
        """A translation mode carries no rotation field, regardless of its own
        axis: `displacement`'s companion rotation is always zero."""
        mode = RigidBodyMode(
            kind=RigidBodyMode.Kind.TRANSLATION,
            axis=np.array([2.0, -1.0, 0.5]),
            label="T_test",
        )
        np.testing.assert_array_equal(mode.rotation, np.zeros(3))

    def test_rotation_property_is_the_axis_for_a_rotation_mode(self) -> None:
        """A rotation mode's rotation field is its own axis."""
        axis = np.array([2.0, -1.0, 0.5])
        mode = RigidBodyMode(
            kind=RigidBodyMode.Kind.ROTATION, axis=axis, label="R_test"
        )
        np.testing.assert_array_equal(mode.rotation, axis)

    def test_translation_displacement_is_uniform(self) -> None:
        """Every point gets the same displacement: the translation axis,
        independent of where the point sits."""
        mode = RigidBodyMode(
            kind=RigidBodyMode.Kind.TRANSLATION,
            axis=np.array([0.0, 1.0, 0.0]),
            label="Ty",
        )
        expected = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ])
        np.testing.assert_array_equal(mode.displacement(self.POINTS), expected)

    def test_rotation_displacement_is_axis_cross_point(self) -> None:
        """`displacement(x) = axis x x`, hand-computed at the three points."""
        mode = RigidBodyMode(
            kind=RigidBodyMode.Kind.ROTATION,
            axis=np.array([1.0, 0.0, 0.0]),
            label="Rx",
        )
        # axis x (0,0,1) = (0,-1,0); axis x (0,1,0) = (0,0,1); axis x (1,0,0) = 0.
        expected = np.array([
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        np.testing.assert_array_equal(mode.displacement(self.POINTS), expected)
