import unittest
from unittest.mock import MagicMock

import drjit as dr
import mitsuba as mi

from forge_sim.meshing.composite_spline_surface_factory import (
    CompositeSplineSurfaceFactory,
)
from forge_sim.meshing.patch import Patch
from forge_sim.meshing.patch_manager import PatchManager
from forge_sim.meshing.patch_manager_factory import PatchManagerFactory
from forge_sim.splines.bspline import BSpline
from forge_sim.splines.bspline_patch import BSplinePatch


class TestPatchManagerFactory(unittest.TestCase):
    class TestPatchManagerFactory(PatchManagerFactory):
        def build(self) -> PatchManager:
            return TestPatchManagerFactory.TestPatchManager(
                name=self._manager_name,
                patch_list=self._patch_list,
                connection_list=self._connection_list,
            )

    class TestPatchManager(PatchManager):
        def update_global_representation(self):
            pass

    class TestPatch(Patch):
        """Concrete implementation of a patch with 4 nodes for testing purposes"""

        def __init__(self, name: str, nodes: "mi.TensorXf"):
            assert len(nodes) == 4, "Patch must have exactly 4 nodes"
            self._name = name
            self._nodes = nodes

        def get_node_indices_for(self, side: Patch.Side) -> "mi.ArrayXu":
            if side == Patch.Side.LEFT:
                return dr.ravel(mi.TensorXu([0, 2]))
            if side == Patch.Side.RIGHT:
                return dr.ravel(mi.TensorXu([1, 3]))
            if side == Patch.Side.TOP:
                return dr.ravel(mi.TensorXu([2, 3]))
            if side == Patch.Side.BOTTOM:
                return dr.ravel(mi.TensorXu([0, 1]))

        @property
        def nodes(self) -> "mi.TensorXf":
            return self._nodes

        @property
        def name(self) -> str:
            return self._name

    def setUp(self):
        # Create mock patches for testing
        mi.set_variant("llvm_ad_rgb")

        self.patch1 = self.TestPatch(
            name="patch1",
            nodes=mi.TensorXf(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ]
            ),
        )
        self.patch2 = self.TestPatch(
            name="patch2",
            nodes=mi.TensorXf(
                [
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [2.0, 1.0, 0.0],
                ]
            ),
        )
        self.patch3 = self.TestPatch(
            name="patch3",
            nodes=mi.TensorXf(
                [
                    [1.0, 1.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [1.0, 2.0, 0.0],
                    [2.0, 2.0, 0.0],
                ]
            ),
        )

    def test_init(self):
        """Test initialization of PatchManagerFactory."""
        factory = self.TestPatchManagerFactory("test_mesh")
        self.assertEqual(factory._manager_name, "test_mesh")
        self.assertEqual(factory._patch_list, [])
        self.assertEqual(factory._connection_list, [])
        self.assertEqual(factory._name_to_index_map, {})

    def test_add_patch(self):
        """Test adding patches to the factory."""
        factory = self.TestPatchManagerFactory("test_mesh")

        # Add first patch
        factory.add_patch(self.patch1)
        self.assertEqual(len(factory._patch_list), 1)
        self.assertEqual(len(factory._connection_list), 1)
        self.assertEqual(factory._name_to_index_map["patch1"], 0)

        # Add second patch
        factory.add_patch(self.patch2)
        self.assertEqual(len(factory._patch_list), 2)
        self.assertEqual(len(factory._connection_list), 2)
        self.assertEqual(factory._name_to_index_map["patch2"], 1)

        # Try adding a patch with an existing name
        with self.assertRaises(ValueError):
            factory.add_patch(self.patch1)

    def test_link_sides(self):
        """Test linking sides between patches."""
        factory = self.TestPatchManagerFactory("test_mesh")
        factory.add_patch(self.patch1)
        factory.add_patch(self.patch2)

        # Link the right side of patch1 to the left side of patch2
        factory.link_sides(
            first_patch_name="patch1",
            first_patch_side=Patch.Side.RIGHT,
            second_patch_name="patch2",
            second_patch_side=Patch.Side.LEFT,
        )

        # Check the connections were established
        self.assertEqual(factory._connection_list[0][Patch.Side.RIGHT], 1)
        self.assertEqual(factory._connection_list[1][Patch.Side.LEFT], 0)

        # Try to link an already linked side
        with self.assertRaises(ValueError):
            factory.link_sides(
                first_patch_name="patch1",
                first_patch_side=Patch.Side.RIGHT,
                second_patch_name="patch2",
                second_patch_side=Patch.Side.TOP,
            )

    def test_link_sides_with_nonexistent_patch(self):
        """Test linking sides with a nonexistent patch."""
        factory = self.TestPatchManagerFactory("test_mesh")
        factory.add_patch(self.patch1)

        # Try to link with non-existent patch
        with self.assertRaises(ValueError):
            factory.link_sides(
                first_patch_name="patch1",
                first_patch_side=Patch.Side.RIGHT,
                second_patch_name="nonexistent",
                second_patch_side=Patch.Side.LEFT,
            )

        with self.assertRaises(ValueError):
            factory.link_sides(
                first_patch_name="nonexistent",
                first_patch_side=Patch.Side.RIGHT,
                second_patch_name="patch1",
                second_patch_side=Patch.Side.LEFT,
            )

    def test_link_sides_with_mismatched_vertices(self):
        """Test linking sides with mismatched vertex counts."""
        factory = self.TestPatchManagerFactory("test_mesh")

        # Create patch with different numbers of vertices on sides
        mismatch_patch = MagicMock(spec=Patch)
        mismatch_patch.name = "mismatch_patch"
        mismatch_patch.get_node_indices_for.return_value = dr.ravel(
            mi.TensorXu([0, 1, 2])
        )  # 3 vertices instead of 2

        factory.add_patch(self.patch1)
        factory.add_patch(mismatch_patch)

        # Try to link sides with mismatched vertex counts
        with self.assertRaises(ValueError):
            factory.link_sides(
                first_patch_name="patch1",
                first_patch_side=Patch.Side.TOP,
                second_patch_name="mismatch_patch",
                second_patch_side=Patch.Side.BOTTOM,
            )

    def test_build(self):
        """Test building a PatchManager from the factory."""
        factory = self.TestPatchManagerFactory("test_mesh")
        factory.add_patch(self.patch1)
        factory.add_patch(self.patch2)
        factory.add_patch(self.patch3)

        # Link patches
        factory.link_sides(
            first_patch_name="patch1",
            first_patch_side=Patch.Side.RIGHT,
            second_patch_name="patch2",
            second_patch_side=Patch.Side.LEFT,
        )
        factory.link_sides(
            first_patch_name="patch2",
            first_patch_side=Patch.Side.TOP,
            second_patch_name="patch3",
            second_patch_side=Patch.Side.BOTTOM,
        )

        # Build the patch manager
        manager = factory.build()

        # Check that the patch manager has the correct data
        self.assertIsInstance(manager, PatchManager)
        self.assertEqual(manager.name, "test_mesh")
        self.assertEqual(len(manager._patch_list), 3)
        self.assertEqual(len(manager._connection_list), 3)

        # Verify the connections
        self.assertEqual(manager._connection_list[0][Patch.Side.RIGHT], 1)
        self.assertEqual(manager._connection_list[1][Patch.Side.LEFT], 0)
        self.assertEqual(manager._connection_list[1][Patch.Side.TOP], 2)
        self.assertEqual(manager._connection_list[2][Patch.Side.BOTTOM], 1)

    def test_spline_factory(self):
        """Test building a PatchManager for spline patches."""
        # The number of control points is 3 in both directions.
        spline_1 = BSpline(knot_vector=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], degree=2)
        spline_2 = BSpline(knot_vector=[0.0, 0.0, 0.5, 1.0, 1.0], degree=1)

        # Control point positions do not matter for this test
        spline_patch_1 = BSplinePatch(
            name="spline_patch_1",
            u_spline=spline_1,
            v_spline=spline_2,
            control_points=dr.zeros(dtype=mi.TensorXf, shape=(9, 3)),
        )
        spline_patch_2 = BSplinePatch(
            name="spline_patch_2",
            u_spline=spline_2,
            v_spline=spline_1,
            control_points=dr.zeros(dtype=mi.TensorXf, shape=(9, 3)),
        )

        factory = CompositeSplineSurfaceFactory(manager_name="test_composite_spline")
        factory.add_patch(spline_patch_1)
        factory.add_patch(spline_patch_2)

        # spline_2 knot spans are exposed on the right side of spline_patch_1
        # and the bottom side of spline_patch_2, so this must pass
        factory.link_sides(
            first_patch_name="spline_patch_1",
            first_patch_side=Patch.Side.RIGHT,
            second_patch_name="spline_patch_2",
            second_patch_side=Patch.Side.BOTTOM,
        )

        # The left side of spline_patch_2 has knot spans that are not present on the
        # right side of spline_patch_1, so this must fail
        with self.assertRaises(ValueError):
            factory.link_sides(
                first_patch_name="spline_patch_1",
                first_patch_side=Patch.Side.RIGHT,
                second_patch_name="spline_patch_2",
                second_patch_side=Patch.Side.LEFT,
            )
