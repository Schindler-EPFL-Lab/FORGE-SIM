import unittest

import jsonpickle
import mitsuba as mi
import numpy as np

from rebel_shape.meshing.composite_iga_spline_surface import CompositeIGASplineSurface
from rebel_shape.meshing.composite_iga_spline_surface_factory import (
    CompositeIGASplineSurfaceFactory,
)
from rebel_shape.meshing.patch import Patch
from rebel_shape.splines.bspline_patch import BSplinePatch
from rebel_shape.splines.iga_spline_patch import IGASplinePatch


class TestCompositeSplineSurface(unittest.TestCase):
    def setUp(self):
        """
        Exploits the fact that control points are also field values in a sense, as
        in they can also be interpolated to get the surface, just like any other
        field values.
        """
        mi.set_variant("scalar_rgb")
        # Use BSplinePatch for testing, not mocks, too complex to mock properly
        # Unit square with 1 knot span in each direction = Bezier patch
        patch1 = BSplinePatch.unit_square(
            patch_name="patch1", degree=2, u_subdivisions=1, v_subdivisions=1
        )
        patch1.control_points = mi.TensorXf(
            patch1.control_points.numpy() + np.array([0.5, 0.5, 0.0])
        )
        self.patch1 = IGASplinePatch(
            spline_patch=patch1, field_values=patch1.control_points
        )

        patch2 = BSplinePatch.unit_square(
            patch_name="patch2", degree=2, u_subdivisions=1, v_subdivisions=1
        )
        patch2.control_points = mi.TensorXf(
            patch2.control_points.numpy() + np.array([1.5, 0.5, 0.0])
        )
        self.patch2 = IGASplinePatch(
            spline_patch=patch2, field_values=patch2.control_points
        )

        patch3 = BSplinePatch.unit_square(
            patch_name="patch3", degree=2, u_subdivisions=1, v_subdivisions=1
        )
        patch3.control_points = mi.TensorXf(
            patch3.control_points.numpy() + np.array([1.5, 1.5, 0.0])
        )
        self.patch3 = IGASplinePatch(
            spline_patch=patch3, field_values=patch3.control_points
        )

        self.composite_factory = CompositeIGASplineSurfaceFactory("test_composite")
        self.composite_factory.add_patch(self.patch1)
        self.composite_factory.add_patch(self.patch2)
        self.composite_factory.add_patch(self.patch3)

        self.composite_factory.link_sides(
            first_patch_name=self.patch1.name,
            first_patch_side=Patch.Side.RIGHT,
            second_patch_name=self.patch2.name,
            second_patch_side=Patch.Side.LEFT,
        )
        self.composite_factory.link_sides(
            first_patch_name=self.patch2.name,
            first_patch_side=Patch.Side.TOP,
            second_patch_name=self.patch3.name,
            second_patch_side=Patch.Side.BOTTOM,
        )

        self.patches = [self.patch1, self.patch2, self.patch3]
        self.connections = self.composite_factory._connection_list

        self.iga_surface = self.composite_factory.build()

    def test_init(self):
        """Test the initialization of CompositeSplineSurface"""
        self.assertIsInstance(self.iga_surface, CompositeIGASplineSurface)
        self.assertEqual(self.iga_surface.name, "test_composite")
        self.assertEqual(self.iga_surface.global_field_values.shape, (21, 3))

    def test_build_global_representation(self):
        """Test building of global control points"""
        global_field_values = self.iga_surface.global_field_values.numpy()
        expected_field_values = self.iga_surface.global_control_points.numpy()

        np.testing.assert_array_almost_equal(global_field_values, expected_field_values)

    def test_update_global_field_values(self):
        """
        Test whether the global field values are updated correctly.

        Since the global control points are also the global field values, they should
        behave the same way when updated. Validity of control point update is tested
        in `test_composite_spline_surface.py`.
        """
        new_patch1_field_values = mi.TensorXf(
            self.patch1.control_points.numpy() + np.array([0.0, 0.0, 1.0])
        )
        self.iga_surface[0].field_values = new_patch1_field_values
        self.iga_surface[0].control_points = new_patch1_field_values
        self.iga_surface.update_global_representation()

        np.testing.assert_array_almost_equal(
            self.iga_surface.global_field_values.numpy(),
            self.iga_surface.global_control_points.numpy(),
        )

    def test_global_field_values_setter(self):
        """Test whether global field values setter updates all patches correctly."""
        new_field_values = mi.TensorXf(
            self.iga_surface.global_field_values.numpy() + np.array([0.0, 0.0, 0.5])
        )
        self.iga_surface.global_field_values = new_field_values

        for patch_idx, patch in enumerate(self.iga_surface):
            np.testing.assert_array_almost_equal(
                patch.field_values.numpy(),
                new_field_values.numpy()[
                    self.iga_surface._patch_to_global_map[patch_idx]
                ],
            )

    def test_serialization(self):
        """
        Tests serialization and deserialization of CompositeSplineSurface.

        Since only global control points, global quads, and averaging weights require
        special handling, only these attributes are tested.
        """
        encoded = jsonpickle.encode(self.iga_surface)
        decoded = jsonpickle.decode(encoded)

        self.assertEqual(len(decoded), len(self.iga_surface))

        np.testing.assert_array_almost_equal(
            decoded.global_field_values.numpy(),
            self.iga_surface.global_field_values.numpy(),
        )

        np.testing.assert_array_almost_equal(
            decoded.global_control_points.numpy(),
            self.iga_surface.global_control_points.numpy(),
        )

        np.testing.assert_array_equal(
            decoded.global_quads.numpy(), self.iga_surface.global_quads.numpy()
        )

        np.testing.assert_array_equal(
            decoded._averaging_weights.numpy(),
            self.iga_surface._averaging_weights.numpy(),
        )
