import unittest

import jsonpickle
import mitsuba as mi
import numpy as np

from rebel_shape.splines.bspline import BSpline
from rebel_shape.splines.bspline_patch import BSplinePatch
from rebel_shape.splines.iga_spline_patch import IGASplinePatch


class TestBSplinePatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mi.set_variant("llvm_ad_rgb")
        cls._parametric_coords_data = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ]

        cls.bezier_u_knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        cls.bezier_v_knots = [0.0, 0.0, 1.0, 1.0]
        cls._bezier_control_points_data = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
        cls.bezier_u_spline = BSpline(knot_vector=cls.bezier_u_knots, degree=2)
        cls.bezier_v_spline = BSpline(knot_vector=cls.bezier_v_knots, degree=1)

        cls._bezier_field_values_data = [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]

    def setUp(self) -> None:
        # Control points, field values, and the surface built from them are
        # constructed per test rather than cached on the class: a Mitsuba
        # object that outlives the test which built it aborts a *later,
        # unrelated* test inside Dr.Jit's thread pool ("Assertion failed ...
        # queue.cpp: remain == 1").
        self.parametric_coords = mi.TensorXf(self._parametric_coords_data)

        self.bezier_control_points = mi.TensorXf(self._bezier_control_points_data)
        self.bezier_surface = BSplinePatch(
            name="bezier_surface",
            u_spline=self.bezier_u_spline,
            v_spline=self.bezier_v_spline,
            control_points=self.bezier_control_points,
        )

        self.bezier_field_values = mi.TensorXf(self._bezier_field_values_data)

    def test_initialization(self):
        """
        Test initialization of IGASplinePatch and whether it raises errors
        for invalid inputs.
        """
        try:
            _ = IGASplinePatch(
                spline_patch=self.bezier_surface,
                field_values=self.bezier_field_values,
            )
        except Exception as e:
            self.fail(f"IGASplinePatch initialization raised {e} unexpectedly!")

        # invalid value shape
        invalid_field_values = mi.TensorXf(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
        with self.assertRaises(ValueError):
            IGASplinePatch(
                spline_patch=self.bezier_surface,
                field_values=invalid_field_values,
            )

        # invalid number of field values
        invalid_field_values = mi.TensorXf(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]
        )
        with self.assertRaises(ValueError):
            IGASplinePatch(
                spline_patch=self.bezier_surface,
                field_values=invalid_field_values,
            )

    def test_serialization(self):
        """
        Test serialization and deserialization of IGASplinePatch.

        Only a single property is updated manually in the `__getstate__` method,
        and that is the private`_field_values` attribute, which is a Mitsuba tensor.
        The rest are a bound to the serialization method of the underlying
        `SplinePatch` class.
        """
        iga_spline_patch = IGASplinePatch(
            spline_patch=self.bezier_surface, field_values=self.bezier_field_values
        )

        serialized = jsonpickle.encode(iga_spline_patch)
        deserialized = jsonpickle.decode(serialized)

        # Check that the basic properties are preserved
        np.testing.assert_array_equal(
            deserialized.field_values.numpy(), iga_spline_patch.field_values.numpy()
        )

    def test_field_values_setter(self):
        """Test the setter for field values."""
        iga_spline_patch = IGASplinePatch(
            spline_patch=self.bezier_surface, field_values=self.bezier_field_values
        )

        new_field_values = mi.TensorXf(
            [[0.1, 0.1], [1.1, 0.1], [2.1, 0.1], [0.1, 1.1], [1.1, 1.1], [2.1, 1.1]]
        )
        iga_spline_patch.field_values = new_field_values

        np.testing.assert_array_equal(
            iga_spline_patch.field_values.numpy(), new_field_values.numpy()
        )

        # Test with invalid shape
        invalid_field_values = mi.TensorXf(np.array([0.1, 1.1, 2.1, 3.1, 4.1, 5.1]))
        with self.assertRaises(ValueError):
            iga_spline_patch.field_values = invalid_field_values

        # Test with mismatched number of control points
        invalid_field_values = mi.TensorXf(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]
        )
        with self.assertRaises(ValueError):
            iga_spline_patch.field_values = invalid_field_values

    def test_evaluate_field_texture(self):
        """
        Simple evaluation of the field texture.

        Values that are interpolated are the same as parametric coordinates.
        """
        spline_patch = IGASplinePatch(
            spline_patch=self.bezier_surface, field_values=self.bezier_field_values
        )

        interpolated_texture = spline_patch.evaluate_field_texture(width=2, height=2)

        expected_texture = mi.TensorXf(
            [[[0.5, 0.25], [1.5, 0.25]], [[0.5, 0.75], [1.5, 0.75]]]
        )
        np.testing.assert_array_almost_equal(
            interpolated_texture.numpy(), expected_texture.numpy()
        )

    def test_h_refinement(self):
        """
        Test horizontal refinement of the spline patch.

        This method should increase the resolution of the patch in both u and v
        directions by a factor of 2.
        """
        u_sample_points = np.linspace(0.0, 1.0, 19)
        v_sample_points = np.linspace(0.0, 1.0, 19)
        parametric_coords = np.array(
            np.meshgrid(u_sample_points, v_sample_points, indexing="xy")
        ).reshape(-1, 2)

        for i in range(4):
            with self.subTest(i=i):
                new_u_spline = BSpline(knot_vector=self.bezier_u_knots, degree=2)
                new_v_spline = BSpline(knot_vector=self.bezier_v_knots, degree=1)

                new_bezier_surface = BSplinePatch(
                    name="refined_bezier_surface",
                    u_spline=new_u_spline,
                    v_spline=new_v_spline,
                    control_points=self.bezier_control_points,
                )
                new_bezier_iga_patch = IGASplinePatch(
                    spline_patch=new_bezier_surface,
                    field_values=self.bezier_field_values,
                )

                original_values = new_bezier_iga_patch.evaluate_field(
                    parametric_coords=mi.TensorXf(parametric_coords)
                )
                original_coordinates = new_bezier_iga_patch.evaluate_coordinates(
                    parametric_coords=mi.TensorXf(parametric_coords)
                )

                new_bezier_iga_patch.h_refinement(factor=i + 1)

                refined_values = new_bezier_iga_patch.evaluate_field(
                    parametric_coords=mi.TensorXf(parametric_coords)
                )
                refined_coordinates = new_bezier_iga_patch.evaluate_coordinates(
                    parametric_coords=mi.TensorXf(parametric_coords)
                )

                # Check that the refined values are consistent with the original
                np.testing.assert_array_almost_equal(
                    original_values.numpy(), refined_values.numpy()
                )
                np.testing.assert_array_almost_equal(
                    original_coordinates.numpy(), refined_coordinates.numpy()
                )
