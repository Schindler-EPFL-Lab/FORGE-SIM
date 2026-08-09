import itertools
import unittest

import mitsuba as mi
import numpy as np

from rebel_shape.splines.bspline import BSpline
from rebel_shape.splines.bspline_patch import BSplinePatch


class TestBSplinePatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mi.set_variant("llvm_ad_rgb")

        cls.bezier_2_u_knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        cls.bezier_2_v_knots = [0.0, 0.0, 1.0, 1.0]
        cls._bezier_2_control_points_data = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
        cls.bezier_2_u_spline = BSpline(knot_vector=cls.bezier_2_u_knots, degree=2)
        cls.bezier_2_v_spline = BSpline(knot_vector=cls.bezier_2_v_knots, degree=1)

        cls._parametric_coords_data = [
            [0.0, 0.0],  # Bottom-left
            [1.0, 0.0],  # Bottom-right
            [0.0, 1.0],  # Top-left
            [1.0, 1.0],  # Top-right
            [0.5, 0.5],  # Center
        ]

        cls.bezier_3_u_knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        cls.bezier_3_v_knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        cls._bezier_3_control_points_data = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.5, 0.0],
            [2.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
        cls.bezier_3_u_spline = BSpline(knot_vector=cls.bezier_3_u_knots, degree=2)
        cls.bezier_3_v_spline = BSpline(knot_vector=cls.bezier_3_v_knots, degree=2)

    def setUp(self) -> None:
        # Control-point tensors and the surfaces built from them are
        # constructed per test rather than cached on the class: a Mitsuba
        # object that outlives the test which built it aborts a *later,
        # unrelated* test inside Dr.Jit's thread pool ("Assertion failed ...
        # queue.cpp: remain == 1").
        self.bezier_2_control_points = mi.TensorXf(
            np.array(self._bezier_2_control_points_data)
        )
        self.bezier_2_surface = BSplinePatch(
            name="bezier_2_surface",
            u_spline=self.bezier_2_u_spline,
            v_spline=self.bezier_2_v_spline,
            control_points=self.bezier_2_control_points,
        )

        self.parametric_coords = mi.TensorXf(np.array(self._parametric_coords_data))

        self.bezier_3_control_points = mi.TensorXf(
            np.array(self._bezier_3_control_points_data)
        )
        self.bezier_3_surface = BSplinePatch(
            name="bezier_3_surface",
            u_spline=self.bezier_3_u_spline,
            v_spline=self.bezier_3_v_spline,
            control_points=self.bezier_3_control_points,
        )

    def test_metric_tensor(self):
        """
        Tests whether the metric tensor is computed correctly for the Bezier surface.
        """
        jacobian = self.bezier_2_surface.evaluate_jacobian(
            parametric_coords=self.parametric_coords
        )

        metric_tensor = self.bezier_2_surface.metric_tensor(jacobian=jacobian)

        expected_metric_tensor = np.array(
            [
                [[8.0, 0.0], [0.0, 1.0]],
                [[8.0, 0.0], [0.0, 1.0]],
                [[8.0, 0.0], [0.0, 1.0]],
                [[8.0, 0.0], [0.0, 1.0]],
                [[4.0, 0.0], [0.0, 1.0]],
            ]
        )

        np.testing.assert_array_almost_equal(
            metric_tensor.numpy(), expected_metric_tensor
        )

    def test_metric_tensor_invalid_input(self):
        """
        Tests whether the metric tensor raises an error when an invalid jacobian is
        provided.
        """
        invalid_jacobian = mi.TensorXf(np.ones((5, 3, 3)))
        with self.assertRaises(ValueError):
            self.bezier_2_surface.metric_tensor(jacobian=invalid_jacobian)

        invalid_jacobian = mi.TensorXf(np.ones((5, 2, 3)))
        with self.assertRaises(ValueError):
            self.bezier_2_surface.metric_tensor(jacobian=invalid_jacobian)

        invalid_jacobian = mi.TensorXf(np.ones((3, 2)))
        with self.assertRaises(ValueError):
            self.bezier_2_surface.metric_tensor(jacobian=invalid_jacobian)

        invalid_jacobian = mi.TensorXf(np.ones((5, 5, 3, 2)))
        with self.assertRaises(ValueError):
            self.bezier_2_surface.metric_tensor(jacobian=invalid_jacobian)

    def test_inv_metric_tensor(self):
        """
        Tests whether the inverse metric tensor and its determinant are computed
        correctly.
        """
        jacobian = self.bezier_2_surface.evaluate_jacobian(
            parametric_coords=self.parametric_coords
        )

        metric_tensor = self.bezier_2_surface.metric_tensor(jacobian=jacobian)
        inv_metric_tensor, det_metric_tensor = self.bezier_2_surface.inv_metric_tensor(
            metric=metric_tensor
        )

        expected_inv_metric_tensor = np.array(
            [
                [[0.125, 0.0], [0.0, 1.0]],
                [[0.125, 0.0], [0.0, 1.0]],
                [[0.125, 0.0], [0.0, 1.0]],
                [[0.125, 0.0], [0.0, 1.0]],
                [[0.25, 0.0], [0.0, 1.0]],
            ]
        )
        expected_det_metric_tensor = np.array([8.0, 8.0, 8.0, 8.0, 4.0])
        np.testing.assert_array_almost_equal(
            inv_metric_tensor.numpy(), expected_inv_metric_tensor
        )
        np.testing.assert_array_almost_equal(
            det_metric_tensor.numpy(), expected_det_metric_tensor
        )

    def test_inv_metric_tensor_invalid_input(self):
        """
        Tests whether the inverse metric tensor raises an error when an invalid metric
        is provided.
        """
        invalid_metric = mi.TensorXf(np.ones((5, 3, 2)))
        with self.assertRaises(ValueError):
            self.bezier_2_surface.inv_metric_tensor(metric=invalid_metric)

        invalid_metric = mi.TensorXf(np.ones((5, 2, 3)))
        with self.assertRaises(ValueError):
            self.bezier_2_surface.inv_metric_tensor(metric=invalid_metric)

        invalid_metric = mi.TensorXf(np.ones((2, 2)))
        with self.assertRaises(ValueError):
            self.bezier_2_surface.inv_metric_tensor(metric=invalid_metric)

        invalid_metric = mi.TensorXf(np.ones((5, 5, 2, 2)))
        with self.assertRaises(ValueError):
            self.bezier_2_surface.inv_metric_tensor(metric=invalid_metric)

    def test_second_fundamental_form(self):
        """
        Tests whether the second fundamental form tensor is computed correctly.
        """

        expected_second_fundamental_form = np.array(
            [
                [[0.0, -2.0], [-2.0, -2.0]],
                [
                    [-4 / np.sqrt(3), 6 / np.sqrt(3)],
                    [6 / np.sqrt(3), -2 / np.sqrt(3)],
                ],
                [
                    [-4 / np.sqrt(3), 6 / np.sqrt(3)],
                    [6 / np.sqrt(3), -2 / np.sqrt(3)],
                ],
                [[0.0, -2.0], [-2.0, -2.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        )

        hessian = self.bezier_3_surface.evaluate_hessian(self.parametric_coords)
        normals = self.bezier_3_surface.evaluate_normals(self.parametric_coords)
        second_fundamental_form = self.bezier_3_surface.second_fundamental_form(
            hessian=hessian, normal=normals
        )

        np.testing.assert_array_almost_equal(
            second_fundamental_form.numpy(), expected_second_fundamental_form
        )

    def test_second_fundamental_form_invalid_input(self):
        """
        Tests whether the second fundamental form raises an error when invalid hessian
        or invalid normals are provided.
        """

        invalid_hessians_cross = [
            mi.TensorXf(np.ones((5, 5, 3, 2, 2))),
            mi.TensorXf(np.ones((3, 2, 2))),
            mi.TensorXf(np.ones((5, 2, 2, 2))),
            mi.TensorXf(np.ones((5, 3, 3, 2))),
            mi.TensorXf(np.ones((5, 3, 2, 3))),
        ]
        invalid_normals_cross = [
            mi.TensorXf(np.ones((5, 5, 3))),
            mi.TensorXf(np.ones((5, 4))),
        ]

        invalid_hessians_and_normals = [
            [mi.TensorXf(np.ones((5, 3, 2, 2))), mi.TensorXf(np.ones((6, 3)))]
        ]

        invalid_hessians_and_normals = invalid_hessians_and_normals + list(
            itertools.product(invalid_hessians_cross, invalid_normals_cross)
        )

        for i, (invalid_hessian, invalid_normal) in enumerate(
            invalid_hessians_and_normals
        ):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    self.bezier_3_surface.second_fundamental_form(
                        hessian=invalid_hessian, normal=invalid_normal
                    )

    def test_principal_curvatures(self):
        """
        Tests whether the principal curvatures are computed correctly.
        """

        excepted_principal_curvatures = np.array(
            [
                [-0.5, 0.5],
                [-np.sqrt(7) / 6, np.sqrt(7) / 6],
                [-np.sqrt(7) / 6, np.sqrt(7) / 6],
                [-0.5, 0.5],
                [0.0, 0.0],
            ]
        )

        jacobian = self.bezier_3_surface.evaluate_jacobian(self.parametric_coords)
        metric = self.bezier_3_surface.metric_tensor(jacobian=jacobian)
        inv_metric_tensor, _ = self.bezier_3_surface.inv_metric_tensor(metric)

        hessian = self.bezier_3_surface.evaluate_hessian(self.parametric_coords)
        normals = self.bezier_3_surface.evaluate_normals(self.parametric_coords)
        second_form = self.bezier_3_surface.second_fundamental_form(
            hessian=hessian, normal=normals
        )

        principal_curvatures = self.bezier_3_surface.principal_curvatures(
            inv_metric_tensor=inv_metric_tensor,
            second_form=second_form,
        )

        np.testing.assert_array_almost_equal(
            principal_curvatures[0].numpy(), excepted_principal_curvatures[:, 1]
        )

        np.testing.assert_array_almost_equal(
            principal_curvatures[1].numpy(), excepted_principal_curvatures[:, 0]
        )

    def test_principal_curvatures_invalid_input(self):
        """
        Tests whether the principal curvatures raises an error when invalid inv metric
        tensor or invalid second form are provided.
        """

        invalid_inv_metric_tensor_cross = [
            mi.TensorXf(np.ones((5, 5, 2, 2))),
            mi.TensorXf(np.ones((2, 2))),
            mi.TensorXf(np.ones((5, 3, 2))),
            mi.TensorXf(np.ones((5, 2, 3))),
        ]
        invalid_second_form_cross = [
            mi.TensorXf(np.ones((6, 5, 2, 2))),
            mi.TensorXf(np.ones((3, 2))),
            mi.TensorXf(np.ones((5, 4, 2))),
            mi.TensorXf(np.ones((5, 2, 4))),
        ]

        invalid_inv_metric_tensor_and_second_form = [
            [mi.TensorXf(np.ones((5, 2, 2))), mi.TensorXf(np.ones((6, 2, 2)))]
        ]

        invalid_inv_metric_tensor_and_second_form = (
            invalid_inv_metric_tensor_and_second_form
            + list(
                itertools.product(
                    invalid_inv_metric_tensor_cross, invalid_second_form_cross
                )
            )
        )

        for i, (
            invalid_inv_metric_tensor,
            invalid_second_form,
        ) in enumerate(invalid_inv_metric_tensor_and_second_form):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    self.bezier_3_surface.principal_curvatures(
                        inv_metric_tensor=invalid_inv_metric_tensor,
                        second_form=invalid_second_form,
                    )
