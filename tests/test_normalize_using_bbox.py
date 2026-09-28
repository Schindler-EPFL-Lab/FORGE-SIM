import unittest

import drjit as dr
import mitsuba as mi

from forge_sim.splines.cube_initialization import _get_normalizing_tf_matrix


class TestNormalizeUsingBBox(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mi.set_variant("llvm_ad_rgb")

    def test_normalize_zero_transform(self):
        """
        Tests that if points are already normalized then the return translation is zero
        and return scale is
        """
        points = mi.TensorXf(
            [
                [-1, -1, -1],
                [-1, 1, -1],
                [-1, 1, 1],
                [-1, -1, 1],
                [1, -1, -1],
                [1, 1, -1],
                [1, 1, 1],
                [1, -1, 1],
            ]
        )

        transform = _get_normalizing_tf_matrix(points)
        transform_ref = mi.Transform4f()
        self.assertTrue(dr.allclose(transform.matrix, transform_ref.matrix))

    def test_normalize_only_translate(self):
        """
        Tests that the point normalization works in 3d
        """
        points = mi.TensorXf(
            [
                [0, 1, 2],
                [0, 3, 2],
                [0, 3, 4],
                [0, 1, 4],
                [2, 3, 2],
                [2, 3, 2],
                [2, 3, 4],
                [2, 1, 4],
            ]
        )

        transform = _get_normalizing_tf_matrix(points)
        transform_ref = mi.Transform4f().translate(mi.Point3f([-1, -2, -3]))
        self.assertTrue(dr.allclose(transform.matrix, transform_ref.matrix))

    def test_normalize_scale_only(self):
        """
        Tests that the normalization correctly scales a bounding box at the origin.
        """
        points = mi.TensorXf(
            [
                [-1, -2, -3],
                [-1, 2, -3],
                [-1, 2, 3],
                [-1, -2, 3],
                [1, -2, -3],
                [1, 2, -3],
                [1, 2, 3],
                [1, -2, 3],
            ]
        )

        transform = _get_normalizing_tf_matrix(points)
        transform_ref = mi.Transform4f().scale(mi.Point3f([1, 1 / 2, 1 / 3]))
        self.assertTrue(dr.allclose(transform.matrix, transform_ref.matrix))

    def test_normalize_scale_and_translate(self):
        """
        Tests that the normalization correctly scales and translates a bounding box not
        centered at the origin, with different scales across all axes x, y, z.
        """
        points = mi.TensorXf(
            [
                [0, 0, 0],
                [0, 4, 0],
                [0, 4, 6],
                [0, 0, 6],
                [2, 0, 0],
                [2, 4, 0],
                [2, 4, 6],
                [2, 0, 6],
            ]
        )

        transform = _get_normalizing_tf_matrix(points)
        transform_ref = (
            mi.Transform4f()
            .scale(mi.Point3f([1, 1 / 2, 1 / 3]))
            .translate(mi.Point3f([-1, -2, -3]))
        )
        self.assertTrue(dr.allclose(transform.matrix, transform_ref.matrix))

    def test_raises_error_if_points_ndim_not_2(self):
        """
        Tests that a RuntimeError is raised if points does not have 2 dimensions.
        """
        points = mi.TensorXf([1, 2, 3])  # 1D tensor
        with self.assertRaises(RuntimeError):
            _get_normalizing_tf_matrix(points)

        points = mi.TensorXf([[[1, 2, 3]]])  # 3D tensor
        with self.assertRaises(RuntimeError):
            _get_normalizing_tf_matrix(points)

    def test_raises_error_if_points_shape1_not_3(self):
        """
        Tests that a RuntimeError is raised if points.shape[1] is not 3.
        """
        points = mi.TensorXf([[1, 2], [3, 4]])  # shape (2, 2)
        with self.assertRaises(RuntimeError):
            _get_normalizing_tf_matrix(points)
