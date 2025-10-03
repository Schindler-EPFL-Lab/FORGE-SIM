try:
    OPEN3D_IMPORTED = True
    import open3d as o3d

    from rebel_shape.metrics.shapes import tessellated_spline_to_open3d_mesh
except ImportError:
    OPEN3D_IMPORTED = False

import unittest

import mitsuba as mi
import numpy as np

from rebel_shape.meshing.composite_spline_surface_factory import (
    CompositeSplineSurfaceFactory,
)
from rebel_shape.meshing.patch import Patch
from rebel_shape.meshing.tessellated_spline_surface import TessellatedSplineSurface
from rebel_shape.splines.bspline_patch import BSplinePatch


@unittest.skipUnless(OPEN3D_IMPORTED, "open3d is not installed.")
class TestSplineToOpen3D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mi.set_variant("llvm_ad_rgb")

    def test_spline_to_open3d_mesh(self):
        """Tests that tessellated spline surface is converted properly to open3d mesh"""
        patch1 = BSplinePatch.unit_square(
            patch_name="patch1", degree=2, u_subdivisions=1, v_subdivisions=1
        )
        patch1.control_points = mi.TensorXf(
            patch1.control_points.numpy() + np.array([-0.5, 0.0, 0.0])
        )

        patch2 = BSplinePatch.unit_square(
            patch_name="patch2", degree=2, u_subdivisions=1, v_subdivisions=1
        )
        patch2.control_points = mi.TensorXf(
            patch2.control_points.numpy() + np.array([0.5, 0.0, 0.0])
        )

        composite_factory = CompositeSplineSurfaceFactory("test_composite")
        composite_factory.add_patch(patch1)
        composite_factory.add_patch(patch2)

        composite_factory.link_sides(
            first_patch_name=patch1.name,
            first_patch_side=Patch.Side.RIGHT,
            second_patch_name=patch2.name,
            second_patch_side=Patch.Side.LEFT,
        )
        spline_surface = composite_factory.build()

        tessellated_surface = TessellatedSplineSurface(
            composite_spline_surface=spline_surface,
            tessellation_subdivisions_per_patch=2,
        )

        gt_vertices = np.array(
            [
                [-1, -0.5, 0],
                [-0.5, -0.5, 0],
                [0, -0.5, 0],
                [-1, 0, 0],
                [-0.5, 0, 0],
                [0, 0, 0],
                [-1, 0.5, 0],
                [-0.5, 0.5, 0],
                [0, 0.5, 0],
                [0.5, -0.5, 0],
                [1, -0.5, 0],
                [0.5, 0, 0],
                [1, 0, 0],
                [0.5, 0.5, 0],
                [1, 0.5, 0],
            ]
        )

        gt_faces = np.array(
            [
                [0, 1, 4],
                [0, 4, 3],
                [1, 2, 5],
                [1, 5, 4],
                [3, 4, 7],
                [3, 7, 6],
                [4, 5, 8],
                [4, 8, 7],
                [2, 9, 11],
                [2, 11, 5],
                [9, 10, 12],
                [9, 12, 11],
                [5, 11, 13],
                [5, 13, 8],
                [11, 12, 14],
                [11, 14, 13],
            ]
        )

        gt_open3d_mesh = o3d.t.geometry.TriangleMesh(
            vertex_positions=o3d.core.Tensor(
                gt_vertices,
                dtype=o3d.core.Dtype.Float32,
            ),
            triangle_indices=o3d.core.Tensor(gt_faces, dtype=o3d.core.Dtype.Int32),
        )

        converted_open3d_mesh = tessellated_spline_to_open3d_mesh(
            tessellated_spline_surface=tessellated_surface
        )

        self.assertTrue(
            np.allclose(
                converted_open3d_mesh.vertex.positions.numpy(),
                gt_open3d_mesh.vertex.positions.numpy(),
            )
        )

        self.assertTrue(
            np.allclose(
                converted_open3d_mesh.triangle.indices.numpy(),
                gt_open3d_mesh.triangle.indices.numpy(),
            )
        )
