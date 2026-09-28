try:
    OPEN3D_IMPORTED = True
    import open3d as o3d

    from forge_sim.metrics.shapes import (
        NERFSTUDIO_AXIS_FIX,
        load_reference_in_scene,
    )
    from forge_sim.scripts.evaluation.stitch_and_eval_cubed_sphere_mesh import (
        CUBE_FACE_ORDER,
        discover_patch_files,
        stitch_patches,
    )
except ImportError:
    OPEN3D_IMPORTED = False

import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

import numpy as np


@unittest.skipUnless(OPEN3D_IMPORTED, "open3d is not installed.")
class TestStitchAndEvalCubedSphereMesh(unittest.TestCase):
    @staticmethod
    def _position_set(points: np.ndarray) -> set:
        """The set of rounded `(x, y, z)` tuples in `points`
        (order-independent)."""
        return {tuple(np.round(point, 4)) for point in np.asarray(points)}

    def _write_triangle_ply(self, path: Path, vertices: Sequence) -> None:
        """Writes a single-triangle mesh with the given `vertices` to `path`."""
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
            o3d.utility.Vector3iVector(np.array([[0, 1, 2]], dtype=np.int32)),
        )
        o3d.io.write_triangle_mesh(str(path), mesh)

    def test_nerfstudio_axis_fix_is_x_rotation_by_90_degrees(self):
        """NERFSTUDIO_AXIS_FIX is a proper +90 deg rotation about X, no translation."""
        rotation = NERFSTUDIO_AXIS_FIX[:3, :3]

        # Proper rotation: orthonormal with determinant +1.
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)

        # +90 deg about X: fixes e_x, maps e_y -> e_z and e_z -> -e_y.
        expected = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        np.testing.assert_allclose(rotation, expected, atol=1e-12)

        # No translation and a homogeneous bottom row.
        np.testing.assert_allclose(NERFSTUDIO_AXIS_FIX[:3, 3], 0.0, atol=1e-12)
        np.testing.assert_allclose(NERFSTUDIO_AXIS_FIX[3, :], [0, 0, 0, 1], atol=1e-12)

    def test_discover_patch_files_picks_highest_iteration(self):
        """The highest `iteration-NNN` file is chosen for each cube face."""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for face in CUBE_FACE_ORDER:
                for iteration in (1, 12, 3):
                    name = f"shape_{face}_20260101-000000_iteration-{iteration:03}.ply"
                    (directory / name).touch()

            patch_files = discover_patch_files(directory, "shape")

            self.assertEqual(set(patch_files), set(CUBE_FACE_ORDER))
            for face in CUBE_FACE_ORDER:
                self.assertIn("iteration-012", patch_files[face].name)

    def test_discover_patch_files_raises_on_missing_face(self):
        """A missing cube face raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for face in CUBE_FACE_ORDER[:-1]:
                (directory / f"shape_{face}_iteration-001.ply").touch()

            with self.assertRaises(FileNotFoundError):
                discover_patch_files(directory, "shape")

    def test_stitch_patches_welds_coincident_vertices(self):
        """Stitching merges coincident seam vertices and preserves every triangle."""
        # `left` and `right` share two vertices; the other four are disjoint.
        triangles = {
            "left": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            "right": [(0, 0, 0), (0, 1, 0), (2, 2, 2)],
            "back": [(10, 0, 0), (11, 0, 0), (10, 1, 0)],
            "front": [(20, 0, 0), (21, 0, 0), (20, 1, 0)],
            "bottom": [(30, 0, 0), (31, 0, 0), (30, 1, 0)],
            "top": [(40, 0, 0), (41, 0, 0), (40, 1, 0)],
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            patch_files = {}
            for face, vertices in triangles.items():
                path = directory / f"{face}.ply"
                self._write_triangle_ply(path, vertices)
                patch_files[face] = path

            welded_vertices, welded_faces = stitch_patches(patch_files, 1e-5)

            # 6 triangles * 3 = 18 raw vertices; left & right share 2 -> 16 unique.
            self.assertEqual(welded_vertices.shape[0], 16)
            self.assertEqual(welded_faces.shape, (6, 3))
            self.assertEqual(
                np.unique(welded_vertices, axis=0).shape[0], welded_vertices.shape[0]
            )

            # The welded surface reproduces every original triangle (as a position set).
            welded_triangles = {
                frozenset(self._position_set(welded_vertices[face]))
                for face in welded_faces
            }
            original_triangles = {
                frozenset(self._position_set(vertices))
                for vertices in triangles.values()
            }
            self.assertEqual(welded_triangles, original_triangles)

    def test_load_reference_in_scene_applies_transform(self):
        """load_reference_in_scene reads the mesh and applies the 4x4 transform."""
        vertices = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.ply"
            self._write_triangle_ply(path, vertices)

            identity = load_reference_in_scene(path, np.eye(4))
            self.assertEqual(
                self._position_set(identity.vertex.positions.numpy()),
                self._position_set(vertices),
            )

            rotated = load_reference_in_scene(path, NERFSTUDIO_AXIS_FIX)
            expected = np.asarray(vertices) @ NERFSTUDIO_AXIS_FIX[:3, :3].T
            self.assertEqual(
                self._position_set(rotated.vertex.positions.numpy()),
                self._position_set(expected),
            )
