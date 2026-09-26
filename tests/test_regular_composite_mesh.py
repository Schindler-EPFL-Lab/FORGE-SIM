import tempfile
import unittest
from pathlib import Path

import jsonpickle
import mitsuba as mi
import numpy as np

from rebel_shape.meshing.patch import Patch
from rebel_shape.meshing.regular_composite_mesh import RegularCompositeMesh


class TestRegularCompositeMesh(unittest.TestCase):
    """
    Tests for `RegularCompositeMesh`: jsonpickle serialization round-trips
    (including the `from_file` convenience) that check the welded global mesh
    and per-patch side-index recomputation, plus the actual `cubed_sphere`
    geometry (vertex positions on the sphere of the given radius and
    outward-pointing vertex normals), which the serialization tests never
    inspect.
    """

    @classmethod
    def setUpClass(cls):
        mi.set_variant("llvm_ad_rgb")

    def setUp(self):
        self.composite_mesh = RegularCompositeMesh.cubed_sphere(
            name="shape", subdivisions=3, radius=1.0
        )

    def _assert_matches(
        self, restored: RegularCompositeMesh, original: RegularCompositeMesh
    ) -> None:
        """Asserts `restored` reproduces `original`'s patches and welded global mesh."""
        self.assertIsInstance(restored, RegularCompositeMesh)
        self.assertEqual(len(restored._patch_list), len(original._patch_list))
        np.testing.assert_allclose(
            restored.global_vertices.numpy(), original.global_vertices.numpy()
        )
        np.testing.assert_array_equal(
            restored.global_faces.numpy(), original.global_faces.numpy()
        )

    def test_jsonpickle_roundtrip_preserves_global_mesh(self):
        """encode -> decode reproduces the welded global vertices and faces."""
        restored = jsonpickle.decode(jsonpickle.encode(self.composite_mesh))
        self._assert_matches(restored, self.composite_mesh)

    def test_roundtrip_recomputes_patch_side_indices(self):
        """Restored patches recompute their side-index arrays from texcoords."""
        restored = jsonpickle.decode(jsonpickle.encode(self.composite_mesh))
        for restored_patch, original_patch in zip(restored, self.composite_mesh):
            for side in Patch.Side:
                np.testing.assert_array_equal(
                    np.array(restored_patch.get_node_indices_for(side)),
                    np.array(original_patch.get_node_indices_for(side)),
                )

    def test_from_file_roundtrip(self):
        """`from_file` reloads a composite written with jsonpickle."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shape.json"
            path.write_text(jsonpickle.encode(self.composite_mesh))
            restored = RegularCompositeMesh.from_file(path)
        self._assert_matches(restored, self.composite_mesh)

    def test_from_file_missing_raises(self):
        """`from_file` raises FileNotFoundError for a non-existent path."""
        with self.assertRaises(FileNotFoundError):
            RegularCompositeMesh.from_file(Path("/nonexistent/shape.json"))

    def test_cubed_sphere_vertices_lie_on_sphere_of_given_radius(self):
        """
        Every global vertex must have Euclidean norm equal to `radius`.

        Tolerance: vertices are stored as float32 (`mi.TensorXf`), whose
        relative machine epsilon is ~1.2e-7; the normalize-and-rescale used to
        build the sphere accumulates only a handful of ULPs on top of that
        (empirically ~1e-7 relative here). `rtol=1e-4` is ~1000x looser than
        that noise floor (comfortable across platforms/build configs) while
        still ~4 orders of magnitude tighter than the defect this pins: an
        unprojected cube corner sits `sqrt(3)/1 ~= 1.73x` farther from the
        origin than a face center, a ~70% relative error.
        """
        radius = 2.5
        composite_mesh = RegularCompositeMesh.cubed_sphere(
            name="geometry_probe", subdivisions=3, radius=radius
        )

        vertex_norms = np.linalg.norm(composite_mesh.global_vertices.numpy(), axis=1)

        np.testing.assert_allclose(vertex_norms, radius, rtol=1e-4, atol=0.0)

    def test_cubed_sphere_vertex_normals_point_outward(self):
        """
        Every global vertex normal should point away from the sphere's center
        (positive dot product with the radial direction at that vertex), not
        the flat per-face direction an unprojected cube would carry and not an
        inverted normal.

        Threshold: the defect this pins is normals left at the cube's flat face
        normals after the vertices are projected. Its value is exact, not
        empirical. The worst such vertex is a face-interior one, at cube
        coordinates `(1, 1/3, 1/3)` for `subdivisions=3`: it keeps the face
        normal `(1, 0, 0)` while its projected radial direction is
        `(3, 1, 1)/sqrt(11)`, so the alignment is `3/sqrt(11) = 0.9045`. An
        inverted normal gives a negative value. A correctly averaged mesh gives
        0.9971 here. `0.95` sits between the defect (0.9045) and the correct
        value (0.9971), 0.046 clear of the first and 0.047 of the second, so it
        separates the two without asserting a particular mesh quality.
        """
        radius = 2.5
        composite_mesh = RegularCompositeMesh.cubed_sphere(
            name="normal_probe", subdivisions=3, radius=radius
        )

        vertices = composite_mesh.global_vertices.numpy()
        normals = composite_mesh.global_vertex_normals.numpy()
        radial_directions = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

        outward_alignment = np.sum(normals * radial_directions, axis=1)

        self.assertGreater(float(outward_alignment.min()), 0.95)
