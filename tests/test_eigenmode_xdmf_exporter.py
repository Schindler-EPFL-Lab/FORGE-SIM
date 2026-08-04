import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import h5py
import mitsuba as mi
import numpy as np

from rebel_shape.isogeometric_analysis.eigenmode_xdmf_exporter import (
    EigenmodeXDMFExporter,
)
from rebel_shape.isogeometric_analysis.reissner_mindlin_modal_problem import (
    ReissnerMindlinModalProblem,
)
from rebel_shape.meshing.composite_iga_spline_surface_factory import (
    CompositeIGASplineSurfaceFactory,
)
from rebel_shape.meshing.patch import Patch
from rebel_shape.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)
from rebel_shape.splines.bspline_patch import BSplinePatch
from rebel_shape.splines.iga_spline_patch import IGASplinePatch

DOFS_PER_CONTROL_POINT = ReissnerMindlinModalProblem.DOFS_PER_CONTROL_POINT


class TestEigenmodeXDMFExporter(unittest.TestCase):
    """
    Contract tests for the eigenmode XDMF/HDF5 export.

    The exported files are read back by the modal post-processing tools (mode
    alignment, MAC metrics) and by ParaView, so the dataset layout, the deformed
    geometry convention and the canonical mode sign are a durable format
    contract. No eigenvalue solve is needed to pin them down: the exporter is
    fed synthetic eigenvectors on a small two-patch surface.
    """

    def setUp(self) -> None:
        mi.set_variant("llvm_ad_rgb")

        # Two unit squares placed side by side in the z = 0 plane, so the
        # surface bounding box measures 2 x 1 x 0: the smallest *positive*
        # extent (1.0) is the one the deformation scale must pick up.
        patches = []
        for patch_index, offset in enumerate(
            [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
        ):
            patch = BSplinePatch.unit_square(
                patch_name=f"patch{patch_index}",
                degree=2,
                u_subdivisions=1,
                v_subdivisions=1,
            )
            patch.control_points = mi.TensorXf(patch.control_points.numpy() + offset)
            patches.append(
                IGASplinePatch(
                    spline_patch=patch,
                    field_values=mi.TensorXf(
                        np.zeros(
                            (patch.control_points.shape[0], DOFS_PER_CONTROL_POINT),
                            dtype=np.float32,
                        )
                    ),
                )
            )

        factory = CompositeIGASplineSurfaceFactory("test_eigenmode_export")
        for patch in patches:
            factory.add_patch(patch)
        factory.link_sides(
            first_patch_name=patches[0].name,
            first_patch_side=Patch.Side.RIGHT,
            second_patch_name=patches[1].name,
            second_patch_side=Patch.Side.LEFT,
        )

        self.surface = TessellatedIGASplineSurface(
            iga_spline_surface=factory.build(),
            tessellation_subdivisions_per_patch=2,
        )
        self.vertices = self.surface.global_vertices.numpy().astype(np.float64)
        self.faces = self.surface.global_faces.numpy().astype(np.int32)
        self.num_control_points = self.surface.global_field_values.shape[0]
        self.num_dofs = DOFS_PER_CONTROL_POINT * self.num_control_points

    def _random_eigenvectors(self, num_modes: int) -> np.ndarray:
        """A deterministic synthetic eigenvector matrix of shape (n_dofs, n_modes)."""
        generator = np.random.default_rng(20240517)
        return generator.standard_normal((self.num_dofs, num_modes))

    def _uniform_eigenvector(self, dofs: np.ndarray) -> np.ndarray:
        """
        A single eigenvector assigning the same six DOFs to every control point.

        The spline basis is a partition of unity, so such a field interpolates
        to exactly `dofs` at every tessellation vertex.
        """
        return np.tile(dofs, self.num_control_points).reshape(-1, 1)

    def _export(
        self,
        output_folder: Path,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        mode_indices: list[int],
        scale_factor: float = 0.1,
    ) -> None:
        EigenmodeXDMFExporter(
            tessellated_iga_spline=self.surface,
            output_folder=output_folder,
            scale_factor=scale_factor,
        ).export_eigenmodes(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            mode_indices=mode_indices,
        )

    def test_hdf5_layout(self):
        """The HDF5 file holds the undeformed mesh plus one group per mode."""
        num_modes = 3
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_folder = Path(temporary_directory)
            self._export(
                output_folder=output_folder,
                eigenvalues=np.arange(num_modes, dtype=np.float64),
                eigenvectors=self._random_eigenvectors(num_modes),
                mode_indices=list(range(num_modes)),
            )

            self.assertTrue((output_folder / "eigenmodes.h5").exists())
            self.assertTrue((output_folder / "eigenmodes.xdmf").exists())

            num_vertices = self.vertices.shape[0]
            with h5py.File(output_folder / "eigenmodes.h5", "r") as h5_file:
                self.assertEqual(h5_file["/Mesh/topology"].dtype, np.int32)
                np.testing.assert_array_equal(h5_file["/Mesh/topology"][()], self.faces)
                np.testing.assert_array_equal(
                    h5_file["/Mesh/vertices"][()], self.vertices
                )

                self.assertEqual(len(h5_file["/Modes"]), num_modes)
                for mode_index in range(num_modes):
                    mode_group = h5_file[f"/Modes/mode_{mode_index}"]
                    self.assertEqual(
                        set(mode_group.keys()),
                        {
                            "geometry",
                            "displacement",
                            "rotation",
                            "displacement_magnitude",
                            "rotation_magnitude",
                        },
                    )
                    for name in ("geometry", "displacement", "rotation"):
                        self.assertEqual(mode_group[name].shape, (num_vertices, 3))
                        self.assertEqual(mode_group[name].dtype, np.float64)
                    for name in ("displacement_magnitude", "rotation_magnitude"):
                        self.assertEqual(mode_group[name].shape, (num_vertices, 1))
                        self.assertEqual(mode_group[name].dtype, np.float64)

                    np.testing.assert_allclose(
                        mode_group["displacement_magnitude"][()][:, 0],
                        np.linalg.norm(mode_group["displacement"][()], axis=1),
                        rtol=0.0,
                        atol=1.0e-15,
                    )
                    np.testing.assert_allclose(
                        mode_group["rotation_magnitude"][()][:, 0],
                        np.linalg.norm(mode_group["rotation"][()], axis=1),
                        rtol=0.0,
                        atol=1.0e-15,
                    )

    def test_fields_are_interpolated_in_global_vertex_order(self):
        """
        A control-point-uniform mode interpolates to that same value everywhere.

        This pins both the partition-of-unity interpolation and the scatter into
        global vertex order: every vertex of every patch must be written. The
        tolerance is 1e-5 because the spline basis is evaluated and accumulated
        in float32 (~1e-7 relative), on values of order 10.
        """
        dofs = np.array([1.0, -2.0, 3.0, 0.5, -0.25, 10.0])
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_folder = Path(temporary_directory)
            self._export(
                output_folder=output_folder,
                eigenvalues=np.array([1.0]),
                eigenvectors=self._uniform_eigenvector(dofs),
                mode_indices=[0],
            )

            with h5py.File(output_folder / "eigenmodes.h5", "r") as h5_file:
                displacement = h5_file["/Modes/mode_0/displacement"][()]
                rotation = h5_file["/Modes/mode_0/rotation"][()]

        num_vertices = self.vertices.shape[0]
        np.testing.assert_allclose(
            displacement,
            np.tile(dofs[:3], (num_vertices, 1)),
            rtol=0.0,
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            rotation,
            np.tile(dofs[3:], (num_vertices, 1)),
            rtol=0.0,
            atol=1.0e-5,
        )

    def test_deformed_geometry_uses_normalized_scaled_displacement(self):
        """
        Geometry is the mesh displaced by the unit-normalized mode shape, scaled
        by ``scale_factor * smallest_positive_bbox_extent`` (1.0 here, since the
        flat surface measures 2 x 1 x 0 and the zero extent is skipped).
        """
        scale_factor = 0.25
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_folder = Path(temporary_directory)
            self._export(
                output_folder=output_folder,
                eigenvalues=np.array([1.0]),
                eigenvectors=self._random_eigenvectors(1),
                mode_indices=[0],
                scale_factor=scale_factor,
            )

            with h5py.File(output_folder / "eigenmodes.h5", "r") as h5_file:
                displacement = h5_file["/Modes/mode_0/displacement"][()]
                geometry = h5_file["/Modes/mode_0/geometry"][()]

        unit_displacement = displacement / np.max(np.linalg.norm(displacement, axis=1))
        expected_geometry = self.vertices - unit_displacement * scale_factor * 1.0

        # Recomputed with the same float64 operations, so only bit-level
        # reassociation could differ.
        np.testing.assert_allclose(geometry, expected_geometry, rtol=0.0, atol=1.0e-15)

    def test_canonical_sign_is_invariant_to_eigenvector_sign(self):
        """
        Eigenvectors are defined up to sign, so the exporter flips a mode until
        its largest-magnitude displacement component is positive. Exporting a
        mode and its negation must therefore produce identical fields.
        """
        eigenvector = self._random_eigenvectors(1)
        exported = []
        with tempfile.TemporaryDirectory() as temporary_directory:
            for sign_index, sign in enumerate((1.0, -1.0)):
                output_folder = Path(temporary_directory) / f"sign_{sign_index}"
                self._export(
                    output_folder=output_folder,
                    eigenvalues=np.array([1.0]),
                    eigenvectors=sign * eigenvector,
                    mode_indices=[0],
                )
                with h5py.File(output_folder / "eigenmodes.h5", "r") as h5_file:
                    exported.append(
                        (
                            h5_file["/Modes/mode_0/displacement"][()],
                            h5_file["/Modes/mode_0/rotation"][()],
                        )
                    )

        (displacement, rotation), (flipped_displacement, flipped_rotation) = exported
        np.testing.assert_array_equal(displacement, flipped_displacement)
        np.testing.assert_array_equal(rotation, flipped_rotation)

        flat_displacement = displacement.reshape(-1)
        largest = flat_displacement[int(np.argmax(np.abs(flat_displacement)))]
        self.assertGreater(largest, 0.0)

    def test_selected_modes_are_written_in_export_order(self):
        """
        The HDF5 groups are numbered by export position, while the XDMF grid
        names and time values keep the original mode index. Exporting modes
        [2, 0] therefore writes ``mode_0`` from eigenvector column 2.
        """
        eigenvectors = np.stack(
            [
                self._uniform_eigenvector(np.full(DOFS_PER_CONTROL_POINT, value))[:, 0]
                for value in (1.0, 2.0, 3.0)
            ],
            axis=1,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_folder = Path(temporary_directory)
            self._export(
                output_folder=output_folder,
                eigenvalues=np.array([10.0, 20.0, 30.0]),
                eigenvectors=eigenvectors,
                mode_indices=[2, 0],
            )

            with h5py.File(output_folder / "eigenmodes.h5", "r") as h5_file:
                self.assertEqual(len(h5_file["/Modes"]), 2)
                first_displacement = h5_file["/Modes/mode_0/displacement"][()]
                second_displacement = h5_file["/Modes/mode_1/displacement"][()]

            grid_names = [
                grid.get("Name")
                for grid in ElementTree.parse(
                    output_folder / "eigenmodes.xdmf"
                ).getroot()[0][0]
            ]

        # Column 2 carries a value of 3.0 per DOF, column 0 a value of 1.0.
        np.testing.assert_allclose(first_displacement, 3.0, rtol=0.0, atol=1.0e-5)
        np.testing.assert_allclose(second_displacement, 1.0, rtol=0.0, atol=1.0e-5)
        self.assertEqual(grid_names, ["mode_2", "mode_0"])

    def test_xdmf_describes_a_temporal_collection(self):
        """
        The XDMF descriptor is a temporal collection with one uniform grid per
        exported mode, referencing the companion HDF5 datasets.
        """
        num_modes = 2
        num_vertices = self.vertices.shape[0]
        num_triangles = self.faces.shape[0]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_folder = Path(temporary_directory)
            self._export(
                output_folder=output_folder,
                eigenvalues=np.array([1.0, 2.0]),
                eigenvectors=self._random_eigenvectors(num_modes),
                mode_indices=list(range(num_modes)),
            )
            root = ElementTree.parse(output_folder / "eigenmodes.xdmf").getroot()

        collection = root[0][0]
        self.assertEqual(collection.get("GridType"), "Collection")
        self.assertEqual(collection.get("CollectionType"), "Temporal")
        self.assertEqual(len(collection), num_modes)

        for export_index, grid in enumerate(collection):
            self.assertEqual(grid.get("GridType"), "Uniform")

            topology = grid.find("Topology")
            self.assertEqual(topology.get("TopologyType"), "Triangle")
            self.assertEqual(topology.get("NumberOfElements"), str(num_triangles))
            self.assertEqual(
                topology.find("DataItem").text, "eigenmodes.h5:/Mesh/topology"
            )

            geometry_item = grid.find("Geometry").find("DataItem")
            self.assertEqual(
                geometry_item.text,
                f"eigenmodes.h5:/Modes/mode_{export_index}/geometry",
            )
            self.assertEqual(geometry_item.get("Dimensions"), f"{num_vertices} 3")

            self.assertEqual(grid.find("Time").get("Value"), str(export_index))

            attributes = grid.findall("Attribute")
            self.assertEqual(
                [attribute.get("Name") for attribute in attributes],
                [
                    "displacement",
                    "rotation",
                    "displacement_magnitude",
                    "rotation_magnitude",
                ],
            )
            for attribute in attributes:
                expected_columns = (
                    3 if attribute.get("AttributeType") == "Vector" else 1
                )
                data_item = attribute.find("DataItem")
                self.assertEqual(
                    data_item.get("Dimensions"),
                    f"{num_vertices} {expected_columns}",
                )
                self.assertEqual(
                    data_item.text,
                    f"eigenmodes.h5:/Modes/mode_{export_index}/{attribute.get('Name')}",
                )

    def test_invalid_arguments_raise(self):
        """A negative scale factor and out-of-range mode indices are rejected."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_folder = Path(temporary_directory)
            with self.assertRaises(ValueError):
                EigenmodeXDMFExporter(
                    tessellated_iga_spline=self.surface,
                    output_folder=output_folder,
                    scale_factor=-0.1,
                )

            eigenvectors = self._random_eigenvectors(2)
            for mode_index in (-1, 2):
                with self.assertRaises(ValueError):
                    self._export(
                        output_folder=output_folder,
                        eigenvalues=np.array([1.0, 2.0]),
                        eigenvectors=eigenvectors,
                        mode_indices=[mode_index],
                    )


if __name__ == "__main__":
    unittest.main()
