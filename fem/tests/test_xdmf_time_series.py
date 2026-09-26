import itertools
import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from _xdmf_fixtures import grid_mesh

from fem.iga_fem_comparison.xdmf_solution import XDMFSolution
from fem.xdmf_time_series import XDMFTimeSeries


class TestWriteTimeSeriesXdmf(unittest.TestCase):
    """
    Writer/reader contract of the shared XDMF time-series writer.

    This is the single writer both the FEM heat runner and the IGA/FEM
    difference output depend on, so what it puts on disk defines the
    interface between the solvers, ParaView, and `compare_iga_fem`. The
    tests pin that interface rather than the XML text: the writer/reader
    round trip is lossless, the two cosmetic layouts describe identical
    data, the on-disk dtypes are normalised whatever the caller passes in,
    the `.h5` is referenced relatively so a written pair can be moved, and
    `create_parent_dirs` behaves as its two callers (which pass opposite
    values) require.

    The XML is deliberately left unpinned as a golden string: pinning it
    would restate the code and break on every cosmetic edit. The archival
    format contract is pinned instead by `test_format_compatibility`,
    against a committed real-format file.

    Everything here is pure numpy + h5py.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # A 5x5 grid: small enough to compare element-wise, large enough that a
        # transposed or truncated array would not accidentally still match.
        cls.vertices, cls.faces = grid_mesh(5)
        cls.times = [0.0, 1.5, 3.0]
        cls.fields = [
            cls.vertices[:, 0] + index * cls.vertices[:, 1]
            for index in range(len(cls.times))
        ]

    def setUp(self) -> None:
        self._directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self._directory, ignore_errors=True)

    def _write(self, path: Path, **overrides) -> Path:
        """Writes the class fixture series to `path`, with keyword overrides."""
        arguments = {
            "attribute_name": "temperature",
            "layout": XDMFTimeSeries.Layout.COMPACT,
            "declare_float_precision": True,
        }
        arguments.update(overrides)
        XDMFTimeSeries.write(
            path, self.vertices, self.faces, self.times, self.fields, **arguments
        )
        return path

    def test_written_series_reads_back_identically(self):
        """
        The pair this writer produces is read back by `XDMFSolution.from_file`
        (the reader every consumer in the repository uses) with the
        vertices, faces, times and every field bit-identical.

        Exact equality is the right gate, not a tolerance: float64 in, float64
        out, with no arithmetic anywhere on the path. A tolerance here would hide
        a silent demotion to float32 (which would show up as ~1e-7 differences).
        """
        path = self._write(self._directory / "series.xdmf")

        solution = XDMFSolution.from_file(path)

        np.testing.assert_array_equal(solution.vertices, self.vertices)
        np.testing.assert_array_equal(solution.faces, self.faces)
        self.assertEqual(list(solution.times), self.times)
        self.assertEqual(
            solution.temperatures.shape, (len(self.times), len(self.vertices))
        )
        for index, field in enumerate(self.fields):
            np.testing.assert_array_equal(solution.temperatures[index], field)

    def test_layout_and_precision_are_purely_cosmetic(self):
        """
        All four combinations of `XDMFTimeSeries.Layout` and
        `declare_float_precision` produce byte-identical HDF5 payloads and
        identical loader output.

        The class docstring calls these differences cosmetic and keeps them only
        so that unifying two historical writers left each consumer's files
        unchanged; that claim is prose today and is the licence for ever
        collapsing the two layouts. This test makes it enforceable.
        """
        reference = None
        for layout, declare_precision in itertools.product(
            XDMFTimeSeries.Layout, (False, True)
        ):
            with self.subTest(layout=layout, precision=declare_precision):
                path = self._write(
                    self._directory / f"{layout.value}_{declare_precision}.xdmf",
                    layout=layout,
                    declare_float_precision=declare_precision,
                )
                with h5py.File(path.with_suffix(".h5"), "r") as h5_file:
                    datasets = {
                        "vertices": h5_file["Mesh/Vertices"][:],
                        "topology": h5_file["Mesh/Topology"][:],
                        **{
                            key: h5_file[f"Function/{key}"][:]
                            for key in h5_file["Function"]
                        },
                    }
                loaded = XDMFSolution.from_file(path)

                if reference is None:
                    reference = (datasets, loaded)
                    continue

                reference_datasets, reference_loaded = reference
                self.assertEqual(datasets.keys(), reference_datasets.keys())
                for key, values in datasets.items():
                    np.testing.assert_array_equal(values, reference_datasets[key])
                np.testing.assert_array_equal(
                    loaded.vertices, reference_loaded.vertices
                )
                np.testing.assert_array_equal(loaded.faces, reference_loaded.faces)
                self.assertEqual(loaded.times, reference_loaded.times)
                np.testing.assert_array_equal(
                    loaded.temperatures, reference_loaded.temperatures
                )

    def test_on_disk_dtypes_are_normalised(self):
        """
        Vertices and fields land as float64 and the connectivity as int64
        whatever the caller passes in: the contract the docstring states and
        the one `compare_iga_fem` relies on when it indexes one file's topology
        into another file's geometry.

        The FEM runner passes float64/uint32 and the difference writer float64/
        int32-or-int64, so the normalisation is load-bearing in production.
        """
        path = self._directory / "narrow.xdmf"
        XDMFTimeSeries.write(
            path,
            self.vertices.astype(np.float32),
            self.faces.astype(np.int32),
            self.times[:1],
            [self.fields[0].astype(np.float32)],
            attribute_name="temperature",
            layout=XDMFTimeSeries.Layout.COMPACT,
            declare_float_precision=True,
        )

        with h5py.File(path.with_suffix(".h5"), "r") as h5_file:
            self.assertEqual(h5_file["Mesh/Vertices"].dtype, np.float64)
            self.assertEqual(h5_file["Mesh/Topology"].dtype, np.int64)
            self.assertEqual(h5_file["Function/Step_0"].dtype, np.float64)
            # Widening must not perturb the values: every fixture coordinate is
            # exactly representable in float32, so the widened copy is exact.
            np.testing.assert_array_equal(h5_file["Mesh/Vertices"][:], self.vertices)
            np.testing.assert_array_equal(h5_file["Mesh/Topology"][:], self.faces)

    def test_pair_is_relocatable(self):
        """
        The `.xdmf` refers to its sidecar by bare file name (`stem.h5:/...`),
        never by absolute path, so a written pair can be copied elsewhere and
        still open.

        This is not hypothetical: the recorded runs are read from directories
        they were not written in, and `XDMFSolution.from_file` finds the `.h5` by
        `with_suffix`, so an absolute reference would leave the XML pointing at a
        machine-specific path while the data still loaded, broken for ParaView
        only, i.e. silently.
        """
        path = self._write(self._directory / "series.xdmf")

        xml = path.read_text()
        self.assertIn("series.h5:/Mesh/Vertices", xml)
        self.assertNotIn(str(self._directory), xml)

        moved = self._directory / "elsewhere"
        moved.mkdir()
        for suffix in (".xdmf", ".h5"):
            shutil.copy(path.with_suffix(suffix), moved / f"series{suffix}")

        solution = XDMFSolution.from_file(moved / "series.xdmf")
        np.testing.assert_array_equal(solution.vertices, self.vertices)
        self.assertEqual(list(solution.times), self.times)
        np.testing.assert_array_equal(solution.temperatures[0], self.fields[0])

    def test_create_parent_dirs_contract(self):
        """
        `create_parent_dirs=True` creates the output directory; the default
        `False` refuses to.

        The two production callers rely on opposite settings: the FEM runner
        has already created `output_folder` and passes False, while the
        difference writer is handed an arbitrary `--output-xdmf` and passes True,
        so both halves of this behaviour are in use.
        """
        missing = self._directory / "not" / "there"

        with self.assertRaises(OSError):
            self._write(missing / "series.xdmf")

        self._write(missing / "series.xdmf", create_parent_dirs=True)
        self.assertTrue((missing / "series.xdmf").is_file())
        self.assertTrue((missing / "series.h5").is_file())
