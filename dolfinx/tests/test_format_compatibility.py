"""
On-disk format compatibility of the XDMF reader.

Round-trip tests (`test_xdmf_time_series`) cannot catch the one failure mode that
matters most for an archive: a change to the writer mirrored by a matching change
to the reader passes every round trip while silently making every previously
recorded file unreadable -- or, worse, readable and wrong. The only guard against
that is a file the current code did not write.

`iga_dialect_fixture.{xdmf,h5}` is that file: a 50-vertex, 48-face, 3-step
excerpt carved verbatim out of a recorded IGA heat run
(`suzanne/iga_heat_simulation/iga_heat.*`), keeping the IGA exporter's dialect
and dtypes exactly as they are on disk. It differs from anything
`xdmf_time_series` emits: a `<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>` line, an
`xmlns:xi` attribute on the root, an **int32** topology, and an extra `Normals`
attribute the reader must ignore.

The two tests against the full recorded archive are confirmation, not the guard:
they skip when the archive is absent, which is the normal case in CI. Point
`REBEL_SHAPE_RECORDED_RUNS` at a directory holding `iga_heat_simulation/` and
`fem_heat_simulation/` to enable them.

Run from the repository root::

    conda run -n dolfinx-rebel-shape python -m unittest discover dolfinx/tests
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

# Imported for its module-level side effect: `_fem_fixtures` puts `dolfinx/` and
# the repository root on `sys.path`.
import _fem_fixtures  # noqa: F401

from compare_iga_fem import CompareIGAFEMParameters, compare_solutions, load_xdmf_data

_FIXTURE = Path(__file__).resolve().parent / "iga_dialect_fixture.xdmf"

# Recorded end-to-end runs, if this machine has them. They live outside the
# repository (tens of MB per file), so their tests are confirmation only.
_RECORDED_RUNS = Path(
    os.environ.get(
        "REBEL_SHAPE_RECORDED_RUNS",
        Path.home() / "Schindler/rebel-shape_paper/End-to-end-verification/suzanne",
    )
)
_RECORDED_IGA = _RECORDED_RUNS / "iga_heat_simulation" / "iga_heat.xdmf"
_RECORDED_FEM = _RECORDED_RUNS / "fem_heat_simulation" / "fem_heat_gt.xdmf"
_HAVE_RECORDED = _RECORDED_IGA.exists() and _RECORDED_FEM.exists()
_SKIP_REASON = f"No recorded runs at {_RECORDED_RUNS} (set REBEL_SHAPE_RECORDED_RUNS)."


class TestFormatCompatibility(unittest.TestCase):
    """
    The reader against files it did not write.
    """

    def test_reads_the_recorded_iga_dialect(self):
        """
        The committed excerpt of a real IGA export loads with its geometry,
        connectivity, times and every snapshot intact.

        Each assertion targets one way the dialect could trip the reader: the
        DOCTYPE / `xmlns:xi` prologue (the XML parse), the int32 topology (index
        arithmetic that assumes int64), the extra `Normals` attribute (an
        unexpected dataset in `Mesh/`), and `Step_10`-style keys (which sort
        wrongly as strings, though this excerpt keeps only three steps).
        """
        vertices, faces, times, temperatures = load_xdmf_data(_FIXTURE)

        with h5py.File(_FIXTURE.with_suffix(".h5"), "r") as h5_file:
            # A statement about the fixture as much as the reader: if this file
            # is ever re-carved, it must keep the archival dtypes.
            self.assertEqual(h5_file["Mesh/Topology"].dtype, np.int32)
            self.assertEqual(h5_file["Mesh/Vertices"].dtype, np.float64)
            self.assertIn("Normals", h5_file["Mesh"])
            np.testing.assert_array_equal(vertices, h5_file["Mesh/Vertices"][:])
            np.testing.assert_array_equal(faces, h5_file["Mesh/Topology"][:])
            for index in range(len(times)):
                np.testing.assert_array_equal(
                    temperatures[index], h5_file[f"Function/Step_{index}"][:]
                )

        self.assertEqual(vertices.shape, (50, 3))
        self.assertEqual(faces.shape, (48, 3))
        self.assertEqual(len(times), 3)
        self.assertEqual(temperatures.shape, (3, 50))
        self.assertLess(faces.max(), len(vertices))
        self.assertEqual(times, sorted(times))
        # The excerpt carries a real inpainted temperature field, so a reader
        # that returned zeros or a transposed view would be caught here too.
        self.assertGreater(temperatures.std(), 0.0)

    @unittest.skipUnless(_HAVE_RECORDED, _SKIP_REASON)
    def test_reads_full_recorded_runs(self):
        """
        Confirmation on the full recorded files, at production size (393,218 IGA
        vertices against 129,526 FEM ones): both load, both are internally
        consistent, and their time axes agree.
        """
        for label, path in (("IGA", _RECORDED_IGA), ("FEM", _RECORDED_FEM)):
            with self.subTest(side=label):
                vertices, faces, times, temperatures = load_xdmf_data(path)
                self.assertLess(faces.max(), len(vertices))
                self.assertEqual(temperatures.shape, (len(times), len(vertices)))
                self.assertEqual(times, sorted(times))
                self.assertEqual(len(set(times)), len(times))

    @unittest.skipUnless(_HAVE_RECORDED, _SKIP_REASON)
    def test_compares_a_full_recorded_pair(self):
        """
        The real-data version of the difference-output consistency test, on
        exactly the asymmetry that produced the fault: 393,218 IGA vertices
        against 129,526 FEM ones, i.e. a 3x finer reference side.

        Every recorded comparison in the paper ran through this path.
        """
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        output = directory / "difference.xdmf"

        compare_solutions(
            CompareIGAFEMParameters(
                iga_xdmf=_RECORDED_IGA, fem_xdmf=_RECORDED_FEM, output_xdmf=output
            )
        )

        iga_vertices, iga_faces, _, iga_temperatures = load_xdmf_data(_RECORDED_IGA)
        _, _, _, fem_temperatures = load_xdmf_data(_RECORDED_FEM)
        vertices, faces, times, difference = load_xdmf_data(output)

        np.testing.assert_array_equal(vertices, iga_vertices)
        np.testing.assert_array_equal(faces, iga_faces)
        self.assertLess(faces.max(), len(vertices))
        self.assertEqual(difference.shape, (len(times), len(iga_vertices)))

        # A gather can only ever produce values the FEM field already holds, so
        # the recovered FEM column must stay inside the FEM field's own range at
        # every step. Unwritten slots -- the failure mode the gather direction
        # exists to prevent -- would sit at exactly 0.0, which is outside the
        # range of every step but the first (where the FEM minimum is ~1e-14).
        # The 1e-9 slack is the round-off of recovering `fem = iga - difference`.
        recovered_fem = iga_temperatures[-len(times) :] - difference
        for step in range(1, len(times)):
            with self.subTest(step=step):
                self.assertGreaterEqual(
                    recovered_fem[step].min(), fem_temperatures[step].min() - 1e-9
                )
                self.assertLessEqual(
                    recovered_fem[step].max(), fem_temperatures[step].max() + 1e-9
                )


if __name__ == "__main__":
    unittest.main()
