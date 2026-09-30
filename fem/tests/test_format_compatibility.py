import os
import shutil
import tempfile
import unittest
from pathlib import Path

# Imported for its module-level side effect: `_fem_fixtures` puts the
# repository root on `sys.path`, so `fem` and `forge_sim` resolve.
import _fem_fixtures  # noqa: F401
import h5py
import numpy as np

from fem.compare_iga_fem import CompareIGAFEMParameters, compare_solutions
from fem.iga_fem_comparison.xdmf_solution import XDMFSolution

# Recorded end-to-end runs, if this machine has them. They live outside the
# repository (tens of MB per file), so their tests are confirmation only. An unset
# or empty `FORGE_SIM_RECORDED_RUNS` means there are none.
_RECORDED_RUNS_VALUE = os.environ.get("FORGE_SIM_RECORDED_RUNS", "")
_RECORDED_RUNS = Path(_RECORDED_RUNS_VALUE)
_RECORDED_IGA = _RECORDED_RUNS / "iga_heat_simulation" / "iga_heat.xdmf"
_RECORDED_FEM = _RECORDED_RUNS / "fem_heat_simulation" / "fem_heat_gt.xdmf"
_HAVE_RECORDED = (
    _RECORDED_RUNS_VALUE != "" and _RECORDED_IGA.exists() and _RECORDED_FEM.exists()
)
_SKIP_REASON = (
    f"No recorded runs at {_RECORDED_RUNS} (set FORGE_SIM_RECORDED_RUNS)."
    if _RECORDED_RUNS_VALUE
    else "FORGE_SIM_RECORDED_RUNS is not set."
)


class TestFormatCompatibility(unittest.TestCase):
    """
    The reader against files it did not write.

    Round-trip tests (`test_xdmf_time_series`) cannot catch the one failure
    mode that matters most for an archive: a change to the writer mirrored
    by a matching change to the reader passes every round trip while
    silently making every previously recorded file unreadable, or,
    worse, readable and wrong. The only guard against that is a file the
    current code did not write.

    `iga_dialect_fixture.{xdmf,h5}` is that file: a 50-vertex, 48-face,
    3-step excerpt carved verbatim out of a recorded IGA heat run
    (`suzanne/iga_heat_simulation/iga_heat.*`), keeping the IGA exporter's
    dialect and dtypes exactly as they are on disk. It differs from
    anything `xdmf_time_series` emits: a
    `<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>` line, an `xmlns:xi` attribute on
    the root, a topology stored as int32 rather than the usual int64, and
    an extra `Normals` attribute the reader must ignore.

    The two tests against the full recorded archive are confirmation, not
    the guard: they skip when the archive is absent, which is the normal
    case in CI. Point `FORGE_SIM_RECORDED_RUNS` at a directory holding
    `iga_heat_simulation/` and `fem_heat_simulation/` to enable them.
    """

    _FIXTURE = Path(__file__).resolve().parent / "data" / "iga_dialect_fixture.xdmf"

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
        solution = XDMFSolution.from_file(self._FIXTURE)

        with h5py.File(self._FIXTURE.with_suffix(".h5"), "r") as h5_file:
            # A statement about the fixture as much as the reader: if this file
            # is ever re-carved, it must keep the archival dtypes.
            self.assertEqual(h5_file["Mesh/Topology"].dtype, np.int32)
            self.assertEqual(h5_file["Mesh/Vertices"].dtype, np.float64)
            self.assertIn("Normals", h5_file["Mesh"])
            np.testing.assert_array_equal(
                solution.vertices, h5_file["Mesh/Vertices"][:]
            )
            np.testing.assert_array_equal(
                solution.faces, h5_file["Mesh/Topology"][:]
            )
            for index in range(len(solution.times)):
                np.testing.assert_array_equal(
                    solution.temperatures[index],
                    h5_file[f"Function/Step_{index}"][:],
                )

        self.assertEqual(solution.vertices.shape, (50, 3))
        self.assertEqual(solution.faces.shape, (48, 3))
        self.assertEqual(len(solution.times), 3)
        self.assertEqual(solution.temperatures.shape, (3, 50))
        self.assertLess(solution.faces.max(), len(solution.vertices))
        self.assertEqual(list(solution.times), sorted(solution.times))
        # The excerpt carries a real inpainted temperature field, so a reader
        # that returned zeros or a transposed view would be caught here too.
        self.assertGreater(solution.temperatures.std(), 0.0)

    @unittest.skipUnless(_HAVE_RECORDED, _SKIP_REASON)
    def test_reads_full_recorded_runs(self):
        """
        Confirmation on the full recorded files, at production size (393,218 IGA
        vertices against 129,526 FEM ones): both load, both are internally
        consistent, and their time axes agree.
        """
        for label, path in (("IGA", _RECORDED_IGA), ("FEM", _RECORDED_FEM)):
            with self.subTest(side=label):
                solution = XDMFSolution.from_file(path)
                self.assertLess(solution.faces.max(), len(solution.vertices))
                self.assertEqual(
                    solution.temperatures.shape,
                    (len(solution.times), len(solution.vertices)),
                )
                self.assertEqual(list(solution.times), sorted(solution.times))
                self.assertEqual(len(set(solution.times)), len(solution.times))

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

        iga = XDMFSolution.from_file(_RECORDED_IGA)
        fem = XDMFSolution.from_file(_RECORDED_FEM)
        written = XDMFSolution.from_file(output)

        np.testing.assert_array_equal(written.vertices, iga.vertices)
        np.testing.assert_array_equal(written.faces, iga.faces)
        self.assertLess(written.faces.max(), len(written.vertices))
        self.assertEqual(
            written.temperatures.shape,
            (len(written.times), len(iga.vertices)),
        )

        # A gather can only ever produce values the FEM field already holds, so
        # the recovered FEM column must stay inside the FEM field's own range at
        # every step. Unwritten slots (the failure mode the gather direction
        # exists to prevent) would sit at exactly 0.0, which is outside the
        # range of every step but the first (where the FEM minimum is ~1e-14).
        # The 1e-9 slack is the round-off of recovering `fem = iga - difference`.
        recovered_fem = (
            iga.temperatures[-len(written.times) :] - written.temperatures
        )
        for step in range(1, len(written.times)):
            with self.subTest(step=step):
                self.assertGreaterEqual(
                    recovered_fem[step].min(), fem.temperatures[step].min() - 1e-9
                )
                self.assertLessEqual(
                    recovered_fem[step].max(), fem.temperatures[step].max() + 1e-9
                )
