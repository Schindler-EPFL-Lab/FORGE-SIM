import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from _xdmf_fixtures import grid_mesh, write_pair, write_time_series

from fem.compare_iga_fem import CompareIGAFEMParameters, compare_solutions
from fem.iga_fem_comparison.xdmf_solution import XDMFSolution
from fem.xdmf_time_series import XDMFTimeSeries


class TestCompareSolutions(unittest.TestCase):
    """
    End-to-end behaviour of `compare_solutions` on synthetic IGA/FEM pairs.

    Runs the whole driver on synthetic XDMF pairs whose two sides carry
    unrelated resolutions, and asserts the difference file it writes is
    internally consistent: geometry, topology and attribute all
    describing one mesh. That asymmetry is the regime the production
    comparison always ran in (the recorded suzanne run is 393,218 IGA
    vertices against 129,526 FEM ones) and it is where a latent
    geometry-truncation fault lived: the written vertices were sliced to
    `min(n_iga, n_fem)` while the topology and the field were not, so the
    file referenced vertices it did not contain.

    The error algebra of `ErrorNorms.relative` and `ErrorNorms.absolute` is
    pinned separately, in `test_error_norms.py`.

    Everything here is pure numpy + h5py.
    """

    def setUp(self) -> None:
        self._directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self._directory, ignore_errors=True)

    @staticmethod
    def _smooth_field(vertices: np.ndarray, time: float) -> np.ndarray:
        """
        A resolution-independent field: smooth, non-symmetric, and non-zero mean,
        so that no error norm degenerates and no accidental symmetry can mask an
        index permutation.
        """
        x, y = vertices[:, 0], vertices[:, 1]
        return 300.0 + 10.0 * np.sin(2.0 * x + y) + time * (1.0 + x * y)

    def _compare(self, iga_path: Path, fem_path: Path) -> XDMFSolution:
        """Runs the comparison writing a difference file, and loads it back."""
        output = self._directory / "difference" / "difference.xdmf"
        compare_solutions(
            CompareIGAFEMParameters(
                iga_xdmf=iga_path, fem_xdmf=fem_path, output_xdmf=output
            )
        )
        return XDMFSolution.from_file(output)

    def test_difference_output_is_self_consistent_at_either_asymmetry(self):
        """
        Regression: the difference file describes one mesh, whichever side is
        finer.

        The comparison lives entirely in IGA vertex ordering: the FEM field is
        gathered onto it and the IGA faces index the full IGA vertex array, so
        the written geometry must be the complete IGA mesh. The three assertions
        below are exactly what "one mesh" means on disk: as many vertices as the
        IGA input has, a topology whose largest index is in range, and one field
        value per vertex per step.

        Both asymmetries are run because a `min(n_iga, n_fem)` slice would be a
        silent no-op in one of them and a corruption in the other.
        """
        times = [0.0, 2.0]
        for n_iga, n_fem in ((7, 4), (4, 7)):
            with self.subTest(n_iga=n_iga**2, n_fem=n_fem**2):
                directory = self._directory / f"{n_iga}_{n_fem}"
                directory.mkdir()
                iga_path, fem_path = write_pair(
                    directory, n_iga, n_fem, times, self._smooth_field
                )
                iga = XDMFSolution.from_file(iga_path)

                written = self._compare(iga_path, fem_path)

                np.testing.assert_array_equal(written.vertices, iga.vertices)
                np.testing.assert_array_equal(written.faces, iga.faces)
                self.assertLess(written.faces.max(), len(written.vertices))
                self.assertEqual(
                    written.temperatures.shape, (len(times), len(iga.vertices))
                )
                self.assertEqual(list(written.times), times)

    def test_fem_field_is_gathered_onto_every_iga_vertex(self):
        """
        Every IGA vertex receives its FEM counterpart, with no unwritten slots.

        Construction: the FEM mesh carries the IGA vertices in a known shuffled
        order, jittered by 1e-9 (far below the 1e-4 matching tolerance) to mimic
        the float32 tessellation noise between the two real meshes, plus extra
        vertices sitting between the grid nodes so that the reverse (FEM->IGA)
        nearest map fails to be injective. Its field is the IGA field minus one at
        the corresponding vertex, so a correct gather makes every difference
        exactly +1.

        This is the property `VertexMapping`'s gather-vs-scatter reasoning
        rests on: scattering through a non-injective FEM->IGA map leaves IGA
        slots at zero, which here would show up as a difference of ~300 instead
        of 1 at those vertices.
        """
        iga_vertices, iga_faces = grid_mesh(6)
        times = [0.0, 1.0]
        write_time_series(
            self._directory / "iga.xdmf",
            iga_vertices,
            iga_faces,
            times,
            self._smooth_field,
            layout=XDMFTimeSeries.Layout.INDENTED,
        )

        generator = np.random.default_rng(20260801)
        permutation = generator.permutation(len(iga_vertices))
        extra = np.column_stack(
            [
                generator.uniform(0.05, 0.95, 5),
                generator.uniform(0.05, 0.95, 5),
                np.zeros(5),
            ]
        )
        fem_vertices = np.vstack(
            [
                iga_vertices[permutation]
                + generator.uniform(-1e-9, 1e-9, (len(iga_vertices), 3)),
                extra,
            ]
        )
        fem_fields = [
            np.concatenate(
                [
                    self._smooth_field(iga_vertices[permutation], time) - 1.0,
                    self._smooth_field(extra, time) - 1.0,
                ]
            )
            for time in times
        ]
        XDMFTimeSeries.write(
            self._directory / "fem.xdmf",
            fem_vertices,
            # Any valid connectivity will do: the comparison never reads the FEM
            # topology, only its vertices and fields.
            np.array([[0, 1, 2]], dtype=np.int64),
            times,
            fem_fields,
            attribute_name="temperature",
            layout=XDMFTimeSeries.Layout.COMPACT,
            declare_float_precision=True,
        )

        written = self._compare(
            self._directory / "iga.xdmf", self._directory / "fem.xdmf"
        )

        # Exact in exact arithmetic (the gathered values are bit-identical
        # copies); 1e-12 covers only the float64 round-off of evaluating the
        # field twice at coordinates differing by the 1e-9 jitter.
        np.testing.assert_allclose(written.temperatures, 1.0, atol=1e-12)

    def test_mismatched_step_counts_are_rejected(self):
        """
        Two runs with different snapshot counts are refused, not silently
        truncated to a common tail.

        Snapshot `i` of one file is compared against snapshot `i` of the other,
        so unequal counts mean the paired fields are at different instants and
        every reported error is meaningless. The counts appear in the message,
        because that is what tells the user which of the two runs to redo. No
        output file is produced.
        """
        iga_times = [0.0, 1.0, 2.0, 3.0]
        fem_times = [2.0, 3.0]
        iga_vertices, iga_faces = grid_mesh(4)
        write_time_series(
            self._directory / "iga.xdmf",
            iga_vertices,
            iga_faces,
            iga_times,
            self._smooth_field,
            layout=XDMFTimeSeries.Layout.INDENTED,
        )
        fem_vertices, fem_faces = grid_mesh(4)
        write_time_series(
            self._directory / "fem.xdmf",
            fem_vertices,
            fem_faces,
            fem_times,
            self._smooth_field,
            layout=XDMFTimeSeries.Layout.COMPACT,
        )
        output = self._directory / "difference" / "difference.xdmf"

        with self.assertRaises(ValueError) as raised:
            compare_solutions(
                CompareIGAFEMParameters(
                    iga_xdmf=self._directory / "iga.xdmf",
                    fem_xdmf=self._directory / "fem.xdmf",
                    output_xdmf=output,
                )
            )

        # `assertIn("4", ...)` / `assertIn("2", ...)` alone would also pass if
        # `ComparisonReport.require_matching_time_steps` were deleted entirely:
        # the unequal snapshot counts then reach `ErrorNorms.relative` and NumPy raises
        # its own ValueError ("operands could not be broadcast together with
        # shapes (4,16) (2,16)"), which contains the same two digits. Pinning
        # the "N vs N" phrase from the guard's own message is what tells the two
        # apart.
        self.assertIn("different snapshot counts", str(raised.exception))
        self.assertIn("4 vs 2", str(raised.exception))
        self.assertFalse(output.exists())

    def test_matching_counts_over_different_spans_are_rejected(self):
        """
        Regression: equal snapshot counts alone do not mean the two runs
        describe the same instants.

        Before `require_matching_time_steps` checked the times themselves, two
        runs of the same length but different total spans passed this guard and
        were differenced index-by-index against instants that do not correspond,
        a structural mismatch the snapshot count cannot see. No output file
        is produced.
        """
        iga_times = [0.0, 1.0, 2.0, 3.0]
        fem_times = [0.0, 1.0, 2.0, 6.0]  # same count, last snapshot at a
        # different instant
        iga_vertices, iga_faces = grid_mesh(4)
        write_time_series(
            self._directory / "iga.xdmf",
            iga_vertices,
            iga_faces,
            iga_times,
            self._smooth_field,
            layout=XDMFTimeSeries.Layout.INDENTED,
        )
        fem_vertices, fem_faces = grid_mesh(4)
        write_time_series(
            self._directory / "fem.xdmf",
            fem_vertices,
            fem_faces,
            fem_times,
            self._smooth_field,
            layout=XDMFTimeSeries.Layout.COMPACT,
        )
        output = self._directory / "difference" / "difference.xdmf"

        with self.assertRaises(ValueError) as raised:
            compare_solutions(
                CompareIGAFEMParameters(
                    iga_xdmf=self._directory / "iga.xdmf",
                    fem_xdmf=self._directory / "fem.xdmf",
                    output_xdmf=output,
                )
            )

        self.assertIn("disagree", str(raised.exception))
        self.assertFalse(output.exists())

    def test_parameters_reject_missing_inputs(self):
        """
        `CompareIGAFEMParameters` refuses paths that do not exist, so a mistyped
        CLI argument fails immediately instead of inside h5py.
        """
        iga_path, fem_path = write_pair(
            self._directory, 4, 4, [0.0], self._smooth_field
        )

        with self.assertRaises(ValueError):
            CompareIGAFEMParameters(
                iga_xdmf=self._directory / "absent.xdmf", fem_xdmf=fem_path
            )
        with self.assertRaises(ValueError):
            CompareIGAFEMParameters(
                iga_xdmf=iga_path, fem_xdmf=self._directory / "absent.xdmf"
            )
