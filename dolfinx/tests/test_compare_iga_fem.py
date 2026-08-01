"""
Tests for the IGA/FEM comparison driver.

Two groups. `TestCompareSolutions` runs the whole driver on synthetic XDMF pairs
whose two sides have *independent* resolutions, and asserts the difference file
it writes is internally consistent -- geometry, topology and attribute all
describing one mesh. That asymmetry is the regime the production comparison
always ran in (the recorded suzanne run is 393,218 IGA vertices against 129,526
FEM ones) and it is where a latent geometry-truncation fault lived: the written
vertices were sliced to `min(n_iga, n_fem)` while the topology and the field were
not, so the file referenced vertices it did not contain.

`TestComputeErrors` pins the error algebra of `compute_errors` against closed
forms. Those are cheap and exact, and they are what any future change to the
norms has to reproduce.

Everything here is pure numpy + h5py. Run from the repository root::

    conda run -n dolfinx-rebel-shape python -m unittest discover dolfinx/tests
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from _xdmf_fixtures import grid_mesh, write_pair, write_time_series
from compare_iga_fem import (
    CompareIGAFEMParameters,
    compare_solutions,
    compute_errors,
    load_xdmf_data,
)
from xdmf_time_series import XdmfLayout, write_time_series_xdmf


def _smooth_field(vertices: np.ndarray, time: float) -> np.ndarray:
    """
    A resolution-independent field: smooth, non-symmetric, and non-zero mean, so
    that no error norm degenerates and no accidental symmetry can mask an index
    permutation.
    """
    x, y = vertices[:, 0], vertices[:, 1]
    return 300.0 + 10.0 * np.sin(2.0 * x + y) + time * (1.0 + x * y)


class TestCompareSolutions(unittest.TestCase):
    """
    End-to-end behaviour of `compare_solutions` on synthetic IGA/FEM pairs.
    """

    def setUp(self) -> None:
        self._directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self._directory, ignore_errors=True)

    def _compare(self, iga_path: Path, fem_path: Path) -> tuple:
        """Runs the comparison writing a difference file, and loads it back."""
        output = self._directory / "difference" / "difference.xdmf"
        compare_solutions(
            CompareIGAFEMParameters(
                iga_xdmf=iga_path, fem_xdmf=fem_path, output_xdmf=output
            )
        )
        return load_xdmf_data(output)

    def test_difference_output_is_self_consistent_at_either_asymmetry(self):
        """
        Regression: the difference file describes one mesh, whichever side is
        finer.

        The comparison lives entirely in IGA vertex ordering -- the FEM field is
        gathered onto it and `iga_faces` indexes the full IGA vertex array -- so
        the written geometry must be the complete IGA mesh. The three assertions
        below are exactly what "one mesh" means on disk: as many vertices as the
        IGA input has, a topology whose largest index is in range, and one field
        value per vertex per step.

        Both asymmetries are run because the historical `min(n_iga, n_fem)` slice
        was a silent no-op in one of them and a corruption in the other.
        """
        times = [0.0, 2.0]
        for n_iga, n_fem in ((7, 4), (4, 7)):
            with self.subTest(n_iga=n_iga**2, n_fem=n_fem**2):
                directory = self._directory / f"{n_iga}_{n_fem}"
                directory.mkdir()
                iga_path, fem_path = write_pair(
                    directory, n_iga, n_fem, times, _smooth_field
                )
                iga_vertices, iga_faces, _, _ = load_xdmf_data(iga_path)

                vertices, faces, out_times, difference = self._compare(
                    iga_path, fem_path
                )

                np.testing.assert_array_equal(vertices, iga_vertices)
                np.testing.assert_array_equal(faces, iga_faces)
                self.assertLess(faces.max(), len(vertices))
                self.assertEqual(difference.shape, (len(times), len(iga_vertices)))
                self.assertEqual(out_times, list(times))

    def test_fem_field_is_gathered_onto_every_iga_vertex(self):
        """
        Every IGA vertex receives its FEM counterpart, with no unwritten slots.

        Construction: the FEM mesh carries the IGA vertices in a known shuffled
        order, jittered by 1e-9 (far below the 1e-4 matching tolerance) to mimic
        the float32 tessellation noise between the two real meshes, plus extra
        vertices sitting between the grid nodes so that the reverse (FEM->IGA)
        nearest map is *not* injective. Its field is the IGA field minus one at
        the corresponding vertex, so a correct gather makes every difference
        exactly +1.

        This is the property the gather-vs-scatter comment in `compare_solutions`
        documents: scattering through a non-injective FEM->IGA map leaves IGA
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
            _smooth_field,
            layout=XdmfLayout.INDENTED,
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
                    _smooth_field(iga_vertices[permutation], time) - 1.0,
                    _smooth_field(extra, time) - 1.0,
                ]
            )
            for time in times
        ]
        write_time_series_xdmf(
            self._directory / "fem.xdmf",
            fem_vertices,
            # Any valid connectivity will do: the comparison never reads the FEM
            # topology, only its vertices and fields.
            np.array([[0, 1, 2]], dtype=np.int64),
            times,
            fem_fields,
            attribute_name="temperature",
            layout=XdmfLayout.COMPACT,
            declare_float_precision=True,
        )

        _, _, _, difference = self._compare(
            self._directory / "iga.xdmf", self._directory / "fem.xdmf"
        )

        # Exact in exact arithmetic (the gathered values are bit-identical
        # copies); 1e-12 covers only the float64 round-off of evaluating the
        # field twice at coordinates differing by the 1e-9 jitter.
        np.testing.assert_allclose(difference, 1.0, atol=1e-12)

    def test_mismatched_step_counts_align_on_the_tail(self):
        """
        When the two runs have different numbers of snapshots, the last `n`
        of each are compared and the reported times are the IGA tail -- not the
        head.

        The two runners save at their own intervals but share the same final
        time, so the tail is the aligned end. Reading the times back out of the
        difference file is the only externally visible statement of which steps
        were compared.
        """
        iga_times = [0.0, 1.0, 2.0, 3.0]
        fem_times = [2.0, 3.0]
        iga_vertices, iga_faces = grid_mesh(4)
        write_time_series(
            self._directory / "iga.xdmf",
            iga_vertices,
            iga_faces,
            iga_times,
            _smooth_field,
            layout=XdmfLayout.INDENTED,
        )
        fem_vertices, fem_faces = grid_mesh(4)
        write_time_series(
            self._directory / "fem.xdmf",
            fem_vertices,
            fem_faces,
            fem_times,
            _smooth_field,
            layout=XdmfLayout.COMPACT,
        )

        _, _, out_times, difference = self._compare(
            self._directory / "iga.xdmf", self._directory / "fem.xdmf"
        )

        self.assertEqual(out_times, fem_times)
        # Identical meshes and identical fields at the aligned times: the tail
        # difference is exactly zero, while a head alignment would compare
        # t=0,1 against t=2,3 and differ by the time-dependent term.
        np.testing.assert_array_equal(difference, np.zeros_like(difference))

    def test_parameters_reject_missing_inputs(self):
        """
        `CompareIGAFEMParameters` refuses paths that do not exist, so a mistyped
        CLI argument fails immediately instead of inside h5py.
        """
        iga_path, fem_path = write_pair(self._directory, 4, 4, [0.0], _smooth_field)

        with self.assertRaises(ValueError):
            CompareIGAFEMParameters(
                iga_xdmf=self._directory / "absent.xdmf", fem_xdmf=fem_path
            )
        with self.assertRaises(ValueError):
            CompareIGAFEMParameters(
                iga_xdmf=iga_path, fem_xdmf=self._directory / "absent.xdmf"
            )


class TestComputeErrors(unittest.TestCase):
    """
    Closed-form properties of the error norms.
    """

    @classmethod
    def setUpClass(cls) -> None:
        generator = np.random.default_rng(7)
        cls.reference = 300.0 + generator.normal(0.0, 5.0, (4, 11))
        cls.perturbed = cls.reference + generator.normal(0.0, 0.1, (4, 11))

    def test_identical_solutions_have_exactly_zero_error(self):
        """
        Comparing a solution with itself gives an identically zero difference and
        exactly zero norms -- no epsilon, no fallback. The relative branch
        divides by a reference norm of ~1000 here, so it is genuinely taken.
        """
        difference, l2_errors, linf_errors = compute_errors(
            self.reference, self.reference, relative=True
        )

        np.testing.assert_array_equal(difference, np.zeros_like(self.reference))
        np.testing.assert_array_equal(l2_errors, np.zeros(len(self.reference)))
        np.testing.assert_array_equal(linf_errors, np.zeros(len(self.reference)))

    def test_norms_match_their_closed_forms(self):
        """
        Known-answer test: the per-step norms are `||d||_2 / ||ref||_2` and
        `max|d| / max|ref|`, recomputed here independently of the loop that
        produced them.

        `rtol=1e-12` is round-off for a sum of 11 float64 terms; the point of the
        gate is to catch a changed normalisation (e.g. dividing by sqrt(n), or
        using the difference's own norm), not to measure round-off.
        """
        difference, l2_errors, linf_errors = compute_errors(
            self.reference, self.perturbed, relative=True
        )

        np.testing.assert_array_equal(difference, self.reference - self.perturbed)
        expected_l2 = np.linalg.norm(difference, axis=1) / np.linalg.norm(
            self.reference, axis=1
        )
        expected_linf = np.max(np.abs(difference), axis=1) / np.max(
            np.abs(self.reference), axis=1
        )
        np.testing.assert_allclose(l2_errors, expected_l2, rtol=1e-12)
        np.testing.assert_allclose(linf_errors, expected_linf, rtol=1e-12)

    def test_relative_errors_are_scale_invariant_and_absolute_ones_are_not(self):
        """
        Scaling both solutions by a common factor leaves the relative errors
        unchanged and multiplies the absolute errors by that factor.

        This is what makes the relative numbers comparable between runs at
        different temperature scales -- a Kelvin-vs-Celsius or a rescaled-source
        run -- and it is exactly the property a mis-placed normalisation breaks.
        """
        factor = 37.0
        _, relative_l2, relative_linf = compute_errors(
            self.reference, self.perturbed, relative=True
        )
        _, scaled_relative_l2, scaled_relative_linf = compute_errors(
            factor * self.reference, factor * self.perturbed, relative=True
        )
        np.testing.assert_allclose(scaled_relative_l2, relative_l2, rtol=1e-12)
        np.testing.assert_allclose(scaled_relative_linf, relative_linf, rtol=1e-12)

        _, absolute_l2, absolute_linf = compute_errors(
            self.reference, self.perturbed, relative=False
        )
        _, scaled_absolute_l2, scaled_absolute_linf = compute_errors(
            factor * self.reference, factor * self.perturbed, relative=False
        )
        np.testing.assert_allclose(scaled_absolute_l2, factor * absolute_l2, rtol=1e-12)
        np.testing.assert_allclose(
            scaled_absolute_linf, factor * absolute_linf, rtol=1e-12
        )

    def test_vanishing_reference_falls_back_to_absolute_norms(self):
        """
        When the reference snapshot is (nearly) zero the relative branch would
        divide by ~0, so the code falls back to absolute norms. Pinned because
        the fallback is silent: without it a zero-initial-temperature run would
        report inf/NaN errors at t=0 and nothing else would say why.

        The reference norm here is 2e-3, below the 1e-10 guard only in the second
        case; the first case checks the guard is not accidentally always on.
        """
        reference = np.array([[2.0e-3, 0.0]])
        difference = np.array([[1.0e-3, 0.0]])
        _, l2_errors, linf_errors = compute_errors(
            reference, reference - difference, relative=True
        )
        np.testing.assert_allclose(l2_errors, [0.5], rtol=1e-12)
        np.testing.assert_allclose(linf_errors, [0.5], rtol=1e-12)

        tiny_reference = np.array([[1.0e-14, 0.0]])
        _, l2_errors, linf_errors = compute_errors(
            tiny_reference, tiny_reference - difference, relative=True
        )
        np.testing.assert_allclose(l2_errors, [1.0e-3], rtol=1e-12)
        np.testing.assert_allclose(linf_errors, [1.0e-3], rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
