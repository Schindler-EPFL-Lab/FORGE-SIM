import tempfile
import unittest
from pathlib import Path

import numpy as np

from fem.end_to_end.modal_comparison.scene_modal_comparison import SceneModalComparison


class TestSortedFrequencies(unittest.TestCase):
    """
    `SceneModalComparison.sorted_frequencies` against every modal archive schema
    on disk.

    `SceneModalComparison.run` needs a recorded dataset and a real DOLFINx solve,
    so it is not exercised here; `sorted_frequencies` is the one piece of the
    class checkable without one. Its counterpart, `modal_frequencies`
    (`rebel_shape.scripts.end_to_end.modal.modal_frequencies`), is a plain
    `rebel_shape` function with no DOLFINx dependency and stays tested in
    `tests/test_end_to_end_reports.py`; the two are exercised side by side there
    against ascending-versus-stored order, which is the contract that separates
    them and needs both readers on the same fixture to see.
    """

    _MODAL_ARCHIVE_SCHEMAS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("iga", ("eigenvalues", "eigenvectors")),
        ("fem_recorded", ("eigenvalues", "eigenvectors", "frequencies")),
        (
            "fem_current",
            ("eigenvalues", "eigenvectors", "frequencies", "mode_energies"),
        ),
    )
    """
    Every key set a `modal_analysis_results.npz` carries, oldest first.

    The same three schemas `tests/test_end_to_end_reports.py` pins for
    `modal_frequencies`, duplicated here rather than imported, since nothing
    in this conda-only environment's test tree is importable from the venv
    suite and vice versa.
    """

    def test_every_recorded_modal_archive_schema_still_reads_as_frequencies(self):
        """
        Neither an archive predating `mode_energies` nor one predating
        `frequencies` itself may be indexed on a key it lacks.

        The eigenvalues are exact squares of `2 pi f`, one of them negated: an
        archive carrying no `frequencies` must be converted through
        `sqrt(|lambda|) / (2 pi)`, and that sign is how the rigid-body modes
        arrive, so a conversion dropping the absolute value would yield `nan`
        instead of the frequency the FEM schemas store outright. They are also
        stored out of order, which `sorted_frequencies` must undo: it pairs
        two independent spectra by rank, unlike `modal_frequencies`, which must
        preserve the stored order to stay aligned with the mode shapes it
        indexes.
        """
        as_stored = [3.0, 0.5, 7.0]
        arrays = {
            "eigenvalues": np.array([
                (2.0 * np.pi * 3.0) ** 2,
                -((2.0 * np.pi * 0.5) ** 2),
                (2.0 * np.pi * 7.0) ** 2,
            ]),
            "eigenvectors": np.eye(3),
            "frequencies": np.array(as_stored),
            "mode_energies": np.zeros((3, 4)),
        }
        expected = sorted(as_stored)

        with tempfile.TemporaryDirectory() as tmp_dir:
            for label, keys in self._MODAL_ARCHIVE_SCHEMAS:
                path = Path(tmp_dir) / f"{label}.npz"
                np.savez(path, **{key: arrays[key] for key in keys})

                with self.subTest(schema=label):
                    frequencies = SceneModalComparison.sorted_frequencies(path)

                    self.assertEqual(len(frequencies), 3)
                    for got, want in zip(frequencies, expected):
                        self.assertAlmostEqual(got, want, places=12)
