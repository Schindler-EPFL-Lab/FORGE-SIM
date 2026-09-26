import unittest

import numpy as np

from rebel_shape.scripts.end_to_end.modal.degenerate_mode_runs import (
    degenerate_mode_runs,
)


class TestDegenerateModeRuns(unittest.TestCase):
    def test_single_array_relative_gap_is_inclusive_at_the_boundary(self) -> None:
        """
        A relative gap exactly at `rtol` links; one step beyond it does not: the
        boundary an off-by-one in the comparison would move.
        """
        self.assertEqual(
            degenerate_mode_runs(0.01, np.array([100.0, 101.0])), [[0, 1]]
        )
        self.assertEqual(
            degenerate_mode_runs(0.01, np.array([100.0, 101.00001])), [[0], [1]]
        )

    def test_single_array_zero_frequency_uses_the_absolute_floor(self) -> None:
        """
        The gap is measured against `max(|freq[i-1]|, _RELATIVE_GAP_FLOOR)`, so a
        run starting at zero frequency does not divide by zero.
        """
        self.assertEqual(
            degenerate_mode_runs(1.0, np.array([0.0, 1e-10, 100.0])),
            [[0, 1], [2]],
        )

    def test_two_arrays_require_agreement_on_both_to_link(self) -> None:
        """
        Two indices link only when every array supplied agrees within `rtol`;
        agreeing on one array alone must not merge them into a degenerate run.
        """
        rtol = 1e-3
        fem_freq = np.array([100.0, 100.0])
        iga_freq = np.array([100.0, 101.0])
        self.assertEqual(degenerate_mode_runs(rtol, fem_freq, iga_freq), [[0], [1]])
        self.assertEqual(degenerate_mode_runs(rtol, iga_freq, fem_freq), [[0], [1]])

    def test_two_arrays_exact_threshold_is_inclusive(self) -> None:
        """
        A relative gap exactly at `rtol` links on both arrays; one step beyond it
        does not: the boundary an off-by-one in the comparison would move.
        """
        rtol = 1e-3
        at_threshold = np.array([100.0, 100.1])  # (100.1 - 100) / 100 == 1e-3
        self.assertEqual(
            degenerate_mode_runs(rtol, at_threshold, at_threshold), [[0, 1]]
        )
        past_threshold = np.array([100.0, 100.10001])
        self.assertEqual(
            degenerate_mode_runs(rtol, past_threshold, past_threshold),
            [[0], [1]],
        )

    def test_three_mutually_close_frequencies_form_one_run(self) -> None:
        """A run extends past one adjacent pair when every neighbour stays close."""
        freq = np.array([100.0, 100.05, 100.1])
        self.assertEqual(degenerate_mode_runs(1e-3, freq, freq), [[0, 1, 2]])

    def test_two_arrays_zero_frequency_pair_uses_the_absolute_floor(self) -> None:
        """
        `close` compares against `max(|b|, _RELATIVE_GAP_FLOOR)`, so two
        frequencies at zero do not divide by zero and still compare as equal.
        """
        freq = np.array([0.0, 0.0])
        self.assertEqual(degenerate_mode_runs(1e-6, freq, freq), [[0, 1]])
