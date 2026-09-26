import unittest

from fem.end_to_end.heat_comparison.scene_heat_comparison import SceneHeatComparison


class TestRequireMatchingSnapshotCounts(unittest.TestCase):
    """
    The guard rejecting a scene whose two runs were not saved at the same instants.

    `SceneHeatComparison.run` needs a recorded dataset and a real DOLFINx
    solve, so it is not exercised here; `require_matching_snapshot_counts`
    is the one piece of the class checkable without one. The scene table
    it is called with is pinned separately, in `test_scene_profile.py`.
    """

    def test_matching_counts_are_accepted(self) -> None:
        """The recorded configuration, which must stay a no-op."""
        times = list(range(11))
        SceneHeatComparison.require_matching_snapshot_counts(
            "buildnet-1",
            fem_times=times,
            iga_times=times,
        )

    def test_invalid_mismatched_counts_are_rejected_naming_both_sides(self) -> None:
        """
        Rows are paired by index, so a shorter FEM series once truncated the comparison
        to `min(...)` and reported it as a full one. The message has to say which side
        holds how many, in that order, or the reader cannot tell which run to redo.
        """
        with self.assertRaises(ValueError) as raised:
            SceneHeatComparison.require_matching_snapshot_counts(
                "car",
                fem_times=list(range(7)),
                iga_times=list(range(11)),
            )

        message = str(raised.exception)
        self.assertIn("different snapshot counts (11 vs 7)", message)
        self.assertIn("car", message)

    def test_invalid_surplus_fem_snapshots_are_rejected_too(self) -> None:
        """
        The mismatch is rejected in either direction: a FEM run saving more
        often than IGA pairs its early snapshots against IGA instants they do
        not belong to, which truncation to the shorter series would have
        hidden just as effectively.
        """
        with self.assertRaises(ValueError) as raised:
            SceneHeatComparison.require_matching_snapshot_counts(
                "lion",
                fem_times=list(range(21)),
                iga_times=list(range(11)),
            )

        self.assertIn("different snapshot counts (11 vs 21)", str(raised.exception))

    def test_matching_counts_over_different_spans_are_rejected(self) -> None:
        """
        Regression: equal counts alone do not mean the two runs describe the
        same instants: two runs of the same length but different total spans
        must be rejected too, not differenced index-by-index against instants
        that do not correspond.
        """
        with self.assertRaises(ValueError) as raised:
            SceneHeatComparison.require_matching_snapshot_counts(
                "woodshed",
                fem_times=[0.0, 1.0, 2.0, 6.0],
                iga_times=[0.0, 1.0, 2.0, 3.0],
            )

        message = str(raised.exception)
        self.assertIn("disagree", message)
        self.assertIn("woodshed", message)
