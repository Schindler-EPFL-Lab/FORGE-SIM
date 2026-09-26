import unittest

# Imported for its module-level side effect: `_fem_fixtures` puts the
# repository root on `sys.path`, so `fem` and `rebel_shape` resolve.
import _fem_fixtures  # noqa: F401
import numpy as np
import numpy.typing as npt

from fem.shells.rigid_body_modes import (
    detect_rigid_body_modes,
    report_rigid_body_modes,
)


class TestRigidBodyModeDetection(unittest.TestCase):
    """
    `detect_rigid_body_modes` and the report that reads it.

    The report is the only automatic check a production shell run
    performs on its own output, and the detection behind it is a
    heuristic (the largest relative gap in the low spectrum), so it is
    tested on constructed spectra, where the right answer is known by
    construction rather than by whatever the solver happened to return.
    The spectra are shaped like a real post-shift solve: signed round-off
    residuals for the rigid-body band, then a physical spectrum.
    """

    @staticmethod
    def spectrum(
        n_rigid: int = 6, n_elastic: int = 6
    ) -> npt.NDArray[np.float64]:
        """Round-off-level rigid-body residuals of both signs, then a physical
        spectrum: the shape a real post-shift solve returns."""
        rigid = np.linspace(-5.0e-8, 6.0e-8, n_rigid)
        elastic = 3.5e6 * np.arange(1, n_elastic + 1, dtype=float)
        return np.concatenate([rigid, elastic])

    def report(
        self, eigenvalues: npt.NDArray[np.float64], n_rigid: int = 6
    ) -> list[str]:
        frequencies = np.sqrt(np.abs(eigenvalues)) / (2.0 * np.pi)
        with self.assertLogs("rebel_shape", level="INFO") as captured:
            report_rigid_body_modes(eigenvalues, frequencies, n_rigid)
        return captured.output

    def test_detects_the_expected_rigid_body_count(self) -> None:
        output = self.report(self.spectrum())
        self.assertIn("largest spectral gap after mode 6", "\n".join(output))
        self.assertFalse([line for line in output if line.startswith("WARNING")])

    def test_warns_when_the_count_disagrees(self) -> None:
        """A shell with a spurious soft mode shows up as a seventh near-zero
        eigenvalue; the run must say so rather than quietly reporting a wrong
        first elastic frequency."""
        eigenvalues = self.spectrum()
        eigenvalues[6] = 7.0e-8  # a seventh near-zero mode
        output = self.report(np.sort(eigenvalues))
        self.assertTrue([line for line in output if line.startswith("WARNING")])

    def test_does_not_crash_for_a_single_mode(self) -> None:
        """`--num-modes 1` is a legal invocation. The gap detection needs two
        eigenvalues to form a ratio, so it must decline rather than raise, and
        the decline must leave nothing lost, i.e. come with a warning."""
        output = self.report(np.array([-5.0e-8]), n_rigid=0)
        self.assertTrue([line for line in output if line.startswith("WARNING")])

    def test_is_scale_invariant(self) -> None:
        """Detection must not depend on the units of the problem: multiplying the
        whole spectrum by 1e12 (a metre/millimetre change, say) is the same
        physics and must give the same answer."""
        for scale in (1.0e-12, 1.0, 1.0e12):
            with self.subTest(scale=scale):
                output = self.report(self.spectrum() * scale)
                self.assertIn("largest spectral gap after mode 6", "\n".join(output))

    def test_returns_the_count_and_the_gap_it_found(self) -> None:
        """The detection is also read directly, by the ground-truth driver's run
        summary. The count is the number of modes before the gap, and the ratio
        is the one across it: here the jump from a 6e-8 residual to the first
        elastic 3.5e6, floored, so it is enormous and unambiguous."""
        detection = detect_rigid_body_modes(self.spectrum(), expected_count=6)
        self.assertEqual(detection.count, 6)
        self.assertGreater(detection.gap, 1.0e3)

    def test_declines_rather_than_guessing_on_a_short_spectrum(self) -> None:
        """Fewer than two eigenvalues form no ratio. The documented answer is
        `count=0, gap=nan`, 0 being a count the detection can never otherwise
        return, since it counts only the modes on the near side of a gap that
        has one on each side."""
        for eigenvalues in (np.array([]), np.array([1.0e-8])):
            with self.subTest(size=eigenvalues.size):
                detection = detect_rigid_body_modes(
                    eigenvalues, expected_count=6
                )
                self.assertEqual(detection.count, 0)
                self.assertTrue(np.isnan(detection.gap))

    def test_the_expected_count_only_sizes_the_scan_window(self) -> None:
        """
        The expected count must not steer the answer, or the check would confirm
        whatever it was told to expect. Passing a wrong expectation still finds
        the real band, as long as the window reaches past it.
        """
        for expected in (3, 6, 9):
            with self.subTest(expected_count=expected):
                detection = detect_rigid_body_modes(
                    self.spectrum(n_elastic=12), expected_count=expected
                )
                self.assertEqual(detection.count, 6)

    def test_the_near_zero_floor_prevents_a_spurious_intra_band_gap(self) -> None:
        """
        A post-shift rigid-body residual can land arbitrarily close to zero (a
        clean cancellation, not just generic round-off), and without the floor
        that one value alone can manufacture a bigger relative jump than the
        real rigid-to-elastic one: `spectrum()`'s own linspace never lands on
        zero, so this frontier case is otherwise unexercised. Deleting the
        floor (`capped = scanned`) makes this fail: it reports 1 rigid-body
        mode with a ~2e22 spurious ratio instead of 6.
        """
        rigid = np.array([1.0e-30, -2.0e-8, 3.0e-8, -1.0e-8, 4.0e-8, -5.0e-8])
        elastic = 3.5e6 * np.arange(1, 7, dtype=float)
        eigenvalues = np.concatenate([rigid, elastic])

        detection = detect_rigid_body_modes(eigenvalues, expected_count=6)
        self.assertEqual(detection.count, 6)
        # Floored ratio = elastic[0] / (1e-6 * scanned.max()) = 3.5e6 / 21 =
        # 1.67e5; still two orders above the largest intra-band ratio the floor
        # leaves (~1, since every floored residual is pinned to the same value).
        self.assertGreater(detection.gap, 1.0e4)

    def test_the_scan_window_excludes_a_larger_gap_further_out(self) -> None:
        """
        The docstring's claim is that later, larger gaps cannot outrank the
        rigid-to-elastic one because they sit outside the `2*expected_count+4`
        window, a case none of the other fixtures exercise, since every other
        fixture's elastic tail is monotonic enough that no later gap ever
        competes. Here a genuinely bigger relative jump is placed at the very
        end of a long spectrum, far past the window; removing the window's
        `min(...)` cap (scanning everything) makes this fail by picking that
        later gap instead of the real rigid-body one.
        """
        rigid = np.linspace(-5.0e-8, 6.0e-8, 6)
        mild_elastic = 1.0e6 * 1.05 ** np.arange(20)
        eigenvalues = np.concatenate([rigid, mild_elastic])
        eigenvalues[-1] *= 1.0e9  # far outside the scan window; a huge, later jump

        detection = detect_rigid_body_modes(eigenvalues, expected_count=6)
        self.assertEqual(detection.count, 6)
