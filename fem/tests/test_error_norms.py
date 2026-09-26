import unittest

import numpy as np

from fem.iga_fem_comparison.error_norms import ErrorNorms


class TestErrorNorms(unittest.TestCase):
    """
    Closed-form properties of the error norms.

    `ErrorNorms.relative` and `ErrorNorms.absolute` are cheap and exact, and
    what any future change to the norms has to reproduce. The end-to-end
    behaviour of the comparison driver that calls them is pinned separately,
    in `test_compare_iga_fem.py`.
    """

    @classmethod
    def setUpClass(cls) -> None:
        generator = np.random.default_rng(7)
        cls.reference = 300.0 + generator.normal(0.0, 5.0, (4, 11))
        cls.perturbed = cls.reference + generator.normal(0.0, 0.1, (4, 11))

    def test_identical_solutions_have_exactly_zero_error(self):
        """
        Comparing a solution with itself gives an identically zero difference and
        exactly zero norms, no epsilon, no fallback. The relative branch
        divides by a reference norm of ~1000 here, so it is genuinely taken.
        """
        norms = ErrorNorms.relative(self.reference, self.reference)

        np.testing.assert_array_equal(norms.difference, np.zeros_like(self.reference))
        np.testing.assert_array_equal(norms.l2_errors, np.zeros(len(self.reference)))
        np.testing.assert_array_equal(norms.linf_errors, np.zeros(len(self.reference)))

    def test_norms_match_their_closed_forms(self):
        """
        Known-answer test: the per-step norms are `||d||_2 / ||ref||_2` and
        `max|d| / max|ref|`, recomputed here independently of the loop that
        produced them.

        `rtol=1e-12` is round-off for a sum of 11 float64 terms; the point of the
        gate is to catch a changed normalisation (e.g. dividing by sqrt(n), or
        using the difference's own norm), not to measure round-off.
        """
        norms = ErrorNorms.relative(self.reference, self.perturbed)

        self.assertIs(norms.kind, ErrorNorms.Kind.RELATIVE)
        np.testing.assert_array_equal(
            norms.difference, self.reference - self.perturbed
        )
        expected_l2 = np.linalg.norm(norms.difference, axis=1) / np.linalg.norm(
            self.reference, axis=1
        )
        expected_linf = np.max(np.abs(norms.difference), axis=1) / np.max(
            np.abs(self.reference), axis=1
        )
        np.testing.assert_allclose(norms.l2_errors, expected_l2, rtol=1e-12)
        np.testing.assert_allclose(norms.linf_errors, expected_linf, rtol=1e-12)

    def test_relative_errors_are_scale_invariant_and_absolute_ones_are_not(self):
        """
        Scaling both solutions by a common factor leaves the relative errors
        unchanged and multiplies the absolute errors by that factor.

        This is what makes the relative numbers comparable between runs at
        different temperature scales, a Kelvin-vs-Celsius or a rescaled-source
        run, and it is exactly the property a mis-placed normalisation breaks.
        """
        factor = 37.0
        relative = ErrorNorms.relative(self.reference, self.perturbed)
        scaled_relative = ErrorNorms.relative(
            factor * self.reference, factor * self.perturbed
        )
        np.testing.assert_allclose(
            scaled_relative.l2_errors, relative.l2_errors, rtol=1e-12
        )
        np.testing.assert_allclose(
            scaled_relative.linf_errors, relative.linf_errors, rtol=1e-12
        )

        absolute = ErrorNorms.absolute(self.reference, self.perturbed)
        scaled_absolute = ErrorNorms.absolute(
            factor * self.reference, factor * self.perturbed
        )
        self.assertIs(absolute.kind, ErrorNorms.Kind.ABSOLUTE)
        np.testing.assert_allclose(
            scaled_absolute.l2_errors, factor * absolute.l2_errors, rtol=1e-12
        )
        np.testing.assert_allclose(
            scaled_absolute.linf_errors, factor * absolute.linf_errors, rtol=1e-12
        )

    def test_vanishing_reference_falls_back_to_absolute_norms(self):
        """
        When the reference snapshot is (nearly) zero the relative branch would
        divide by ~0, so the code falls back to absolute norms. Pinned because
        the fallback is silent: without it a zero-initial-temperature run would
        report inf/NaN errors at t=0 and nothing else would say why.

        The reference norm here is 2e-3, below the `REFERENCE_NORM_FLOOR` guard
        only in the second case; the first case checks the guard is not
        accidentally always on.
        """
        reference = np.array([[2.0e-3, 0.0]])
        difference = np.array([[1.0e-3, 0.0]])
        norms = ErrorNorms.relative(reference, reference - difference)
        np.testing.assert_allclose(norms.l2_errors, [0.5], rtol=1e-12)
        np.testing.assert_allclose(norms.linf_errors, [0.5], rtol=1e-12)

        tiny_reference = np.array([[1.0e-14, 0.0]])
        tiny_norms = ErrorNorms.relative(tiny_reference, tiny_reference - difference)
        np.testing.assert_allclose(tiny_norms.l2_errors, [1.0e-3], rtol=1e-12)
        np.testing.assert_allclose(tiny_norms.linf_errors, [1.0e-3], rtol=1e-12)
