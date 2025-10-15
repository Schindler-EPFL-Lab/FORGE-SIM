import unittest

import mitsuba as mi

from rebel_shape.data_processing.masked_pose_image import MaskedPoseImage
from rebel_shape.scene_optimizers.scene_optimizer import (
    MitsubaOptimizationType,
    SceneOptimizer,
)


class TestSceneOptimizers(unittest.TestCase):
    """
    Tests for the _batch_sizes method that computes batch sizes to split
    num_images into batches whose sizes differ by at most 1 and do not
    exceed max_batch_size.
    """

    class MockSceneOptimizer(SceneOptimizer):
        """
        Mock implementation of the abstract SceneOptimizer for testing purposes.
        Only implements the necessary methods and properties to allow instantiation
        and testing of the _batch_sizes method.
        """

        def __init__(
            self,
            batch_size: int | None = None,
            randomize_image_order: bool = True,
            image_order_seed: int = 42,
        ) -> None:
            super().__init__(
                max_batch_size=batch_size,
                randomize_image_order=randomize_image_order,
                image_order_seed=image_order_seed,
            )
            # Add dummy parameters to satisfy abstract base class requirements
            self._parameters: dict[str, MitsubaOptimizationType] = {}
            self._learning_rates: dict[str, float] = {}

        def set_parameters(
            self, parameters: dict[str, MitsubaOptimizationType]
        ) -> None:
            self._parameters = parameters

        def set_learning_rate(self, learning_rates: dict[str, float]) -> None:
            self._learning_rates = learning_rates

        @property
        def parameters(self) -> dict[str, MitsubaOptimizationType]:
            return self._parameters

        def run_optimization_step(
            self,
            scene: "mi.Scene",
            scene_params: "mi.SceneParams",
            param_optimization_configs: list[SceneOptimizer.ParamOptimizationConfig],
            ref_images: list[MaskedPoseImage],
            seed: int,
        ) -> tuple["mi.TensorXf", "mi.TensorXf"]:
            # Not needed for these tests
            raise NotImplementedError()

    def test_single_batch_cases(self):
        """
        Test various scenarios where a single batch should be returned,
        including when max_batch_size is None, equals num_images, or
        exceeds num_images.
        """
        # No batch size limit
        optimizer = self.MockSceneOptimizer(batch_size=None)
        for n_images in [1, 5, 10, 100]:
            self.assertEqual(optimizer._batch_sizes(n_images), [n_images])

        # Batch size equals or exceeds num_images
        optimizer = self.MockSceneOptimizer(batch_size=50)
        self.assertEqual(optimizer._batch_sizes(50), [50])
        self.assertEqual(optimizer._batch_sizes(25), [25])
        self.assertEqual(optimizer._batch_sizes(1), [1])

    def test_even_and_uneven_batch_distributions(self):
        """
        Test that images are distributed appropriately across batches,
        with even distributions when divisible and uneven distributions
        with larger batches first when not divisible.
        """
        optimizer = self.MockSceneOptimizer(batch_size=50)

        # Even distributions
        self.assertEqual(optimizer._batch_sizes(100), [50, 50])
        self.assertEqual(optimizer._batch_sizes(150), [50, 50, 50])

        # Uneven distributions - larger batches come first
        result = optimizer._batch_sizes(119)
        self.assertEqual(result, [40, 40, 39])
        self.assertEqual(sum(result), 119)

        result = optimizer._batch_sizes(118)
        self.assertEqual(sum(result), 118)
        self.assertEqual(len(result), 3)
        self.assertEqual(result.count(40), 1)
        self.assertEqual(result.count(39), 2)

    def test_edge_cases_around_batch_boundaries(self):
        """
        Test edge cases where num_images is at or near multiples of max_batch_size.
        """
        optimizer = self.MockSceneOptimizer(batch_size=50)

        # Exactly one more than batch_size
        result = optimizer._batch_sizes(51)
        self.assertEqual(len(result), 2)
        self.assertEqual(sum(result), 51)
        self.assertEqual(result, [26, 25])

        # Exactly twice batch_size
        optimizer = self.MockSceneOptimizer(batch_size=30)
        result = optimizer._batch_sizes(60)
        self.assertEqual(result, [30, 30])

        # Twice batch_size plus one
        result = optimizer._batch_sizes(61)
        self.assertEqual(len(result), 3)
        self.assertEqual(sum(result), 61)
        self.assertTrue(all(size <= 30 for size in result))

    def test_small_batch_sizes(self):
        """
        Test behavior with very small batch sizes like 1 and 2.
        """
        # Batch size of 1
        optimizer = self.MockSceneOptimizer(batch_size=1)
        self.assertEqual(optimizer._batch_sizes(5), [1, 1, 1, 1, 1])
        self.assertEqual(optimizer._batch_sizes(1), [1])

        # Batch size of 2 with odd number of images
        optimizer = self.MockSceneOptimizer(batch_size=2)
        result = optimizer._batch_sizes(5)
        self.assertEqual(result, [2, 2, 1])
        self.assertEqual(sum(result), 5)

    def test_batch_size_invariants(self):
        """
        Test that key invariants hold across various configurations:
        batch sizes sum to num_images, no batch exceeds max_batch_size,
        batches differ by at most 1, and minimal number of batches used.
        """
        for batch_size in [1, 5, 11, 25, 49, 71, 100]:
            with self.subTest(batch_size=batch_size):
                optimizer = self.MockSceneOptimizer(batch_size=batch_size)
                for num_images in range(1, 150):
                    result = optimizer._batch_sizes(num_images)

                    # Sum equals num_images
                    self.assertEqual(sum(result), num_images)

                    # No batch exceeds max_batch_size
                    self.assertTrue(all(size <= batch_size for size in result))

                    # Batches differ by at most 1
                    if len(result) > 1:
                        self.assertTrue(max(result) - min(result) <= 1)

                    # Minimal number of batches
                    if batch_size < num_images:
                        expected_min_batches = (
                            num_images + batch_size - 1
                        ) // batch_size
                        self.assertEqual(len(result), expected_min_batches)

    def test_batch_sizes_invalid_num_images(self):
        """Test that invalid num_images values raise ValueError."""
        optimizer = self.MockSceneOptimizer(batch_size=10)

        with self.assertRaises(ValueError):
            optimizer._batch_sizes(0)

        with self.assertRaises(ValueError):
            optimizer._batch_sizes(-5)

    def test_iteration_batches_deterministic_ordering(self):
        """
        Test that without randomization, indices are sequential and
        deterministic across single and multiple batches.
        """
        # Single batch
        optimizer = self.MockSceneOptimizer(
            batch_size=None, randomize_image_order=False
        )

        result = optimizer.iteration_batch_indices(10)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], list(range(10)))

        # Multiple batches
        optimizer = self.MockSceneOptimizer(batch_size=3, randomize_image_order=False)

        result = optimizer.iteration_batch_indices(10)
        self.assertEqual(len(result), 4)
        flattened = [idx for batch in result for idx in batch]
        self.assertEqual(sorted(flattened), list(range(10)))

    def test_iteration_indices_randomization_behavior(self):
        """
        Test that randomization changes order, is reproducible with the same seed,
        and differs with different seeds.

        Note that these tests could nominally fail due to random chance, but the
        probability is extremely low for reasonable batch sizes (>10), on the
        order of 1 in 10! (3.6 million) for batch size 10.
        """
        # Randomization changes order
        optimizer = self.MockSceneOptimizer(
            batch_size=5, randomize_image_order=True, image_order_seed=42
        )
        result = optimizer.iteration_batch_indices(20)
        flattened = [idx for batch in result for idx in batch]
        self.assertEqual(sorted(flattened), list(range(20)))
        # For the given seed the ordering should not be sequential.
        self.assertNotEqual(flattened, list(range(20)))

        # Same seed produces same result
        optimizer1 = self.MockSceneOptimizer(
            batch_size=5, randomize_image_order=True, image_order_seed=42
        )
        optimizer2 = self.MockSceneOptimizer(
            batch_size=5, randomize_image_order=True, image_order_seed=42
        )
        result1 = optimizer1.iteration_batch_indices(20)
        result2 = optimizer2.iteration_batch_indices(20)
        self.assertEqual(result1, result2)

        # Different seeds produce different results
        # Probability of collision is very low for reasonable batch sizes
        optimizer1 = self.MockSceneOptimizer(
            batch_size=5, randomize_image_order=True, image_order_seed=42
        )
        optimizer2 = self.MockSceneOptimizer(
            batch_size=5, randomize_image_order=True, image_order_seed=123
        )
        result1 = optimizer1.iteration_batch_indices(50)
        result2 = optimizer2.iteration_batch_indices(50)
        self.assertNotEqual(result1, result2)

        # Consecutive calls differ due to RNG state
        optimizer = self.MockSceneOptimizer(
            batch_size=5, randomize_image_order=True, image_order_seed=42
        )
        result1 = optimizer.iteration_batch_indices(50)
        result2 = optimizer.iteration_batch_indices(50)
        self.assertNotEqual(result1, result2)

    def test_iteration_batch_indices_invariants(self):
        """
        Test that key invariants hold for both randomized and non-randomized cases:
        all indices present, no duplicates, indices within valid range, no empty
        batches, and batch sizes match expectations.
        """
        for randomize in [True, False]:
            optimizer = self.MockSceneOptimizer(
                batch_size=7, randomize_image_order=randomize
            )
            for num_images in [1, 5, 10, 20, 49, 71, 89, 100]:
                with self.subTest(randomize=randomize, num_images=num_images):
                    result = optimizer.iteration_batch_indices(num_images)
                    flattened = [idx for batch in result for idx in batch]

                    # All indices present
                    self.assertEqual(sorted(flattened), list(range(num_images)))

                    # No duplicates
                    self.assertEqual(len(flattened), len(set(flattened)))

                    # Indices within valid range
                    for batch in result:
                        for idx in batch:
                            self.assertGreaterEqual(idx, 0)
                            self.assertLess(idx, num_images)

                    # No empty batches
                    for batch in result:
                        self.assertGreater(len(batch), 0)

                    # Batch sizes match expectations
                    expected_sizes = optimizer._batch_sizes(num_images)
                    actual_sizes = [len(batch) for batch in result]
                    if randomize:
                        # With randomization, sizes are permuted
                        self.assertEqual(sorted(actual_sizes), sorted(expected_sizes))
                    else:
                        # Without randomization, sizes match exactly
                        self.assertEqual(actual_sizes, expected_sizes)

    def test_iteration_batch_size_randomization(self):
        """
        If randomization is enabled, batch sizes should be shuffled as well as
        indices. Make sure that the sizes are correct but in different order.

        The number of batch sizes is chosen to be large enough to make the failing of
        the test due to random chance extremely unlikely.
        """
        optimizer_1 = self.MockSceneOptimizer(
            batch_size=40, randomize_image_order=True, image_order_seed=42
        )
        optimizer_2 = self.MockSceneOptimizer(
            batch_size=40, randomize_image_order=True, image_order_seed=123
        )

        # The chance of a collision is 1/2^20 ~ 1 in a million
        result_1 = optimizer_1.iteration_batch_indices(1220)
        result_2 = optimizer_2.iteration_batch_indices(1220)
        sizes_1 = [len(batch) for batch in result_1]
        sizes_2 = [len(batch) for batch in result_2]

        self.assertEqual(sorted(sizes_1), sorted(sizes_2))
        self.assertNotEqual(sizes_1, sizes_2)
