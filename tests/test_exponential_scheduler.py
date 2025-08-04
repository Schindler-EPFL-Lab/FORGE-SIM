import unittest

from rebel_shape.schedulers.exponential_scheduler import ExponentialScheduler


class TestExponentialScheduler(unittest.TestCase):
    def test_invalid_learning_rates(self):
        """Test that invalid learning rates raise ValueError."""
        # Negative start learning rate
        with self.assertRaises(ValueError) as context:
            ExponentialScheduler(
                start_learning_rate=-0.01,
                end_learning_rate=0.001,
                start_iteration=0,
                end_iteration=100
            )
        self.assertIn("non-negative", str(context.exception))

        # Zero end learning rate
        with self.assertRaises(ValueError) as context:
            ExponentialScheduler(
                start_learning_rate=0.01,
                end_learning_rate=-0.001,
                start_iteration=0,
                end_iteration=100
            )
        self.assertIn("non-negative", str(context.exception))

        # Equal learning rates
        with self.assertRaises(ValueError) as context:
            ExponentialScheduler(
                start_learning_rate=0.01,
                end_learning_rate=0.01,
                start_iteration=0,
                end_iteration=100
            )
        self.assertIn("different", str(context.exception))

    def test_invalid_steepness(self):
        """Test that invalid steepness values raise ValueError."""
        with self.assertRaises(ValueError) as context:
            ExponentialScheduler(
                start_learning_rate=0.01,
                end_learning_rate=0.001,
                start_iteration=0,
                end_iteration=100,
                steepness=0.0
            )
        self.assertIn("positive", str(context.exception))

        with self.assertRaises(ValueError) as context:
            ExponentialScheduler(
                start_learning_rate=0.01,
                end_learning_rate=0.001,
                start_iteration=0,
                end_iteration=100,
                steepness=-1.0
            )
        self.assertIn("positive", str(context.exception))

    def test_learning_rate_before_start(self):
        """Test that learning rate equals start_learning_rate before start iteration."""
        scheduler = ExponentialScheduler(
            start_learning_rate=0.01,
            end_learning_rate=0.001,
            start_iteration=10,
            end_iteration=100
        )

        self.assertEqual(scheduler.learning_rate_for_step(0), 0.01)
        self.assertEqual(scheduler.learning_rate_for_step(9), 0.01)

    def test_learning_rate_after_end(self):
        """Test that learning rate equals end_learning_rate after end iteration."""
        scheduler = ExponentialScheduler(
            start_learning_rate=0.01,
            end_learning_rate=0.001,
            start_iteration=10,
            end_iteration=100
        )

        self.assertEqual(scheduler.learning_rate_for_step(100), 0.001)
        self.assertEqual(scheduler.learning_rate_for_step(200), 0.001)

    def test_decreasing_learning_rate(self):
        """Test that learning rate decreases properly (start_lr > end_lr)."""
        scheduler = ExponentialScheduler(
            start_learning_rate=0.1,
            end_learning_rate=0.01,
            start_iteration=0,
            end_iteration=100,
            steepness=2.0
        )

        # Check the values at key points
        self.assertAlmostEqual(scheduler.learning_rate_for_step(0), 0.1)

        mid_step = 50
        mid_lr = scheduler.learning_rate_for_step(mid_step)
        self.assertAlmostEqual(mid_lr, 0.034204727923)
        self.assertAlmostEqual(scheduler.learning_rate_for_step(100), 0.01)

        # Verify that learning rates are monotonically decreasing
        prev_lr = scheduler.learning_rate_for_step(0)
        for step in range(1, 101):
            current_lr = scheduler.learning_rate_for_step(step)
            self.assertLessEqual(current_lr, prev_lr)
            prev_lr = current_lr

    def test_increasing_learning_rate(self):
        """Test that learning rate increases properly (start_lr < end_lr)."""
        scheduler = ExponentialScheduler(
            start_learning_rate=0.01,
            end_learning_rate=0.1,
            start_iteration=0,
            end_iteration=100,
            steepness=2.0
        )

        # Check the values at key points
        self.assertAlmostEqual(scheduler.learning_rate_for_step(0), 0.01)

        mid_step = 50
        mid_lr = scheduler.learning_rate_for_step(mid_step)
        self.assertAlmostEqual(mid_lr, 0.034204727923)
        self.assertAlmostEqual(scheduler.learning_rate_for_step(100), 0.1)

        # Verify that learning rates are monotonically increasing
        prev_lr = scheduler.learning_rate_for_step(0)
        for step in range(1, 101):
            current_lr = scheduler.learning_rate_for_step(step)
            self.assertGreaterEqual(current_lr, prev_lr)
            prev_lr = current_lr
