import unittest
from unittest import mock

import drjit as dr
import mitsuba as mi

from forge_sim.scene_optimizers.batch_maker import BatchMaker
from forge_sim.scene_optimizers.scene_adam_interleaved import SceneAdamInterleaved
from forge_sim.scene_optimizers.scene_adam_sequential import SceneAdamSequential
from forge_sim.scene_optimizers.scene_adam_simultaneous import SceneAdamSimultaneous
from forge_sim.scene_optimizers.scene_optimizer import SceneOptimizer
from forge_sim.scripts.configs.scene_optimization_config import (
    SceneOptimizationConfig,
)


class TestSceneAdam(unittest.TestCase):
    """
    The three `SceneAdam` optimizers construct through the batching seam and step
    over the batches `BatchMaker` prescribes.

    `SceneOptimizer` takes a `BatchMaker` and owns no batching arguments of its own,
    and `SceneAdam` once still passed it the old ones, so none of the three could be
    constructed. The construction tests pin that through the real call paths; the
    step tests replace only the rendering (`_objective_function`) so that the batching,
    the backward pass and the Adam update are the real ones.
    """

    ADAM_CLASSES = (SceneAdamSequential, SceneAdamInterleaved, SceneAdamSimultaneous)
    """Every concrete `SceneAdam` optimizer."""

    NUM_IMAGES = 5
    """Reference images per step: more than one batch at `BATCH_SIZE`, and uneven."""
    BATCH_SIZE = 2
    """Maximum images per update; 5 images split into batches of 2, 2 and 1."""
    IMAGE_ORDER_SEED = 7
    """A seed other than the default 42, so that a dropped seed is visible."""
    PARAMETER_KEY = "shape"
    """Key of the single optimized parameter."""

    @classmethod
    def setUpClass(cls) -> None:
        mi.set_variant("llvm_ad_rgb")

    def test_every_adam_optimizer_constructs_with_its_documented_arguments(self):
        for adam_class in self.ADAM_CLASSES:
            with self.subTest(adam_class.__name__):
                optimizer = adam_class(
                    learning_rate=0.05,
                    batch_size=self.BATCH_SIZE,
                    uniform=True,
                    randomize_image_order=False,
                    image_order_seed=self.IMAGE_ORDER_SEED,
                )

                self.assertIsInstance(optimizer, SceneOptimizer)

    def test_the_config_builds_an_adam_optimizer_when_lbfgs_is_off(self):
        config = SceneOptimizationConfig(
            learning_rate_start=0.1,
            learning_rate_end=0.01,
            learning_rate_steepness=1.0,
            batch_size=self.BATCH_SIZE,
            uniform_adam=True,
            use_lbfgs=False,
        )

        optimizer = config.scene_optimizer

        self.assertIsInstance(optimizer, SceneAdamSequential)

    def test_an_adam_optimizer_refuses_a_batch_size_below_one(self):
        for adam_class in self.ADAM_CLASSES:
            with self.subTest(adam_class.__name__):
                with self.assertRaises(ValueError):
                    adam_class(batch_size=0)

    def step_batches(
        self, adam_class: type[SceneOptimizer], **arguments: object
    ) -> tuple[list[list[int]], float, float]:
        """
        Run one optimization step of `adam_class` on the one-parameter quadratic
        loss `mean((x - 1)^2)`, with `NUM_IMAGES` stand-in reference images.

        Only the rendering is replaced. Each image is identified by its position, so
        the images every update saw can be read back.

        :returns: The image indices of every update in order, the parameter before
            the step and the parameter after it.
        """
        optimizer = adam_class(learning_rate=0.1, **arguments)
        initial = mi.Float([0.0])
        optimizer.set_parameters({self.PARAMETER_KEY: initial})
        param_config = SceneOptimizer.ParamOptimizationConfig(
            parameter_key=self.PARAMETER_KEY,
            image_loss=mock.Mock(),
            scheduler=mock.Mock(),
            scene_update_callback=mock.Mock(),
        )
        images = list(range(self.NUM_IMAGES))
        batches: list[list[int]] = []

        def objective(scene, scene_params, ref_images, param_config, parameters, seed):
            batches.append(list(ref_images))
            loss = mi.TensorXf(dr.mean((parameters - 1.0) ** 2))
            return loss, mi.TensorXf(0.0)

        with mock.patch.object(
            SceneOptimizer, "_objective_function", staticmethod(objective)
        ):
            optimizer.run_optimization_step(
                scene=mock.Mock(),
                scene_params=mock.Mock(),
                param_optimization_configs=[param_config],
                ref_images=images,
                seed=0,
            )

        return batches, 0.0, float(optimizer.parameters[self.PARAMETER_KEY][0])

    def test_an_unshuffled_step_updates_over_batches_in_image_order(self):
        for adam_class in self.ADAM_CLASSES:
            with self.subTest(adam_class.__name__):
                batches, before, after = self.step_batches(
                    adam_class,
                    batch_size=self.BATCH_SIZE,
                    randomize_image_order=False,
                )

                self.assertEqual(batches, [[0, 1], [2, 3], [4]])
                self.assertGreater(after, before)

    def test_a_shuffled_step_follows_the_batch_maker_for_the_same_seed(self):
        expected = BatchMaker(
            max_batch_size=self.BATCH_SIZE,
            randomize_image_order=True,
            image_order_seed=self.IMAGE_ORDER_SEED,
        ).iteration_batch_indices(self.NUM_IMAGES)
        for adam_class in self.ADAM_CLASSES:
            with self.subTest(adam_class.__name__):
                batches, _, _ = self.step_batches(
                    adam_class,
                    batch_size=self.BATCH_SIZE,
                    randomize_image_order=True,
                    image_order_seed=self.IMAGE_ORDER_SEED,
                )

                self.assertEqual(batches, expected)
