import unittest
from pathlib import Path

import mitsuba as mi

from rebel_shape.data_processing.thermoscenes_loader import ThermoScenesLoader
from rebel_shape.plugin_wrappers.perspective_camera import PerspectiveCamera


class TestThermoScenesDataloader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mi.set_variant("llvm_ad_rgb")

        local_dir = Path(__file__).parent.resolve()
        dataset_path = local_dir / "data/thermoscenes_example/buildingA_winter"

        cls.loader = ThermoScenesLoader(dataset_path)

    def test_number_of_loaded_training_images(self):
        self.assertEqual(len(self.loader.train_images_rgb), 2)
        self.assertEqual(len(self.loader.train_images_thermal), 2)
        self.assertEqual(len(self.loader.train_foreground_masks), 2)
        self.assertEqual(len(self.loader.train_cameras), 2)

    def test_number_of_loaded_eval_images(self):
        self.assertEqual(len(self.loader.eval_images_rgb), 1)
        self.assertEqual(len(self.loader.eval_images_thermal), 1)
        self.assertEqual(len(self.loader.eval_foreground_masks), 1)
        self.assertEqual(len(self.loader.eval_cameras), 1)

    def test_camera_settings(self):
        for camera in self.loader.train_cameras + self.loader.eval_cameras:
            self.assertIsInstance(camera, PerspectiveCamera)
            self.assertAlmostEqual(camera.mitsuba_dict["fov"], 67.60170686)
            self.assertEqual(camera.mitsuba_dict["film"]["width"], 480)
            self.assertEqual(camera.mitsuba_dict["film"]["height"], 640)
            self.assertEqual(camera.mitsuba_dict["film"]["height"], 640)
