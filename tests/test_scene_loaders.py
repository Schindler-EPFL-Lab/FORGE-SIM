import itertools
import unittest
from pathlib import Path

import mitsuba as mi

from rebel_shape.data_processing.image_scene_loader import ImageSceneLoader
from rebel_shape.data_processing.pose_scene_loader import PoseSceneLoader
from rebel_shape.data_processing.single_channel_scene_loader import (
    SingleChannelImageSceneLoader,
)
from rebel_shape.plugin_wrappers.perspective_camera import PerspectiveCamera


class TestScenesDataloaders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mi.set_variant("llvm_ad_rgb")

        local_dir = Path(__file__).parent.resolve()
        cls.dataset_path = local_dir / "data/thermoscenes_example/buildingA_winter"

        cls.pose_loader = PoseSceneLoader(cls.dataset_path)
        cls.image_loader = ImageSceneLoader(cls.dataset_path)
        cls.thermal_loader = SingleChannelImageSceneLoader.thermal_scene(
            cls.dataset_path
        )
        cls.single_channel_loader = SingleChannelImageSceneLoader.windows_masks_scene(
            cls.dataset_path
        )

    def test_number_of_loaded_training_data_pose(self):
        """
        Tests that the pose_loader loads exactly 2 data entries.
        """
        self.assertEqual(len(self.pose_loader.train_cameras), 2)
        self.assertEqual(len(self.pose_loader.train_filenames), 2)
        self.assertEqual(len(list(self.pose_loader.train_data)), 2)

    def test_number_of_loaded_eval_data_pose(self):
        """
        Tests that the pose_loader loads exactly 1 data entry.
        """
        self.assertEqual(len(self.pose_loader.eval_cameras), 1)
        self.assertEqual(len(self.pose_loader.eval_filenames), 1)
        self.assertEqual(len(list(self.pose_loader.eval_data)), 1)

    def test_number_of_loaded_training_data_images(self):
        """
        Tests that the image_loader loads exactly 2 data entries.
        """
        self.assertEqual(len(list(self.image_loader.train_data)), 2)

    def test_number_of_loaded_eval_data_images(self):
        """
        Tests that the image_loader loads exactly 1 data entry.
        """
        self.assertEqual(len(list(self.image_loader.eval_data)), 1)

    def test_number_of_loaded_training_images_thermal(self):
        """
        Tests that the thermal_loader loads exactly 2 data entries.
        """
        self.assertEqual(len(list(self.thermal_loader.train_images)), 2)

    def test_number_of_loaded_eval_images_thermal(self):
        """
        Tests that the thermal_loader loads exactly 1 data entry.
        """
        self.assertEqual(len(list(self.thermal_loader.eval_images)), 1)

    def test_number_of_loaded_training_images_windows_mask(self):
        """
        Tests that the windows mask loader loads exactly 2 and data entries.
        """
        self.assertEqual(len(list(self.thermal_loader.train_images)), 2)

    def test_number_of_loaded_eval_images_windows_mask(self):
        """
        Tests that the windows mask loader loads exactly 1 data entry.
        """
        self.assertEqual(len(list(self.thermal_loader.eval_images)), 1)

    def test_camera_settings(self):
        """
        Tests that the camera in pose_loader have correct settings
        """
        for camera in self.pose_loader.train_cameras + self.pose_loader.eval_cameras:
            self.assertIsInstance(camera, PerspectiveCamera)
            self.assertAlmostEqual(camera.mitsuba_dict["fov"], 67.60170686)
            self.assertEqual(camera.mitsuba_dict["film"]["width"], 480)
            self.assertEqual(camera.mitsuba_dict["film"]["height"], 640)
            self.assertEqual(camera.mitsuba_dict["film"]["height"], 640)

    def test_camera_settings_luminance(self):
        """
        Tests that the camera in thermal loader have correct settings
        """
        for image in itertools.chain(
            self.thermal_loader.train_images, self.thermal_loader.eval_images
        ):
            camera = image.camera
            self.assertIsInstance(camera, PerspectiveCamera)
            self.assertIsInstance(camera, PerspectiveCamera)
            self.assertAlmostEqual(camera.mitsuba_dict["fov"], 67.60170686)
            self.assertEqual(camera.mitsuba_dict["film"]["width"], 480)
            self.assertEqual(camera.mitsuba_dict["film"]["height"], 640)
            self.assertEqual(camera.mitsuba_dict["film"]["height"], 640)
            self.assertEqual(camera.mitsuba_dict["film"]["pixel_format"], "luminance")
