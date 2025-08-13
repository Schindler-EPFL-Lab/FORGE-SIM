import unittest
from pathlib import Path

import mitsuba as mi

from rebel_shape.data_processing.input_preprocessing import preprocess_rgb_images


class TestInputPreprocessing(unittest.TestCase):
    def test_input_preprocessing(self):
        mi.set_variant("llvm_ad_rgb")

        local_dir = Path(__file__).parent.resolve()

        dataset_path = local_dir / "data/thermoscenes_example/buildingA_winter"
        envmap_path = local_dir / "data/textures/overcast_soil_1k.exr"

        images = preprocess_rgb_images(
            scene_path=dataset_path,
            environment_map_path=envmap_path,
            spp=1,
        )

        self.assertEqual(len(images), 2)
