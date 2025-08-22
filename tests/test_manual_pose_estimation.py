import unittest
from pathlib import Path

import drjit as dr
import mitsuba as mi

from rebel_shape.data_processing.manual_initialization import (
    load_bounding_bbox_transforms,
)


class TestManualPoseEstimation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mi.set_variant("llvm_ad_rgb")

        local_dir = Path(__file__).parent.resolve()
        cls.dataset_path = local_dir / "data/bbox_transforms/buildingA-spring.json"

    def test_loading(self) -> None:
        """
        Test that the loaded transforms are the same
        """
        tf_loaded = load_bounding_bbox_transforms(self.dataset_path)
        tf_expected = mi.Transform4f(
            [
                (
                    2.6316609382629395,
                    -3.4852536856533334e-08,
                    0.1921636462211609,
                    0.7498473525047302,
                ),
                (
                    -0.09508349001407623,
                    1.994611382484436,
                    0.4876357614994049,
                    -0.3293049931526184,
                ),
                (
                    -0.2992772161960602,
                    -0.6337091326713562,
                    1.5348427295684814,
                    5.261859893798828,
                ),
                (0.0, 0.0, 0.0, 1.0),
            ]
        )

        self.assertTrue(dr.allclose(tf_loaded.matrix, tf_expected.matrix))
