import itertools
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import mitsuba as mi
import numpy as np

from rebel_shape import logger as rebel_shape_logger
from rebel_shape.data_processing.image_scene_loader import ImageSceneLoader
from rebel_shape.data_processing.pose_scene_loader import PoseSceneLoader
from rebel_shape.data_processing.silhouette_scene_loader import SilhouetteSceneLoader
from rebel_shape.data_processing.single_channel_scene_loader import (
    SingleChannelImageSceneLoader,
)
from rebel_shape.plugin_wrappers.perspective_camera import PerspectiveCamera


class TestScenesDataloaders(unittest.TestCase):
    """
    Tests for the ThermoScenes/VGGT scene loaders: basic loading counts and
    camera settings against a real fixture dataset, the hard-failure guards
    against a missing alpha (object mask) channel, the VGGT
    `allow_missing_images` skip-with-warning behavior for gaps that are a hard
    error on the plain ThermoScenes path, and linear (non-sRGB) decoding of
    single-channel images.
    """

    # A single frame's pose; only its shape matters for the tests below since
    # none of them exercise camera geometry.
    _IDENTITY_TRANSFORM = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    @staticmethod
    def _thermoscenes_transforms(frames: list) -> dict:
        """Minimal ThermoScenes-format `transforms.json` payload wrapping
        `frames`."""
        return {"w": 8, "h": 8, "fl_x": 6.0, "frames": frames}

    @staticmethod
    def _vggt_frame(file_path: str, translation_x: float, thermal_file_path) -> dict:
        """
        A single VGGT-format frame (`{"rgb": {...}}`). `thermal_file_path` is
        included verbatim (including `""`) unless it is `None`, in which case
        the key is omitted entirely, mirroring the three ways a
        single-channel reference can be absent.
        """
        rgb = {
            "file_path": file_path,
            "transform_matrix": [
                [1.0, 0.0, 0.0, translation_x],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            "w": 8,
            "h": 8,
            "fovx": 50.0,
        }
        if thermal_file_path is not None:
            rgb["thermal_file_path"] = thermal_file_path
        return {"rgb": rgb}

    @staticmethod
    def _write_luminance_png(path: Path, byte_value: int) -> None:
        """Writes an 8-bit single-channel PNG whose every pixel equals
        `byte_value`."""
        array = np.full((8, 8), byte_value, dtype=np.uint8)
        bitmap = mi.Bitmap(array, pixel_format=mi.Bitmap.PixelFormat.Y)
        bitmap.write(str(path))

    @staticmethod
    def _copy_scene_in_pixel_format(
        src_scene: Path, dst_scene: Path, pixel_format: "mi.Bitmap.PixelFormat"
    ) -> None:
        """
        Copies `transforms.json` and `images/` from `src_scene` into
        `dst_scene`, re-encoding every RGB image into `pixel_format`
        (everything else about the scene, such as poses and resolution, stays
        identical).
        """
        dst_scene.mkdir(parents=True)
        shutil.copy(src_scene / "transforms.json", dst_scene / "transforms.json")
        dst_images = dst_scene / "images"
        dst_images.mkdir()
        for image_path in (src_scene / "images").iterdir():
            bitmap = mi.Bitmap(str(image_path))
            converted = bitmap.convert(
                pixel_format=pixel_format,
                component_format=mi.Struct.Type.UInt8,
                srgb_gamma=True,
            )
            converted.write(str(dst_images / image_path.name))

    @classmethod
    def setUpClass(cls):
        mi.set_variant("llvm_ad_rgb")

        local_dir = Path(__file__).parent.resolve()
        cls.dataset_path = local_dir / "data/thermoscenes_example/buildingA_winter"

    def setUp(self):
        # The loaders hold real `mi.TensorXf` images and `PerspectiveCamera`s, so
        # they are built per test rather than cached on the class: a Mitsuba object
        # that outlives the test which built it aborts a *later, unrelated* test
        # inside Dr.Jit's thread pool ("Assertion failed ... queue.cpp: remain ==
        # 1"). Only `dataset_path`, which is plain data, stays at class level.
        self.pose_loader = PoseSceneLoader(self.dataset_path)
        self.image_loader = ImageSceneLoader(
            scene_path=self.dataset_path, pose_scene_loader=self.pose_loader
        )
        self.thermal_loader = SingleChannelImageSceneLoader.thermal_scene(
            self.dataset_path
        )
        self.single_channel_loader = SingleChannelImageSceneLoader.windows_masks_scene(
            self.dataset_path
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

    # TestMissingAlphaFailsLoud (merged)
    #
    # The alpha channel of the RGB images *is* the object mask. `Bitmap.convert()`
    # synthesizes an opaque alpha for a source that has none, so a source without
    # a mask used to be silently accepted, and, through the silhouette loader,
    # produced a solid all-black frame instead of a real silhouette. Both loaders
    # must now raise instead of degrading silently. Reuses `dataset_path` above
    # as the intact (alpha-carrying) reference scene: the two fixtures were
    # identical.

    def test_image_scene_loader_raises_when_rgb_has_no_alpha(self):
        """A source RGB image without an alpha channel must raise `RuntimeError`
        from `ImageSceneLoader`, not silently synthesize an opaque mask."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_path = Path(tmp_dir) / "scene"
            self._copy_scene_in_pixel_format(
                self.dataset_path, scene_path, mi.Bitmap.PixelFormat.RGB
            )

            with self.assertRaises(RuntimeError):
                ImageSceneLoader.from_thermoscene(scene_path)

    def test_silhouette_scene_loader_raises_when_rgb_has_no_alpha(self):
        """Same guard through `SilhouetteSceneLoader`: without it, the missing
        mask used to be read back as fully opaque, so the whole frame silently
        became solid black."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_path = Path(tmp_dir) / "scene"
            self._copy_scene_in_pixel_format(
                self.dataset_path, scene_path, mi.Bitmap.PixelFormat.RGB
            )

            with self.assertRaises(RuntimeError):
                SilhouetteSceneLoader.from_thermoscene(scene_path)

    def test_image_scene_loader_raises_when_rgb_is_grey_with_alpha(self):
        """A grey+alpha (`PixelFormat.YA`) source carries an object mask but no
        colour, and `Bitmap.has_alpha()` is true for it. Converting it to RGBA
        replicates the grey value into R, G and B, so it would be optimized
        against as if it were a colour photograph: it must raise `RuntimeError`,
        and the message must name the pixel format that was actually found."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_path = Path(tmp_dir) / "scene"
            self._copy_scene_in_pixel_format(
                self.dataset_path, scene_path, mi.Bitmap.PixelFormat.YA
            )

            with self.assertRaises(RuntimeError) as context:
                ImageSceneLoader.from_thermoscene(scene_path)

            self.assertIn("YA", str(context.exception))

    def test_silhouette_scene_loader_intact_scene_silhouette_is_not_degenerate(self):
        """Sanity check the two tests above are pinning the right failure mode:
        on the intact (alpha-carrying) scene, every silhouette must actually
        separate object (0.0) from background (1.0), not collapse to a
        constant image the way the missing-alpha bug produced."""
        loader = SilhouetteSceneLoader.from_thermoscene(self.dataset_path)

        for masked_image in loader.train_images + loader.eval_images:
            silhouette = masked_image.image.numpy()
            self.assertAlmostEqual(float(silhouette.min()), 0.0, places=5)
            self.assertAlmostEqual(float(silhouette.max()), 1.0, places=5)

    # TestSingleChannelAllowMissingImages (merged)
    #
    # `allow_missing_images` is the switch between two contracts on
    # `SingleChannelImageSceneLoader`: ThermoScenes datasets are assumed complete,
    # so any gap in a frame's single-channel reference is a dataset bug (hard
    # failure, via the plain `thermal_scene`/`windows_masks_scene` factories).
    # VGGT-style datasets only annotate a subset of frames, so the identical gaps
    # must instead be skipped, with a warning, via the `_vggt` factories.

    def test_thermal_scene_missing_image_key_raises_key_error(self):
        """An absent `thermal_file_path` entry on a frame is a ThermoScenes
        dataset bug: it must raise `KeyError`, not silently skip the frame."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_path = Path(tmp_dir)
            frame = {
                "file_path": "images/frame_train_000.png",
                "transform_matrix": self._IDENTITY_TRANSFORM,
            }
            (scene_path / "transforms.json").write_text(
                json.dumps(self._thermoscenes_transforms([frame]))
            )

            with self.assertRaises(KeyError):
                SingleChannelImageSceneLoader.thermal_scene(scene_path)

    def test_thermal_scene_blank_image_entry_raises_runtime_error(self):
        """A blank (empty-string) `thermal_file_path` is a ThermoScenes dataset
        bug distinct from an absent key: it must raise `RuntimeError`."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_path = Path(tmp_dir)
            frame = {
                "file_path": "images/frame_train_000.png",
                "transform_matrix": self._IDENTITY_TRANSFORM,
                "thermal_file_path": "",
            }
            (scene_path / "transforms.json").write_text(
                json.dumps(self._thermoscenes_transforms([frame]))
            )

            with self.assertRaises(RuntimeError):
                SingleChannelImageSceneLoader.thermal_scene(scene_path)

    def test_thermal_scene_file_missing_on_disk_raises_runtime_error(self):
        """A `thermal_file_path` naming a file absent from the `thermal` folder
        is a ThermoScenes dataset bug: it must raise `RuntimeError`."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_path = Path(tmp_dir)
            (scene_path / "thermal").mkdir()
            frame = {
                "file_path": "images/frame_train_000.png",
                "transform_matrix": self._IDENTITY_TRANSFORM,
                "thermal_file_path": "thermal/does_not_exist.PNG",
            }
            (scene_path / "transforms.json").write_text(
                json.dumps(self._thermoscenes_transforms([frame]))
            )

            with self.assertRaises(RuntimeError):
                SingleChannelImageSceneLoader.thermal_scene(scene_path)

    def test_thermal_scene_vggt_allow_missing_images_skips_broken_frames(self):
        """The same two gaps above (a dangling reference and a blank entry)
        must instead be skipped, with a logged warning, via the VGGT
        (`allow_missing_images=True`) path, yielding a strictly smaller image
        count than an otherwise-identical intact scene's."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            intact_scene = Path(tmp_dir) / "intact"
            intact_scene.mkdir()
            intact_thermal_dir = intact_scene / "thermal"
            intact_thermal_dir.mkdir()
            for name in ("t0.png", "t1.png", "t2.png"):
                self._write_luminance_png(intact_thermal_dir / name, byte_value=128)

            intact_frames = [
                self._vggt_frame("images/train_000.png", 0.0, "thermal/t0.png"),
                self._vggt_frame("images/train_001.png", 1.0, "thermal/t1.png"),
                self._vggt_frame("images/eval_000.png", 2.0, "thermal/t2.png"),
            ]
            (intact_scene / "transforms.json").write_text(
                json.dumps({"frames": intact_frames})
            )

            intact_loader = SingleChannelImageSceneLoader.thermal_scene_vggt(
                intact_scene
            )
            intact_count = len(intact_loader.train_images) + len(
                intact_loader.eval_images
            )
            self.assertEqual(intact_count, 3)

            broken_scene = Path(tmp_dir) / "broken"
            broken_scene.mkdir()
            broken_thermal_dir = broken_scene / "thermal"
            broken_thermal_dir.mkdir()
            self._write_luminance_png(broken_thermal_dir / "t0.png", byte_value=128)

            broken_frames = [
                self._vggt_frame("images/train_000.png", 0.0, "thermal/t0.png"),
                # Dangling reference: never written to disk.
                self._vggt_frame("images/train_001.png", 1.0, "thermal/missing.png"),
                # Blank entry.
                self._vggt_frame("images/eval_000.png", 2.0, ""),
            ]
            (broken_scene / "transforms.json").write_text(
                json.dumps({"frames": broken_frames})
            )

            with self.assertLogs(rebel_shape_logger, level="WARNING") as captured:
                broken_loader = SingleChannelImageSceneLoader.thermal_scene_vggt(
                    broken_scene
                )
            broken_count = len(broken_loader.train_images) + len(
                broken_loader.eval_images
            )

            self.assertEqual(broken_count, 1)
            self.assertLess(broken_count, intact_count)
            self.assertEqual(len(captured.output), 2)
            self.assertTrue(
                any("missing.png" in message for message in captured.output),
                captured.output,
            )
            self.assertTrue(
                any("eval_000.png" in message for message in captured.output),
                captured.output,
            )

    # TestSingleChannelLinearDecode (merged)
    #
    # Regression test for commit 89331d6 ("Fix Spurious sRGB Conversion"):
    # Mitsuba auto-tags an 8-bit PNG/JPEG source as sRGB purely from its bit
    # depth, so a plain `convert(srgb_gamma=False)` still silently applies an
    # sRGB -> linear decode unless the source bitmap is first marked linear.
    # That decode darkens every mid-tone value (byte 128 -> ~0.216 instead of
    # the verbatim ~0.502), which corrupts downstream heat estimation.

    def test_thermal_scene_mid_tone_byte_decodes_linearly(self):
        """An 8-bit single-channel PNG whose pixels are all byte 128 must load
        as `128 / 255`, not as the sRGB-decoded ~0.216."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_path = Path(tmp_dir)
            thermal_dir = scene_path / "thermal"
            thermal_dir.mkdir()
            self._write_luminance_png(thermal_dir / "t0.PNG", byte_value=128)

            frame = {
                "file_path": "images/frame_train_000.png",
                "transform_matrix": self._IDENTITY_TRANSFORM,
                "thermal_file_path": "thermal/t0.PNG",
            }
            (scene_path / "transforms.json").write_text(
                json.dumps(self._thermoscenes_transforms([frame]))
            )

            loader = SingleChannelImageSceneLoader.thermal_scene(scene_path)
            value = loader.train_images[0].image.numpy()

            np.testing.assert_allclose(value, 128.0 / 255.0, rtol=0, atol=1e-4)
