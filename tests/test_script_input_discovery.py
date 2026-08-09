"""
How the entry-point scripts and their batch drivers resolve directories on disk: which
checkpoint an optimization result folder means, which texture belongs to which mesh,
refusing a dataset that has already been renamed, and refusing an output folder that
still holds an earlier run's metrics.

Every case here is a regression test for a directory layout that a real run actually
produces and that the previous, order-dependent implementations resolved wrongly, so
the fixtures are built as files rather than mocked away.
"""

import json
import tempfile
import unittest
from pathlib import Path

import mitsuba as mi
import numpy as np

from rebel_shape.scripts.evaluation.eval_all_common import (
    prepare_metrics_output_folder,
)
from rebel_shape.scripts.render_eval_images import discover_spline_checkpoint
from rebel_shape.scripts.render_eval_images_mesh import (
    discover_mesh_textures,
    face_key,
)
from rebel_shape.scripts.rename_dataset import RenameDatasetParams, rename_dataset


def _touch(directory: Path, *names: str) -> None:
    """Creates every file in `names` under `directory`, with no content."""
    for name in names:
        (directory / name).touch()


def _build_vggt_dataset(root: Path, frame_count: int) -> None:
    """
    Writes a minimal VGGT-style dataset under `root`: `frame_count` frames, each with
    an `images/<frame>.jpeg`, a `depths/<frame>.npy`, and an entry in
    `transforms.json`. Only the file names matter to `rename_dataset`, so the image
    bytes are a bare JPEG marker pair.
    """
    (root / "images").mkdir(parents=True)
    (root / "depths").mkdir()

    frames = []
    for index in range(frame_count):
        frame_name = f"f{index:04d}"
        (root / "images" / f"{frame_name}.jpeg").write_bytes(b"\xff\xd8\xff\xd9")
        np.save(root / "depths" / f"{frame_name}.npy", np.zeros((2, 2)))
        frames.append(
            {
                "rgb": {
                    "file_path": f"images/{frame_name}.jpeg",
                    "depth_file_path": f"depths/{frame_name}.npy",
                }
            }
        )
    (root / "transforms.json").write_text(json.dumps({"frames": frames}, indent=4))


class TestScriptInputDiscovery(unittest.TestCase):
    """
    Tests for `render_eval_images.discover_spline_checkpoint`,
    `render_eval_images_mesh.discover_mesh_textures`,
    `eval_all_common.prepare_metrics_output_folder` and `rename_dataset`'s
    already-renamed guard.
    """

    @classmethod
    def setUpClass(cls):
        mi.set_variant("llvm_ad_rgb")

    # -- discover_spline_checkpoint ---------------------------------------------

    def test_spline_checkpoint_ignores_an_evaluation_metrics_json(self):
        """The recorded result folders hold an `eval_*.json` metrics dump next to
        the spline checkpoint, and picking the first entry of an unsorted
        `iterdir()` selected that instead. Only a name carrying an
        `iteration-<N>` marker may be considered."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            _touch(
                folder,
                "eval_spline_chamfer.json",
                "shape_20260601-023350_iteration-249.json",
                "shape_20260601-023350_iteration-249.step",
            )

            self.assertEqual(
                discover_spline_checkpoint(folder).name,
                "shape_20260601-023350_iteration-249.json",
            )

    def test_spline_checkpoint_picks_the_highest_iteration(self):
        """With several periodic checkpoints present, the latest iteration wins -
        and by iteration number, not by the lexicographic order in which `9` sorts
        after `10`."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            _touch(
                folder,
                "shape_20260601-023350_iteration-009.json",
                "shape_20260601-023350_iteration-249.json",
                "shape_20260601-023350_iteration-100.json",
            )

            self.assertEqual(
                discover_spline_checkpoint(folder).name,
                "shape_20260601-023350_iteration-249.json",
            )

    def test_spline_checkpoint_absent_raises_naming_the_candidates(self):
        """A folder with JSON files but no checkpoint must fail with a message
        naming what it did find, rather than handing a metrics file downstream."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            _touch(folder, "eval_spline_chamfer.json", "chamfer.json")

            with self.assertRaises(FileNotFoundError) as context:
                discover_spline_checkpoint(folder)

            self.assertIn("eval_spline_chamfer.json", str(context.exception))
            self.assertIn("chamfer.json", str(context.exception))

    # -- discover_mesh_textures --------------------------------------------------

    def test_mesh_textures_keys_the_final_save_without_its_suffix(self):
        """`shape_left.ply`, the unnumbered final save, used to be keyed
        `shape_left.ply` - a key no `.exr` name can contain, so the final mesh was
        rendered with the placeholder texture. It must key as `shape_left` and find
        `shape_left_texture.exr`."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            _touch(folder, "shape_left.ply", "shape_left_texture.exr")

            discovered = discover_mesh_textures(folder)

            self.assertEqual(list(discovered), ["shape_left"])
            self.assertEqual(
                discovered["shape_left"],
                (
                    (folder / "shape_left.ply").resolve(),
                    (folder / "shape_left_texture.exr").resolve(),
                ),
            )

    def test_mesh_textures_excludes_the_welded_global_mesh(self):
        """The whole-surface PLY saved beside the patches names no cube face and
        must not enter the scene as a seventh patch, duplicating geometry."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            _touch(folder, "shape_left.ply", "thermoscene_global_mesh.ply")

            self.assertEqual(list(discover_mesh_textures(folder)), ["shape_left"])

    def test_mesh_textures_prefers_the_final_save_over_a_checkpoint(self):
        """Numbered checkpoints are written during the run and the unnumbered save
        after it, so the unnumbered one is the latest state and must win. The same
        ranking applies to the patch's texture."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            _touch(
                folder,
                "shape_left_20260601-023350_iteration-020.ply",
                "shape_left_20260601-023350_iteration-100.ply",
                "shape_left.ply",
                "shape_left-bsdf-reflectance-data_20260601-023350_iteration-100.exr",
                "shape_left_texture.exr",
            )

            self.assertEqual(
                discover_mesh_textures(folder)["shape_left"],
                (
                    (folder / "shape_left.ply").resolve(),
                    (folder / "shape_left_texture.exr").resolve(),
                ),
            )

    def test_mesh_textures_picks_the_highest_iteration_checkpoint(self):
        """Without a final save, the highest-numbered checkpoint is the answer -
        again by number, so `100` beats `020` where a sort by name would not."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            _touch(
                folder,
                "shape_top_20260601-023350_iteration-020.ply",
                "shape_top_20260601-023350_iteration-100.ply",
                "shape_top-bsdf-reflectance-data_20260601-023350_iteration-020.exr",
                "shape_top-bsdf-reflectance-data_20260601-023350_iteration-100.exr",
            )

            texture_name = (
                "shape_top-bsdf-reflectance-data_20260601-023350_iteration-100.exr"
            )
            self.assertEqual(
                discover_mesh_textures(folder)["shape_top"],
                (
                    (folder / "shape_top_20260601-023350_iteration-100.ply").resolve(),
                    (folder / texture_name).resolve(),
                ),
            )

    def test_mesh_textures_untextured_patch_yields_none(self):
        """A run that saved no texture for a patch must report `None` rather than
        mis-attaching a neighbour's texture; `initialize_scene` then substitutes a
        grey placeholder."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            _touch(
                folder, "shape_left.ply", "shape_right.ply", "shape_left_texture.exr"
            )

            discovered = discover_mesh_textures(folder)

            self.assertIsNotNone(discovered["shape_left"][1])
            self.assertIsNone(discovered["shape_right"][1])

    def test_mesh_textures_two_runs_in_one_folder_raise(self):
        """Two composites claiming the same cube face cannot both be the answer;
        silently taking whichever `iterdir()` returned first is what this
        replaces."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            _touch(folder, "shape_left.ply", "other_left.ply")

            with self.assertRaises(ValueError) as context:
                discover_mesh_textures(folder)

            self.assertIn("left", str(context.exception))

    def test_mesh_textures_empty_folder_raises_naming_the_expected_pattern(self):
        """A folder with no mesh at all must say what it was looking for - both the
        per-face pattern and the single-mesh alternative."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            _touch(folder, "views_20260605-190856_iteration-249.png")

            with self.assertRaises(FileNotFoundError) as context:
                discover_mesh_textures(folder)

            self.assertIn("bottom", str(context.exception))
            self.assertIn("single whole-surface PLY", str(context.exception))

    def test_mesh_textures_composite_layout_of_a_recorded_run(self):
        """The recorded `buildnet-2/results/large_step` layout: six numbered patch
        saves, their six textures, and the welded `fuzed.ply` beside them. All six
        patches resolve and `fuzed.ply` stays out - the per-face behaviour this
        must not disturb."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            faces = ("back", "bottom", "front", "left", "right", "top")
            _touch(folder, "fuzed.ply")
            for face in faces:
                _touch(
                    folder,
                    f"shape_{face}_20260604-093436_iteration-249.ply",
                    f"shape_{face}-bsdf-reflectance-data_20260604-093436"
                    f"_iteration-249.exr",
                )

            discovered = discover_mesh_textures(folder)

            self.assertEqual(sorted(discovered), [f"shape_{face}" for face in faces])
            for face in faces:
                mesh, texture = discovered[f"shape_{face}"]
                self.assertEqual(
                    mesh.name, f"shape_{face}_20260604-093436_iteration-249.ply"
                )
                self.assertIsNotNone(texture)
                self.assertIn(face, texture.name)

    # -- discover_mesh_textures, single-mesh layout -------------------------------

    def test_mesh_textures_single_mesh_pairs_the_only_mesh_and_texture(self):
        """The recorded `car` and `buildinga_spring` LargeSteps layout: one mesh and
        one texture, named alike but with no cube face and no shared stem. Being the
        only candidates is what pairs them."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            _touch(
                folder,
                "thermoscene_shape.ply",
                "thermoscene_rgb_texture.exr",
                "views_20260522-031529_iteration-049.png",
            )

            discovered = discover_mesh_textures(folder)

            self.assertEqual(
                discovered,
                {
                    "thermoscene_shape": (
                        (folder / "thermoscene_shape.ply").resolve(),
                        (folder / "thermoscene_rgb_texture.exr").resolve(),
                    )
                },
            )

    def test_mesh_textures_single_mesh_pairs_across_a_name_mismatch(self):
        """The recorded `woodshed` layout, where the texture's stem
        (`shape-bsdf-reflectance-data_...`) does not start with the mesh's stem
        (`shape_...`), so the per-patch prefix rule would find nothing. Uniqueness
        must still pair them."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            _touch(
                folder,
                "shape_20260521-225221_iteration-049.ply",
                "shape-bsdf-reflectance-data_20260521-225221_iteration-049.exr",
            )

            mesh, texture = discover_mesh_textures(folder)[
                "shape_20260521-225221_iteration-049"
            ]

            self.assertEqual(mesh.name, "shape_20260521-225221_iteration-049.ply")
            self.assertEqual(
                texture.name,
                "shape-bsdf-reflectance-data_20260521-225221_iteration-049.exr",
            )

    def test_mesh_textures_lone_welded_mesh_is_the_single_reconstruction(self):
        """A welded whole-surface mesh is skipped only because patch meshes are there
        to prefer. On its own it is the one mesh the folder holds, and rendering it is
        the only reading available - so it is now paired rather than refused."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            _touch(folder, "thermoscene_global_mesh.ply")

            self.assertEqual(
                list(discover_mesh_textures(folder)), ["thermoscene_global_mesh"]
            )

    def test_mesh_textures_single_mesh_without_a_texture_yields_none(self):
        """A single-mesh run that saved no texture reports `None`, exactly as an
        untextured patch does; `initialize_scene` then substitutes grey."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            _touch(folder, "thermoscene_shape.ply")

            self.assertEqual(
                discover_mesh_textures(folder)["thermoscene_shape"],
                ((folder / "thermoscene_shape.ply").resolve(), None),
            )

    def test_mesh_textures_two_unkeyed_meshes_raise_naming_both(self):
        """Two whole-surface meshes carry nothing to rank or match on - the
        `iteration-<N>` ordering only orders saves of one patch - so picking either
        would be a guess. Both names must appear in the refusal."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            _touch(folder, "thermoscene_shape.ply", "fuzed.ply")

            with self.assertRaises(ValueError) as context:
                discover_mesh_textures(folder)

            self.assertIn("thermoscene_shape.ply", str(context.exception))
            self.assertIn("fuzed.ply", str(context.exception))

    def test_mesh_textures_single_mesh_with_two_textures_raises_naming_both(self):
        """With one mesh but two textures there is no key to match on, so the pairing
        is ambiguous and must be refused rather than resolved by sort order."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            _touch(
                folder,
                "thermoscene_shape.ply",
                "thermoscene_rgb_texture.exr",
                "thermoscene_thermal_texture.exr",
            )

            with self.assertRaises(ValueError) as context:
                discover_mesh_textures(folder)

            self.assertIn("thermoscene_rgb_texture.exr", str(context.exception))
            self.assertIn("thermoscene_thermal_texture.exr", str(context.exception))

    def test_mesh_textures_a_patch_mesh_keeps_the_composite_layout(self):
        """One cube-face mesh is enough to mean the per-face layout, so the extra
        whole-surface meshes beside it are ignored rather than making the folder
        ambiguous."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            _touch(folder, "shape_left.ply", "fuzed.ply", "thermoscene_global_mesh.ply")

            self.assertEqual(list(discover_mesh_textures(folder)), ["shape_left"])

    def test_face_key_separates_a_patch_key_from_a_whole_surface_key(self):
        """`eval_all_buildnet3d_large_steps` hands one mesh to the Chamfer step and
        must refuse a composite. Counting the discovered keys is not enough: a run
        that saved one patch and then died leaves a single key that is still one
        sixth of a surface, so the test has to be whether the key names a face."""
        self.assertEqual(face_key("shape_left"), "shape_left")
        self.assertEqual(
            face_key("shape_left_20260605-190857_iteration-249"), "shape_left"
        )
        self.assertIsNone(face_key("thermoscene_shape"))
        self.assertIsNone(face_key("lion_thermoscene_shape"))
        self.assertIsNone(face_key("shape_20260521-225221_iteration-049"))
        self.assertIsNone(face_key("fuzed"))

    def test_mesh_textures_a_lone_surviving_patch_still_keys_as_a_patch(self):
        """A directory left with one patch mesh discovers one entry, exactly as a
        single-mesh directory does - so the entry count cannot tell them apart and
        the key must stay face-derived for the Chamfer guard to see the difference."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            _touch(folder, "shape_left_20260605-190857_iteration-249.ply")

            discovered = discover_mesh_textures(folder)

            self.assertEqual(len(discovered), 1)
            self.assertIsNotNone(face_key(next(iter(discovered))))

    # -- prepare_metrics_output_folder -------------------------------------------

    def test_metrics_output_folder_refuses_an_earlier_run_s_metrics(self):
        """A driver reads back whatever metrics file the per-scene script left in the
        folder, and cannot tell this run's from the last one's. Re-running into a
        populated output root used to aggregate the stale numbers while this run's
        renders failed, so the folder must be refused - naming the file, and leaving
        it in place."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir) / "extracted-car_0"
            folder.mkdir()
            (folder / "eval_nvs.json").write_text('{"psnr": 1.0}')

            with self.assertRaises(FileExistsError) as context:
                prepare_metrics_output_folder(folder, "eval_nvs.json")

            self.assertIn("eval_nvs.json", str(context.exception))
            self.assertEqual((folder / "eval_nvs.json").read_text(), '{"psnr": 1.0}')

    def test_metrics_output_folder_refuses_on_any_of_the_named_files(self):
        """The buildnet3d drivers read two metrics files, and a folder holding either
        one is stale. A leftover `eval_shape.json` must be caught even when
        `eval_nvs.json` is absent."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir) / "scene_0"
            folder.mkdir()
            _touch(folder, "eval_shape.json")

            with self.assertRaises(FileExistsError) as context:
                prepare_metrics_output_folder(
                    folder, "eval_nvs.json", "eval_shape.json"
                )

            self.assertIn("eval_shape.json", str(context.exception))

    def test_metrics_output_folder_creates_a_fresh_folder_and_reuses_a_clean_one(self):
        """The guard must not cost the normal cases: a folder that does not exist yet
        is created with its parents, and one holding only a previous run's renders -
        which the driver overwrites anyway - is accepted."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fresh = Path(tmp_dir) / "metrics" / "extracted-car_0"

            prepare_metrics_output_folder(fresh, "eval_nvs.json")

            self.assertTrue(fresh.is_dir())

            _touch(fresh, "eval_images.png", "rendered_images.png")
            prepare_metrics_output_folder(fresh, "eval_nvs.json")

            self.assertTrue((fresh / "eval_images.png").exists())

    # -- rename_dataset ----------------------------------------------------------

    def test_rename_dataset_refuses_an_already_renamed_dataset(self):
        """Running twice used to re-scan the prefixed names, reshuffle the split
        over them and produce `eval_eval_f0000.jpeg`, logging success both times.
        The second run must refuse, and must leave the first run's result exactly
        as it stands."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _build_vggt_dataset(dataset_path, frame_count=5)
            params = RenameDatasetParams(dataset_path=dataset_path)

            rename_dataset(params=params)
            after_first_run = sorted(
                path.name for path in (dataset_path / "images").iterdir()
            )

            with self.assertRaises(ValueError) as context:
                rename_dataset(params=params)

            self.assertIn("already been renamed", str(context.exception))
            self.assertEqual(
                sorted(path.name for path in (dataset_path / "images").iterdir()),
                after_first_run,
            )
            self.assertTrue(
                all(name.startswith(("train_f", "eval_f")) for name in after_first_run),
                after_first_run,
            )

    def test_rename_dataset_first_run_prefixes_images_depths_and_transforms(self):
        """The guard must not have cost the first run anything: every image and
        depth file is prefixed exactly once, and `transforms.json` points at the
        new names."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _build_vggt_dataset(dataset_path, frame_count=5)

            rename_dataset(params=RenameDatasetParams(dataset_path=dataset_path))

            image_names = sorted(
                path.name for path in (dataset_path / "images").iterdir()
            )
            depth_names = sorted(
                path.name for path in (dataset_path / "depths").iterdir()
            )
            self.assertEqual(len(image_names), 5)
            self.assertEqual(len(depth_names), 5)

            transforms = json.loads((dataset_path / "transforms.json").read_text())
            for frame in transforms["frames"]:
                image_path = Path(frame["rgb"]["file_path"])
                depth_path = Path(frame["rgb"]["depth_file_path"])
                self.assertIn(image_path.name, image_names)
                self.assertIn(depth_path.name, depth_names)
                self.assertTrue((dataset_path / image_path).exists())
                self.assertTrue((dataset_path / depth_path).exists())


if __name__ == "__main__":
    unittest.main()
