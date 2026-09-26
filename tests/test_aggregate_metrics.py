import json
import tempfile
import unittest
from pathlib import Path

from rebel_shape.scripts.evaluation.aggregate_metrics import (
    AggregateMetricsParameters,
    aggregate_metrics,
)


class TestAggregateMetrics(unittest.TestCase):
    """
    Tests for `aggregate_metrics.py`, which averages the per-target metrics files
    the evaluation scripts write, and which the `main_scripts/evaluation/` sweeps
    spawn once per scene.
    """

    @staticmethod
    def _write_metrics(folder: Path, file_name: str, metrics: dict) -> None:
        """Writes `metrics` to `folder`/`file_name` as the evaluation scripts do."""
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / file_name, "w") as f:
            json.dump(metrics, f, indent=4)

    @staticmethod
    def _read(path: Path) -> dict:
        """The JSON object stored at `path`."""
        with open(path) as f:
            return json.load(f)

    def test_averages_each_metric_over_targets_merging_both_files(self):
        """Each metric is the mean over the three targets, with the NVS and shape
        files merged per target: hand-computed means 21.0, 0.5 and 2.0. Catches a
        reader that keeps only the first file, or averages across metrics instead
        of across targets."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            targets = [
                (10.0, 0.25, 0.5),
                (20.0, 0.5, 1.5),
                (33.0, 0.75, 4.0),
            ]
            folders = []
            for index, (psnr, ssim, chamfer) in enumerate(targets):
                folder = root / f"scene_{index}"
                self._write_metrics(
                    folder, "eval_nvs.json", {"psnr": psnr, "ssim": ssim}
                )
                self._write_metrics(
                    folder, "eval_shape.json", {"chamfer_distance": chamfer}
                )
                folders.append(folder)
            output = root / "scene_aggregated.json"

            aggregate_metrics(
                AggregateMetricsParameters(
                    metrics_folders=folders,
                    metrics_file_names=[Path("eval_nvs.json"), Path("eval_shape.json")],
                    output_file_path=output,
                )
            )

            aggregated = self._read(output)
            self.assertEqual(list(aggregated), ["psnr", "ssim", "chamfer_distance"])
            self.assertAlmostEqual(aggregated["psnr"], 21.0)
            self.assertAlmostEqual(aggregated["ssim"], 0.5)
            self.assertAlmostEqual(aggregated["chamfer_distance"], 2.0)

    def test_single_target_is_written_through_unchanged(self):
        """The large-steps sweeps evaluate one reconstruction per scene and record
        its metrics as they are. The written values must equal the inputs exactly,
        not merely closely, so a rounding or a reformatting pass is caught."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir) / "scene"
            metrics = {"psnr": 0.1 + 0.2, "lpips": 1.0 / 3.0}
            self._write_metrics(folder, "eval_nvs.json", metrics)
            output = Path(tmp_dir) / "scene_metrics.json"

            aggregate_metrics(
                AggregateMetricsParameters(
                    metrics_folders=[folder],
                    metrics_file_names=[Path("eval_nvs.json")],
                    output_file_path=output,
                )
            )

            self.assertEqual(self._read(output), metrics)

    def test_output_folder_is_created(self):
        """The sweeps hand a path under a metrics root that may not exist yet."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir) / "scene"
            self._write_metrics(folder, "eval_shape.json", {"chamfer_distance": 1.0})
            output = Path(tmp_dir) / "not" / "yet" / "there.json"

            aggregate_metrics(
                AggregateMetricsParameters(
                    metrics_folders=[folder],
                    metrics_file_names=[Path("eval_shape.json")],
                    output_file_path=output,
                )
            )

            self.assertTrue(output.is_file())

    def test_invalid_no_target_refuses_to_write_an_empty_mean(self):
        """A scene with no evaluated reconstruction has no mean. Writing `{}` would
        read as a completed scene, so the empty case is refused."""
        with self.assertRaises(ValueError):
            AggregateMetricsParameters(
                metrics_folders=[],
                metrics_file_names=[Path("eval_nvs.json")],
                output_file_path=Path("unused.json"),
            )

    def test_invalid_missing_metrics_file_is_named(self):
        """A target whose evaluation wrote no metrics file must stop the aggregate,
        naming the file, instead of averaging over the targets that did."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            present = Path(tmp_dir) / "scene_0"
            absent = Path(tmp_dir) / "scene_1"
            self._write_metrics(present, "eval_nvs.json", {"psnr": 1.0})
            absent.mkdir()

            with self.assertRaises(FileNotFoundError) as context:
                AggregateMetricsParameters(
                    metrics_folders=[present, absent],
                    metrics_file_names=[Path("eval_nvs.json")],
                    output_file_path=Path(tmp_dir) / "out.json",
                )

            self.assertIn(str(absent / "eval_nvs.json"), str(context.exception))

    def test_invalid_metric_reported_by_two_files(self):
        """Merging two files that both report `psnr` would keep one value and
        silently drop the other."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir) / "scene_0"
            self._write_metrics(folder, "eval_nvs.json", {"psnr": 1.0})
            self._write_metrics(folder, "eval_shape.json", {"psnr": 2.0})

            with self.assertRaisesRegex(ValueError, "psnr"):
                aggregate_metrics(
                    AggregateMetricsParameters(
                        metrics_folders=[folder],
                        metrics_file_names=[
                            Path("eval_nvs.json"),
                            Path("eval_shape.json"),
                        ],
                        output_file_path=Path(tmp_dir) / "out.json",
                    )
                )

    def test_invalid_targets_reporting_different_metrics(self):
        """If one target lacks `lpips`, its mean would be taken over the others
        only while reading as a mean over every target."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            full = Path(tmp_dir) / "scene_0"
            partial = Path(tmp_dir) / "scene_1"
            self._write_metrics(full, "eval_nvs.json", {"psnr": 1.0, "lpips": 0.1})
            self._write_metrics(partial, "eval_nvs.json", {"psnr": 2.0})
            output = Path(tmp_dir) / "out.json"

            with self.assertRaisesRegex(ValueError, "lpips"):
                aggregate_metrics(
                    AggregateMetricsParameters(
                        metrics_folders=[full, partial],
                        metrics_file_names=[Path("eval_nvs.json")],
                        output_file_path=output,
                    )
                )

            self.assertFalse(output.exists())
