import json
import shutil
import tempfile
import unittest
from pathlib import Path

from fem.end_to_end.ground_truth_bake.ground_truth_bake_report import (
    GroundTruthBakeReport,
)
from fem.end_to_end.ground_truth_bake.ground_truth_bake_runner import (
    GroundTruthBakeRunner,
)


class TestWriteBakeReport(unittest.TestCase):
    """
    `GroundTruthBakeRunner.write_bake_report` writes a `GroundTruthBakeReport` to
    JSON, verbatim.

    `ground_truth_bake.py` is the assembly step that welds/remeshes the
    source mesh into the two FEM analysis inputs (`heat_gt.npz`,
    `modal_gt.npz`); running either path costs on the order of 20 seconds
    and needs gpytoolbox, so the bake itself is not exercised here. The
    report writer is cheap and self-contained by comparison: it takes an
    already-built `GroundTruthBakeReport` and writes it to disk, and that
    step is worth pinning on its own, because `dataclasses.asdict` does
    not know how to serialise a `Path` (the two `path` fields in the
    record), so the writer's `default=str` is doing real work, not
    decoration.
    """

    def setUp(self) -> None:
        self._directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self._directory, ignore_errors=True)

    def test_write_bake_report_round_trips_every_field_with_curved_geometry(self):
        """
        Every scalar in a full record (order-2 modal mesh, `curved` populated)
        survives the write, both `Path` fields land as their string form (the
        one thing `asdict` cannot do unassisted, which is why `default=str` is
        passed at all), and the file ends with a trailing newline.

        Six distinct sentinel values per nested record (not e.g. all `1.0`)
        make a field transposed with its neighbour visible.
        """
        report = GroundTruthBakeReport(
            heat=GroundTruthBakeReport.HeatMesh(
                path=self._directory / "heat_gt.npz",
                num_triangles=1234,
                num_vertices=617,
                steel_fraction=0.375,
                hot_fraction=0.125,
            ),
            modal=GroundTruthBakeReport.ModalMesh(
                path=self._directory / "modal_gt.npz",
                num_triangles=456,
                num_vertices=230,
                curved=GroundTruthBakeReport.CurvedGeometry(
                    geometry_order=2,
                    num_edge_nodes=789,
                    num_left_straight=3,
                    flat_area_fraction=0.97,
                    curved_area_fraction=0.995,
                ),
            ),
        )
        out_path = self._directory / "ground_truth_bake_report.json"

        GroundTruthBakeRunner.write_bake_report(report, out_path)

        raw = out_path.read_text()
        self.assertTrue(raw.endswith("\n"))
        written = json.loads(raw)
        self.assertEqual(written["heat"]["path"], str(report.heat.path))
        self.assertEqual(written["heat"]["num_triangles"], 1234)
        self.assertEqual(written["heat"]["num_vertices"], 617)
        self.assertEqual(written["heat"]["steel_fraction"], 0.375)
        self.assertEqual(written["heat"]["hot_fraction"], 0.125)
        self.assertEqual(written["modal"]["path"], str(report.modal.path))
        self.assertEqual(written["modal"]["num_triangles"], 456)
        self.assertEqual(written["modal"]["num_vertices"], 230)
        self.assertEqual(written["modal"]["curved"]["geometry_order"], 2)
        self.assertEqual(written["modal"]["curved"]["num_edge_nodes"], 789)
        self.assertEqual(written["modal"]["curved"]["num_left_straight"], 3)
        self.assertEqual(written["modal"]["curved"]["flat_area_fraction"], 0.97)
        self.assertEqual(written["modal"]["curved"]["curved_area_fraction"], 0.995)

    def test_write_bake_report_writes_null_for_a_straight_sided_modal_mesh(self):
        """
        An order-1 modal mesh (the common case, absent an explicit request for
        curved geometry) carries `curved=None`. `asdict` recurses into a nested
        dataclass but leaves a plain `None` alone, so the writer must not choke
        on it, and the file should say so explicitly (a JSON `null`) rather
        than omitting the key, since a reader keying off its presence would
        otherwise mistake "no curving" for "the field was never computed".
        """
        report = GroundTruthBakeReport(
            heat=GroundTruthBakeReport.HeatMesh(
                path=self._directory / "heat_gt.npz",
                num_triangles=10,
                num_vertices=8,
                steel_fraction=1.0,
                hot_fraction=0.0,
            ),
            modal=GroundTruthBakeReport.ModalMesh(
                path=self._directory / "modal_gt.npz",
                num_triangles=6,
                num_vertices=5,
                curved=None,
            ),
        )
        out_path = self._directory / "ground_truth_bake_report.json"

        GroundTruthBakeRunner.write_bake_report(report, out_path)

        written = json.loads(out_path.read_text())
        self.assertIn("curved", written["modal"])
        self.assertIsNone(written["modal"]["curved"])
