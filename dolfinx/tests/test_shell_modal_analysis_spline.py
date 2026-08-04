"""
Tests for the FEM shell modal entry point.

Three things are pinned here, in decreasing order of how expensive they would be
to get wrong:

1. **The on-disk eigenmode format.** Recorded modal runs exist outside this
   repository and must stay readable, so the writer's layout -- dataset paths,
   shapes, dtypes, the temporal-collection XDMF -- is asserted explicitly, and
   confirmed against a recorded archive when this machine has one. This is the
   FEM writer's own contract; the IGA exporter is a deliberately independent
   implementation and is not imported here.
2. **The rigid-body report.** It is the only automatic check a production run
   performs on its own output, and it is a heuristic (largest relative gap),
   so it is tested on constructed spectra where the right answer is known.
3. **Parameter validation.** Every rule in `__post_init__`, from one table.

The solver itself is covered by `test_geometrically_exact_shell_solver.py`; the
export tests here build a tiny gmsh sphere directly rather than going through a
spline, so only the one end-to-end test pays for Mitsuba.

Run from the repository root::

    conda run -n dolfinx-rebel-shape python -m unittest discover dolfinx/tests
"""

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock
from xml.etree import ElementTree

import h5py
import numpy as np

# Imported for its module-level side effect: `_fem_fixtures` puts `dolfinx/` and
# the repository root on `sys.path`.
import _fem_fixtures  # noqa: F401

from geometrically_exact_shell_solver import (
    GeometricallyExactShellSolver,
    ShellElement,
    make_curved_sphere_mesh,
)
from shell_modal_analysis_spline import (
    ShellModalAnalysisParameters,
    _export_mode_shapes_xdmf,
    _report_rigid_body_modes,
    run_modal_analysis,
)

# Recorded end-to-end modal runs, if this machine has them. They live outside the
# repository, so the tests against them skip by default -- the explicit layout
# assertions above are the guard, these are confirmation. Same convention (and
# environment variable) as `test_format_compatibility.py`.
_RECORDED_RUNS = Path(
    os.environ.get(
        "REBEL_SHAPE_RECORDED_RUNS",
        Path.home() / "Schindler/rebel-shape_paper/End-to-end-verification/suzanne",
    )
)
_RECORDED_MODAL = _RECORDED_RUNS / "fem_modal_simulation" / "eigenmodes.xdmf"
_SKIP_REASON = (
    f"No recorded modal run at {_RECORDED_MODAL} (set REBEL_SHAPE_RECORDED_RUNS)."
)

# Datasets every mode carries, and the trailing dimension of each.
_MODE_DATASETS = {
    "geometry": 3,
    "displacement": 3,
    "rotation": 3,
    "displacement_magnitude": 1,
    "rotation_magnitude": 1,
}


def tiny_solver() -> GeometricallyExactShellSolver:
    """The smallest sphere shell solver that still has a spectrum, for exercising
    the export path. Concrete-like material; the numbers are irrelevant here."""
    return GeometricallyExactShellSolver(
        make_curved_sphere_mesh(1.0, 0.9, order=2),
        youngs_modulus=30.0e9,
        poisson_ratio=0.2,
        density=2250.0,
        thickness=0.2,
        element=ShellElement(bbar=True, psri=False),
    )


class TestParameterValidation(unittest.TestCase):
    """`__post_init__` is the only thing between a typo on the command line and a
    run that fails minutes later, or silently computes the wrong problem."""

    INVALID = (
        ("num_modes zero", {"num_modes": 0}),
        ("num_modes negative", {"num_modes": -3}),
        ("geometry order 0", {"geometry_order": 0}),
        ("geometry order 3", {"geometry_order": 3}),
        ("field degree 1", {"field_degree": 1}),
        ("field degree 4", {"field_degree": 4}),
        ("subdivisions zero", {"subdivisions_per_patch": 0}),
        ("target length scale zero", {"target_length_scale": 0.0}),
        ("target length scale negative", {"target_length_scale": -1.0}),
    )

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.spline_path = Path(self._temporary.name) / "surface.json"
        self.spline_path.write_text("{}")  # existence is all __post_init__ checks

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def valid_parameters(self, **overrides) -> ShellModalAnalysisParameters:
        return ShellModalAnalysisParameters(
            spline_path=self.spline_path,
            output_folder=Path(self._temporary.name) / "out",
            **overrides,
        )

    def test_valid_parameters_are_accepted(self) -> None:
        self.valid_parameters()

    def test_invalid_parameters_are_rejected(self) -> None:
        for label, overrides in self.INVALID:
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    self.valid_parameters(**overrides)

    def test_missing_spline_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ShellModalAnalysisParameters(
                spline_path=Path(self._temporary.name) / "absent.json"
            )

    def test_target_length_scale_may_be_none(self) -> None:
        """`None` is the documented "leave the geometry alone" value, and must not
        trip the positivity rule."""
        self.assertIsNone(self.valid_parameters(target_length_scale=None)
                          .target_length_scale)

    def test_bbar_is_the_default(self) -> None:
        """Every recorded production run used `use_bbar: True`; the default must
        not drift away from what the archive was produced with."""
        parameters = self.valid_parameters()
        self.assertTrue(parameters.use_bbar)
        self.assertFalse(parameters.use_psri)


class TestRigidBodyReport(unittest.TestCase):
    """
    `_report_rigid_body_modes` locates the rigid-body band as the largest
    relative gap in the low spectrum rather than by thresholding, so that it
    works at any geometry scale. Tested on constructed spectra: the detection is
    a property of the numbers, and building them by hand is the only way to know
    what the answer should be.
    """

    @staticmethod
    def spectrum(n_rigid: int = 6, n_elastic: int = 6) -> np.ndarray:
        """Round-off-level rigid-body residuals of both signs, then a physical
        spectrum -- the shape a real post-shift solve returns."""
        rigid = np.linspace(-5.0e-8, 6.0e-8, n_rigid)
        elastic = 3.5e6 * np.arange(1, n_elastic + 1, dtype=float)
        return np.concatenate([rigid, elastic])

    def report(self, eigenvalues: np.ndarray, n_rigid: int = 6):
        frequencies = np.sqrt(np.abs(eigenvalues)) / (2.0 * np.pi)
        with self.assertLogs("rebel_shape", level="INFO") as captured:
            _report_rigid_body_modes(eigenvalues, frequencies, n_rigid)
        return captured.output

    def test_detects_the_expected_rigid_body_count(self) -> None:
        output = self.report(self.spectrum())
        self.assertIn("largest spectral gap after mode 6", "\n".join(output))
        self.assertFalse([line for line in output if line.startswith("WARNING")])

    def test_warns_when_the_count_disagrees(self) -> None:
        """A shell with a spurious soft mode shows up as a seventh near-zero
        eigenvalue; the run must say so rather than quietly reporting a wrong
        first elastic frequency."""
        eigenvalues = self.spectrum()
        eigenvalues[6] = 7.0e-8  # a seventh near-zero mode
        output = self.report(np.sort(eigenvalues))
        self.assertTrue([line for line in output if line.startswith("WARNING")])

    def test_does_not_crash_for_a_single_mode(self) -> None:
        """`--num-modes 1` is a legal invocation. The gap detection needs two
        eigenvalues to form a ratio, so it must decline rather than raise -- and
        it must decline *after* nothing has been lost, i.e. with a warning."""
        output = self.report(np.array([-5.0e-8]), n_rigid=0)
        self.assertTrue([line for line in output if line.startswith("WARNING")])

    def test_is_scale_invariant(self) -> None:
        """Detection must not depend on the units of the problem: multiplying the
        whole spectrum by 1e12 (a metre/millimetre change, say) is the same
        physics and must give the same answer."""
        for scale in (1.0e-12, 1.0, 1.0e12):
            with self.subTest(scale=scale):
                output = self.report(self.spectrum() * scale)
                self.assertIn("largest spectral gap after mode 6", "\n".join(output))


class TestEigenmodeExport(unittest.TestCase):
    """
    The on-disk contract. Recorded runs under
    `<archive>/*/fem_modal_simulation/eigenmodes.{xdmf,h5}` were written by this
    layout and must stay readable, so it is asserted explicitly rather than
    round-tripped.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # No Mitsuba anywhere in this class, so a shared solver is safe.
        cls.solver = tiny_solver()
        cls.result = cls.solver.solve_modes_slepc(num_modes=8)

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def export(self, indices=(0, 1, 2), scale_factor: float = 0.1) -> Path:
        _export_mode_shapes_xdmf(
            self.solver, self.result, list(indices), self.folder, scale_factor
        )
        return self.folder / "eigenmodes.xdmf"

    def test_hdf5_layout(self) -> None:
        """Dataset paths, shapes and dtypes exactly as the archive carries them:
        an int32 `/Mesh/topology` of triangles, a float64 `/Mesh/vertices`, and
        `/Modes/mode_<k>/<field>` per exported mode, indexed by EXPORT position
        `k` (not by the original mode index, which lives in the XDMF)."""
        self.export(indices=(0, 3, 7))
        with h5py.File(self.folder / "eigenmodes.h5", "r") as h5_file:
            topology = h5_file["/Mesh/topology"]
            vertices = h5_file["/Mesh/vertices"]
            self.assertEqual(topology.dtype, np.int32)
            self.assertEqual(topology.shape[1], 3)
            self.assertEqual(vertices.dtype, np.float64)
            self.assertEqual(vertices.shape[1], 3)
            self.assertEqual(
                vertices.shape[0], self.solver.msh.geometry.x.shape[0]
            )
            for k in range(3):
                for name, columns in _MODE_DATASETS.items():
                    dataset = h5_file[f"/Modes/mode_{k}/{name}"]
                    self.assertEqual(
                        dataset.shape, (vertices.shape[0], columns), msg=name
                    )
                    self.assertEqual(dataset.dtype, np.float64, msg=name)
            self.assertNotIn("mode_3", h5_file["/Modes"])

    def test_undeformed_vertices_are_stored_for_alignment(self) -> None:
        """`/Mesh/vertices` is the undeformed geometry; the post-hoc MAC alignment
        step reads it, and it must not be the deformed copy that `/Modes/*/geometry`
        holds."""
        self.export()
        with h5py.File(self.folder / "eigenmodes.h5", "r") as h5_file:
            np.testing.assert_allclose(
                np.asarray(h5_file["/Mesh/vertices"]),
                self.solver.msh.geometry.x[:, :3],
                rtol=0.0,
            )

    def test_mode_geometry_is_the_scaled_deformation(self) -> None:
        """
        `geometry = vertices - unit_displacement * scale_factor * smallest_extent`.
        This is what ParaView shows, so an error here is invisible in the numbers
        and glaring in the picture. Checked by re-deriving it from the stored
        displacement; exact, since both come from the same array.
        """
        scale_factor = 0.25
        self.export(indices=(0,), scale_factor=scale_factor)
        vertices = self.solver.msh.geometry.x[:, :3]
        extents = vertices.max(axis=0) - vertices.min(axis=0)
        expected_scale = scale_factor * float(extents[extents > 1.0e-12].min())

        with h5py.File(self.folder / "eigenmodes.h5", "r") as h5_file:
            displacement = np.asarray(h5_file["/Modes/mode_0/displacement"])
            geometry = np.asarray(h5_file["/Modes/mode_0/geometry"])
        unit = displacement / np.linalg.norm(displacement, axis=1).max()
        np.testing.assert_allclose(
            geometry, vertices - unit * expected_scale, atol=1.0e-12
        )

    def test_magnitudes_match_their_vectors(self) -> None:
        self.export(indices=(0, 1))
        with h5py.File(self.folder / "eigenmodes.h5", "r") as h5_file:
            for k in range(2):
                for field in ("displacement", "rotation"):
                    vector = np.asarray(h5_file[f"/Modes/mode_{k}/{field}"])
                    magnitude = np.asarray(
                        h5_file[f"/Modes/mode_{k}/{field}_magnitude"]
                    )
                    np.testing.assert_allclose(
                        magnitude.ravel(),
                        np.linalg.norm(vector, axis=1),
                        rtol=1.0e-14,
                    )

    def test_mode_sign_is_canonical(self) -> None:
        """An eigenvector is defined up to sign, so the exporter fixes one: the
        largest-magnitude displacement component is positive. Without it, a FEM
        and an IGA run of the same mode can render as mirror images."""
        self.export(indices=(0, 1, 2))
        with h5py.File(self.folder / "eigenmodes.h5", "r") as h5_file:
            for k in range(3):
                flat = np.asarray(h5_file[f"/Modes/mode_{k}/displacement"]).ravel()
                self.assertGreater(flat[int(np.argmax(np.abs(flat)))], 0.0)

    def test_rotation_follows_the_displacement_sign_flip(self) -> None:
        """The flip is applied to the whole mode. Flipping only the displacement
        would leave a physically inconsistent mode shape on disk."""
        result = replace(
            self.result, eigenvectors=-self.result.eigenvectors.copy()
        )
        _export_mode_shapes_xdmf(self.solver, result, [0], self.folder, 0.1)
        with h5py.File(self.folder / "eigenmodes.h5", "r") as h5_file:
            flipped = {
                field: np.asarray(h5_file[f"/Modes/mode_0/{field}"])
                for field in ("displacement", "rotation")
            }
        self.export(indices=(0,))
        with h5py.File(self.folder / "eigenmodes.h5", "r") as h5_file:
            for field, values in flipped.items():
                # Negating the input then re-canonicalising restores the same mode.
                # IEEE negation is exact and the interpolation is linear, so this
                # should be bit-identical; 1e-12 leaves room for that not to hold.
                np.testing.assert_allclose(
                    values,
                    np.asarray(h5_file[f"/Modes/mode_0/{field}"]),
                    rtol=1.0e-12,
                )

    def test_xdmf_is_a_temporal_collection_of_uniform_grids(self) -> None:
        """The structure ParaView needs to animate the modes, and the structure a
        recorded file has. Time value is the EXPORT index, so modes advance one
        per frame however they were selected."""
        path = self.export(indices=(0, 4, 7))
        root = ElementTree.parse(path).getroot()
        collection = root.find("Domain/Grid")
        self.assertEqual(collection.get("GridType"), "Collection")
        self.assertEqual(collection.get("CollectionType"), "Temporal")

        grids = collection.findall("Grid")
        self.assertEqual([grid.get("Name") for grid in grids],
                         ["mode_0", "mode_4", "mode_7"])
        for k, grid in enumerate(grids):
            self.assertEqual(grid.find("Time").get("Value"), str(k))
            self.assertEqual(
                grid.find("Topology").get("TopologyType"), "Triangle"
            )
            self.assertEqual(grid.find("Geometry").get("GeometryType"), "XYZ")
            attributes = {a.get("Name"): a.get("AttributeType")
                          for a in grid.findall("Attribute")}
            self.assertEqual(
                attributes,
                {
                    "displacement": "Vector",
                    "rotation": "Vector",
                    "displacement_magnitude": "Scalar",
                    "rotation_magnitude": "Scalar",
                },
            )

    def test_xdmf_references_only_the_companion_hdf5_by_name(self) -> None:
        """Relative reference, so the pair stays readable after being moved or
        copied out of `outputs/` -- which is how the archive was assembled."""
        path = self.export()
        for item in ElementTree.parse(path).getroot().iter("DataItem"):
            self.assertTrue(item.text.strip().startswith("eigenmodes.h5:"))

    def test_out_of_range_indices_are_skipped_with_a_warning(self) -> None:
        """The default index list is a fixed range, so asking for more modes than
        were computed is routine -- but it must be reported, not silent."""
        with self.assertLogs("rebel_shape", level="WARNING"):
            path = self.export(indices=(0, 99))
        grids = ElementTree.parse(path).getroot().findall("Domain/Grid/Grid")
        self.assertEqual([grid.get("Name") for grid in grids], ["mode_0"])

    def test_no_valid_indices_writes_nothing(self) -> None:
        with self.assertLogs("rebel_shape", level="WARNING"):
            _export_mode_shapes_xdmf(self.solver, self.result, [99], self.folder)
        self.assertFalse((self.folder / "eigenmodes.xdmf").exists())

    @unittest.skipUnless(_RECORDED_MODAL.exists(), _SKIP_REASON)
    def test_recorded_archive_has_the_same_layout(self) -> None:
        """
        Confirmation against a file this code did not write. A writer change
        mirrored by a reader change would pass every round trip here while making
        the recorded runs unreadable; only a recorded file catches that.
        """
        self.export(indices=(0, 1))
        with h5py.File(self.folder / "eigenmodes.h5", "r") as written, h5py.File(
            _RECORDED_MODAL.with_suffix(".h5"), "r"
        ) as recorded:
            written_paths = {
                path for path in _visit_datasets(written) if "mode_0/" in path
                or path.startswith("Mesh/")
            }
            recorded_paths = {
                path for path in _visit_datasets(recorded) if "mode_0/" in path
                or path.startswith("Mesh/")
            }
            self.assertEqual(written_paths, recorded_paths)
            for path in written_paths:
                self.assertEqual(
                    written[path].dtype, recorded[path].dtype, msg=path
                )
                self.assertEqual(
                    written[path].shape[1:], recorded[path].shape[1:], msg=path
                )

        def structure(xdmf_path: Path) -> list[str]:
            root = ElementTree.parse(xdmf_path).getroot()
            grid = root.find("Domain/Grid/Grid")
            return [element.tag for element in grid] + sorted(
                attribute.get("Name") for attribute in grid.findall("Attribute")
            )

        self.assertEqual(
            structure(self.folder / "eigenmodes.xdmf"), structure(_RECORDED_MODAL)
        )


def _visit_datasets(h5_file) -> list[str]:
    paths: list[str] = []
    h5_file.visit(paths.append)
    return [path for path in paths if isinstance(h5_file[path], h5py.Dataset)]


class TestRunModalAnalysis(unittest.TestCase):
    """
    The entry point end to end, on the smallest spline that has patch seams. One
    test, because it is the only one here that pays for Mitsuba and the spline
    load.
    """

    def setUp(self) -> None:
        # Mitsuba/Dr.Jit objects are built per test, never held on the class.
        import jsonpickle
        import _spline_fixtures  # noqa: F401
        from rebel_shape.splines.cube_initialization import cubed_sphere

        self._temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self._temporary.name)
        self.spline_path = self.folder / "sphere.json"
        self.spline_path.write_text(
            jsonpickle.encode(
                cubed_sphere(
                    name="modal_entry_sphere",
                    degree=3,
                    spline_subdivisions=4,
                    tessellation_subdivisions=8,
                    optimize_g1=False,
                )
            )
        )

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_end_to_end_run_writes_results_and_modes(self) -> None:
        """
        Everything a production run produces: a spectrum with six rigid-body
        modes, an `.npz` carrying the four result arrays, and an eigenmode pair
        on disk. The scipy path is used so `mode_energies` is populated -- that
        is the one output the SLEPc path deliberately leaves zero.
        """
        parameters = ShellModalAnalysisParameters(
            spline_path=self.spline_path,
            output_folder=self.folder / "results",
            num_modes=8,
            subdivisions_per_patch=2,
            field_degree=2,
            geometry_order=2,
            target_length_scale=3.0,
            use_scipy_solver=True,
            export_xdmf_indices=[0, 6, 7],
        )
        result = run_modal_analysis(params=parameters)

        rigid = np.abs(result.eigenvalues[:6]).max()
        self.assertGreater(
            abs(result.eigenvalues[6]) / max(rigid, 1.0e-300),
            1.0e8,
            msg=f"Spectrum lacks a clean rigid-body band: {result.eigenvalues}",
        )

        saved = np.load(parameters.output_folder / "modal_analysis_results.npz")
        self.assertEqual(
            set(saved.keys()),
            {"eigenvalues", "eigenvectors", "frequencies", "mode_energies"},
        )
        np.testing.assert_allclose(saved["eigenvalues"], result.eigenvalues, rtol=0.0)
        self.assertEqual(saved["mode_energies"].shape, (8, 4))
        self.assertGreater(
            saved["mode_energies"][6].sum(),
            0.0,
            msg="The scipy path must report a non-zero energy split.",
        )

        self.assertTrue((parameters.output_folder / "eigenmodes.xdmf").exists())
        with h5py.File(parameters.output_folder / "eigenmodes.h5", "r") as h5_file:
            self.assertEqual(len(h5_file["/Modes"]), 3)

    def test_spectral_shift_reaches_the_scipy_path(self) -> None:
        """The parameter used to be threaded into the SLEPc branch only, so a
        `--use-scipy-solver --spectral-shift X` run silently ignored X."""
        parameters = ShellModalAnalysisParameters(
            spline_path=self.spline_path,
            output_folder=self.folder / "shift",
            num_modes=8,
            subdivisions_per_patch=2,
            use_scipy_solver=True,
            spectral_shift=1234.5,
            export_xdmf=False,
        )
        with mock.patch.object(
            GeometricallyExactShellSolver,
            "solve_modes",
            autospec=True,
            side_effect=GeometricallyExactShellSolver.solve_modes,
        ) as spied:
            run_modal_analysis(params=parameters)
        self.assertEqual(spied.call_args.kwargs["spectral_shift"], 1234.5)


if __name__ == "__main__":
    unittest.main()
