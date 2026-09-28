import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Imported for its module-level side effect: `_fem_fixtures` puts the
# repository root on `sys.path`, so `fem` and `forge_sim` resolve.
import _fem_fixtures  # noqa: F401
import h5py
import jsonpickle
import mitsuba as mi
import numpy as np

from fem.shells.geometrically_exact_shell_modal_solver import (
    GeometricallyExactShellModalSolver,
)
from fem.shells.shell_element import ShellElement
from fem.shells.shell_modal_analysis_spline import (
    ShellModalAnalysisParameters,
    run_modal_analysis,
)
from forge_sim.splines.cube_initialization import cubed_sphere


class TestShellModalAnalysisSpline(unittest.TestCase):
    """
    Parameter validation, then the entry point end to end.

    Two things are pinned here: every rule in `__post_init__`, from one
    table, plus the translation of the two locking-cure flags.
    `__post_init__` is all that stands between a typo on the command line
    and a run that fails minutes later, or silently computes the wrong
    problem, and the run itself, end to end on the smallest spline that
    has patch seams, which is the only pair of tests here that pays for
    Mitsuba and a real solve.

    What the entry point calls is tested next to what it calls: the solver
    in `test_geometrically_exact_shell_modal_solver.py`, the mesh bridge in
    `test_spline_to_shell_mesh.py`, the eigenmode writer in
    `test_eigenmode_xdmf.py` and the rigid-body report in
    `test_rigid_body_modes.py`.
    """

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
        ("both locking cures", {"use_bbar": True, "use_psri": True}),
    )

    @classmethod
    def setUpClass(cls) -> None:
        mi.set_variant("llvm_ad_mono")

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self._temporary.name)
        self.spline_path = self.folder / "surface.json"
        self.spline_path.write_text("{}")  # existence is all __post_init__ checks

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def valid_parameters(self, **overrides) -> ShellModalAnalysisParameters:
        return ShellModalAnalysisParameters(
            spline_path=self.spline_path,
            output_folder=self.folder / "out",
            **overrides,
        )

    def _real_spline_path(self) -> Path:
        """
        Writes a real six-patch cubed sphere and returns its path, for the two
        tests below that run the entry point end to end. Kept out of `setUp`
        so the seven parameter-validation tests above, which never load a
        spline, do not pay for building one.
        """
        path = self.folder / "sphere.json"
        path.write_text(
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
        return path

    def test_valid_parameters_are_accepted(self) -> None:
        self.valid_parameters()

    def test_invalid_parameters_are_rejected(self) -> None:
        for label, overrides in self.INVALID:
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    self.valid_parameters(**overrides)

    def test_missing_spline_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ShellModalAnalysisParameters(spline_path=self.folder / "absent.json")

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

    def test_the_flags_map_onto_the_locking_cure(self) -> None:
        """
        The two CLI booleans are the archive's spelling; the solver takes an
        enum. This is the whole of the translation, and getting it wrong would
        silently run a different discretisation than the flags asked for;
        `both set` is rejected in `__post_init__`, so it cannot arrive here.
        """
        for use_bbar, use_psri, expected in (
            (True, False, ShellElement.LockingCure.BBAR),
            (False, True, ShellElement.LockingCure.PSRI),
            (False, False, ShellElement.LockingCure.FULL),
        ):
            with self.subTest(use_bbar=use_bbar, use_psri=use_psri):
                parameters = self.valid_parameters(
                    use_bbar=use_bbar, use_psri=use_psri
                )
                self.assertIs(parameters.locking_cure, expected)

    def test_end_to_end_run_writes_results_and_modes(self) -> None:
        """
        Everything a production run produces: a spectrum with six rigid-body
        modes, an `.npz` carrying the four result arrays, and an eigenmode pair
        on disk. The scipy path is used so `mode_energies` is populated; that
        is the one output the SLEPc path deliberately leaves zero.
        """
        parameters = ShellModalAnalysisParameters(
            spline_path=self._real_spline_path(),
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
        np.testing.assert_allclose(saved["frequencies"], result.frequencies, rtol=0.0)
        self.assertGreater(float(np.abs(result.eigenvectors).max()), 0.0)
        np.testing.assert_allclose(saved["eigenvectors"], result.eigenvectors, rtol=0.0)
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
        """Without this wiring the parameter reaches only the SLEPc branch, so a
        `--use-scipy-solver --spectral-shift X` run would silently ignore X."""
        parameters = ShellModalAnalysisParameters(
            spline_path=self._real_spline_path(),
            output_folder=self.folder / "shift",
            num_modes=8,
            subdivisions_per_patch=2,
            use_scipy_solver=True,
            spectral_shift=1234.5,
            export_xdmf=False,
        )
        with mock.patch.object(
            GeometricallyExactShellModalSolver,
            "solve_modes",
            autospec=True,
            side_effect=GeometricallyExactShellModalSolver.solve_modes,
        ) as spied:
            run_modal_analysis(params=parameters)
        self.assertEqual(spied.call_args.kwargs["spectral_shift"], 1234.5)
