import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
from _fem_fixtures import TETRAHEDRON_FACES, TETRAHEDRON_VERTICES

from fem.end_to_end.modal_gt.fem_shell_modal_gt_mesh import (
    FEMShellModalGTMeshParameters,
)
from fem.shells.shell_element import ShellElement
from forge_sim.scripts.configs.structural_material_config import (
    StructuralMaterialConfig,
)


class TestParameterValidation(unittest.TestCase):
    """
    `__post_init__` is all that stands between a mistyped command line and a run
    that either fails minutes later or silently solves a different problem.

    `run_modal_analysis` has no behaviour beyond translating this parameter
    set into a `ShellElement` and a `FEMShellModalGTRunner`; what that
    runner does with the result (the raw-array handover at both geometry
    orders and the archive a finished run leaves behind) is pinned in
    `test_fem_shell_modal_gt_runner.py` against `FEMShellModalGTRunner`
    directly. The solve's accuracy is not re-validated in either file: it
    is the same `GeometricallyExactShellProblem` /
    `GeometricallyExactShellModalSolver` pair that
    `test_geometrically_exact_shell_modal_solver.py` pins against the steel
    sphere.
    """

    INVALID = (
        ("num_modes zero", {"num_modes": 0}),
        ("field degree 1", {"field_degree": 1}),
        ("field degree 4", {"field_degree": 4}),
        ("both locking cures", {"use_bbar": True, "use_psri": True}),
    )

    INVALID_MATERIALS = (
        ("youngs modulus zero", {"youngs_modulus": 0.0}),
        ("poisson ratio at 0.5", {"poisson_ratio": 0.5}),
        ("poisson ratio at -1", {"poisson_ratio": -1.0}),
        ("density negative", {"density": -1.0}),
        ("thickness zero", {"thickness": 0.0}),
        ("shear correction zero", {"shear_correction_factor": 0.0}),
    )
    """
    Materials that must not reach the solver, as overrides on the driver's own default.

    They are rejected by `StructuralMaterialConfig` rather than by the driver, which is
    the point of nesting it: the IGA side is held to the same rules by the same class,
    and the rejection now happens where the material is built rather than a field at a
    time on the way past.
    """

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.mesh_npz = Path(self._temporary.name) / "modal_gt.npz"
        np.savez(
            self.mesh_npz,
            vertices=TETRAHEDRON_VERTICES,
            faces=TETRAHEDRON_FACES,
            scale=1.0,
        )

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def valid_parameters(self, **overrides) -> FEMShellModalGTMeshParameters:
        return FEMShellModalGTMeshParameters(
            mesh_npz=self.mesh_npz,
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

    def test_invalid_materials_are_rejected(self) -> None:
        """
        Both statements sit inside the assertion because the rejection has moved: the
        material is validated as it is built, so an invalid one never becomes a
        parameter set at all.
        """
        default_material = asdict(self.valid_parameters().material_config)
        for label, overrides in self.INVALID_MATERIALS:
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    self.valid_parameters(
                        material_config=StructuralMaterialConfig(
                            **{**default_material, **overrides}
                        )
                    )

    def test_the_material_defaults_are_the_car_case_not_the_shared_concrete(
        self,
    ) -> None:
        """
        A standalone run must keep integrating what it integrated before the material
        became a sub-config. `StructuralMaterialConfig`'s own defaults are a 30 GPa,
        20 cm concrete; inheriting them here would leave the script runnable and its
        spectrum unrecognisable.
        """
        material = self.valid_parameters().material_config

        self.assertEqual(material.youngs_modulus, 210.0e9)
        self.assertEqual(material.poisson_ratio, 0.3)
        self.assertEqual(material.density, 7850.0)
        self.assertEqual(material.thickness, 0.0015)
        self.assertEqual(material.shear_correction_factor, 5.0 / 6.0)

    def test_missing_mesh_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FEMShellModalGTMeshParameters(
                mesh_npz=Path(self._temporary.name) / "absent.npz"
            )

    def test_the_flags_map_onto_the_locking_cure(self) -> None:
        """
        Same translation as the spline driver's, and it must stay the same: the
        two runs are compared against each other, so a driver that quietly used
        a different discretisation would make that comparison meaningless.
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

    def test_bbar_is_the_default(self) -> None:
        """The recorded end-to-end runs used B-bar; the default must not drift
        away from what the archive was produced with."""
        self.assertTrue(self.valid_parameters().use_bbar)
        self.assertFalse(self.valid_parameters().use_psri)
