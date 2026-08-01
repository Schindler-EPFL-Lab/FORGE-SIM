"""
Tests for the two FEM heat entry scripts.

They are ~90 % the same file, so they are tested together: the validation rules
are asserted against both parameter classes from one table, and the wiring test
covers the one place they genuinely differ -- which material's (rho, c, h) the
`target_delta_temp` source scaling is derived from.

The parity test is the important one here. The README's central claim is that the
FEM parameters deliberately mirror the IGA ones, so that a given configuration
resolves to a near-identical physical time step under either discretisation. That
claim is enforced by nothing but eyesight today, and a silent divergence in a
single default -- `dt_over_t_relax`, `save_interval`, a material property -- would
not break either run, it would just make every published comparison compare two
different problems.

No solving happens here: the parameter classes only touch the filesystem, and the
wiring test replaces the projector and the runner with mocks.

Run from the repository root::

    conda run -n dolfinx-rebel-shape python -m unittest discover dolfinx/tests
"""

import shutil
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from unittest import mock

# Imported for its module-level side effect: `_fem_fixtures` puts `dolfinx/` and
# the repository root on `sys.path`.
import _fem_fixtures  # noqa: F401

from rebel_shape.scripts.heat_simulation.heat_simulation import HeatSimulationParameters
from rebel_shape.scripts.heat_simulation.heat_simulation_two_materials import (
    HeatSimulationParametersTwoMaterials,
)

import heat_simulation_spline
import heat_simulation_spline_two_materials
from heat_simulation_spline import HeatSimulationSplineParameters
from heat_simulation_spline_two_materials import (
    HeatSimulationSplineTwoMaterialsParameters,
)

# Each entry: a label, and the overrides that must be rejected. They are applied
# on top of an otherwise valid parameter set, so exactly one rule is under test
# at a time.
_INVALID_OVERRIDES = (
    ("final_time not positive", {"final_time": 0.0}),
    ("both step specifications", {"time_step_size": 1.0, "dt_over_t_relax": 1e-5}),
    ("neither step specification", {"time_step_size": None, "dt_over_t_relax": None}),
    (
        "time step not positive",
        {"time_step_size": -1.0, "dt_over_t_relax": None},
    ),
    ("normalized step not positive", {"dt_over_t_relax": 0.0}),
    ("no time steps", {"num_time_steps": 0}),
    ("no save interval", {"save_interval": 0}),
    ("no tessellation", {"tessellation_subdivisions": 0}),
    ("degree below one", {"fem_degree": 0}),
    ("length scale not positive", {"target_length_scale": 0.0}),
    ("target rise not positive", {"target_delta_temp": -1.0}),
)


class TestEntryPointParameters(unittest.TestCase):
    """
    Validation, IGA parity, and source-material routing of the two entry scripts.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # The parameter classes only check that these paths exist, so empty
        # files are enough and no spline has to be built.
        cls._directory = Path(tempfile.mkdtemp())
        cls.addClassCleanup(shutil.rmtree, cls._directory, ignore_errors=True)
        cls.spline_path = cls._directory / "spline.json"
        cls.spline_path.touch()
        cls.mask_path = cls._directory / "mask.json"
        cls.mask_path.touch()

    def _single_material(self, **overrides) -> HeatSimulationSplineParameters:
        arguments = {"spline_path": self.spline_path}
        arguments.update(overrides)
        return HeatSimulationSplineParameters(**arguments)

    def _two_materials(self, **overrides) -> HeatSimulationSplineTwoMaterialsParameters:
        arguments = {
            "spline_path": self.spline_path,
            "facade_mask_path": self.mask_path,
        }
        arguments.update(overrides)
        return HeatSimulationSplineTwoMaterialsParameters(**arguments)

    def test_parameter_validation(self):
        """
        Every `__post_init__` rule of both parameter classes rejects its bad
        input, and the defaults are themselves valid.

        These are the only guards between a mistyped CLI flag and a run that
        either crashes hours later or, worse, silently simulates something else
        (a zero `save_interval` and a negative `target_length_scale` both fail
        deep inside, far from the cause).
        """
        for build in (self._single_material, self._two_materials):
            with self.subTest(script=build.__name__, case="defaults are valid"):
                build()
            for label, overrides in _INVALID_OVERRIDES:
                with self.subTest(script=build.__name__, case=label):
                    with self.assertRaises(ValueError):
                        build(**overrides)

        with self.subTest(case="missing spline"):
            with self.assertRaises(ValueError):
                self._single_material(spline_path=self._directory / "absent.json")
        with self.subTest(case="missing facade mask"):
            with self.assertRaises(ValueError):
                self._two_materials(facade_mask_path=self._directory / "absent.json")

    def test_defaults_match_the_iga_counterparts(self):
        """
        Every knob the FEM scripts share with their IGA counterparts defaults to
        the same value, materials included.

        `dt_over_t_relax` is the load-bearing one: both sides derive `T_relax`
        from the smallest non-zero generalized eigenvalue of their own assembled
        (K, M), so the same normalized step resolves to a near-identical physical
        dt only if the defaults agree. The materials matter for the same reason
        -- they enter both the eigenvalue and the source scaling.

        Comparing dataclass field-by-field rather than object identity, since the
        two sides construct their own `ThermalMaterialConfig` instances.
        """
        pairs = (
            (
                HeatSimulationSplineParameters(spline_path=self.spline_path),
                HeatSimulationParameters(spline_path=self.spline_path),
                ("concrete_config",),
            ),
            (
                self._two_materials(),
                HeatSimulationParametersTwoMaterials(
                    spline_path=self.spline_path, facade_mask_path=self.mask_path
                ),
                ("concrete_config", "glass_config"),
            ),
        )
        # Knobs common to at least one of the two pairs; the ones a given pair
        # does not share (`facade_*` on the single-material scripts) are skipped.
        shared_scalars = (
            "num_time_steps",
            "save_interval",
            "dt_over_t_relax",
            "target_length_scale",
            "target_delta_temp",
            "heat_source",
            "facade_heat_source",
            "facade_mask_threshold",
        )

        for fem_params, iga_params, material_names in pairs:
            for name in shared_scalars:
                if not (hasattr(fem_params, name) and hasattr(iga_params, name)):
                    continue
                with self.subTest(parameters=type(fem_params).__name__, knob=name):
                    self.assertEqual(
                        getattr(fem_params, name), getattr(iga_params, name)
                    )
            for material_name in material_names:
                fem_material = getattr(fem_params, material_name)
                iga_material = getattr(iga_params, material_name)
                for material_field in fields(fem_material):
                    with self.subTest(
                        parameters=type(fem_params).__name__,
                        material=material_name,
                        property=material_field.name,
                    ):
                        self.assertEqual(
                            getattr(fem_material, material_field.name),
                            getattr(iga_material, material_field.name),
                        )

    def test_target_delta_temp_routes_the_heated_material(self):
        """
        The source scaling uses the material that is actually being heated:
        concrete in the single-material run, glass (the windows) in the
        two-material run. In both cases the projected source field is unit-valued
        and the scaling supplies the magnitude.

        Routing concrete into the two-material run is a plausible copy-paste
        error that would mis-scale every window source by the ratio of the two
        thermal masses (here 42x) -- and nothing downstream would flag it,
        because the run would still be perfectly well-posed.

        The projector and the runner are mocked out: this test is about wiring,
        and building a spline or solving would say nothing more about it.
        """
        parameters = self._single_material(target_delta_temp=2.0)
        with (
            mock.patch.object(heat_simulation_spline, "SplineToFEMProjector") as proj,
            mock.patch.object(
                heat_simulation_spline, "run_fem_heat_simulation"
            ) as runner,
        ):
            heat_simulation_spline.run(parameters)

        project_call = proj.return_value.project_fields.call_args
        self.assertEqual(project_call.kwargs["source"], 1.0)
        config = runner.call_args.kwargs["config"]
        self.assertIs(config.heat_source_material, parameters.concrete_config)
        self.assertEqual(config.target_delta_temp, 2.0)

        parameters = self._two_materials(target_delta_temp=2.0)
        with (
            mock.patch.object(
                heat_simulation_spline_two_materials, "SplineToFEMProjector"
            ) as proj,
            mock.patch.object(
                heat_simulation_spline_two_materials, "run_fem_heat_simulation"
            ) as runner,
        ):
            heat_simulation_spline_two_materials.run(parameters)

        project_call = proj.return_value.project_two_material_fields.call_args
        self.assertEqual(project_call.kwargs["glass_source"], 1.0)
        config = runner.call_args.kwargs["config"]
        self.assertIs(config.heat_source_material, parameters.glass_config)

        # Without a target rise the configured source is passed through verbatim
        # and no material is routed, so the scaling stays off.
        parameters = self._two_materials(target_delta_temp=None, facade_heat_source=0.1)
        with (
            mock.patch.object(
                heat_simulation_spline_two_materials, "SplineToFEMProjector"
            ) as proj,
            mock.patch.object(
                heat_simulation_spline_two_materials, "run_fem_heat_simulation"
            ) as runner,
        ):
            heat_simulation_spline_two_materials.run(parameters)

        project_call = proj.return_value.project_two_material_fields.call_args
        self.assertEqual(project_call.kwargs["glass_source"], 0.1)
        self.assertIsNone(runner.call_args.kwargs["config"].heat_source_material)


if __name__ == "__main__":
    unittest.main()
