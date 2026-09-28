import tempfile
import unittest
from pathlib import Path

from _fem_fixtures import ACTIVE, PASSIVE, write_fixture_npz

from fem.end_to_end.heat_gt.fem_heat_gt_mesh import FEMHeatGTMeshParameters
from forge_sim.scripts.configs.thermal_material_config import ThermalMaterialConfig
from forge_sim.scripts.end_to_end.heat.thermal_sim_config import ThermalSimConfig


class TestParameterValidation(unittest.TestCase):
    """
    `__post_init__` is all that stands between a mistyped command line and a run that
    either fails minutes later or silently solves a different problem.

    `run_fem_heat` has no behaviour beyond constructing a `FEMHeatGTRunner`
    from this parameter set and calling `.run()`; what that runner does
    with the result (where the source lands, what `FEMSimulationConfig`
    it assembles, and the diagnostics sidecar it leaves behind) is
    pinned in `test_fem_heat_gt_runner.py` against `FEMHeatGTRunner`
    directly, `run_fem_heat` included among its call sites.
    """

    INVALID = (
        ("no time steps", {"num_time_steps": 0}),
        ("normalized step not positive", {"dt_over_t_relax": 0.0}),
        ("no save interval", {"save_interval": 0}),
        ("degree below one", {"fem_degree": 0}),
        ("negative heat source", {"heat_source": -1.0}),
        ("target rise not positive", {"target_delta_temp": 0.0}),
        ("region as its serialised string", {"heat_source_region": "wood"}),
    )

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self._temporary.name)
        self.npz_path = self.folder / "heat_gt.npz"
        write_fixture_npz(self.npz_path)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def valid_parameters(self, **overrides) -> FEMHeatGTMeshParameters:
        arguments = {"npz_path": self.npz_path, "output_folder": self.folder}
        arguments.update(overrides)
        return FEMHeatGTMeshParameters(**arguments)

    def test_valid_parameters_are_accepted(self) -> None:
        self.valid_parameters()

    def test_invalid_parameters_are_rejected(self) -> None:
        for label, overrides in self.INVALID:
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    self.valid_parameters(**overrides)

    def test_missing_npz_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.valid_parameters(npz_path=self.folder / "absent.npz")

    def test_an_invalid_material_is_rejected_by_the_shared_validator(self) -> None:
        """
        The material rules are `ThermalMaterialConfig`'s, which is where the IGA side
        gets them too, so an invalid material cannot even be built on the way in:
        the rejection happens at the sub-config, before the driver sees it.
        """
        for label, overrides in (
            ("conductivity zero", {"conductivity": 0.0}),
            ("density negative", {"density": -1.0}),
            ("specific heat zero", {"specific_heat": 0.0}),
            ("thickness negative", {"thickness": -0.01}),
        ):
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    self.valid_parameters(
                        steel_config=ThermalMaterialConfig(
                            **{
                                "conductivity": ACTIVE.conductivity,
                                "density": ACTIVE.density,
                                "specific_heat": ACTIVE.specific_heat,
                                "thickness": ACTIVE.thickness,
                                **overrides,
                            }
                        )
                    )

    def test_the_source_material_is_the_one_the_region_names(self) -> None:
        """
        `source_material` is what the target-rise scaling divides by (Q = dT rho c h /
        t_sim), so naming the wrong slot mis-scales every source by the ratio of the
        two thermal masses (here 39x) while leaving the run perfectly well-posed.
        """
        parameters = self.valid_parameters(
            steel_config=ACTIVE,
            wood_config=PASSIVE,
            heat_source_region=ThermalSimConfig.HeatSourceRegion.STEEL,
        )
        self.assertIs(parameters.source_material, parameters.steel_config)

        parameters = self.valid_parameters(
            steel_config=ACTIVE,
            wood_config=PASSIVE,
            heat_source_region=ThermalSimConfig.HeatSourceRegion.WOOD,
        )
        self.assertIs(parameters.source_material, parameters.wood_config)

        parameters = self.valid_parameters(
            steel_config=ACTIVE,
            wood_config=PASSIVE,
            heat_source_region=ThermalSimConfig.HeatSourceRegion.NONE,
        )
        self.assertIsNone(parameters.source_material)
