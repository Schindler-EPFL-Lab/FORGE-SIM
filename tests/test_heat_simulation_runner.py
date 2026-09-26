import tempfile
import unittest
from pathlib import Path

import mitsuba as mi

from rebel_shape.isogeometric_analysis.heat_simulation import HeatSimulation
from rebel_shape.meshing.composite_iga_spline_surface_factory import (
    CompositeIGASplineSurfaceFactory,
)
from rebel_shape.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)
from rebel_shape.scripts.configs.thermal_material_config import ThermalMaterialConfig
from rebel_shape.scripts.heat_simulation.heat_simulation_runner import (
    HeatSimulationRunner,
)
from rebel_shape.splines.bspline_patch import BSplinePatch
from rebel_shape.splines.iga_spline_patch import IGASplinePatch


class TestHeatSimulationRunner(unittest.TestCase):
    """
    `HeatSimulationRunner` has no XDMF-driving test of its own elsewhere in the
    suite, so these cover the two things the rendering removal touches: that
    the solve itself is unaffected, and that the exposed `heat_simulation` and
    `cn_scheme` properties hand back the live, post-scaling objects rather
    than a stale reconstruction.
    """

    @classmethod
    def setUpClass(cls) -> None:
        mi.set_variant("llvm_ad_rgb")

    def setUp(self) -> None:
        # A single degree-1 unit-square patch (area 1, in the XY plane) is
        # the cheapest surface that still exercises a real Crank-Nicolson
        # solve; a coarse 3x3 grid gives the source term room to vary.
        patch = BSplinePatch.unit_square(
            patch_name="patch", degree=1, u_subdivisions=2, v_subdivisions=2
        )
        # A non-uniform, non-zero-mean initial temperature (the control
        # points' own x coordinate, offset off the unit square's symmetric
        # zero mean) so the solve has something to actually diffuse and the
        # total heat is not trivially zero.
        initial_temperature = mi.TensorXf(
            patch.control_points.numpy()[:, 0:1] + 1.0
        )
        iga_patch = IGASplinePatch(spline_patch=patch, field_values=initial_temperature)

        factory = CompositeIGASplineSurfaceFactory("test_surface")
        factory.add_patch(iga_patch)
        composite_iga_spline = factory.build()

        self.tessellated_iga_spline = TessellatedIGASplineSurface(
            iga_spline_surface=composite_iga_spline,
            tessellation_subdivisions_per_patch=1,
        )
        self.material_config = ThermalMaterialConfig(
            conductivity=1.0, density=1.0, specific_heat=1.0, thickness=1.0
        )

    def _new_runner(self) -> HeatSimulationRunner:
        return HeatSimulationRunner(
            tessellated_iga_spline=self.tessellated_iga_spline,
            material_config=self.material_config,
        )

    def test_solve_conserves_energy_without_a_source(self) -> None:
        """
        With no source term, the Crank-Nicolson scheme conserves total heat by
        construction (see `crank_nicolson_scheme.py`'s own docstring); this is
        an analytic property of the scheme, not a recorded number, so it
        stays valid regardless of what the runner's rendering removal did or
        did not touch.
        """
        runner = self._new_runner()
        initial_total_heat = float(
            HeatSimulation(
                composite_iga_spline=self.tessellated_iga_spline.composite_spline,
                conductivity=runner.conductivity,
                density=runner.density,
                specific_heat=runner.specific_heat,
                thickness=runner.thickness,
                source_term=runner.source,
            ).total_heat.numpy().reshape(-1)[0]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            runner.run_heat_simulation_with_xdmf(
                num_time_steps=5,
                xdmf_path=Path(tmp_dir) / "test.xdmf",
                tessellation_subdivisions=2,
                time_step_size=0.01,
                xdmf_save_interval=5,
            )

        assert runner.heat_simulation is not None
        final_total_heat = float(
            runner.heat_simulation.total_heat.numpy().reshape(-1)[0]
        )

        self.assertAlmostEqual(final_total_heat / initial_total_heat, 1.0, places=6)

    def test_heat_simulation_property_reflects_target_delta_temp_scaling(
        self,
    ) -> None:
        """
        `CrankNicolsonScheme.scale_source` rebinds `HeatSimulation.source_term`
        to a new tensor rather than mutating it in place; the `heat_simulation`
        property must still resolve to that same, now-scaled instance.
        """
        # A uniform, non-scaled reference at the pre-scaling magnitude (1.0),
        # to give the expected scaled rate an independent baseline: the
        # source integral is linear in its magnitude, so the run's rate must
        # equal this baseline times the target-delta-temp scaling factor Q.
        ones = mi.TensorXf(
            [[1.0]] * self.tessellated_iga_spline.global_field_values.shape[0]
        )
        baseline_rate = float(
            HeatSimulation(
                composite_iga_spline=self.tessellated_iga_spline.composite_spline,
                conductivity=ones,
                density=ones,
                specific_heat=ones,
                thickness=ones,
                source_term=ones,
            ).heat_generation_rate.numpy().reshape(-1)[0]
        )

        target_delta_temp = 2.0
        heat_source_material = ThermalMaterialConfig(
            conductivity=1.0, density=3.0, specific_heat=5.0, thickness=0.1
        )
        num_time_steps = 4
        time_step_size = 0.01
        t_sim = num_time_steps * time_step_size
        expected_Q = (
            target_delta_temp
            * heat_source_material.density
            * heat_source_material.specific_heat
            * heat_source_material.thickness
            / t_sim
        )

        runner = self._new_runner()
        runner.source = ones
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner.run_heat_simulation_with_xdmf(
                num_time_steps=num_time_steps,
                xdmf_path=Path(tmp_dir) / "test.xdmf",
                tessellation_subdivisions=2,
                time_step_size=time_step_size,
                xdmf_save_interval=num_time_steps,
                target_delta_temp=target_delta_temp,
                heat_source_material=heat_source_material,
            )

        assert runner.heat_simulation is not None
        reported_rate = float(
            runner.heat_simulation.heat_generation_rate.numpy().reshape(-1)[0]
        )

        self.assertAlmostEqual(
            reported_rate / (baseline_rate * expected_Q), 1.0, places=6
        )

    def test_run_heat_simulation_with_xdmf_exposes_resolved_cn_scheme(self) -> None:
        runner = self._new_runner()
        self.assertIsNone(runner.cn_scheme)

        with tempfile.TemporaryDirectory() as tmp_dir:
            runner.run_heat_simulation_with_xdmf(
                num_time_steps=2,
                xdmf_path=Path(tmp_dir) / "test.xdmf",
                tessellation_subdivisions=2,
                time_step_size=0.01,
            )

        assert runner.cn_scheme is not None
        self.assertAlmostEqual(runner.cn_scheme.time_step_size, 0.01, places=12)

    def test_on_snapshot_fires_at_every_saved_step(self) -> None:
        snapshots: list[tuple[int, float]] = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = self._new_runner()
            runner.run_heat_simulation_with_xdmf(
                num_time_steps=6,
                xdmf_path=Path(tmp_dir) / "test.xdmf",
                tessellation_subdivisions=2,
                time_step_size=0.01,
                xdmf_save_interval=3,
                on_snapshot=lambda step, time: snapshots.append((step, time)),
            )

        self.assertEqual([s for s, _ in snapshots], [0, 3, 6])

    def test_runner_no_longer_offers_rendering(self) -> None:
        """
        Rendering (scene construction, `mi.render`, PNG contact sheets) was
        removed rather than made optional; nothing in the runner's surface
        should still name it.
        """
        runner = self._new_runner()
        for removed_attribute in (
            "render_temperature",
            "scene",
            "scene_params",
            "sensors",
        ):
            self.assertFalse(hasattr(runner, removed_attribute))
