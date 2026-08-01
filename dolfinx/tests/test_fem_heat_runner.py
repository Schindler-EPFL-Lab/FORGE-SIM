"""
Tests for the shared FEM heat-run orchestration.

`fem_heat_runner` is where everything meets: the projected spline mesh becomes a
DOLFINx mesh, vertex-based fields become DOFs, the configuration resolves to a
physical time step, and the result travels back from DOF order to vertex order to
be written. None of that has an invariant a physicist would notice being
violated -- a run with a scrambled field or a mis-resolved time step completes,
conserves energy, and writes a perfectly well-formed file -- so it is pinned here
against schedules, closed forms and known permutations instead.

The two loads-bearing ones:

* `test_vertex_interpolant_*` uses a *spatially varying* field on purpose. The
  mesh round trip genuinely permutes vertices (measured maximum coordinate
  difference 1.95 between the tessellation order and the DOF order), so the
  nearest-vertex matching is doing real work; a constant field would pass with a
  completely scrambled permutation.
* `test_exported_snapshots_are_in_vertex_order` closes the loop the other way,
  which is the only check that the file describes the simulation that ran.

Runs happen on the 386-vertex spline fixture from `_spline_fixtures`; a full
six-step simulation on it costs ~0.1 s.

Run from the repository root::

    conda run -n dolfinx-rebel-shape python -m unittest discover dolfinx/tests
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dolfinx import fem
from mpi4py import MPI

from rebel_shape.scripts.configs.thermal_material_config import ThermalMaterialConfig

from _spline_fixtures import affine_field, make_iga_spline_file
from compare_iga_fem import load_xdmf_data
from fem_heat_runner import (
    FEMSimulationConfig,
    SnapshotDiagnostics,
    _build_vertex_interpolant,
    create_dolfinx_mesh,
    run_fem_heat_simulation,
)
from spline_to_fem_projector import SplineToFEMProjector

_MATERIAL = ThermalMaterialConfig(
    conductivity=2.0, density=2250.0, specific_heat=900.0, thickness=0.2
)

# Initial temperature: 300 K plus a top-to-bottom gradient. The nonzero mean
# matters -- an odd field integrates to ~0 over a closed sphere, which makes the
# runner's `relative_drift` a ratio of two near-zero numbers and meaningless.
_INITIAL_TEMPERATURE = affine_field(c=20.0, d=300.0)


class TestFEMHeatRunner(unittest.TestCase):
    """
    Mesh construction, field placement, time-step resolution and the run loop.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.comm = MPI.COMM_SELF
        cls._directory = Path(tempfile.mkdtemp())
        cls.addClassCleanup(shutil.rmtree, cls._directory, ignore_errors=True)

        spline_path = make_iga_spline_file(
            cls._directory / "thermal.json", _INITIAL_TEMPERATURE
        )
        cls.projector = SplineToFEMProjector(
            thermal_spline_path=spline_path, tessellation_subdivisions=8
        )
        cls.mesh_data = cls.projector.get_mesh_data()
        cls.field_data = cls.projector.project_fields(
            conductivity=_MATERIAL.conductivity,
            density=_MATERIAL.density,
            specific_heat=_MATERIAL.specific_heat,
            thickness=_MATERIAL.thickness,
            source=0.0,
        )

    def _config(self, **overrides) -> FEMSimulationConfig:
        """A short, explicit-step run; `overrides` replace individual fields."""
        arguments = {
            "num_time_steps": 4,
            "save_interval": 2,
            "time_step_size": 1.0e4,
            "dt_over_t_relax": None,
            "write_xdmf": False,
        }
        arguments.update(overrides)
        return FEMSimulationConfig(**arguments)

    def _run(self, config: FEMSimulationConfig, name: str = "run", **kwargs):
        """Runs a simulation into a fresh subdirectory of the class temp dir."""
        return run_fem_heat_simulation(
            projector=self.projector,
            field_data=self.field_data,
            output_folder=self._directory / name,
            xdmf_name=f"{name}.xdmf",
            config=config,
            comm=self.comm,
            **kwargs,
        )

    # ------------------------------------------------------------------ mesh
    def test_dolfinx_mesh_carries_the_tessellation_geometry(self):
        """
        The DOLFINx mesh built from the tessellation has the same vertices and
        the same number of triangles -- as *sets*, since the construction is free
        to renumber.

        Stated on the sorted coordinate arrays rather than element-wise because
        the permutation is the implementation's business (the nearest-vertex
        matching downstream is what makes it irrelevant); the geometry is not.

        `atol=1e-9` leaves room for any coordinate rewriting inside mesh
        construction, while being twelve orders below the 1e-3 spacing of the
        fixture's vertices.
        """
        dolfinx_mesh = create_dolfinx_mesh(self.mesh_data, self.comm)

        vertices = dolfinx_mesh.geometry.x
        self.assertEqual(vertices.shape, self.mesh_data.vertices.shape)

        topology = dolfinx_mesh.topology
        topology.create_entities(2)
        self.assertEqual(topology.index_map(2).size_local, len(self.mesh_data.faces))

        order = np.lexsort(vertices.T)
        expected_order = np.lexsort(self.mesh_data.vertices.T)
        np.testing.assert_allclose(
            vertices[order], self.mesh_data.vertices[expected_order], atol=1e-9
        )

    def test_vertex_interpolant_places_values_at_the_right_coordinates(self):
        """
        A field given per tessellation vertex arrives at the DOF that sits at
        that vertex's position -- not merely somewhere in the array.

        Tested on the private helper deliberately: this property is invisible
        end-to-end, because the export path gathers back through the inverse of
        the same nearest-neighbour map, so an inverted or scrambled placement
        would cancel out and every round-trip test would still pass. The value
        `3x + 1` is affine in the coordinate, so the expected DOF value can be
        stated directly from `tabulate_dof_coordinates()`.

        P1 DOFs *are* the vertices, so the placement is a permutation and the
        values are copied, not interpolated. The only slack needed is that the
        DOF coordinate and the tessellation coordinate it came from can differ in
        the last bit or two of their float64 representation (measured: 8.9e-16
        absolute), which `3x + 1` amplifies by three. `rtol=1e-12` is three
        orders above that and twelve below the ~0.1 gap between distinct
        vertices, so any misplacement fails outright.
        """
        dolfinx_mesh = create_dolfinx_mesh(self.mesh_data, self.comm)
        V = fem.functionspace(dolfinx_mesh, ("Lagrange", 1))

        values = 3.0 * self.mesh_data.vertices[:, 0] + 1.0
        function = _build_vertex_interpolant(self.mesh_data.vertices)(V, values)

        expected = 3.0 * V.tabulate_dof_coordinates()[:, 0] + 1.0
        np.testing.assert_allclose(function.x.array, expected, rtol=1e-12)

    # -------------------------------------------------------- time step
    def test_time_step_precedence(self):
        """
        The three ways of specifying the step resolve as documented, and the
        reported relaxation time says which path was taken.

        `final_time` wins outright and returns `relaxation_time is None`, which
        is the observable proof that the eigensolve was skipped -- that matters
        because the eigensolve is the expensive part of setting up a run, and
        because `final_time` exists precisely to match a reference run's end time
        without it. An explicit `time_step_size` is used verbatim. The normalized
        path reports the `T_relax` it divided by, so `dt == dt_over_t_relax *
        T_relax` closes the loop.

        Equality is exact (`rtol=1e-15`) because all three are single float
        operations on the configured values.
        """
        diagnostics = self._run(
            self._config(final_time=1.0e5, num_time_steps=5, time_step_size=None),
            name="final_time",
        )
        self.assertAlmostEqual(diagnostics.dt, 2.0e4, delta=2.0e4 * 1e-15)
        self.assertIsNone(diagnostics.relaxation_time)

        diagnostics = self._run(self._config(time_step_size=7.5e3), name="explicit")
        self.assertEqual(diagnostics.dt, 7.5e3)
        self.assertIsNone(diagnostics.relaxation_time)

        dt_over_t_relax = 0.01
        diagnostics = self._run(
            self._config(time_step_size=None, dt_over_t_relax=dt_over_t_relax),
            name="normalized",
        )
        self.assertIsNotNone(diagnostics.relaxation_time)
        self.assertAlmostEqual(
            diagnostics.dt,
            dt_over_t_relax * diagnostics.relaxation_time,
            delta=abs(diagnostics.dt) * 1e-15,
        )

    def test_exactly_one_time_step_specification_is_required(self):
        """
        Without `final_time`, setting both `time_step_size` and
        `dt_over_t_relax` -- or neither -- is refused.

        Both are silent hazards otherwise: "both" would quietly ignore one of
        them, and "neither" has no defensible default, since the two runners are
        only comparable when their steps are resolved the same way.
        """
        for time_step_size, dt_over_t_relax in ((1.0e4, 0.01), (None, None)):
            with self.subTest(dt=time_step_size, normalized=dt_over_t_relax):
                with self.assertRaises(ValueError):
                    self._run(
                        self._config(
                            time_step_size=time_step_size,
                            dt_over_t_relax=dt_over_t_relax,
                        ),
                        name="invalid",
                    )

    # ------------------------------------------------------------- source
    def test_target_delta_temp_reaches_its_target(self):
        """
        With a uniform initial temperature and a uniform source, the whole
        surface rises by exactly `target_delta_temp` over the run, and the
        reported source is `dT rho c h / t_sim`.

        The oracle is analytic: a uniform field has zero conduction (the constant
        vector spans the kernel of K), so the problem reduces to the ODE
        `rho c h u' = Q`, which Crank-Nicolson integrates exactly. That makes
        this a closed-form end-to-end test of the scaling *and* of its
        interaction with the time-step resolution -- `t_sim = num_steps * dt`
        must use the same dt the solver stepped with.

        Two different tolerances, for two different reasons. The reported source
        is pure float64 arithmetic, so it is pinned at 1e-12 relative. The final
        temperature carries the projector's float32 floor: a "constant 300 K"
        spline field evaluates to 300 K within ~300 * 1.2e-7 = 3.6e-5, and
        diffusion only partly damps that ripple over four steps. `atol=1e-4`
        clears it threefold while still failing on a 1e-5 relative error in Q --
        four orders tighter than any plausible scaling mistake.
        """
        target_delta_temp = 5.0
        num_time_steps, time_step_size = 4, 1.0e4

        uniform_path = make_iga_spline_file(
            self._directory / "uniform.json", affine_field(c=0.0, d=300.0)
        )
        projector = SplineToFEMProjector(
            thermal_spline_path=uniform_path, tessellation_subdivisions=8
        )
        field_data = projector.project_fields(
            conductivity=_MATERIAL.conductivity,
            density=_MATERIAL.density,
            specific_heat=_MATERIAL.specific_heat,
            thickness=_MATERIAL.thickness,
            source=1.0,
        )

        config = FEMSimulationConfig(
            num_time_steps=num_time_steps,
            save_interval=num_time_steps,
            time_step_size=time_step_size,
            dt_over_t_relax=None,
            target_delta_temp=target_delta_temp,
            heat_source_material=_MATERIAL,
        )
        diagnostics = run_fem_heat_simulation(
            projector=projector,
            field_data=field_data,
            output_folder=self._directory / "target_delta",
            xdmf_name="target_delta.xdmf",
            config=config,
            comm=self.comm,
        )

        thermal_mass = _MATERIAL.density * _MATERIAL.specific_heat * _MATERIAL.thickness
        expected_source = (
            target_delta_temp * thermal_mass / (num_time_steps * time_step_size)
        )
        self.assertAlmostEqual(
            diagnostics.applied_heat_source,
            expected_source,
            delta=expected_source * 1e-12,
        )

        _, _, _, temperatures = load_xdmf_data(
            self._directory / "target_delta" / "target_delta.xdmf"
        )
        np.testing.assert_allclose(
            temperatures[-1], 300.0 + target_delta_temp, atol=1e-4
        )

    def test_target_delta_temp_requires_a_source_material(self):
        """
        `target_delta_temp` without `heat_source_material` is refused: the
        scaling `Q = dT rho c h / t_sim` has no meaning without (rho, c, h), and
        guessing a default would silently mis-scale every source.
        """
        with self.assertRaises(ValueError):
            self._run(
                self._config(target_delta_temp=1.0, heat_source_material=None),
                name="no_material",
            )

    # --------------------------------------------------------- run loop
    def test_snapshot_schedule_and_energy_bookkeeping(self):
        """
        Snapshots land at `t = 0`, at every `save_interval`-th step, and always
        at the final step even when it is not a multiple of the interval; and at
        each of them the assembled energy matches `H(0) + t Q` to round-off.

        The final-step clause is what makes a run's last snapshot comparable to a
        reference run's; with `num_steps=7, save_interval=3` its absence would
        end the series at `6 dt`. The drift is the runner's own reported
        diagnostic and the reason a source term is switched on here: with `Q = 0`
        the identity degenerates to the solver's conservation test, which the
        solver suite already owns.

        `1e-10` is the same gate the solver's conservation test uses, justified
        there; the measured drift is ~1e-15.
        """
        num_time_steps, save_interval, dt = 7, 3, 1.0e4
        field_data = self.projector.project_fields(
            conductivity=_MATERIAL.conductivity,
            density=_MATERIAL.density,
            specific_heat=_MATERIAL.specific_heat,
            thickness=_MATERIAL.thickness,
            source=25.0,
        )

        diagnostics = run_fem_heat_simulation(
            projector=self.projector,
            field_data=field_data,
            output_folder=self._directory / "schedule",
            xdmf_name="schedule.xdmf",
            config=FEMSimulationConfig(
                num_time_steps=num_time_steps,
                save_interval=save_interval,
                time_step_size=dt,
                dt_over_t_relax=None,
                write_xdmf=False,
            ),
            comm=self.comm,
        )

        self.assertEqual(
            [snapshot.time for snapshot in diagnostics.snapshots],
            [0.0, 3 * dt, 6 * dt, 7 * dt],
        )
        self.assertGreater(diagnostics.heat_generation_rate, 0.0)
        for snapshot in diagnostics.snapshots:
            with self.subTest(time=snapshot.time):
                self.assertLess(abs(snapshot.relative_drift), 1e-10)

    def test_exported_snapshots_are_in_vertex_order(self):
        """
        The first snapshot written to XDMF is the projected initial field, vertex
        for vertex.

        This is the only end-to-end statement that the exported file describes
        the simulation that ran. The export gathers PETSc DOF values back into
        tessellation-vertex order; inverting that gather scrambles every written
        file while the in-memory solution stays correct, the energy diagnostics
        stay correct, and nothing downstream complains.

        Exact equality holds: for P1 the two nearest-neighbour maps are mutually
        inverse permutations and every value is copied, never combined.
        """
        self._run(self._config(write_xdmf=True), name="export")

        vertices, faces, times, temperatures = load_xdmf_data(
            self._directory / "export" / "export.xdmf"
        )

        np.testing.assert_array_equal(vertices, self.mesh_data.vertices)
        np.testing.assert_array_equal(faces, self.mesh_data.faces)
        self.assertEqual(times[0], 0.0)
        np.testing.assert_array_equal(temperatures[0], self.field_data.temperature)

    def test_diagnostics_callback_sees_every_snapshot_in_order(self):
        """
        The optional callback receives exactly the snapshots the run reports, in
        time order.

        It is the hook the entry scripts use for MLflow-style logging, so a
        dropped or reordered snapshot would corrupt a recorded run's metrics
        while the simulation itself stayed correct.
        """
        received: list[SnapshotDiagnostics] = []
        diagnostics = self._run(
            self._config(num_time_steps=6, save_interval=2),
            name="callback",
            diagnostics_callback=received.append,
        )

        self.assertEqual(received, list(diagnostics.snapshots))
        self.assertEqual(
            [snapshot.time for snapshot in received],
            sorted(snapshot.time for snapshot in received),
        )


if __name__ == "__main__":
    unittest.main()
