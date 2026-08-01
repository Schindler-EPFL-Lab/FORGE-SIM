"""
Characterization (golden-value) test for `CrankNicolsonFEMSolver`.

Where `test_crank_nicolson_fem_solver.py` pins *properties* that any correct heat
solver has, this module pins the *numbers this particular solver produces*, so
that a refactor can be shown not to have moved them. It runs one small, fully
deterministic two-material scenario -- non-uniform initial field, source on the
active material, a relaxation-time-normalized step -- and compares a handful of
scalar diagnostics against the committed fixture `crank_nicolson_golden.json`.

Regenerating the fixture is deliberate, never automatic. After an *intended*
change to the scheme, from the repository root::

    conda run -n dolfinx-rebel-shape python \\
        dolfinx/tests/test_crank_nicolson_characterization.py \\
        --regenerate-golden-fixture

and commit the resulting JSON together with the change that justifies it.
Running the module without that flag simply runs the test.

Caveat: the scenario is pinned to the mesh gmsh produces for these parameters.
`num_dofs` / `num_cells` are asserted first and exactly, so a gmsh version bump
that remeshes the sphere fails with an obvious message instead of an opaque
numerical mismatch -- in that case the fixture has to be regenerated, and the
regenerated values are only meaningful for the new mesh.
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

from dolfinx import fem
from mpi4py import MPI

from _fem_fixtures import (
    CONCRETE,
    STEEL,
    function_from_dof_values,
    make_sphere_mesh,
    state_array,
    two_material_funcs,
)
from crank_nicolson_fem_solver import CrankNicolsonFEMSolver

GOLDEN_FIXTURE_PATH = Path(__file__).resolve().parent / "crank_nicolson_golden.json"

REGENERATION_FLAG = "--regenerate-golden-fixture"

# Scenario constants. Everything about the run is fixed here so the recorded
# numbers are reproducible from the fixture file alone.
MESH_SIZE = 0.35
SPHERE_RADIUS = 1.0
DT_OVER_T_RELAX = 0.02
NUM_STEPS = 12
SNAPSHOT_INTERVAL = 3
ACTIVE_SOURCE = 40.0


def run_reference_scenario() -> dict:
    """
    Runs the reference two-material heat scenario and returns its scalar
    diagnostics.

    The setup mirrors the two-material pipeline in miniature: the upper
    hemisphere is the "active" material (steel-like, thin, highly conductive)
    and carries the whole heat source, the lower hemisphere is the passive
    concrete-like base, and the initial field is non-uniform so that both
    conduction and generation contribute to every recorded number. The time step
    comes from the relaxation eigensolve, so `relaxation_time` is characterized
    as well.

    :returns: JSON-serializable dictionary of diagnostics.
    """
    msh = make_sphere_mesh(
        radius=SPHERE_RADIUS, mesh_size=MESH_SIZE, comm=MPI.COMM_SELF
    )
    V = fem.functionspace(msh, ("Lagrange", 1))
    dof_coordinates = V.tabulate_dof_coordinates()
    x, y, z = dof_coordinates.T

    u0 = function_from_dof_values(V, 300.0 + 20.0 * z + 5.0 * x * y)
    material_funcs = two_material_funcs(
        V,
        active_mask=z > 0.0,
        active_material=STEEL,
        passive_material=CONCRETE,
        active_source=ACTIVE_SOURCE,
    )

    solver = CrankNicolsonFEMSolver.from_normalized_time_step(
        msh=msh,
        V=V,
        u0=u0,
        material_funcs=material_funcs,
        dt_over_t_relax=DT_OVER_T_RELAX,
        comm=MPI.COMM_SELF,
    )

    # Snapshots are taken exactly the way `run_fem_heat_simulation` takes them:
    # standalone state copies during the loop, diagnostics afterwards.
    snapshots = [solver.snapshot_state()]
    for step in range(1, NUM_STEPS + 1):
        solver.step()
        if step % SNAPSHOT_INTERVAL == 0:
            snapshots.append(solver.snapshot_state())

    total_heat = []
    for snapshot in snapshots:
        solver.load_state(snapshot)
        total_heat.append(solver.compute_total_heat())
        snapshot.destroy()

    final_temperature = state_array(solver)
    return {
        "num_dofs": int(V.dofmap.index_map.size_global),
        "num_cells": int(msh.topology.index_map(2).size_global),
        "relaxation_time": float(solver.relaxation_time),
        "dt": float(solver.dt),
        "heat_generation_rate": float(solver.compute_heat_generation_rate()),
        "total_heat": [float(value) for value in total_heat],
        "final_temperature_min": float(final_temperature.min()),
        "final_temperature_max": float(final_temperature.max()),
        "final_temperature_mean": float(final_temperature.mean()),
        "final_temperature_l2_norm": float(np.linalg.norm(final_temperature)),
    }


class TestCrankNicolsonCharacterization(unittest.TestCase):
    """
    Compares the reference scenario against the committed golden values.
    """

    # Empirically, this solver reproduces itself to ~7e-16 relative across
    # independent process invocations (the residual noise is MUMPS' threaded
    # factorization; at OMP_NUM_THREADS=1 it is bitwise exact), and the SLEPc
    # relaxation eigensolve reproduces to ~2.5e-16. 1e-12 therefore leaves
    # roughly four orders of margin over the observed noise while still catching
    # any refactor that actually perturbs the arithmetic.
    RELATIVE_TOLERANCE = 1e-12

    @classmethod
    def setUpClass(cls) -> None:
        cls.golden = json.loads(GOLDEN_FIXTURE_PATH.read_text())
        cls.result = run_reference_scenario()

    def test_mesh_matches_the_recorded_discretization(self):
        """
        Guards the rest of the comparison: the golden values only mean anything
        on the mesh they were recorded on. A different gmsh version (or a change
        to `make_sphere_mesh`) shows up here rather than as an unexplained
        numerical drift downstream.
        """
        self.assertEqual(self.result["num_dofs"], self.golden["num_dofs"])
        self.assertEqual(self.result["num_cells"], self.golden["num_cells"])

    def test_time_step_characterization(self):
        """
        Pins the relaxation eigensolve and the step it resolves to. These fix the
        physical time axis of every FEM run, and are what makes an FEM run
        comparable with the IGA run configured with the same `dt_over_t_relax`.
        """
        for key in ("relaxation_time", "dt"):
            with self.subTest(quantity=key):
                self.assertAlmostEqual(
                    self.result[key] / self.golden[key],
                    1.0,
                    delta=self.RELATIVE_TOLERANCE,
                )

    def test_energy_characterization(self):
        """
        Pins the assembled source power and the total-heat time series -- the
        integrated quantities the runner reports and `compare_iga_fem` diffs. A
        change in any material coefficient, in the quadrature, or in the
        Crank-Nicolson weighting moves at least one of these.
        """
        self.assertAlmostEqual(
            self.result["heat_generation_rate"] / self.golden["heat_generation_rate"],
            1.0,
            delta=self.RELATIVE_TOLERANCE,
        )
        self.assertEqual(len(self.result["total_heat"]), len(self.golden["total_heat"]))
        np.testing.assert_allclose(
            self.result["total_heat"],
            self.golden["total_heat"],
            rtol=self.RELATIVE_TOLERANCE,
        )

    def test_final_field_characterization(self):
        """
        Pins the spatial solution itself, not just its integrals: the extremes,
        the mean and the Euclidean norm of the final temperature field. The
        integrals above are blind to any change that redistributes heat without
        creating or destroying it (a wrong conductivity contrast, say); these
        summaries are not.
        """
        for key in (
            "final_temperature_min",
            "final_temperature_max",
            "final_temperature_mean",
            "final_temperature_l2_norm",
        ):
            with self.subTest(quantity=key):
                self.assertAlmostEqual(
                    self.result[key] / self.golden[key],
                    1.0,
                    delta=self.RELATIVE_TOLERANCE,
                )


def _regenerate_golden_fixture() -> None:
    """
    Recomputes the reference scenario and overwrites the committed fixture.
    Only reachable through the explicit `--regenerate-golden-fixture` flag.
    """
    from rebel_shape import logger

    result = run_reference_scenario()
    GOLDEN_FIXTURE_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    logger.info(f"Regenerated golden fixture at {GOLDEN_FIXTURE_PATH}.")


if __name__ == "__main__":
    if REGENERATION_FLAG in sys.argv:
        _regenerate_golden_fixture()
    else:
        unittest.main()
