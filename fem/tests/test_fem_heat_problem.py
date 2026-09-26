import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
from _fem_fixtures import (
    CONCRETE,
    function_from_dof_values,
    make_sphere_mesh,
    state_array,
    uniform_material_funcs,
)
from dolfinx import fem
from mpi4py import MPI

from fem.heat.crank_nicolson_fem_scheme import CrankNicolsonFEMScheme
from fem.heat.fem_heat_problem import FEMHeatProblem
from fem.heat.material_functions import MaterialFunctions

_MPIRUN = shutil.which("mpirun")
_SKIP_REASON = "No mpirun on PATH; the multi-rank load-vector check needs a launcher."


class TestFEMHeatProblem(unittest.TestCase):
    """
    Property tests on a coarse closed sphere with homogeneous material.

    Where `test_crank_nicolson_fem_scheme.py` pins the numerical properties
    of the Crank-Nicolson time integration, this module pins the
    invariants of the problem class itself: that its state accessors
    round-trip losslessly, and that reading its diagnostics between steps
    cannot feed back into the stepping. Both matter precisely because the
    split moved the canonical stepping state here, decoupled from the
    `CrankNicolsonFEMScheme` that advances it.

    Everything runs on a coarse closed sphere built by `_fem_fixtures`; no
    external data is needed.
    """

    PROBE = Path(__file__).resolve().parent / "_mpi_source_probe.py"
    """Driver the multi-rank check spawns; it cannot be imported, only launched."""

    ROUND_OFF_BOUND = 1.0e-12
    """
    Bound on the disagreement between two summations of one source integral.

    Naive summation over the probe mesh's 169 dofs bounds it at n * eps = 3.8e-14,
    so this allows a factor of 26 while staying thirteen orders below the 4e-2
    shortfall an unaccumulated load vector produces on two ranks.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Mesh generation is the slow part of the suite, so it happens once.
        cls.comm = MPI.COMM_SELF
        cls.msh = make_sphere_mesh(comm=cls.comm)
        cls.V = fem.functionspace(cls.msh, ("Lagrange", 1))
        cls.dof_coordinates = cls.V.tabulate_dof_coordinates()

    def _make_problem(
        self,
        u0: fem.Function,
        source: float = 0.0,
        material_funcs: MaterialFunctions | None = None,
    ) -> FEMHeatProblem:
        if material_funcs is None:
            material_funcs = uniform_material_funcs(self.V, CONCRETE, source=source)
        return FEMHeatProblem(
            msh=self.msh, V=self.V, u0=u0, material_funcs=material_funcs, comm=self.comm
        )

    def _make_scheme(
        self, problem: FEMHeatProblem, dt: float = 1.0e4
    ) -> CrankNicolsonFEMScheme:
        return CrankNicolsonFEMScheme(problem=problem, dt=dt, comm=self.comm)

    def _linear_temperature_field(self) -> fem.Function:
        """
        A smooth, strongly non-uniform initial field: 300 K plus a 20 K
        top-to-bottom gradient and a small azimuthal ripple, so that many
        DOFs carry distinguishable values.
        """
        x, y, z = self.dof_coordinates.T
        return function_from_dof_values(self.V, 300.0 + 20.0 * z + 5.0 * x * y)

    def test_snapshot_and_load_state_round_trip(self):
        """
        The runners collect `snapshot_state()` vectors during the time loop and
        replay them with `load_state()` afterwards to compute diagnostics and
        write output. That replay has to be lossless: the snapshot is a
        standalone copy, so stepping past it and coming back must restore the
        field and its total heat bit-for-bit.
        """
        problem = self._make_problem(self._linear_temperature_field())
        scheme = self._make_scheme(problem)
        for _ in range(3):
            scheme.step()

        snapshot = problem.snapshot_state()
        snapshot_temperatures = state_array(problem)
        snapshot_heat = problem.compute_total_heat()

        for _ in range(5):
            scheme.step()
        self.assertGreater(
            np.abs(state_array(problem) - snapshot_temperatures).max(), 0.0
        )

        problem.load_state(snapshot)
        # A vector copy: restoration is exact, so this is asserted bitwise. Any
        # tolerance here would hide an aliasing bug (e.g. a snapshot that is a
        # view of the live state rather than a copy).
        np.testing.assert_array_equal(state_array(problem), snapshot_temperatures)
        self.assertEqual(problem.compute_total_heat(), snapshot_heat)
        snapshot.destroy()

    @unittest.skipUnless(_MPIRUN is not None, _SKIP_REASON)
    def test_the_load_vector_carries_the_whole_source_across_ranks(self):
        """
        The assembled load vector sums to the source integral on two ranks.

        `assemble_vector` leaves a dof shared across a partition boundary
        holding only the contribution of the cells its rank owns, so a load
        vector that is never finalised under-counts the source by the fraction
        of the integral living on shared dofs (measured at 4 % on two ranks
        and 10 % on four). Nothing else observes it: the scalar diagnostics
        integrate the source function rather than the vector and go on
        reporting the correct power, so the run's own energy-drift line is the
        only symptom, and it is logged rather than asserted on.

        One rank has no shared dofs and cannot see the defect, which is why
        this spawns a launcher instead of asserting in process.
        """
        environment = dict(os.environ)
        # OpenMPI probes TCP interfaces before falling back to shared memory and
        # fails outright on hosts with no routable pairing between them. Naming
        # the local transports skips the probe; other launchers ignore it.
        environment.setdefault("OMPI_MCA_btl", "self,vader,sm")
        environment.setdefault("OMPI_MCA_btl_base_warn_component_unused", "0")

        completed = subprocess.run(
            [_MPIRUN, "-n", "2", sys.executable, str(self.PROBE)],
            capture_output=True,
            text=True,
            env=environment,
        )
        measured = [
            line for line in completed.stdout.splitlines() if "relative_error=" in line
        ]
        if not measured:
            self.skipTest(
                f"Could not launch two ranks here: {completed.stderr.strip()[-300:]}"
            )

        relative_error = float(measured[0].split("relative_error=")[1])
        self.assertLess(relative_error, self.ROUND_OFF_BOUND)

    def test_diagnostics_between_steps_do_not_perturb_the_trajectory(self):
        """
        The stepping state deliberately lives in plain PETSc vectors, decoupled
        from `u_n`: reading `solution` or `compute_total_heat()` between steps
        writes into the Function and must never feed back. Without that
        decoupling, taking snapshots mid-run would silently corrupt the results,
        which is why `FEMHeatRun.march` defers all form assembly to after the
        loop. This test locks the decoupling in, so that a future refactor
        cannot quietly reintroduce the feedback.
        """
        num_steps = 10

        undisturbed_problem = self._make_problem(
            self._linear_temperature_field(), source=4.0
        )
        undisturbed_scheme = self._make_scheme(undisturbed_problem)
        for _ in range(num_steps):
            undisturbed_scheme.step()

        observed_problem = self._make_problem(
            self._linear_temperature_field(), source=4.0
        )
        observed_scheme = self._make_scheme(observed_problem)
        observed_first_dof = []
        for _ in range(num_steps):
            observed_scheme.step()
            observed_problem.compute_total_heat()
            observed_first_dof.append(observed_problem.solution.x.array[0])
            observed_problem.snapshot_state().destroy()

        # The observation itself must not have corrupted the field it read.
        self.assertTrue(np.isfinite(observed_first_dof).all())

        # The two schemes execute the same sequence of operations, but MUMPS is
        # only bitwise reproducible at OMP_NUM_THREADS=1: with threading on, two
        # identical runs in one process already differ by ~7e-16 relative, so a
        # bitwise assertion would be flaky. 1e-13 sits three orders above that
        # floor and still an order of magnitude *below* anything a feedback path
        # could cause: a feedback bug produces visibly wrong peak
        # temperatures, not last-bit noise.
        np.testing.assert_allclose(
            state_array(observed_problem),
            state_array(undisturbed_problem),
            rtol=1e-13,
        )
