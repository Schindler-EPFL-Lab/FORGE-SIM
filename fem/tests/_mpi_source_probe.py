"""
Driver for the multi-rank half of `test_fem_heat_problem`, run under `mpirun`.

The defect it exists to catch is invisible on one rank: `assemble_vector`
returns a vector whose shared dofs hold only the contribution of each rank's
own cells, so the loss appears only once a partition boundary exists. This
module is therefore spawned as a subprocess rather than imported, and prints
one line the parent parses.

It lives here rather than inside the test module because a test module is
imported by `unittest discover`, and a file that runs a solve at import time
would run it once per discovery.
"""

import sys
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fem.heat.fem_heat_problem import FEMHeatProblem  # noqa: E402
from fem.heat.material_functions import MaterialFunctions  # noqa: E402

SOURCE = 5.0
"""Uniform source density, in W/m^2; any nonzero value exposes the defect."""

SUBDIVISIONS = 12
"""Cells per side of the unit square, enough to give every rank interior dofs."""


def relative_source_error(comm: MPI.Comm) -> float:
    """
    Returns the relative shortfall of the assembled load vector against the
    integral of the source it discretizes.

    Summing the load vector over owned dofs is a partition of the same integral
    `assemble_scalar` computes, so the two agree to round-off exactly when the
    shared-dof contributions have been accumulated onto their owners.
    """
    msh = mesh.create_unit_square(comm, SUBDIVISIONS, SUBDIVISIONS)
    V = fem.functionspace(msh, ("Lagrange", 1))

    def uniform(value: float) -> fem.Function:
        function = fem.Function(V)
        function.x.array[:] = value
        return function

    materials = MaterialFunctions(
        conductivity=uniform(2.0),
        density=uniform(2250.0),
        specific_heat=uniform(900.0),
        thickness=uniform(0.2),
        source=uniform(SOURCE),
    )
    problem = FEMHeatProblem(
        msh=msh, V=V, u0=fem.Function(V), material_funcs=materials, comm=comm
    )

    load_vector = problem.assemble_source_vector()
    owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    integral = comm.allreduce(
        fem.assemble_scalar(fem.form(materials.source * ufl.dx)), op=MPI.SUM
    )
    assembled = comm.allreduce(
        float(np.sum(load_vector.array_r[:owned])), op=MPI.SUM
    )
    return abs(assembled - integral) / abs(integral)


if __name__ == "__main__":
    communicator = MPI.COMM_WORLD
    error = relative_source_error(communicator)
    if communicator.rank == 0:
        print(f"ranks={communicator.size} relative_error={error:.6e}")
