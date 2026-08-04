"""
Shared fixtures for the DOLFINx heat-solver tests.

Importing this module makes the `dolfinx/` scripts importable as *top-level*
modules (`from crank_nicolson_fem_solver import CrankNicolsonFEMSolver`), the
same convention the drivers in `dolfinx/` use among themselves, and puts the
repository root on `sys.path` so `rebel_shape` resolves.

The `dolfinx/` directory shares its name with the installed DOLFINx package, but
having the repository root on `sys.path` is harmless: a directory without
`__init__.py` is only a PEP 420 namespace *portion*, and a regular installed
package always wins over one, so `from dolfinx import fem` keeps binding to the
real package. This is the same situation `dolfinx/README.md` describes.

Run the suite from the repository root (so `rebel_shape/__init__.py` can create
`outputs/rebel_shape.log`)::

    conda run -n dolfinx-rebel-shape python -m unittest discover dolfinx/tests

The geometry used throughout is a coarse gmsh sphere: a *closed* 2-manifold
embedded in 3D with no boundary, which is exactly the setting the heat pipeline
runs in (pure Neumann, no boundary conditions applied anywhere) and the setting
in which the stiffness matrix has its constant nullspace.
"""

import sys
from pathlib import Path

import numpy as np

_DOLFINX_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _DOLFINX_SCRIPTS_DIR.parent
for _path in (_REPOSITORY_ROOT, _DOLFINX_SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dolfinx import fem, mesh  # noqa: E402
from mpi4py import MPI  # noqa: E402

from geometrically_exact_shell_solver import make_curved_sphere_mesh  # noqa: E402

# Material constants for the tests. `CONCRETE` matches the defaults of
# `ThermalMaterialConfig` (the single-material case the FEM runner uses);
# `STEEL` matches the "active" material of `fem_heat_gt_mesh.py` and gives a
# ~300x conductivity contrast, which is the regime the direct (Cholesky) solve
# was chosen for.
CONCRETE = {
    "conductivity": 2.0,
    "density": 2250.0,
    "specific_heat": 900.0,
    "thickness": 0.2,
}
STEEL = {
    "conductivity": 50.0,
    "density": 7850.0,
    "specific_heat": 500.0,
    "thickness": 0.01,
}

MATERIAL_FIELDS = ("conductivity", "density", "specific_heat", "thickness", "source")


def make_sphere_mesh(
    radius: float = 1.0,
    mesh_size: float = 0.35,
    comm: MPI.Comm = MPI.COMM_SELF,
) -> mesh.Mesh:
    """
    Builds a coarse *flat* (order-1) closed triangular surface mesh of a sphere,
    the geometry the heat pipeline operates on: a closed shell with no boundary
    facets, so the stiffness matrix carries its constant nullspace.

    This is `geometrically_exact_shell_solver.make_curved_sphere_mesh` at order
    1. That function is the single gmsh sphere builder in `dolfinx/`; it is
    production code (the shell solver's own validation geometry) and it also
    covers the curved order-2 case, so the tests call it rather than keeping a
    second copy of the same gmsh sequence in step with it. The order-1 meshes the
    two produced were verified byte-identical before the copy was removed.

    :param radius: Sphere radius in metres.
    :param mesh_size: Target element size; the default gives ~250 triangles,
        enough for a non-trivial spectrum while keeping the suite fast.
    :param comm: MPI communicator the mesh is built on.
    :returns: DOLFINx mesh (topological dim 2, geometric dim 3).
    """
    return make_curved_sphere_mesh(
        radius=radius, mesh_size=mesh_size, order=1, comm=comm
    )


def constant_function(V: fem.FunctionSpace, value: float) -> fem.Function:
    """
    Returns a `fem.Function` on `V` whose every DOF carries `value`.
    """
    function = fem.Function(V)
    function.x.array[:] = value
    function.x.scatter_forward()
    return function


def function_from_dof_values(V: fem.FunctionSpace, values: np.ndarray) -> fem.Function:
    """
    Returns a `fem.Function` on `V` from an array of per-DOF values (in the DOF
    ordering of `V.tabulate_dof_coordinates()`).
    """
    function = fem.Function(V)
    function.x.array[:] = values
    function.x.scatter_forward()
    return function


def uniform_material_funcs(
    V: fem.FunctionSpace,
    material: dict[str, float] = CONCRETE,
    source: float = 0.0,
) -> dict[str, fem.Function]:
    """
    Builds the `material_funcs` dictionary `CrankNicolsonFEMSolver` expects, with
    a single homogeneous material and a spatially constant source.

    A *fresh* set of Functions is returned on every call. This matters:
    `scale_source` mutates the source Function in place, so sharing one
    dictionary between two solvers would silently couple them.

    :param material: Mapping with 'conductivity', 'density', 'specific_heat' and
        'thickness' entries, e.g. `CONCRETE` or `STEEL`.
    :param source: Heat source per unit area [W/m^2].
    """
    material_funcs = {
        name: constant_function(V, material[name])
        for name in MATERIAL_FIELDS
        if name != "source"
    }
    material_funcs["source"] = constant_function(V, source)
    return material_funcs


def two_material_funcs(
    V: fem.FunctionSpace,
    active_mask: np.ndarray,
    active_material: dict[str, float] = STEEL,
    passive_material: dict[str, float] = CONCRETE,
    active_source: float = 0.0,
) -> dict[str, fem.Function]:
    """
    Builds a two-material `material_funcs` dictionary, selecting per DOF between
    an active and a passive material and putting the heat source on the active
    material only.

    This mirrors the "windows heat up" split of the two-material heat scripts
    (`heat_simulation_spline_two_materials.py`, `fem_heat_gt_mesh.py`), where the
    material fields are per-vertex `np.where` selections rather than cell-wise
    constants.

    :param active_mask: Boolean array over the DOFs of `V`; True = active.
    :param active_source: Heat source per unit area [W/m^2] on active DOFs.
    """
    material_funcs = {
        name: function_from_dof_values(
            V,
            np.where(active_mask, active_material[name], passive_material[name]),
        )
        for name in MATERIAL_FIELDS
        if name != "source"
    }
    material_funcs["source"] = function_from_dof_values(
        V, np.where(active_mask, active_source, 0.0)
    )
    return material_funcs


def state_array(solver) -> np.ndarray:
    """
    Returns a copy of the solver's internal stepping state as a numpy array,
    without disturbing the integration (it goes through `snapshot_state`, which
    never touches a `fem.Function`).
    """
    snapshot = solver.snapshot_state()
    values = snapshot.getArray().copy()
    snapshot.destroy()
    return values
