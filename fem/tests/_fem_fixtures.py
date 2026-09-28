"""
Shared fixtures for the DOLFINx heat-solver and shell-modal tests.

Importing this module puts the repository root on `sys.path`, which is what makes
both `fem` (this suite's subject, imported as `fem.heat.…`, `fem.shells.…`) and
`forge_sim` resolve. The conda environment installs neither, so without it the
suite only runs when the interpreter's own directory happens to be the root.

`fem` is a top-level package name distinct from DOLFINx's `dolfinx.fem`
submodule, so `from dolfinx import fem` inside these modules keeps binding to the
real submodule; nothing here shadows it.

Run the suite from the repository root (so `forge_sim/__init__.py` can create
`outputs/forge_sim.log`)::

    conda run -n dolfinx-forge-sim python -m unittest discover fem/tests

The geometry used throughout is a coarse gmsh sphere: a closed 2-manifold
embedded in 3D with no boundary, which is exactly the setting the heat pipeline
runs in (pure Neumann, no boundary conditions applied anywhere) and the setting
in which the stiffness matrix has its constant nullspace.
"""

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from dolfinx import fem, mesh  # noqa: E402
from mpi4py import MPI  # noqa: E402

from fem.heat.material_functions import MaterialFunctions  # noqa: E402
from fem.shells.curved_sphere_mesh import make_curved_sphere_mesh  # noqa: E402
from fem.shells.geometrically_exact_shell_modal_solver import (  # noqa: E402
    GeometricallyExactShellModalSolver,
)
from fem.shells.geometrically_exact_shell_problem import (  # noqa: E402
    GeometricallyExactShellProblem,
)
from fem.shells.shell_element import ShellElement  # noqa: E402
from forge_sim.scripts.configs.thermal_material_config import (  # noqa: E402
    ThermalMaterialConfig,
)

CONCRETE = ThermalMaterialConfig(
    conductivity=2.0, density=2250.0, specific_heat=900.0, thickness=0.2
)
"""
Matches the defaults of `ThermalMaterialConfig`, the single-material case the
FEM runner uses.
"""

STEEL = ThermalMaterialConfig(
    conductivity=50.0, density=7850.0, specific_heat=500.0, thickness=0.01
)
"""
Matches the "active" material of `fem_heat_gt_mesh.py` and gives a ~300x
conductivity contrast, the regime the direct (Cholesky) solve suits.
"""


def make_sphere_mesh(
    radius: float = 1.0,
    mesh_size: float = 0.35,
    comm: MPI.Comm = MPI.COMM_SELF,
) -> mesh.Mesh:
    """
    Builds a coarse flat (order-1) closed triangular surface mesh of a sphere,
    the geometry the heat pipeline operates on: a closed shell with no boundary
    facets, so the stiffness matrix carries its constant nullspace.

    This is `curved_sphere_mesh.make_curved_sphere_mesh` at order 1. That
    function is the single gmsh sphere builder in `fem/`; it is production
    code (the shell problem's own validation geometry) and it also covers the
    curved order-2 case, so the tests call it rather than keeping a second
    copy of the same gmsh sequence in step with it. The order-1 meshes the two
    produced were verified byte-identical before the copy was removed.

    `radius` is the sphere radius in metres. `mesh_size` is the target element
    size; the default gives ~250 triangles, enough for a non-trivial spectrum
    while keeping the suite fast. `comm` is the MPI communicator the mesh is
    built on.

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
    material: ThermalMaterialConfig = CONCRETE,
    source: float = 0.0,
) -> MaterialFunctions:
    """
    Builds the `MaterialFunctions` record `FEMHeatProblem` expects, with
    a single homogeneous material and a spatially constant source.

    A fresh set of Functions is returned on every call. This matters:
    `scale_source` mutates the source Function in place, so sharing one record
    between two problems would silently couple them.

    `material` is e.g. `CONCRETE` or `STEEL`. `source` is the heat source per
    unit area [W/m^2].
    """
    return MaterialFunctions(
        conductivity=constant_function(V, material.conductivity),
        density=constant_function(V, material.density),
        specific_heat=constant_function(V, material.specific_heat),
        thickness=constant_function(V, material.thickness),
        source=constant_function(V, source),
    )


def two_material_funcs(
    V: fem.FunctionSpace,
    active_mask: np.ndarray,
    active_material: ThermalMaterialConfig = STEEL,
    passive_material: ThermalMaterialConfig = CONCRETE,
    active_source: float = 0.0,
) -> MaterialFunctions:
    """
    Builds a two-material `MaterialFunctions` record, selecting per DOF between
    an active and a passive material and putting the heat source on the active
    material only.

    This mirrors the "windows heat up" split of the two-material heat scripts
    (`heat_simulation_spline_two_materials.py`, `fem_heat_gt_mesh.py`), where the
    material fields are per-vertex `np.where` selections rather than cell-wise
    constants.

    `active_mask` is a boolean array over the DOFs of `V`; True = active.
    `active_source` is the heat source per unit area [W/m^2] on active DOFs.
    """

    def selected(active_value: float, passive_value: float) -> fem.Function:
        return function_from_dof_values(
            V, np.where(active_mask, active_value, passive_value)
        )

    return MaterialFunctions(
        conductivity=selected(
            active_material.conductivity, passive_material.conductivity
        ),
        density=selected(active_material.density, passive_material.density),
        specific_heat=selected(
            active_material.specific_heat, passive_material.specific_heat
        ),
        thickness=selected(active_material.thickness, passive_material.thickness),
        source=function_from_dof_values(V, np.where(active_mask, active_source, 0.0)),
    )


def state_array(problem) -> np.ndarray:
    """
    Returns a copy of the problem's internal stepping state as a numpy array,
    without disturbing the integration (it goes through `snapshot_state`, which
    never touches a `fem.Function`).
    """
    snapshot = problem.snapshot_state()
    values = snapshot.getArray().copy()
    snapshot.destroy()
    return values


RADIUS = 1.0
"""
Sphere radius [m] for the shell tests: the canonical unit sphere
`tests/test_steel_sphere_modal_analysis.py` pins independently on the IGA side.
"""

ANCHOR_MESH_SIZE = 0.5
"""
Element size for the shell anchor mesh. An h-refinement sweep (h = 0.8 to 0.3)
puts the first elastic cluster at 579.05, 576.89, 576.63, 576.57, 576.38 Hz, so
by h = 0.5 the discretisation error is already down to ~0.06 % while the mesh
is still only ~150 cells and the whole solve takes a fraction of a second.
"""

STEEL_SHELL = {
    "youngs_modulus": 193.05e9,
    "poisson_ratio": 0.28,
    "density": 8025.9,
    "thickness": 1.5875e-3,
}
"""
Canonical unit steel sphere material, identical to
`tests/test_steel_sphere_modal_analysis.py`, the IGA reference the shell tests
cross-validate against.
"""

ANALYTIC_CLUSTERS = ((576.3, 5), (681.0, 7), (721.8, 9))
"""
Analytical thin-spherical-shell frequencies (Hz) of the first three
membrane-dominated spheroidal clusters at `STEEL_SHELL` and `RADIUS`, together
with their degeneracies (2n+1), for n = 2, 3, 4. Quoted in the class docstring
of `GeometricallyExactShellProblem`; the IGA solver reproduces them too.
"""

NUM_RIGID_BODY_MODES = 6
"""A closed, unsupported shell has exactly six zero-energy rigid-body modes."""

IGA_FIRST_ELASTIC_HZ = 577.0
"""
The IGA solver's first elastic mode on the canonical steel sphere
(`tests/test_steel_sphere_modal_analysis.py`), the cross-validation reference
the shell tests reproduce.
"""

RIGID_BODY_LABELS = ("Tx", "Ty", "Tz", "Rx", "Ry", "Rz")
"""
`RigidBodyMode.standard_basis()`'s labels, in its own order: translation then
rotation about x, y, z. Pins the count and the identity of
`GeometricallyExactShellModalSolver.rigid_body_energies()`'s result in one
shot, since that result is a computed sequence rather than six named fields.
"""

# Cache meshes and problems across every test module in this package: building a
# problem JIT-compiles its variational forms, which costs far more than the
# eigensolve at these sizes, and several modules want the same configuration.
_MESH_CACHE: dict[tuple[float, int], mesh.Mesh] = {}
_PROBLEM_CACHE: dict[tuple, GeometricallyExactShellProblem] = {}


def sphere_mesh(mesh_size: float = ANCHOR_MESH_SIZE, order: int = 2) -> mesh.Mesh:
    """A cached curved sphere mesh of the given element size and geometry order."""
    key = (mesh_size, order)
    if key not in _MESH_CACHE:
        _MESH_CACHE[key] = make_curved_sphere_mesh(RADIUS, mesh_size, order=order)
    return _MESH_CACHE[key]


def steel_sphere_problem(
    mesh_size: float = ANCHOR_MESH_SIZE, order: int = 2, **element_kwargs
) -> GeometricallyExactShellProblem:
    """A cached steel-sphere problem; `element_kwargs` go to `ShellElement`."""
    key = (mesh_size, order) + tuple(sorted(element_kwargs.items()))
    if key not in _PROBLEM_CACHE:
        _PROBLEM_CACHE[key] = GeometricallyExactShellProblem(
            sphere_mesh(mesh_size, order),
            element=ShellElement(**element_kwargs),
            **STEEL_SHELL,
        )
    return _PROBLEM_CACHE[key]


def steel_sphere_solver(
    mesh_size: float = ANCHOR_MESH_SIZE, order: int = 2, **element_kwargs
) -> GeometricallyExactShellModalSolver:
    """A modal solver over a cached steel-sphere problem; `element_kwargs` go to
    `ShellElement`."""
    return GeometricallyExactShellModalSolver(
        steel_sphere_problem(mesh_size, order, **element_kwargs)
    )


def cluster_slices() -> Iterator[tuple[float, slice]]:
    """Index slices of `ANALYTIC_CLUSTERS` in the sorted spectrum."""
    start = NUM_RIGID_BODY_MODES
    for frequency, degeneracy in ANALYTIC_CLUSTERS:
        yield frequency, slice(start, start + degeneracy)
        start += degeneracy


def relative_asymmetry(matrix: sp.spmatrix) -> float:
    """Largest entry of `matrix - matrix.T`, relative to `matrix`'s own
    largest entry."""
    largest = abs(matrix).max()
    if largest == 0.0:
        return 0.0
    return float(abs(matrix - matrix.T).max() / largest)


# Ground-truth-mesh heat driver fixtures, shared between test_fem_heat_gt_mesh.py
# and test_fem_heat_gt_runner.py.

ACTIVE = ThermalMaterialConfig(
    conductivity=50.0, density=7850.0, specific_heat=500.0, thickness=0.01
)
"""Active-material fixture for the heat ground-truth driver tests."""

PASSIVE = ThermalMaterialConfig(
    conductivity=0.15, density=600.0, specific_heat=2400.0, thickness=0.02
)
"""Passive-material fixture, paired with `ACTIVE`."""

SQUARE_VERTICES = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
)
"""Four-vertex square, the baked-mesh fixture for the heat ground-truth driver
tests."""

SQUARE_FACES = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
"""Two triangles over `SQUARE_VERTICES`."""

MATERIAL_MASK = np.array([1.0, 0.75, 0.5, 0.0])
"""
Per-vertex material mask of the fixture: two active vertices, then two passive.

The mask is continuous rather than binary, as the baked one is, and the third value
sits exactly on the 0.5 cut. It is passive there, because both solvers select the
active material with a strict `> 0.5`, the IGA side at its control points, this
driver at the analysis-mesh vertices, and a boundary that disagreed between them
would move the material interface by one vertex ring.
"""

INITIAL_TEMPERATURE = np.array([1.0, 0.8, 0.2, 0.0])
"""Initial condition of the fixture; distinct per vertex so it cannot be confused
with the material mask if the two were ever swapped on the way in."""


def write_fixture_npz(path: Path) -> None:
    """Writes the baked-mesh bundle in the shape the heat ground-truth driver's
    contract names."""
    np.savez(
        path,
        vertices=SQUARE_VERTICES,
        faces=SQUARE_FACES,
        temperature=INITIAL_TEMPERATURE,
        material=MATERIAL_MASK,
        scale=1.0,
    )


# Ground-truth-mesh shell modal driver fixtures, shared between
# test_fem_shell_modal_gt_mesh.py and test_fem_shell_modal_gt_runner.py.

TETRAHEDRON_VERTICES = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
"""
A closed tetrahedron: the smallest mesh with no boundary, so it exercises the
shell modal ground-truth driver's handover without any of the degenerate cases
the driver guards against. Its four cells also solve instantly, so the same
fixture doubles as the driver's one end-to-end test.
"""

TETRAHEDRON_FACES = np.array(
    [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64
)
"""Outward-wound faces of `TETRAHEDRON_VERTICES`."""
