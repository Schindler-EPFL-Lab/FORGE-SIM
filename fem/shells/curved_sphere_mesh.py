"""
The curved benchmark sphere the shell formulation is validated on.
"""

from dolfinx import mesh
from mpi4py import MPI


def make_curved_sphere_mesh(radius: float, mesh_size: float, order: int = 2,
                            comm: MPI.Comm = MPI.COMM_SELF) -> mesh.Mesh:
    """
    Curved (isoparametric `order`) triangular surface mesh of a sphere via gmsh.

    Only the OCC sphere's surface is meshed (the volume is never generated), so
    the result is a closed 2-manifold with no boundary facets, the setting both
    the shell and the heat solvers run in. `order=1` gives flat facets;
    `order >= 2` adds mid-edge nodes projected onto the exact sphere, which is
    what the shell solver wants (flat facets only degrade accuracy, see the class
    docstring of
    :class:`fem.shells.geometrically_exact_shell_problem.GeometricallyExactShellProblem`).

    `radius` is the sphere radius in metres. `mesh_size` is the target element
    size, passed as both the gmsh min and max. `order` is the geometry order;
    1 gives flat facets, >= 2 curved. `comm` is the MPI communicator the mesh
    is built on.

    :returns: DOLFINx mesh (topological dim 2, geometric dim 3).
    """
    # Deferred: gmsh is a process-global singleton with a heavy native library,
    # and it is needed only by this one benchmark-geometry builder; importing it
    # at module scope would load it into every run of the shell solver.
    import gmsh
    from dolfinx.io import gmshio

    # gmsh is a process-global singleton: a session leaked by an earlier caller
    # (or an aborted run) makes the initialize/finalize pair below misbehave, so
    # start from a known-clean state.
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("sphere")
        gmsh.model.occ.addSphere(0.0, 0.0, 0.0, radius)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [e[1] for e in gmsh.model.getEntities(2)], 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.model.mesh.generate(2)
        if order >= 2:
            gmsh.model.mesh.setOrder(order)
        msh, _, _ = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=3)
    finally:
        gmsh.finalize()  # always finalize, else a later gmsh.initialize() errors
    return msh
