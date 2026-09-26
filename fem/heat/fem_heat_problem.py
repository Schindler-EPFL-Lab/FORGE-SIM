import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from mpi4py import MPI
from petsc4py import PETSc

from fem.heat.material_functions import MaterialFunctions


class FEMHeatProblem:
    """
    Assembles and owns the discretized transient heat equation on a 2D manifold.

    The heat equation with material properties is:

        rho c d du/dt - div(k d grad(u)) = f

    where rho is density, c is specific heat, d is shell thickness, k is thermal
    conductivity, u is temperature and f is the source term per unit area. The
    mass coefficient is rho*c*d, the stiffness coefficient is k*d, and the
    source is integrated per unit area (no thickness factor), identical to
    the IGA problem.

    This class builds the variational forms and assembles the (time-independent)
    mass matrix M and stiffness matrix K once, at construction, and owns the
    current temperature field: as a `fem.Function` for I/O and diagnostics
    (`solution`), and as a plain PETSc vector for time stepping
    (`snapshot_state`/`load_state`).

    Nothing here time-steps. `CrankNicolsonFEMScheme` reads `mass_matrix`,
    `stiffness_matrix` and `assemble_source_vector()` to build its own operators
    and, on every `step`, commits the new temperature back here via
    `load_state`. This mirrors the IGA-side split between `HeatSimulation` (the
    problem) and `CrankNicolsonScheme` (the scheme).
    """

    def __init__(
        self,
        msh: mesh.Mesh,
        V: fem.FunctionSpace,
        u0: fem.Function,
        material_funcs: MaterialFunctions,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        """
        Builds the variational forms and assembles the mass and stiffness
        matrices for the given mesh, function space and material properties.

        `msh` is the DOLFINx mesh (2D manifold in 3D) and `V` the function
        space for temperature (typically P1). `u0` is the initial temperature
        function and `material_funcs` the five material property functions, on
        `V`. `comm` is the MPI communicator for parallel execution.
        """
        self._msh = msh
        self._V = V
        self._comm = comm
        self._material_funcs = material_funcs

        self._u_n = fem.Function(V, name="u_n")
        """
        Previous-step temperature as a DOLFINx Function, used only for form
        assembly (diagnostics) and output, never for the time-stepping
        itself, which runs on `_state`. Initialized from `u0` immediately
        below, before any stepping.
        """
        self._u_n.x.array[:] = u0.x.array

        self._a_mass, self._a_stiff, self._L_source = self._build_forms(
            V, material_funcs
        )
        self._M, self._K = self._assemble_mass_stiffness(self._a_mass, self._a_stiff)

        self._state = self._u_n.x.petsc_vec.duplicate()
        """
        Crank-Nicolson stepping state, held as a plain PETSc vector decoupled
        from any `fem.Function`: touching a Function's `.x` between steps
        desynchronizes the wrapper `CrankNicolsonFEMScheme.step` relies on and
        corrupts the stepping, so snapshots and diagnostics read this instead.
        `_u_n` is synced from it only on demand.
        """
        self._u_n.x.petsc_vec.copy(self._state)

    @staticmethod
    def _build_forms(
        V: fem.FunctionSpace, material_funcs: MaterialFunctions
    ) -> tuple[ufl.Form, ufl.Form, ufl.Form]:
        """
        Builds the UFL variational forms for the mass and stiffness matrices and
        the source vector.

        The mass form carries the volumetric heat capacity:
            M = integral( rho c d phi_i phi_j ) dA
        The stiffness form carries the conductivity:
            K = integral( k d grad(phi_i) . grad(phi_j) ) dA
        The source form is per unit area (no thickness factor):
            F = integral( Q phi_i ) dA

        :returns: The mass, stiffness and source forms, in that order.
        """
        v = ufl.TestFunction(V)
        u_tr = ufl.TrialFunction(V)

        mass_coeff = (
            material_funcs.density
            * material_funcs.specific_heat
            * material_funcs.thickness
        )
        stiff_coeff = material_funcs.conductivity * material_funcs.thickness

        a_mass = mass_coeff * ufl.inner(u_tr, v) * ufl.dx
        a_stiff = stiff_coeff * ufl.inner(ufl.grad(u_tr), ufl.grad(v)) * ufl.dx
        L_source = material_funcs.source * v * ufl.dx
        return a_mass, a_stiff, L_source

    @staticmethod
    def _assemble_mass_stiffness(
        a_mass: ufl.Form, a_stiff: ufl.Form
    ) -> tuple[PETSc.Mat, PETSc.Mat]:
        """
        Assembles the time-independent mass matrix M and stiffness matrix K.

        :returns: The mass and stiffness matrices, in that order.
        """
        M = assemble_matrix(fem.form(a_mass))
        M.assemble()

        K = assemble_matrix(fem.form(a_stiff))
        K.assemble()
        return M, K

    def assemble_source_vector(self) -> PETSc.Vec:
        """
        Assembles the load vector F = integral(Q phi_i) dA from the compiled
        source form.

        Called once, by the owning `CrankNicolsonFEMScheme` at construction;
        the returned vector is then scaled in place by
        `CrankNicolsonFEMScheme.scale_source`, never reassembled from this
        method again.

        `assemble_vector` returns an unfinalised vector: a dof shared across a
        partition boundary holds only the contribution of the cells owned by the
        rank that computed it. The reverse scatter accumulates those partial
        contributions onto the owning rank, which is the vector counterpart of
        the `assemble` call the mass and stiffness matrices make. Without it the
        load vector under-counts the source by the fraction of the integral that
        lives on shared dofs (about 4 % on two ranks and 10 % on four, growing
        with the number of partition boundaries), while every scalar diagnostic
        continues to report the correct source power, so nothing observes the
        loss. The forward scatter then republishes the accumulated values, so a
        reader of the ghost region sees what the owner holds.

        :returns: The finalised load vector F.
        """
        F = assemble_vector(fem.form(self._L_source))
        F.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        F.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
        )
        return F

    def compute_total_heat(self) -> float:
        """
        Computes the total thermal energy in the system:
            H = integral( rho c d u ) dA

        This matches the energy computation in the IGA problem for comparison.
        `assemble_scalar` returns only the rank-local contribution, so the
        partial integrals are reduced across ranks for the global value. The
        stepping state is synced into `u_n` first so the form sees it.

        :returns: Total heat in Joules.
        """
        self._state.copy(self._u_n.x.petsc_vec)
        self._u_n.x.scatter_forward()
        rho = self._material_funcs.density
        c = self._material_funcs.specific_heat
        d = self._material_funcs.thickness
        u_form = rho * c * d * self._u_n * ufl.dx
        local_heat = fem.assemble_scalar(fem.form(u_form))
        return self._comm.allreduce(local_heat, op=MPI.SUM)

    def compute_heat_generation_rate(self) -> float:
        """
        Computes the total heat generation rate from the source term:
            Q_total = integral( Q ) dA

        where Q is power per unit area [W/m^2], so the integral gives total
        power [W]. This matches the IGA formulation where the source is per
        unit area. The rank-local partial integrals are reduced across ranks.

        :returns: Heat generation rate in Watts.
        """
        Q = self._material_funcs.source
        q_form = Q * ufl.dx
        local_rate = fem.assemble_scalar(fem.form(q_form))
        return self._comm.allreduce(local_rate, op=MPI.SUM)

    def snapshot_state(self) -> PETSc.Vec:
        """
        Returns a standalone PETSc-vector copy of the current state. Safe to
        call between steps (it does not touch any fem.Function), so callers
        can collect time snapshots during stepping without disturbing the
        integration. The caller owns the returned vector and should
        `destroy()` it when done.
        """
        return self._state.copy()

    def load_state(self, state: PETSc.Vec) -> None:
        """
        Sets the current state from a snapshot (e.g. to re-evaluate
        diagnostics or write output for a stored time level after the loop),
        or commits the state `CrankNicolsonFEMScheme.step` just solved for.
        """
        state.copy(self._state)

    @property
    def material_funcs(self) -> MaterialFunctions:
        """
        The five coefficient fields this problem assembled its operators from.

        The record is immutable but the DOLFINx Functions it holds are not:
        `CrankNicolsonFEMScheme.scale_source` rewrites `source` in place, as
        its own docstring records, so a caller reading `source` after a
        scaling sees the scaled field.
        """
        return self._material_funcs

    @property
    def mass_matrix(self) -> PETSc.Mat:
        """The assembled, time-independent mass matrix M."""
        return self._M

    @property
    def stiffness_matrix(self) -> PETSc.Mat:
        """The assembled, time-independent stiffness matrix K."""
        return self._K

    @property
    def state(self) -> PETSc.Vec:
        """
        Live PETSc vector holding the current temperature, the same object
        `load_state` writes into, never reallocated. Read directly by
        `CrankNicolsonFEMScheme.step` for the stepping math; use
        `snapshot_state` for a standalone copy, or `solution` for the
        `fem.Function` view.
        """
        return self._state

    @property
    def solution(self) -> fem.Function:
        """
        Returns the current temperature as a DOLFINx Function (synced from the
        internal state vector).
        """
        self._state.copy(self._u_n.x.petsc_vec)
        self._u_n.x.scatter_forward()
        return self._u_n
