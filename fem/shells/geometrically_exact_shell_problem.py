import warnings

import basix.ufl
import numpy as np
import scipy.sparse as sp
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from scipy.sparse.linalg import spsolve
from ufl.core.expr import Expr

from fem.shells.shell_element import ShellElement
from fem.shells.shell_forms import ShellForms
from fem.shells.shell_system_assembler import ShellSystemAssembler


class GeometricallyExactShellProblem:
    """
    Assembles a higher-order, locking-free Reissner-Mindlin shell modal problem
    on a 2-manifold embedded in 3D; it does not solve; that is
    :class:`fem.shells.geometrically_exact_shell_modal_solver.GeometricallyExactShellModalSolver`.
    Its compiled bilinear forms live in :class:`fem.shells.shell_forms.ShellForms`,
    private to this class; its element and integration choice in
    :class:`fem.shells.shell_element.ShellElement`; and the curved benchmark
    sphere the formulation is validated on in
    :func:`fem.shells.curved_sphere_mesh.make_curved_sphere_mesh`. Material is
    uniform (floats); extend to per-vertex fields by replacing the scalar
    constants with `fem.Function` objects.

    This uses only identity-mapped Lagrange elements (DOLFINx note: sidesteps
    the broken Piola-map-on-orientable-manifolds machinery, DOLFINx #3845, that
    rules out a clean MITC / TDNNS / Regge element on a closed surface). Run
    serially; both solve paths use the same mass-shifted shift-invert as the
    IGA solver. Environment: conda `dolfinx-forge-sim` (DOLFINx 0.9.x).

    A coordinate-free geometrically-exact Naghdi shell built from the surface
    normal n and the tangential projector P = I - n(x)n, on geometry that lies
    on the surface:

        membrane   eps   = sym(P . grad_s u . P)
        shear      gamma = P . ( (grad_s u)^T . n + theta x n )          # no +B*u
        bending    rho   = sym( P . [ (grad_s(theta x n))^T P
                                      + (grad_s n)^T grad_s u ] . P )     # exact dB
        drilling   w_dr  = 1/2 n.curl_s(u) - theta.n                      # 0 for RB

    These vanish for u = c + omega x x, theta = omega, so all six rigid-body (RB)
    modes are exactly zero-energy. That is what a flat-facet linear shell (P2
    displacement + Crouzeix-Raviart rotation on flat triangular facets, as in
    comet-fenics) achieves on flat plates but not on curved geometry, where two of its
    strain measures are wrong: the transverse shear  grad~(u.n) - P^T(n x theta)
    carries a spurious +B*u term (B = curvature) relative to the exact (grad_s u)^T n,
    breaking translation invariance once the normal varies, and the membrane strain
    omits the curvature coupling, breaking rotation invariance once curved. Only
    3 of its 6 RB modes are then zero-energy; the other 3 appear as spurious soft
    modes at a frequency proportional to (1/R)*sqrt(E/rho) that pollutes the whole low
    spectrum. The defect is independent of element order (P1/P2/P3 alike), and
    Crouzeix-Raviart is order-1 only, so raising the order cures neither it nor the
    locking.

    Locking is cured by PSRI (Partial Selective Reduced Integration, fenics-shells
    style): membrane and shear are split  alpha*full + (1-alpha)*reduced  with
    alpha = (t/h_cell)^2 and an order-aware reduced quadrature degree 2*(p-1)
    (degree 2 hourglasses at P3; 2*(p-1) does not). `LockingCure.BBAR` selects a
    stronger, parameter-free alternative in its place; see :meth:`_build_bbar_forms`.

    Validated on the canonical unit steel sphere (R=1, E=193.05 GPa, nu=0.28,
    rho=8025.9, h=1.5875 mm): P2 and P3 both reproduce, and converge to, the analytical
    thin-shell spectrum and the forge-sim IGA spectrum to < 0.01 %:
        6 RB modes @ 0 | n=2: 576.3 Hz (x5) | n=3: 681.0 Hz (x7) | n=4: 721.8 Hz (x9).

    Geometry order, normals and drilling
    ------------------------------------
    The measures are rigid-body invariant at any geometry order under the default
    `normal="geometric"` (verified symbolically and numerically: a flat order-1 mesh
    still yields six exactly-zero RB modes). Curved (isoparametric order >= 2)
    geometry is therefore about accuracy, not RB correctness: flat facets give an
    O(h^2) geometry error in the spectrum (n=2 at 573 Hz, cluster spread ~5 Hz at
    h=0.2) whereas order-2 geometry that lies on the mid-surface reaches 576.32 Hz
    with spread < 1e-3. Order-2 is recommended; use
    :func:`fem.shells.curved_sphere_mesh.make_curved_sphere_mesh` or a gmsh
    `setOrder(2)` mesh.

    Caveat: `normal="analytic_sphere"` is only consistent with curved geometry that
    actually lies on the sphere; pairing it with a flat (order-1) mesh makes the
    analytic normal inconsistent with the discrete gradient and corrupts the RB modes
    (the constructor warns about this). The default `normal="geometric"` is safe at
    every order.

    The drilling rotation `theta.n` is not a shell degree of freedom: no strain
    measure constrains it, so it must be neutralised by `drilling=True` (a
    stabilisation stiffness) or by `project_rotary_inertia=True` (zero inertia,
    which sends those modes to +inf), or spurious low modes appear. Either alone
    cleans up the spectrum: on the steel sphere at h = 0.5, disabling one of the two
    leaves the first elastic mode at 576.6 Hz, whereas disabling both drops the
    lowest non-rigid mode to 69.6 Hz (and, since nothing then constrains the drilling
    rotation at all, it softens under refinement, 380 Hz at h = 0.8, 24 Hz at
    h = 0.3, eventually swamping the low spectrum).

    They are not redundant, because they do different jobs: the penalty gives the
    spurious rotation stiffness, the projection removes its mass. Only the penalty
    leaves `K` with the six-dimensional nullspace it should have: with the
    projection alone and the penalty off, the nullity of `K` is 65 at h = 0.6 and
    91 at h = 0.5, growing with the number of rotation nodes. `K + epsilon*M` still
    factorises there (the discrete drilling modes keep a small residual mass, because
    `theta = f n` is not exactly representable in the Lagrange space), but its
    condition number degrades from 4.3e7 to 2.1e11 at h = 0.5 and to 1.2e12 at
    h = 0.4, so the margin is consumed by refinement, and a static solve, which has
    no `epsilon*M` at all, is simply singular. Treat the penalty as the primary cure
    and the inertia projection as a modal-specific complement; see
    `project_rotary_inertia` on :class:`ShellElement` for why that complement
    matters on complex geometry. The penalty is objective on rigid-body motions
    (exactly zero energy) but is a genuine penalty on physical states, where it
    carries ~1e-5 of the strain energy.
    """

    def __init__(
        self,
        msh: mesh.Mesh,
        youngs_modulus: float,
        poisson_ratio: float,
        density: float,
        thickness: float,
        element: ShellElement | None = None,
        shear_correction: float = 5.0 / 6.0,
        comm: MPI.Comm = MPI.COMM_SELF,
    ) -> None:
        """
        Initialize the problem: builds the mixed function space, the surface
        geometry (normal, tangential projector, tangent frame, cell size) and
        every bilinear form the two solve paths of
        :class:`fem.shells.geometrically_exact_shell_modal_solver.GeometricallyExactShellModalSolver`
        need.

        The shell theory, the locking cure and the validated spectrum are
        documented on the class docstring; this constructor only wires up the
        DOLFINx/UFL objects from the material and element parameters.

        `msh` is the surface mesh (topological dim 2, geometric dim 3) the
        shell is defined on. `youngs_modulus` is Young's modulus in Pa,
        `poisson_ratio` is Poisson's ratio, `density` is the mass density in
        kg/m^3, and `thickness` is the shell thickness in m. `element` is the
        element / integration choice; `None` uses the `ShellElement()`
        defaults. `shear_correction` is the transverse-shear correction factor
        kappa. `comm` is the MPI communicator; it is accepted for interface
        symmetry with `msh` but not stored, since the solve paths use
        `msh.comm` directly.
        """
        self.msh = msh
        self.E = youngs_modulus
        self.nu = poisson_ratio
        self.rho = density
        self.h = thickness
        self.kappa = shear_correction
        self.el = element or ShellElement()

        self.mu = self.E / 2.0 / (1.0 + self.nu)
        lmbda = self.E * self.nu / (1.0 + self.nu) / (1.0 - 2.0 * self.nu)
        self.lmbda_ps = 2.0 * lmbda * self.mu / (lmbda + 2.0 * self.mu)

        # Full-integration degree. A stiffness integrand pairs two strains, each of
        # polynomial degree `degree - 1` in the field, so 2*(degree-1) integrates the
        # field part exactly; the remaining four degrees pay for the geometry, whose
        # normal, metric and Jacobian factors are rational on a curved element and
        # therefore not integrable exactly by any finite rule. Raising it further
        # leaves the sphere spectrum unchanged to the digits reported in the class
        # docstring, so this is the point where the rule has stopped mattering.
        self.quad_degree = 2 * self.el.degree + 2
        self.reduced_quad = (
            self.el.reduced_quad if self.el.reduced_quad is not None
            else 2 * (self.el.degree - 1)
        )

        self._bbar_cache: tuple[sp.csr_matrix, sp.csr_matrix] | None = None
        self.V = self._build_space()
        self.n, self.P, self.a1, self.a2, self.h_cell = self._build_geometry()
        self._forms = self._build_forms()
        self.ndofs = self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs
        self._scipy_assembler: ShellSystemAssembler | None = None

    # ------------------------------------------------------------------ #
    #  Public assembly interface                                          #
    # ------------------------------------------------------------------ #

    def bbar_blocks(self) -> tuple[sp.csr_matrix, sp.csr_matrix] | None:
        """
        Returns the cached (membrane, shear) B-bar stiffness blocks in scipy
        CSR, or `None` when `el.locking_cure` is not B-bar.

        Both solve paths need them under B-bar, and each costs a sparse triple
        product, so the result is computed once and cached on `self`.

        :returns: The tuple `(K_mem, K_shear)` of B-bar stiffness blocks, or
            `None`.
        """
        if self.el.locking_cure is not ShellElement.LockingCure.BBAR:
            return None
        if self._bbar_cache is None:
            self._bbar_cache = (self._bbar_block(self._forms.bbar_membrane),
                                self._bbar_block(self._forms.bbar_shear))
        return self._bbar_cache

    def assemble_scipy_system(self) -> "ShellSystemAssembler.ScipySystem":
        """
        Assembles the shell's split stiffness blocks and mass as scipy CSR.

        Used by the small-scale diagnostic solve (`solve_modes`), for its
        per-mode energy split, and by `rigid_body_energies`. The underlying
        `ShellSystemAssembler` is built once and cached on `self`, so a
        second call reuses the already-assembled matrices rather than
        reassembling.

        :returns: The split stiffness blocks, their sum, and the mass.
        :raises RuntimeError: If run on more than one rank: `ShellSystemAssembler
            ._to_scipy` pairs `PETSc.Mat.getValuesCSR`'s rank-local rows with
            `getSize`'s comm-global count, which only agree at `comm.size == 1`.
        """
        if self.msh.comm.size > 1:
            raise RuntimeError(
                "assemble_scipy_system is implemented for serial runs only "
                "(MPI.COMM_SELF); ShellSystemAssembler._to_scipy pairs "
                "rank-local getValuesCSR() rows with comm-global getSize(), "
                "which only agree at one rank."
            )
        if self._scipy_assembler is None:
            self._scipy_assembler = ShellSystemAssembler(
                forms=self._forms,
                comm=self.msh.comm,
                backend=ShellSystemAssembler.Backend.SCIPY,
                bbar_blocks=self.bbar_blocks(),
            )
        return self._scipy_assembler.assemble()

    def assemble_petsc_system(self) -> "ShellSystemAssembler.PetscSystem":
        """
        Assembles the shell's combined stiffness and mass as PETSc AIJ.

        Used by the memory-scalable `solve_modes_slepc`, which is deliberately
        not memoized here: that solve path assembles once, factorises, and
        destroys the matrices itself when the eigensolve is done, so nothing
        is served by keeping a second copy alive on `self`.

        :returns: The combined stiffness and the mass.
        """
        assembler = ShellSystemAssembler(
            forms=self._forms,
            comm=self.msh.comm,
            backend=ShellSystemAssembler.Backend.PETSC,
            bbar_blocks=self.bbar_blocks(),
        )
        return assembler.assemble()

    # ------------------------------------------------------------------ #
    #  Setup                                                               #
    # ------------------------------------------------------------------ #

    def _build_space(self) -> fem.FunctionSpace:
        """
        Builds the mixed displacement/rotation function space.

        Both fields are vector Lagrange spaces of degree `self.el.degree`
        over the mesh cell, combined into one `basix.ufl.mixed_element`.

        :returns: The mixed space the trial and test functions live in.
        """
        gdim = self.msh.geometry.dim
        cell = self.msh.basix_cell()
        Ue = basix.ufl.element("P", cell, self.el.degree, shape=(gdim,))
        Te = basix.ufl.element("P", cell, self.el.degree, shape=(gdim,))
        return fem.functionspace(self.msh, basix.ufl.mixed_element([Ue, Te]))

    def _build_geometry(self) -> tuple[Expr, Expr, Expr, Expr, fem.Function]:
        """
        Builds the per-point surface geometry the strain measures are built from.

        :returns: The surface normal `n` (per `self.el.normal`), the
            tangential projector `P = I - n(x)n`, the tangent frame vectors
            `a1` and `a2`, and the per-cell size field `h_cell`.
        """
        x = ufl.SpatialCoordinate(self.msh)
        if self.el.normal is ShellElement.NormalKind.ANALYTIC_SPHERE:
            # The analytic normal is only consistent with curved geometry that lies
            # on the sphere. On a flat (order-1) mesh it is inconsistent with the
            # discrete surface gradient and corrupts the rigid-body modes.
            nodes_per_cell = self.msh.geometry.dofmap.shape[1]
            if nodes_per_cell <= 3:  # order-1 triangle
                warnings.warn(
                    "normal='analytic_sphere' with a flat (order-1) mesh corrupts "
                    "the rigid-body modes (analytic normal inconsistent with the "
                    "flat geometry). Use order>=2 geometry, or normal='geometric'.",
                    RuntimeWarning, stacklevel=2,
                )
            normal = x / ufl.sqrt(ufl.dot(x, x))
        else:
            J = ufl.Jacobian(self.msh)
            t1 = ufl.as_vector([J[0, 0], J[1, 0], J[2, 0]])
            t2 = ufl.as_vector([J[0, 1], J[1, 1], J[2, 1]])
            ng = ufl.cross(t1, t2)
            normal = ng / ufl.sqrt(ufl.dot(ng, ng))
        projector = ufl.Identity(3) - ufl.outer(normal, normal)
        a1, a2 = self._tangent_frame(normal, projector)
        return normal, projector, a1, a2, self._cell_size()

    def _tangent_frame(self, normal: Expr, projector: Expr) -> tuple[Expr, Expr]:
        """
        Builds the pointwise orthonormal basis `(a1, a2)` of the tangent plane
        of `normal`, the surface normal, as built by :meth:`_build_geometry`,
        with its matching tangential projector `projector` (`I - n(x)n`).

        Used only to coordinatize the B-bar membrane assumed-strain space, where
        the paper's `eps_{alpha beta}` carries Greek indices, i.e. lives in the
        tangent plane. Built by projecting the first Jacobian column onto that
        plane and completing with `n x a1`, so `a1` and `a2` are orthogonal
        to `normal` exactly, whichever normal was selected. (Taking the raw
        Jacobian columns would not be orthonormal, and under
        `normal="analytic_sphere"` would not even be tangent to `n`.)

        :returns: The tangent-plane basis vectors `(a1, a2)`, as UFL expressions.
        """
        J = ufl.Jacobian(self.msh)
        t1 = ufl.as_vector([J[0, 0], J[1, 0], J[2, 0]])
        p1 = ufl.dot(projector, t1)
        a1 = p1 / ufl.sqrt(ufl.dot(p1, p1))
        return a1, ufl.cross(normal, a1)

    def _tangential_tensor(self, components: Expr) -> Expr:
        """
        Rebuilds a symmetric tangential 3x3 tensor from its three independent
        components `(E11, E12, E22)` in the `(a1, a2)` frame.

        `components` carries those three independent components, as a UFL
        expression indexable by 0, 1, 2. This is the image of the membrane
        assumed-strain space: exactly the symmetric tangential tensors, which
        is where `_Cps` is self-adjoint (it needs `<P, e> == tr(e)`, true
        iff `e` is tangential) and hence where the projected stiffness comes
        out symmetric.

        :returns: The reassembled symmetric tangential 3x3 tensor.
        """
        return (components[0] * ufl.outer(self.a1, self.a1)
                + components[1] * (ufl.outer(self.a1, self.a2)
                                   + ufl.outer(self.a2, self.a1))
                + components[2] * ufl.outer(self.a2, self.a2))

    def _cell_size(self) -> fem.Function:
        """
        Builds the per-cell characteristic size field used by PSRI and the
        drilling stabilization.

        The field is piecewise-constant (DG0), one value per cell, equal to
        the longest of the three edges of that cell's triangle.

        :returns: The DG0 max-edge-length field.
        """
        DG0 = fem.functionspace(self.msh, ("DG", 0))
        hf = fem.Function(DG0)
        x = self.msh.geometry.x
        g = self.msh.geometry.dofmap
        # First 3 geometry nodes are the triangle vertices (basix ordering), also on
        # order-2 meshes, so the max edge length is geometry-order-independent.
        v = x[g[:, :3]]
        edges = [np.linalg.norm(v[:, a] - v[:, b], axis=1)
                 for a, b in [(0, 1), (1, 2), (0, 2)]]
        hf.x.array[DG0.dofmap.list[:, 0]] = np.maximum.reduce(edges)
        return hf

    def _Cps(self, e: Expr) -> Expr:
        """
        Applies the plane-stress isotropic constitutive tensor to a tangential
        strain `e` (membrane or bending, given as a UFL expression),
        `lambda_ps * tr(e) * P + 2*mu*e`.

        :returns: The corresponding plane-stress stress tensor.
        """
        return self.lmbda_ps * ufl.tr(e) * self.P + 2.0 * self.mu * e

    def _strains(
        self, u: Expr, theta: Expr
    ) -> tuple[Expr, Expr, Expr, Expr]:
        """
        Computes the four geometrically-exact strain measures of the shell for
        the displacement field `u` and the rotation field `theta`, each given
        as a UFL expression:

            membrane   eps   = sym(P . grad_s u . P)
            shear      gamma = P . ( (grad_s u)^T . n + theta x n )
            bending    rho   = sym( P . [ (grad_s(theta x n))^T P
                                          + (grad_s n)^T grad_s u ] . P )
            drilling   w_dr  = 1/2 n.curl_s(u) - theta.n

        All four vanish for a rigid motion `u = c + omega x x`,
        `theta = omega`.

        :returns: The tuple `(eps, rho, gamma, drill)`, i.e.
            `(membrane, bending, shear, drilling)`. Note that this differs
            from the `(membrane, shear, bending, drilling)` order the
            formulas are listed in above.
        """
        P, n = self.P, self.n
        gu = ufl.grad(u)
        gn = ufl.grad(n)
        eps = ufl.sym(P * gu * P)
        d = ufl.cross(theta, n)
        gd = ufl.grad(d)
        gamma = P * (ufl.dot(ufl.transpose(gu), n) + d)
        dB = ufl.dot(ufl.transpose(gd), P) + ufl.dot(ufl.transpose(gn), gu)
        rho = ufl.sym(P * dB * P)
        curl_u = ufl.as_vector([gu[2, 1] - gu[1, 2],
                                gu[0, 2] - gu[2, 0],
                                gu[1, 0] - gu[0, 1]])
        drill = 0.5 * ufl.dot(n, curl_u) - ufl.dot(theta, n)
        return eps, rho, gamma, drill

    def _build_forms(self) -> ShellForms:
        """
        Builds every bilinear form the two solve paths need, from the strain
        measures of the trial and test functions.

        Membrane and shear each take one of three locking-cure branches,
        selected by `self.el`:

          * `BBAR`: membrane and shear leave UFL entirely and become
            element-local matrix triple products, built by
            :meth:`_build_bbar_forms` and assembled by
            :meth:`fem.shells.shell_system_assembler.ShellSystemAssembler`.
          * `PSRI`: alpha-blended full/reduced integration, with
            `alpha = min((h/h_cell)^2, 1)`.
          * otherwise: plain full integration.

        Bending is always fully integrated (projecting it would be rank
        deficient). The drilling stabilization stiffness is added when
        `self.el.drilling` is set. The mass form uses either the full or the
        projected (drilling-free) rotary inertia, per
        `self.el.project_rotary_inertia`.

        :returns: The split forms (membrane, bending, shear, drilling, mass),
            which the scipy diagnostic path uses for the per-mode energy split,
            together with the combined stiffness the PETSc/SLEPc production path
            assembles and, under B-bar, the two projection form sets.
        """
        du, dth = ufl.split(ufl.TrialFunction(self.V))
        u_, th_ = ufl.split(ufl.TestFunction(self.V))
        eps_t, rho_t, gam_t, dr_t = self._strains(du, dth)
        eps_s, rho_s, gam_s, dr_s = self._strains(u_, th_)

        def mem_dens(e: Expr) -> Expr:
            return self.h * self._Cps(e)

        def sh_dens(g: Expr) -> Expr:
            return self.kappa * self.mu * self.h * g

        dx = ufl.dx(metadata={"quadrature_degree": self.quad_degree})
        dxr = ufl.dx(metadata={"quadrature_degree": self.reduced_quad})

        bend_ufl = ufl.inner(self.h**3 / 12 * self._Cps(rho_t), rho_s) * dx
        bbar_mem, bbar_shear = None, None
        if self.el.locking_cure is ShellElement.LockingCure.BBAR:
            # Membrane and shear stiffness come from element-local B-bar triple
            # products assembled separately (see _build_bbar_forms /
            # ShellSystemAssembler), so they stay out of the UFL stiffness.
            # Bending stays full.
            bbar_mem, bbar_shear = self._build_bbar_forms(eps_t, gam_t)
            mem_ufl = None
            shear_ufl = None
        elif self.el.locking_cure is ShellElement.LockingCure.PSRI:
            alpha = ufl.min_value((self.h / self.h_cell) ** 2, 1.0)
            mem_ufl = (alpha * ufl.inner(mem_dens(eps_t), eps_s) * dx
                       + (1 - alpha) * ufl.inner(mem_dens(eps_t), eps_s) * dxr)
            shear_ufl = (alpha * ufl.dot(sh_dens(gam_t), gam_s) * dx
                         + (1 - alpha) * ufl.dot(sh_dens(gam_t), gam_s) * dxr)
        else:
            mem_ufl = ufl.inner(mem_dens(eps_t), eps_s) * dx
            shear_ufl = ufl.dot(sh_dens(gam_t), gam_s) * dx

        if self.el.drilling:
            # Drilling stabilization. `w_dr` is a dimensionless rotation, so its
            # modulus must carry the units of an energy per area: `E h^3 / h_cell^2`
            # is the bending modulus `E h^3` spread over one cell, i.e. the same
            # order as the bending energy of a cell-sized mode. That is the right
            # magnitude for a stabilization: large enough to lift the
            # unconstrained drilling rotation clear of the physical spectrum, small
            # enough not to stiffen it, and, being tied to `h_cell`, it keeps that
            # property under mesh refinement. It is a penalty, not a physical
            # stiffness: the spectrum is insensitive to `drilling_stiffness_factor`
            # over orders of magnitude (pinned by
            # `test_spectrum_is_insensitive_to_the_drilling_stiffness`).
            k_dr = (
                self.el.drilling_stiffness_factor * self.E * self.h**3 / self.h_cell**2
            )
            drill_ufl = k_dr * dr_t * dr_s * dx
        else:
            drill_ufl = None

        if self.el.project_rotary_inertia:
            # zero inertia on the drilling rotation theta.n: keep only the tangential
            # (physical bending) rotation P*theta. Matches the IGA's (I - D(x)D).
            rot_inertia = ufl.dot(ufl.dot(self.P, dth), ufl.dot(self.P, th_))
        else:
            rot_inertia = ufl.dot(dth, th_)
        mass_ufl = (self.rho * self.h * ufl.dot(du, u_)
                    + self.rho * self.h**3 / 12 * rot_inertia) * dx

        # The combined stiffness form drives the PETSc/SLEPc production path. Under
        # B-bar it holds only bending (+drilling); the B-bar membrane/shear blocks
        # are added as assembled matrices in the solve paths.
        stiff_ufl = (
            bend_ufl
            if self.el.locking_cure is ShellElement.LockingCure.BBAR
            else (mem_ufl + bend_ufl + shear_ufl)
        )
        if drill_ufl is not None:
            stiff_ufl = stiff_ufl + drill_ufl

        return ShellForms(
            membrane=fem.form(mem_ufl) if mem_ufl is not None else None,
            bending=fem.form(bend_ufl),
            shear=fem.form(shear_ufl) if shear_ufl is not None else None,
            drilling=fem.form(drill_ufl) if drill_ufl is not None else None,
            mass=fem.form(mass_ufl),
            stiffness=fem.form(stiff_ufl),
            bbar_membrane=bbar_mem,
            bbar_shear=bbar_shear,
        )

    def _build_bbar_forms(
        self, eps_trial: Expr, gam_trial: Expr
    ) -> tuple[ShellForms.BBar, ShellForms.BBar]:
        """
        Builds the element-local B-bar projection forms for membrane and shear.

        For a strain `s(u, theta)` with energy density `<D s, s>`, the B-bar
        (assumed-strain) stiffness is `B^T M_W^{-1} G M_W^{-1} B` on the per-cell
        reduced space `W = DG_{degree-1}`, where

            B   = <Q, s(u, theta)>   couples the reduced space to the primal DOFs,
            M_W = <Q, R>             reduced-space mass (block-diagonal per cell),
            G   = <D Q, R>           material-weighted reduced-space mass.

        The same projector `Pi = M_W^{-1} B` acts on trial and test, so each
        block `Pi^T G Pi` is symmetric PSD, and the rigid-body nullspace is
        preserved exactly because the projected strains still vanish for rigid
        motions (`K phi_rb = 0` to round-off, not merely
        `phi_rb^T K phi_rb = 0`). Bending is intentionally excluded (projecting
        it is rank-deficient). `scalar_modulus` stands in for `G` when `D`
        is a scalar multiple of the identity, so that `G = scalar_modulus * M_W`.

        That symmetry argument constrains what `W` may be, and the two strains
        answer differently. The shear strain is a tangent vector and its modulus
        `kappa mu h` is a positive scalar, so any three-component space works.
        The membrane strain is a symmetric tangential tensor (the paper's
        `eps_{alpha beta}`, Greek indices), and its modulus `C_ps` is
        self-adjoint only there: `<C_ps(q), r> - <q, C_ps(r)>` reduces to
        `lambda_ps (tr(q) <P, r> - tr(r) <P, q>)`, which vanishes iff
        `<P, e> == tr(e)`, i.e. iff `e` is tangential. `W` is therefore
        carried as the three independent components `(E11, E12, E22)` in the
        local orthonormal tangent frame `(a1, a2)` and rebuilt into a tensor by
        :meth:`_tangential_tensor`, rather than as unconstrained 3x3 tensors,
        which would be a strictly larger space, admitting the non-tangential and
        non-symmetric directions on which `C_ps` is not self-adjoint, and would
        leave the assembled membrane block asymmetric by ~1e-3.

        `eps_trial` and `gam_trial` are the membrane and transverse-shear
        strains of the trial function, from :meth:`_strains`.

        :returns: The membrane and shear projection form sets, in that order.
        """
        cell = self.msh.basix_cell()
        p = self.el.degree
        dx = ufl.dx(metadata={"quadrature_degree": self.quad_degree})

        Ws = fem.functionspace(
            self.msh, basix.ufl.element("DG", cell, p - 1, shape=(3,)))
        qs, rs = ufl.TestFunction(Ws), ufl.TrialFunction(Ws)
        bbar_shear = ShellForms.BBar(
            coupling=fem.form(ufl.inner(qs, gam_trial) * dx),
            reduced_mass=fem.form(ufl.inner(qs, rs) * dx),
            material_mass=None,
            scalar_modulus=self.kappa * self.mu * self.h,  # D = kappa*mu*h * I
        )

        # Three components per point, not nine: the symmetric tangential tensors.
        Wm = fem.functionspace(
            self.msh, basix.ufl.element("DG", cell, p - 1, shape=(3,)))
        qm = self._tangential_tensor(ufl.TestFunction(Wm))
        rm = self._tangential_tensor(ufl.TrialFunction(Wm))
        bbar_mem = ShellForms.BBar(
            coupling=fem.form(ufl.inner(qm, eps_trial) * dx),
            reduced_mass=fem.form(ufl.inner(qm, rm) * dx),
            # D = h * C_ps
            material_mass=fem.form(ufl.inner(self.h * self._Cps(qm), rm) * dx),
            scalar_modulus=None,
        )
        return bbar_mem, bbar_shear

    def _block_diag_inverse(self, mw: sp.spmatrix) -> sp.csr_matrix | None:
        """
        Inverts a per-cell block-diagonal matrix (the DG_{p-1} reduced-space
        mass matrix) by inverting each contiguous cell block, instead of a
        global sparse solve.

        The DG dofs are cell-local, so `M_W` is block-diagonal with uniform
        block size `n / n_cells`; batch-inverting those small blocks is
        orders of magnitude faster than `spsolve` with a wide sparse RHS.
        `mw` is the matrix to invert, expected to be block-diagonal with
        uniform, contiguous per-cell blocks.

        :returns: The inverse as scipy CSR, or `None` (so the caller falls
            back to the global solve) if the structure is not the expected
            contiguous block-diagonal.
        """
        n = mw.shape[0]
        ncells = self.msh.topology.index_map(self.msh.topology.dim).size_local
        if ncells == 0 or n % ncells != 0:
            return None
        bs = n // ncells
        coo = mw.tocoo()
        if np.any(coo.row // bs != coo.col // bs):  # not contiguous block-diagonal
            return None
        blocks = np.zeros((ncells, bs, bs))
        blocks[coo.row // bs, coo.row % bs, coo.col % bs] = coo.data
        inv = np.linalg.inv(blocks)
        cell = np.repeat(np.arange(ncells), bs * bs)
        local_r = np.tile(np.repeat(np.arange(bs), bs), ncells)
        local_c = np.tile(np.arange(bs), bs * ncells)
        return sp.csr_matrix(
            (inv.ravel(), (cell * bs + local_r, cell * bs + local_c)), shape=(n, n)
        )

    def _bbar_block(self, bbar: ShellForms.BBar) -> sp.csr_matrix:
        """
        Assembles one B-bar stiffness block `B^T M_W^{-1} G M_W^{-1} B` from
        `bbar`, the projection forms of one strain as :meth:`_build_bbar_forms`
        builds them.

        `material_mass is None` is the scalar-material shortcut: when `D` is
        a scalar multiple of the identity (as for the shear modulus), `G`
        reduces to `scalar_modulus * M_W` and the triple product simplifies to
        `scalar_modulus * B^T M_W^{-1} B`, so no `G` matrix needs to be
        assembled.

        :returns: The assembled stiffness block as scipy CSR.
        """
        B = ShellSystemAssembler._to_scipy(bbar.coupling)       # (W, V)
        mw = ShellSystemAssembler._to_scipy(bbar.reduced_mass)  # (W, W), per-cell block
        mw_inv = self._block_diag_inverse(mw)
        if mw_inv is not None:
            mw_inv_b = (mw_inv @ B).tocsr()
        else:                                   # fallback: global sparse solve
            mw_inv_b = spsolve(mw.tocsc(), B.tocsc())
            if not sp.issparse(mw_inv_b):
                mw_inv_b = sp.csr_matrix(mw_inv_b)
        if bbar.material_mass is None:          # scalar material: G = scalar * M_W
            block = bbar.scalar_modulus * (B.transpose() @ mw_inv_b)
        else:
            g = ShellSystemAssembler._to_scipy(bbar.material_mass)
            block = mw_inv_b.transpose() @ g @ mw_inv_b
        return block.tocsr()
