"""
Geometrically-exact Reissner-Mindlin (Naghdi) shell modal analysis in DOLFINx.

The module provides :class:`GeometricallyExactShellSolver`, a higher-order,
locking-free shell eigensolver for a 2-manifold embedded in 3D, together with its
element and integration choice (:class:`ShellElement`), the record it returns
(:class:`ModalResult`), and :func:`make_curved_sphere_mesh`, which builds the
curved benchmark sphere the formulation is validated on. Why the formulation is
written the way it is, how locking is cured, and the spectra it reproduces are
documented on :class:`GeometricallyExactShellSolver`.

DOLFINx note: this uses ONLY identity-mapped Lagrange elements, so it sidesteps the
broken Piola-map-on-orientable-manifolds machinery (DOLFINx #3845) that rules out a
clean MITC / TDNNS / Regge element on a closed surface. Run serially; both solve
paths use the same mass-shifted shift-invert as the IGA solver.

Environment: conda `dolfinx-rebel-shape` (DOLFINx 0.9.x).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, spsolve

import ufl
import basix.ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix

from rebel_shape import logger

# PETSc's own factorization is always available; every other `MatSolverType`
# names an external package PETSc must have been configured with.
_BUILTIN_FACTOR_SOLVER = "petsc"


@dataclass
class ShellElement:
    """Finite-element / integration choice for the shell."""

    degree: int = 2
    """Lagrange degree of the displacement and rotation fields."""

    psri: bool = True
    """Partial selective reduced integration (locking cure). Ignored when
    ``bbar`` is True."""

    bbar: bool = False
    """B-bar: element-local L2 projection of the membrane and transverse-shear
    strains onto a per-cell DG_{degree-1} space (assumed-strain method, Hu et al.
    arXiv:1709.00402). Stronger, parameter-free locking cure than PSRI: it removes
    the spurious thin-limit shear/membrane constraints exactly instead of
    down-weighting them by alpha=(t/h)^2, and restores better-than-O(h^2)
    eigenvalue convergence at P3 on thin bending-dominated plates where full
    integration locks (rate ~0) and PSRI only partially relieves it. Bending is
    left fully integrated (projecting it would cause rank deficiency). Takes
    precedence over ``psri`` when True. The projection is a symmetric orthogonal
    projector, so the stiffness stays symmetric PSD with the correct 6-dim
    rigid-body nullspace (required by the shift-invert solver)."""

    reduced_quad: int | None = None
    """Quadrature degree of the PSRI reduced term; ``None`` selects the
    order-aware default 2*(degree-1)."""

    drilling: bool = True
    """Add the drilling stabilization stiffness."""

    drilling_stiffness_factor: float = 1.0
    """Scale on the drilling stabilization stiffness."""

    project_rotary_inertia: bool = True
    """Zero the inertia of the drilling rotation (theta.n), matching the IGA's
    (I - D(x)D) projection. This sends drilling modes to +inf regardless of mesh,
    so they cannot intrude into the physical spectrum on graded/complex meshes.
    With full (unprojected) inertia the drilling modes sit at a finite,
    MESH-SIZE-DEPENDENT frequency ~ (1/h_cell) sqrt(12 E / rho) that can drop into
    the spectrum where elements are large -- a risk a uniform sphere mesh hides.
    Strongly recommended True for complex geometry."""

    normal: str = "geometric"
    """Surface-normal source: ``"geometric"`` (general, from the Jacobian) or
    ``"analytic_sphere"`` (x/|x|, valid only on curved sphere geometry)."""

    def label(self) -> str:
        """
        Short human-readable tag of this element choice, for run logs.

        :returns: A string of the form ``"P2/P2+Bbar+geometric"``, naming the
            displacement/rotation degrees, the active locking cure and the normal.
        """
        s = f"P{self.degree}/P{self.degree}"
        s += "+Bbar" if self.bbar else ("+PSRI" if self.psri else "+full")
        return s + f"+{self.normal}"


@dataclass
class ModalResult:
    """Eigenpairs of one shell modal solve, as returned by both solve paths."""

    eigenvalues: np.ndarray
    """Squared angular frequencies omega^2, ascending, shape (num_modes,)."""

    frequencies: np.ndarray
    """Natural frequencies in Hz, sqrt(|eigenvalue|) / (2 pi)."""

    eigenvectors: np.ndarray
    """Mode shapes in the mixed (displacement, rotation) space, shape
    (ndofs, num_modes), column i belonging to ``eigenvalues[i]``."""

    mode_energies: np.ndarray
    """Per-mode strain-energy split (membrane, bending, shear, drilling), shape
    (num_modes, 4). Left all-zero by ``solve_modes_slepc``, which does not
    assemble the split blocks."""

    ndofs: int
    """Global number of degrees of freedom of the mixed space."""


class GeometricallyExactShellSolver:
    """
    Higher-order, locking-free Reissner-Mindlin shell modal solver on a 2-manifold
    embedded in 3D. Material is uniform (floats); extend to per-vertex fields by
    replacing the scalar constants with ``fem.Function`` objects.

    WHY THIS EXISTS
    ---------------
    The earlier comet-fenics (Bleyer) flat-facet linear shell -- P2 displacement + CR
    (Crouzeix-Raviart) rotation on FLAT triangular facets -- is locking-free for FLAT
    plates but FAILS on curved shells (e.g. a sphere): it is not invariant under
    rigid-body rotations on curved geometry, so only
    3 of the 6 rigid-body (RB) modes are zero-energy. The 3 "rotational" RB modes
    appear as spurious soft modes whose frequency scales as (1/R)*sqrt(E/rho)
    (proportional to curvature) and which pollute the whole low spectrum. This defect
    is INDEPENDENT of element order (P1/P2/P3 all fail), so simply raising the
    polynomial order does not help -- and CR is order-1-only anyway, so the
    comet-fenics locking cure cannot be raised to higher order at all.

    Two distinct mistakes in the flat-facet measures, exposed on curved geometry:
      * the transverse-shear strain  grad~(u.n) - P^T(n x theta)  carries a spurious
        +B*u term (B = curvature) relative to the geometrically-exact (grad_s u)^T n,
        which breaks TRANSLATION invariance once the normal varies;
      * the membrane strain omits the curvature coupling, which breaks ROTATION
        invariance once the geometry is curved.

    THE FIX (this class)
    --------------------
    A coordinate-free geometrically-exact Naghdi shell built from the surface normal
    n and the tangential projector P = I - n(x)n, on CURVED (isoparametric order >= 2)
    geometry that actually lies on the surface:

        membrane   eps   = sym(P . grad_s u . P)
        shear      gamma = P . ( (grad_s u)^T . n + theta x n )          # no +B*u
        bending    rho   = sym( P . [ (grad_s(theta x n))^T P
                                      + (grad_s n)^T grad_s u ] . P )     # exact dB
        drilling   w_dr  = 1/2 n.curl_s(u) - theta.n                      # 0 for RB

    These vanish for u = c + omega x x, theta = omega, so ALL SIX RB modes are exactly
    zero-energy. Locking is cured by PSRI (Partial Selective Reduced Integration,
    fenics-shells style): membrane & shear are split  alpha*full + (1-alpha)*reduced
    with alpha = (t/h_cell)^2 and an ORDER-AWARE reduced quadrature degree 2*(p-1)
    (degree 2 hourglasses at P3; 2*(p-1) does not). ``ShellElement.bbar`` selects a
    stronger, parameter-free alternative in its place; see :meth:`_build_bbar_forms`.

    Validated on the canonical unit steel sphere (R=1, E=193.05 GPa, nu=0.28,
    rho=8025.9, h=1.5875 mm): P2 and P3 both reproduce, and converge to, the analytical
    thin-shell spectrum and the rebel-shape IGA spectrum to < 0.01 %:
        6 RB modes @ 0 | n=2: 576.3 Hz (x5) | n=3: 681.0 Hz (x7) | n=4: 721.8 Hz (x9).

    GEOMETRY ORDER, NORMALS AND DRILLING
    ------------------------------------
    These geometrically-exact measures are rigid-body invariant at ANY geometry
    order when the default ``normal="geometric"`` is used (verified symbolically and
    numerically: a flat order-1 mesh still yields six exactly-zero RB modes). Curved
    (isoparametric order >= 2) geometry is therefore about ACCURACY, not RB
    correctness: flat facets give an O(h^2) geometry error in the spectrum (e.g.
    n=2 = 573 Hz, cluster spread ~5 Hz at h=0.2) whereas order-2 geometry that lies
    on the mid-surface is far more accurate (576.32 Hz, spread < 1e-3). Order-2 is
    recommended; use :func:`make_curved_sphere_mesh` or a gmsh ``setOrder(2)`` mesh.

    Caveat: ``normal="analytic_sphere"`` is only consistent with curved geometry that
    actually lies on the sphere; pairing it with a flat (order-1) mesh makes the
    analytic normal inconsistent with the discrete gradient and corrupts the RB modes
    (the constructor warns about this). The default ``normal="geometric"`` is safe at
    every order.

    The drilling rotation ``theta.n`` is not a shell degree of freedom -- no strain
    measure constrains it -- so it must be neutralised by ``drilling=True`` (a
    stabilisation stiffness) or by ``project_rotary_inertia=True`` (zero inertia,
    which sends those modes to +inf), or spurious low modes appear. Either alone
    cleans up the SPECTRUM: on the steel sphere at h = 0.5, disabling one of the two
    leaves the first elastic mode at 576.6 Hz, whereas disabling both drops the
    lowest non-rigid mode to 69.6 Hz (and, since nothing then constrains the drilling
    rotation at all, it softens under refinement -- 380 Hz at h = 0.8, 24 Hz at
    h = 0.3 -- eventually swamping the low spectrum).

    They are NOT redundant, however, because they do different jobs: the penalty
    gives the spurious rotation STIFFNESS, the projection removes its MASS. Only the
    penalty leaves ``K`` with the six-dimensional nullspace it should have -- with
    the projection alone and the penalty off, the nullity of ``K`` is 65 at h = 0.6
    and 91 at h = 0.5, growing with the number of rotation nodes. ``K + epsilon*M``
    still factorises there (the discrete drilling modes keep a small residual mass,
    because ``theta = f n`` is not exactly representable in the Lagrange space), but
    its condition number degrades from 4.3e7 to 2.1e11 at h = 0.5 and to 1.2e12 at
    h = 0.4 -- so the margin is consumed by refinement, and a STATIC solve, which
    has no ``epsilon*M`` at all, is simply singular. Treat the penalty as the primary
    cure and the inertia projection as a modal-specific complement; see
    ``project_rotary_inertia`` on :class:`ShellElement` for why that complement
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
        Initialize the solver: builds the mixed function space, the surface
        geometry (normal, tangential projector, tangent frame, cell size) and
        every bilinear form needed by the two solve paths.

        The shell theory, the locking cure and the validated spectrum are
        documented on the class docstring; this constructor only wires up the
        DOLFINx/UFL objects from the material and element parameters.

        :param msh: Surface mesh (topological dim 2, geometric dim 3) the shell
            is defined on.
        :param youngs_modulus: Young's modulus in Pa.
        :param poisson_ratio: Poisson's ratio.
        :param density: Mass density in kg/m^3.
        :param thickness: Shell thickness in m.
        :param element: Element / integration choice; ``None`` uses the
            ``ShellElement()`` defaults.
        :param shear_correction: Transverse-shear correction factor kappa.
        :param comm: MPI communicator; accepted for interface symmetry with
            ``msh`` but not stored, since the solve paths use ``msh.comm``
            directly.
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
        self._build_space()
        self._build_geometry()
        self._build_forms()
        imap = self.V.dofmap.index_map
        self.ndofs = imap.size_global * self.V.dofmap.index_map_bs
        self._scipy_assembled = False

    # -- setup ---------------------------------------------------------------
    def _build_space(self) -> None:
        """
        Builds the mixed displacement/rotation function space ``self.V``.

        Both fields are vector Lagrange spaces of degree ``self.el.degree``
        over the mesh cell, combined into one ``basix.ufl.mixed_element``.
        """
        gdim = self.msh.geometry.dim
        cell = self.msh.basix_cell()
        Ue = basix.ufl.element("P", cell, self.el.degree, shape=(gdim,))
        Te = basix.ufl.element("P", cell, self.el.degree, shape=(gdim,))
        self.V = fem.functionspace(self.msh, basix.ufl.mixed_element([Ue, Te]))

    def _build_geometry(self) -> None:
        """
        Builds the per-point surface geometry the strain measures are built from.

        Sets the surface normal ``self.n`` (per ``self.el.normal``), the
        tangential projector ``self.P = I - n(x)n``, the tangent frame
        ``(self.a1, self.a2)`` and the per-cell size field ``self.h_cell``.
        """
        x = ufl.SpatialCoordinate(self.msh)
        if self.el.normal == "analytic_sphere":
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
            self.n = x / ufl.sqrt(ufl.dot(x, x))
        else:
            J = ufl.Jacobian(self.msh)
            t1 = ufl.as_vector([J[0, 0], J[1, 0], J[2, 0]])
            t2 = ufl.as_vector([J[0, 1], J[1, 1], J[2, 1]])
            ng = ufl.cross(t1, t2)
            self.n = ng / ufl.sqrt(ufl.dot(ng, ng))
        self.P = ufl.Identity(3) - ufl.outer(self.n, self.n)
        self.a1, self.a2 = self._tangent_frame()
        self.h_cell = self._cell_size()

    def _tangent_frame(self):
        """
        Builds the pointwise orthonormal basis ``(a1, a2)`` of the tangent plane
        of ``n``.

        Used only to coordinatize the B-bar membrane assumed-strain space, where
        the paper's ``eps_{alpha beta}`` carries Greek indices -- i.e. lives in the
        tangent plane. Built by projecting the first Jacobian column onto that
        plane and completing with ``n x a1``, so ``a1`` and ``a2`` are orthogonal
        to ``self.n`` exactly, whichever normal was selected. (Taking the raw
        Jacobian columns would not be orthonormal, and under
        ``normal="analytic_sphere"`` would not even be tangent to ``n``.)

        :returns: The tangent-plane basis vectors ``(a1, a2)``, as UFL expressions.
        """
        J = ufl.Jacobian(self.msh)
        t1 = ufl.as_vector([J[0, 0], J[1, 0], J[2, 0]])
        p1 = ufl.dot(self.P, t1)
        a1 = p1 / ufl.sqrt(ufl.dot(p1, p1))
        return a1, ufl.cross(self.n, a1)

    def _tangential_tensor(self, components):
        """
        Rebuilds a symmetric tangential 3x3 tensor from its three independent
        components ``(E11, E12, E22)`` in the ``(a1, a2)`` frame.

        This is the image of the membrane assumed-strain space: exactly the
        symmetric tangential tensors, which is where ``_Cps`` is self-adjoint
        (it needs ``<P, e> == tr(e)``, true iff ``e`` is tangential) and hence
        where the projected stiffness comes out symmetric.

        :param components: The three independent components ``(E11, E12, E22)``, as
            a UFL expression indexable by 0, 1, 2.
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
        # order-2 meshes -- so the max edge length is geometry-order-independent.
        v = x[g[:, :3]]
        edges = [np.linalg.norm(v[:, a] - v[:, b], axis=1)
                 for a, b in [(0, 1), (1, 2), (0, 2)]]
        hf.x.array[DG0.dofmap.list[:, 0]] = np.maximum.reduce(edges)
        return hf

    def _Cps(self, e):
        """
        Applies the plane-stress isotropic constitutive tensor to a tangential
        strain, ``lambda_ps * tr(e) * P + 2*mu*e``.

        :param e: A tangential strain tensor (membrane or bending), as a UFL
            expression.
        :returns: The corresponding plane-stress stress tensor.
        """
        return self.lmbda_ps * ufl.tr(e) * self.P + 2.0 * self.mu * e

    def _strains(self, u, theta):
        """
        Computes the four geometrically-exact strain measures of the shell for a
        displacement field ``u`` and a rotation field ``theta``.

            membrane   eps   = sym(P . grad_s u . P)
            shear      gamma = P . ( (grad_s u)^T . n + theta x n )
            bending    rho   = sym( P . [ (grad_s(theta x n))^T P
                                          + (grad_s n)^T grad_s u ] . P )
            drilling   w_dr  = 1/2 n.curl_s(u) - theta.n

        All four vanish for a rigid motion ``u = c + omega x x``,
        ``theta = omega``.

        :param u: Displacement field, as a UFL expression.
        :param theta: Rotation field, as a UFL expression.
        :returns: The tuple ``(eps, rho, gamma, drill)``, i.e.
            ``(membrane, bending, shear, drilling)``. Note that this differs
            from the ``(membrane, shear, bending, drilling)`` order the
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

    def _build_forms(self) -> None:
        """
        Builds every bilinear form the solver needs, from the strain measures
        of the trial and test functions.

        Membrane and shear each take one of three locking-cure branches,
        selected by ``self.el``:

          * ``bbar``: membrane and shear leave UFL entirely and become
            element-local matrix triple products, built by
            :meth:`_build_bbar_forms` and assembled by :meth:`_bbar_blocks`.
          * ``psri``: alpha-blended full/reduced integration, with
            ``alpha = min((h/h_cell)^2, 1)``.
          * otherwise: plain full integration.

        Bending is always fully integrated (projecting it would be rank
        deficient). The drilling stabilization stiffness is added when
        ``self.el.drilling`` is set. The mass form uses either the full or the
        projected (drilling-free) rotary inertia, per
        ``self.el.project_rotary_inertia``.

        Stores both a split set of forms (``self._a_mem``, ``self._a_bend``,
        ``self._a_shear``, ``self._a_drill``, ``self._a_mass``), used by the
        scipy diagnostic path for the per-mode energy split, and a combined
        ``self._a_stiff``, used by the PETSc/SLEPc production path -- which
        under B-bar holds only bending (+drilling), since the B-bar
        membrane/shear blocks are assembled separately as matrices.
        """
        du, dth = ufl.split(ufl.TrialFunction(self.V))
        u_, th_ = ufl.split(ufl.TestFunction(self.V))
        eps_t, rho_t, gam_t, dr_t = self._strains(du, dth)
        eps_s, rho_s, gam_s, dr_s = self._strains(u_, th_)

        h = self.h
        def mem_dens(e):
            return h * self._Cps(e)

        def sh_dens(g):
            return self.kappa * self.mu * h * g

        dx = ufl.dx(metadata={"quadrature_degree": self.quad_degree})
        dxr = ufl.dx(metadata={"quadrature_degree": self.reduced_quad})

        bend_ufl = ufl.inner(h**3 / 12 * self._Cps(rho_t), rho_s) * dx
        if self.el.bbar:
            # Membrane and shear stiffness come from element-local B-bar triple
            # products assembled separately (see _build_bbar_forms / _bbar_blocks);
            # they are NOT placed in the UFL stiffness. Bending stays full.
            self._build_bbar_forms(eps_t, gam_t)
            mem_ufl = None
            shear_ufl = None
        elif self.el.psri:
            alpha = ufl.min_value((h / self.h_cell) ** 2, 1.0)
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
            # magnitude for a stabilization -- large enough to lift the
            # unconstrained drilling rotation clear of the physical spectrum, small
            # enough not to stiffen it -- and, being tied to `h_cell`, it keeps that
            # property under mesh refinement. It is a penalty, not a physical
            # stiffness: the spectrum is insensitive to `drilling_stiffness_factor`
            # over orders of magnitude (pinned by
            # `test_spectrum_is_insensitive_to_the_drilling_stiffness`).
            k_dr = self.el.drilling_stiffness_factor * self.E * h**3 / self.h_cell**2
            drill_ufl = k_dr * dr_t * dr_s * dx
        else:
            drill_ufl = None

        if self.el.project_rotary_inertia:
            # zero inertia on the drilling rotation theta.n: keep only the tangential
            # (physical bending) rotation P*theta. Matches the IGA's (I - D(x)D).
            rot_inertia = ufl.dot(ufl.dot(self.P, dth), ufl.dot(self.P, th_))
        else:
            rot_inertia = ufl.dot(dth, th_)
        mass_ufl = (self.rho * h * ufl.dot(du, u_)
                    + self.rho * h**3 / 12 * rot_inertia) * dx

        # Split forms drive the scipy diagnostic path (per-mode energy split).
        # Under B-bar, membrane and shear are matrix triple products, not forms.
        self._a_mem = fem.form(mem_ufl) if mem_ufl is not None else None
        self._a_bend = fem.form(bend_ufl)
        self._a_shear = fem.form(shear_ufl) if shear_ufl is not None else None
        self._a_drill = fem.form(drill_ufl) if drill_ufl is not None else None
        self._a_mass = fem.form(mass_ufl)
        # Combined stiffness form drives the PETSc/SLEPc production path. Under
        # B-bar it holds only bending (+drilling); the B-bar membrane/shear blocks
        # are added as assembled matrices in the solve paths.
        stiff_ufl = bend_ufl if self.el.bbar else (mem_ufl + bend_ufl + shear_ufl)
        if drill_ufl is not None:
            stiff_ufl = stiff_ufl + drill_ufl
        self._a_stiff = fem.form(stiff_ufl)

    @staticmethod
    def _scipy(form: fem.Form) -> sp.csr_matrix:
        """
        Assembles a compiled DOLFINx form into a scipy CSR matrix and releases
        the underlying PETSc ``Mat``.

        :param form: A compiled bilinear form. Its trial and test spaces need not
            match -- the B-bar coupling forms are rectangular.
        :returns: The assembled matrix as scipy CSR, of shape (rows, cols)
            taken from the form.
        """
        A = assemble_matrix(form)
        A.assemble()
        ai, aj, av = A.getValuesCSR()
        m, k = A.getSize()  # rows, cols (rectangular for B-bar coupling forms)
        matrix = sp.csr_matrix((av, aj, ai), shape=(m, k)).copy()
        A.destroy()  # release the PETSc Mat; its CSR data is now owned by `matrix`
        return matrix

    def _scipy_to_petsc(self, matrix: sp.csr_matrix) -> PETSc.Mat:
        """
        Wraps a square scipy CSR matrix as a PETSc AIJ matrix (serial).

        :param matrix: A square scipy CSR (or CSR-convertible) matrix.
        :returns: The same matrix as an assembled PETSc AIJ ``Mat``.
        """
        a = matrix.tocsr()
        out = PETSc.Mat().createAIJ(
            size=a.shape,
            csr=(a.indptr.astype(PETSc.IntType), a.indices.astype(PETSc.IntType),
                 a.data),
            comm=self.msh.comm,
        )
        out.assemble()
        return out

    def _build_bbar_forms(self, eps_trial, gam_trial) -> None:
        """
        Builds the element-local B-bar projection forms for membrane and shear.

        For a strain ``s(u, theta)`` with energy density ``<D s, s>``, the B-bar
        (assumed-strain) stiffness is ``B^T M_W^{-1} G M_W^{-1} B`` on the per-cell
        reduced space ``W = DG_{degree-1}``, where

            B   = <Q, s(u, theta)>   couples the reduced space to the primal DOFs,
            M_W = <Q, R>             reduced-space mass (block-diagonal per cell),
            G   = <D Q, R>           material-weighted reduced-space mass.

        The same projector ``Pi = M_W^{-1} B`` acts on trial and test, so each
        block ``Pi^T G Pi`` is symmetric PSD, and the rigid-body nullspace is
        preserved exactly because the projected strains still vanish for rigid
        motions (``K phi_rb = 0`` to round-off, not merely
        ``phi_rb^T K phi_rb = 0``). Bending is intentionally excluded (projecting
        it is rank-deficient). Stored as ``(B, M_W, G, scalar)`` tuples;
        ``scalar`` is set (and ``G`` is ``None``) when ``D`` is a scalar multiple
        of the identity, so ``G = scalar * M_W``.

        That symmetry argument constrains what ``W`` may be, and the two strains
        answer differently. The shear strain is a tangent VECTOR and its modulus
        ``kappa mu h`` is a positive scalar, so any three-component space works.
        The membrane strain is a symmetric TANGENTIAL tensor -- the paper's
        ``eps_{alpha beta}``, Greek indices -- and its modulus ``C_ps`` is
        self-adjoint only there: ``<C_ps(q), r> - <q, C_ps(r)>`` reduces to
        ``lambda_ps (tr(q) <P, r> - tr(r) <P, q>)``, which vanishes iff
        ``<P, e> == tr(e)``, i.e. iff ``e`` is tangential. ``W`` is therefore
        carried as the three independent components ``(E11, E12, E22)`` in the
        local orthonormal tangent frame ``(a1, a2)`` and rebuilt into a tensor by
        :meth:`_tangential_tensor`, rather than as unconstrained 3x3 tensors --
        which would be a strictly larger space, admitting the non-tangential and
        non-symmetric directions on which ``C_ps`` is not self-adjoint, and would
        leave the assembled membrane block asymmetric by ~1e-3.

        :param eps_trial: Membrane strain of the trial function, from
            :meth:`_strains`.
        :param gam_trial: Transverse-shear strain of the trial function, from
            :meth:`_strains`.
        """
        cell = self.msh.basix_cell()
        p = self.el.degree
        dx = ufl.dx(metadata={"quadrature_degree": self.quad_degree})

        Ws = fem.functionspace(
            self.msh, basix.ufl.element("DG", cell, p - 1, shape=(3,)))
        qs, rs = ufl.TestFunction(Ws), ufl.TrialFunction(Ws)
        self._bbar_shear = (
            fem.form(ufl.inner(qs, gam_trial) * dx),
            fem.form(ufl.inner(qs, rs) * dx),
            None,
            self.kappa * self.mu * self.h,  # D = kappa*mu*h * I
        )

        # Three components per point, not nine: the symmetric tangential tensors.
        Wm = fem.functionspace(
            self.msh, basix.ufl.element("DG", cell, p - 1, shape=(3,)))
        qm = self._tangential_tensor(ufl.TestFunction(Wm))
        rm = self._tangential_tensor(ufl.TrialFunction(Wm))
        self._bbar_mem = (
            fem.form(ufl.inner(qm, eps_trial) * dx),
            fem.form(ufl.inner(qm, rm) * dx),
            fem.form(ufl.inner(self.h * self._Cps(qm), rm) * dx),  # D = h * C_ps
            None,
        )

    def _block_diag_inverse(self, mw: sp.spmatrix) -> sp.csr_matrix | None:
        """
        Inverts a per-cell block-diagonal matrix (the DG_{p-1} reduced-space
        mass matrix) by inverting each contiguous cell block, instead of a
        global sparse solve.

        The DG dofs are cell-local, so ``M_W`` is block-diagonal with uniform
        block size ``n / n_cells``; batch-inverting those small blocks is
        orders of magnitude faster than ``spsolve`` with a wide sparse RHS.

        :param mw: The matrix to invert; expected to be block-diagonal with
            uniform, contiguous per-cell blocks.
        :returns: The inverse as scipy CSR, or ``None`` (so the caller falls
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

    def _bbar_block(
        self,
        b_form: fem.Form,
        mw_form: fem.Form,
        g_form: fem.Form | None,
        scalar: float | None,
    ) -> sp.csr_matrix:
        """
        Assembles one B-bar stiffness block ``B^T M_W^{-1} G M_W^{-1} B``.

        ``g_form is None`` is the scalar-material shortcut: when ``D`` is a
        scalar multiple of the identity (as for the shear modulus), ``G``
        reduces to ``scalar * M_W`` and the triple product simplifies to
        ``scalar * B^T M_W^{-1} B``, so no ``G`` matrix needs to be assembled.

        :param b_form: The coupling form ``B = <Q, s(u, theta)>`` between the
            reduced space and the primal (trial/test) DOFs.
        :param mw_form: The reduced-space mass form ``M_W = <Q, R>``.
        :param g_form: The material-weighted reduced-space mass form
            ``G = <D Q, R>``, or ``None`` when ``scalar`` is used instead.
        :param scalar: The scalar material modulus used when ``g_form`` is
            ``None``, or ``None`` when ``g_form`` is given.
        :returns: The assembled stiffness block as scipy CSR.
        """
        B = self._scipy(b_form)             # (W, V)
        mw = self._scipy(mw_form)           # (W, W), block-diagonal per cell
        mw_inv = self._block_diag_inverse(mw)
        if mw_inv is not None:
            mw_inv_b = (mw_inv @ B).tocsr()
        else:                               # fallback: global sparse solve
            mw_inv_b = spsolve(mw.tocsc(), B.tocsc())
            if not sp.issparse(mw_inv_b):
                mw_inv_b = sp.csr_matrix(mw_inv_b)
        if g_form is None:                  # scalar material: G = scalar * M_W
            block = scalar * (B.transpose() @ mw_inv_b)
        else:
            g = self._scipy(g_form)
            block = mw_inv_b.transpose() @ g @ mw_inv_b
        return block.tocsr()

    def _bbar_blocks(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """
        Returns the cached (membrane, shear) B-bar stiffness blocks in scipy
        CSR.

        Both solve paths need them, and each costs a sparse triple product, so
        the result is computed once and cached on ``self._bbar_cache``.

        :returns: The tuple ``(K_mem, K_shear)`` of B-bar stiffness blocks.
        """
        if self._bbar_cache is None:
            self._bbar_cache = (self._bbar_block(*self._bbar_mem),
                                self._bbar_block(*self._bbar_shear))
        return self._bbar_cache

    def _ensure_scipy_assembled(self) -> None:
        """
        Assembles the split scipy matrices on first use (diagnostic path).

        Builds four stiffness blocks (membrane/bending/shear/drilling) plus mass,
        which the small-scale ``solve_modes`` and ``rigid_body_energies`` use for
        the per-mode energy split. For large meshes prefer ``solve_modes_slepc``,
        which keeps the assembled stiffness and mass in PETSc rather than holding
        five scipy matrices at once. (Under ``bbar`` it still builds the two B-bar
        blocks in scipy -- see ``_bbar_block`` -- so the saving there is the split
        blocks and the dense-ish triple products, not scipy altogether.)
        """
        if self._scipy_assembled:
            return
        if self.el.bbar:
            self.K_mem, self.K_shear = self._bbar_blocks()
        else:
            self.K_mem = self._scipy(self._a_mem)
            self.K_shear = self._scipy(self._a_shear)
        self.K_bend = self._scipy(self._a_bend)
        self.K_drill = (self._scipy(self._a_drill) if self._a_drill is not None
                        else sp.csr_matrix(self.K_mem.shape))
        self.K = self.K_mem + self.K_bend + self.K_shear + self.K_drill
        self.M = self._scipy(self._a_mass)
        self._scipy_assembled = True

    def _spectral_shift(self) -> float:
        """
        Computes a mesh-independent mass-proportional spectral shift: a small
        fraction of the membrane eigenvalue scale E/(rho*L^2).

        Lifts the 6-dim rigid-body nullspace so K + shift*M is invertible,
        while staying well below the first elastic eigenvalue. Shared by both
        solve paths; a Kdiag/Mdiag heuristic instead scales like 1/h^4 and
        overshoots the elastic spectrum on fine meshes.

        :returns: The spectral shift epsilon.
        """
        x = self.msh.geometry.x
        length = float((x.max(axis=0) - x.min(axis=0)).max())
        return 1.0e-3 * self.E / (self.rho * max(length, 1.0e-30) ** 2)

    # -- solve ---------------------------------------------------------------
    def solve_modes(
        self, num_modes: int = 20, spectral_shift: float | None = None
    ) -> ModalResult:
        """
        Runs a small-scale scipy shift-invert solve with a per-mode energy
        split.

        Takes the same mass-proportional shift as ``solve_modes_slepc``, and for
        the same reason, so that the two paths differ only in the eigensolver.

        :param num_modes: Number of eigenpairs to return.
        :param spectral_shift: Shift lifting the rigid-body nullspace; ``None``
            uses :meth:`_spectral_shift`.
        :returns: The eigenpairs, natural frequencies and per-mode energy
            split, as a :class:`ModalResult`.
        """
        self._ensure_scipy_assembled()
        K, M = self.K.tocsc(), self.M.tocsc()
        eps = (spectral_shift if spectral_shift is not None
               else self._spectral_shift())
        vals, vecs = eigsh((K + eps * M).tocsc(), k=num_modes, M=M,
                           sigma=0.0, which="LM")
        vals = vals - eps
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
        energies = np.array([
            [float(vecs[:, i] @ (X @ vecs[:, i]))
             for X in (self.K_mem, self.K_bend, self.K_shear, self.K_drill)]
            for i in range(vecs.shape[1])
        ])
        return ModalResult(
            eigenvalues=vals,
            frequencies=np.sqrt(np.abs(vals)) / (2 * np.pi),
            eigenvectors=vecs,
            mode_energies=energies,
            ndofs=self.ndofs,
        )

    def solve_modes_slepc(
        self,
        num_modes: int = 20,
        spectral_shift: float | None = None,
        tol: float = 1.0e-9,
        max_it: int = 2000,
        factor_solver: str = "mumps",
    ) -> ModalResult:
        """
        Runs a memory-scalable modal solve via SLEPc shift-invert with a
        direct factorization (MUMPS by default).

        Suitable for large meshes where the scipy path would be too
        memory-hungry.

        A mass-proportional shift ``epsilon`` is added so ``K + epsilon*M`` is SPD
        (the bare stiffness has a 6-dim rigid-body nullspace, so it is only
        positive *semi*-definite and cannot be factorized directly). Eigenvalues
        of the pair ``(K + epsilon*M, M)`` nearest zero are found by shift-invert
        and ``epsilon`` is subtracted afterwards. This mirrors the IGA
        ``ModalAnalysisSolver``. No per-mode energy split is computed here.

        :param num_modes: Number of eigenpairs to return.
        :param spectral_shift: Shift lifting the rigid-body nullspace; ``None``
            uses :meth:`_spectral_shift`.
        :param tol: SLEPc convergence tolerance.
        :param max_it: Maximum number of SLEPc iterations.
        :param factor_solver: PETSc direct-factorization backend for the
            shift-invert (e.g. ``"mumps"``); must be configured into PETSc.
        :returns: The eigenpairs and natural frequencies, as a
            :class:`ModalResult` with ``mode_energies`` left all-zero.
        :raises ImportError: If slepc4py is not available.
        :raises RuntimeError: If run on more than one rank, or if PETSc was not
            configured with ``factor_solver``.
        """
        from dolfinx.fem.petsc import assemble_matrix as petsc_assemble_matrix

        try:
            from slepc4py import SLEPc
        except ImportError as exc:
            raise ImportError(
                "SLEPc (slepc4py) is required for the shift-invert modal solve. "
                "Install with `conda install -c conda-forge slepc4py`, or use "
                "`solve_modes` (scipy) on a small enough mesh."
            ) from exc

        if self.msh.comm.size > 1:
            raise RuntimeError(
                "solve_modes_slepc is implemented for serial runs only "
                "(MPI.COMM_SELF); the eigenvector gather assumes local == global.")

        # `PC.setFactorSolverType` is a pure setter: it accepts a misspelt or
        # unconfigured backend without complaint and only fails much later, inside
        # the factorization, with a misleading error. Check up front instead.
        if (factor_solver != _BUILTIN_FACTOR_SOLVER
                and not PETSc.Sys.hasExternalPackage(factor_solver)):
            raise RuntimeError(
                f"PETSc was not configured with the direct solver "
                f"'{factor_solver}', which is required for the shift-invert modal "
                f"solve. Install a PETSc build that provides it (the conda "
                f"environment in `dolfinx/environment.yml` ships MUMPS), or pass a "
                f"`factor_solver` this PETSc has "
                f"(`'{_BUILTIN_FACTOR_SOLVER}'` is always available).")

        K = petsc_assemble_matrix(self._a_stiff)
        K.assemble()
        if self.el.bbar:
            # _a_stiff holds only bending(+drilling) under B-bar; add the
            # element-local B-bar membrane+shear blocks (assembled in scipy).
            k_mem, k_shear = self._bbar_blocks()
            k_bbar = self._scipy_to_petsc((k_mem + k_shear).tocsr())
            K.axpy(1.0, k_bbar,
                   structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            k_bbar.destroy()
        M = petsc_assemble_matrix(self._a_mass)
        M.assemble()

        epsilon = (spectral_shift if spectral_shift is not None
                   else self._spectral_shift())
        K.axpy(epsilon, M)  # K <- K + epsilon*M (SPD); bare K no longer needed
        logger.info(
            f"SLEPc shift-invert ({factor_solver}), spectral shift "
            f"epsilon={epsilon:.4e}, num_modes={num_modes}, ndofs={self.ndofs}.")

        eps_solver = SLEPc.EPS().create(self.msh.comm)
        eps_solver.setOperators(K, M)
        eps_solver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        # Generous Krylov subspace: the 6 rigid-body modes form a near-degenerate
        # cluster at the target, which a small ncv can fail to resolve fully.
        # `2 * num_modes` is SLEPc's own guidance; the `num_modes + 30` floor is
        # what covers a small request, where doubling still leaves less headroom
        # than the rigid-body cluster plus the first degenerate elastic cluster
        # (5-fold on a sphere) need.
        ncv = min(max(2 * num_modes, num_modes + 30), self.ndofs)
        eps_solver.setDimensions(nev=num_modes, ncv=ncv)
        eps_solver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        eps_solver.setTarget(0.0)
        eps_solver.setTolerances(tol=tol, max_it=max_it)
        st = eps_solver.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        ksp = st.getKSP()
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType(factor_solver)
        eps_solver.setFromOptions()
        eps_solver.solve()

        nconv = eps_solver.getConverged()
        if nconv < num_modes:
            logger.warning(f"SLEPc converged only {nconv}/{num_modes} modes.")
        n_extract = min(nconv, num_modes)
        vals = np.zeros(n_extract)
        vecs = np.zeros((self.ndofs, n_extract))
        vr = K.createVecRight()
        for i in range(n_extract):
            lam = eps_solver.getEigenpair(i, vr)
            vals[i] = lam.real - epsilon
            vecs[:, i] = vr.array_r.real
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
        vr.destroy()
        eps_solver.destroy()
        K.destroy()
        M.destroy()
        return ModalResult(
            eigenvalues=vals,
            frequencies=np.sqrt(np.abs(vals)) / (2 * np.pi),
            eigenvectors=vecs,
            mode_energies=np.zeros((n_extract, 4)),
            ndofs=self.ndofs,
        )

    def rigid_body_energies(self) -> dict[str, float]:
        """
        Computes the Rayleigh-quotient "frequency" of each analytic rigid-body
        field (should be ~0).

        :returns: A dict with keys ``"Tx", "Ty", "Tz", "Rx", "Ry", "Rz"``
            (the three translations and three rotations), each mapping to a
            Rayleigh-quotient frequency in Hz that should be approximately
            zero, or ``nan`` when the modal mass of that field is zero.
        """
        self._ensure_scipy_assembled()
        Vu, u_map = self.V.sub(0).collapse()
        Vt, t_map = self.V.sub(1).collapse()
        fields = {
            "Tx": (lambda x: np.tile([[1.0], [0], [0]], (1, x.shape[1])), np.zeros(3)),
            "Ty": (lambda x: np.tile([[0], [1.0], [0]], (1, x.shape[1])), np.zeros(3)),
            "Tz": (lambda x: np.tile([[0], [0], [1.0]], (1, x.shape[1])), np.zeros(3)),
            "Rx": (lambda x: np.cross([1.0, 0, 0], x.T).T, np.array([1.0, 0, 0])),
            "Ry": (lambda x: np.cross([0, 1.0, 0], x.T).T, np.array([0, 1.0, 0])),
            "Rz": (lambda x: np.cross([0, 0, 1.0], x.T).T, np.array([0, 0, 1.0])),
        }
        out = {}
        for name, (u_cb, w) in fields.items():
            phi = np.zeros(self.ndofs)
            uf = fem.Function(Vu)
            uf.interpolate(u_cb)
            tf = fem.Function(Vt)
            tf.interpolate(lambda x, w=w: np.tile(w[:, None], (1, x.shape[1])))
            phi[u_map] = uf.x.array
            phi[t_map] = tf.x.array
            tot = float(phi @ (self.K @ phi))
            mass = float(phi @ (self.M @ phi))
            out[name] = (np.sqrt(max(tot, 0.0) / mass) / (2 * np.pi)
                         if mass > 0 else float("nan"))
        return out


def make_curved_sphere_mesh(radius: float, mesh_size: float, order: int = 2,
                            comm: MPI.Comm = MPI.COMM_SELF) -> mesh.Mesh:
    """
    Curved (isoparametric ``order``) triangular surface mesh of a sphere via gmsh.

    Only the OCC sphere's *surface* is meshed (the volume is never generated), so
    the result is a closed 2-manifold with no boundary facets -- the setting both
    the shell and the heat solvers run in. ``order=1`` gives flat facets;
    ``order >= 2`` adds mid-edge nodes projected onto the exact sphere, which is
    what the shell solver wants (flat facets only degrade accuracy, see the class
    docstring of :class:`GeometricallyExactShellSolver`).

    :param radius: Sphere radius in metres.
    :param mesh_size: Target element size, passed as both the gmsh min and max.
    :param order: Geometry order; 1 (flat) or >= 2 (curved).
    :param comm: MPI communicator the mesh is built on.
    :returns: DOLFINx mesh (topological dim 2, geometric dim 3).
    """
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
