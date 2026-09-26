import drjit as dr
import mitsuba as mi
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from rebel_shape.isogeometric_analysis.iga_manifold_problem import IGAManifoldProblem
from rebel_shape.meshing.composite_iga_spline_surface import CompositeIGASplineSurface
from rebel_shape.splines.iga_spline_patch import IGASplinePatch


class ReissnerMindlinModalProblem(IGAManifoldProblem):
    """
    Assembles stiffness and mass matrices for Reissner-Mindlin modal analysis
    on a composite spline surface using a 6-DOF global Cartesian formulation.

    For more details about shell theory on surfaces, see "Isogeometric shell
    analysis: The Reissner-Mindlin shell" by D.J. Benson, Y. Bazilevs, M.C. Hsu,
    T.J.R. Hughes (2010) and "Isogeometric Reissner-Mindlin shell analysis with
    exactly calculated director vectors" by W. Dornisch, S. Klinkel, B. Simeon (2013).

    Following the 10 steps from the `IGAProblem` class, we obtain the following:

    1. Problem statement (modal analysis):

        Find eigenpairs (lambda, phi) such that

            K * phi = lambda * M * phi

        where K is the stiffness matrix of the Reissner-Mindlin shell and
        M is the consistent mass matrix. The eigenvalues lambda correspond to
        squared angular frequencies omega^2 of the natural vibration modes.

    2. Unknowns and functional space:

        The primary unknowns at each control point are 6 DOFs in global Cartesian:

            v_x, v_y, v_z, omega_x, omega_y, omega_z

        where v is the mid-surface displacement and omega is the rotation vector.
        Both fields live in H^1(M) and are approximated using spline basis functions.
        The director change (linearized change of the surface normal) is derived from
        the rotation as

            w = omega x D

        where D is the unit surface normal. This automatically satisfies the
        inextensibility constraint w . D = 0. Its parametric derivative is

            w_{,alpha} = omega_{,alpha} x D + omega x D_{,alpha}

        Because the rotation is expressed in global Cartesian coordinates (constant
        basis vectors), the parametric derivative omega_{,alpha} does not generate
        Christoffel symbol corrections. Instead, the curvature of the surface enters
        through the cross products with D and D_{,alpha}.

    3. Weak formulation:

        The total potential energy combines membrane, bending, shear, and drilling
        contributions. In weak form, the bilinear form reads:

            a(u, v) = a_membrane(u, v) + a_bending(u, v) + a_shear(u, v)
                       + a_drilling(u, v)

        where u and v are the vector-valued test and trial functions with the same
        DOF layout as above. Each term is integrated over the physical surface area.
        The mass bilinear form is

            m(u, v) = int_M (rho h v_u . v_v + rho h^3/12 w_u . w_v) dA

        where the first term is translational inertia and the second is rotary inertia.

    4. Strain measures (covariant components on the mid-surface):

        Membrane strain (in-plane stretching):

            epsilon_{alpha beta} = 1/2 (a_alpha . v_{,beta} + a_beta . v_{,alpha})

        Bending strain (curvature change):

            kappa_{alpha beta} = 1/2 (a_alpha . w_{,beta} + a_beta . w_{,alpha}
                                    + D_{,alpha} . v_{,beta} + D_{,beta} . v_{,alpha})

        The last two terms couple the displacement to the curvature through the
        normal derivatives D_{,alpha}. These are essential for curved shells and
        vanish identically for flat plates where D is constant.

        Using the director change w = omega x D and the scalar triple product
        identity a . (b x c) = b . (c x a), the bending strain decomposes into
        three contributions:

            - Rotation gradient: omega_{,beta} . (D x a_alpha)
            - Rotation value (curvature coupling): omega . (D_{,beta} x a_alpha)
            - Displacement (curvature coupling): D_{,alpha} . v_{,beta}

        The vectors L_alpha = D x a_alpha are the "bending lever" vectors that
        convert rotation gradients into curvature changes.

        Transverse shear strain:

            gamma_alpha = a_alpha . w + D . v_{,alpha}
                        = omega . L_alpha + D . v_{,alpha}

    5. Constitutive matrices:

        For isotropic materials with Young's modulus E and Poisson ratio nu,
        the constitutive law is expressed in covariant components using the
        plane-stress modified Lame parameters:

            mu = E / (2 (1 + nu))
            lambda = E nu / ((1 + nu)(1 - 2 nu))
            lambda_bar = 2 lambda mu / (lambda + 2 mu)

        The elastic energy density for a covariant strain e_{alpha beta} with
        thickness scale factor s (s = h for membrane, s = h^3/12 for bending) is:

            W(e) = s * (lambda_bar (g^{ab} e_{ab})^2
                       + 2 mu g^{ac} e_{cd} g^{db} e_{ba})

        where g^{alpha beta} is the contravariant (inverse) metric tensor, used
        to raise indices. On a flat surface with orthonormal coordinates
        (g^{ab} = delta^{ab}), this reduces to the standard plane-stress
        constitutive matrix D = E/(1-nu^2) [[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]].

        The transverse shear stiffness uses:

            W_shear = kappa_s mu h g^{alpha beta} gamma_alpha gamma_beta

        where kappa_s = 5/6 is the Timoshenko shear correction factor and
        the inverse metric raises the covariant shear strain indices.

    5.5. Drilling stabilization:

        The drilling rotation omega . D has no physical stiffness in the
        Reissner-Mindlin theory. An objective drilling penalty ties it to the
        physical in-plane rotation extracted from the displacement gradient:

            omega_phys = 1/2 (v_{,1} . a^2 - v_{,2} . a^1)

        where a^alpha = g^{alpha beta} a_beta are the contravariant tangent vectors.
        The penalty energy is:

            W_drill = alpha * (omega . D - omega_phys)^2

        with alpha = drilling_stiffness_factor * mu * h. This formulation is
        "objective" in the sense that it vanishes for rigid body motions (where
        the drilling rotation exactly matches the displacement-derived rotation),
        preserving the 6 zero eigenvalues of a free-free shell.

    6. Local assembly:

        For each knot span (element) of each patch, strain arrays of shape
        (n_spans, 6 * n_basis) are assembled for all 8 covariant strain
        components (epsilon_{11}, epsilon_{22}, epsilon_{12}, kappa_{11},
        kappa_{22}, kappa_{12}, gamma_1, gamma_2). The column index maps to
        DOF (I, k) as column = I * 6 + k, where I is the basis function index
        and k is the global Cartesian component (0-5).

        For displacement DOF (I, k), k in {0,1,2}, the strain contributions are:

            epsilon_{ab}^{I,k} = 1/2 (a_{a,k} dN_I/dxi^b + a_{b,k} dN_I/dxi^a)
            kappa_{ab}^{I,k} = 1/2 (D_{a,k} dN_I/dxi^b + D_{b,k} dN_I/dxi^a)
            gamma_a^{I,k} = D_k dN_I/dxi^a

        For rotation DOF (I, 3+k), k in {0,1,2}, the strain contributions are:

            kappa_{ab}^{I,3+k} = 1/2 (L_{a,k} dN_I/dxi^b + L_{b,k} dN_I/dxi^a)
                + 1/2 ((D_{,b} x a_a)_k + (D_{,a} x a_b)_k) N_I
            gamma_a^{I,3+k} = L_{a,k} N_I

        where L_alpha = D x a_alpha and (A x B)_k = A_{k+1} B_{k+2} - A_{k+2} B_{k+1}
        with cyclic indexing.

        These strain arrays are then contracted row by row through the covariant
        elastic bilinear form (Section 5) and scattered into the local stiffness
        matrix using `_scatter_values_to_local_matrices` from `IGAManifoldProblem`.

        The drilling penalty operator is assembled separately within the same
        quadrature loop and added to the stiffness matrix.

        Time complexity: O(n_patches * n_knot_spans * (6 * n_basis)^2 *
        n_quadrature_points^2), where n_basis ~ (degree + 1)^2. The dominant
        cost is the row-by-row elastic contraction, which scales linearly with
        the total number of DOFs per knot span (6 * n_basis).

    7. Global assembly:

        Local matrices from all patches are concatenated using the
        `_concatenate_patch_local_matrices` method from `IGAManifoldProblem`
        with `dofs_per_cp=6`, which remaps local DOF indices to global DOF
        indices via the composite spline surface mapping. The global sparse
        matrix is then assembled by `IGAProblem.assemble_global_matrix`.

    8. Boundary conditions:

        For a free-free shell, no boundary conditions are imposed. Dirichlet
        conditions can be applied externally by constraining specific DOFs.
        Since all 6 DOFs are in global Cartesian coordinates, applying constraints
        is straightforward (no local-to-global rotation needed).

    9. Eigenvalue problem:

        The generalized eigenvalue problem K * phi = lambda * M * phi is solved
        using sparse solvers (e.g. scipy.sparse.linalg.eigsh with shift-invert).
        The first 6 eigenvalues of a free-free shell correspond to rigid body
        modes (3 translations + 3 rotations) and should be near zero.

        The mass matrix uses projected rotational inertia:

            M_rot^{IJ}_{ij} = rho h^3/12 N_I N_J (delta_{ij} - D_i D_j)

        The projection (I - D D^T) gives zero mass to the drilling rotation,
        which pushes spurious drilling modes to infinite eigenvalue (they do not
        appear in the spectrum at all for shift-invert solvers targeting sigma=0).

    10. Validation:

        The implementation is validated against:
        a) Symmetry of the assembled stiffness and mass matrices.
        b) Linear scaling of membrane stiffness with thickness h.
        c) Correct shear correction factor application.
        d) Six rigid body modes on a free-free flat plate and cubed sphere.
        e) Analytical eigenvalues for a free-free square plate (Leissa, 1969,
           Table 4.4: lambda_1 = 13.49 for a/b = 1, nu = 0.3).
        f) Cubed sphere eigenspectrum: 6 degenerate rigid modes, then 5-fold
           degenerate first elastic mode (l=2 spherical harmonic), matching
           known reference values for a steel sphere (E = 193.05e9, nu = 0.28,
           rho = 8025.9, h = 1.5875e-3, R = 0.1135: f_1 ~ 5064 Hz).
    """

    DISPLACEMENT_X = 0
    DISPLACEMENT_Y = 1
    DISPLACEMENT_Z = 2
    ROTATION_X = 3
    ROTATION_Y = 4
    ROTATION_Z = 5

    DOFS_PER_CONTROL_POINT = 6

    def __init__(
        self,
        composite_iga_spline: CompositeIGASplineSurface,
        youngs_modulus: "mi.TensorXf",
        poisson_ratio: "mi.TensorXf",
        density: "mi.TensorXf",
        thickness: "mi.TensorXf",
        shear_correction_factor: float = 5.0 / 6.0,
        drilling_stiffness_factor: float = 1.0,
        integration_order: int | None = None,
        max_batch_size: int | None = None,
    ) -> None:
        """
        Initialize the Reissner-Mindlin modal problem with a composite spline
        surface and material properties.

        The problem is a generalized eigenvalue problem K phi = lambda M phi
        on the `composite_iga_spline` surface, where K is the shell stiffness
        matrix and M is the consistent mass matrix.

        All scalar material fields (`youngs_modulus`, `poisson_ratio`, `density`,
        `thickness`) must be 2D tensors of shape (n_control_points, 1), where
        n_control_points is the number of global control points in the composite
        spline surface. This allows spatially varying material properties.

        All patches in the composite spline surface must have the same degree,
        so that all local matrices have the same shape and can be concatenated
        into a single tensor.

        `shear_correction_factor` is the Timoshenko correction for transverse
        shear (default 5/6 for homogeneous cross-sections).

        `drilling_stiffness_factor` scales the objective drilling penalty as
        `drilling_stiffness_factor * G * h`. The objective formulation ties
        the drilling rotation to the physical in-plane rotation from the
        displacement gradient, so this factor can be O(1) without affecting
        physical modes. Set to 0.0 to disable drilling stabilization entirely
        (this will result in a singular stiffness matrix).

        `integration_order` is the order of Gauss-Legendre quadrature used
        for numerical integration along both parametric directions. If not
        specified, it defaults to 2 * degree + 1, which provides sufficient
        accuracy for the rational integrands that arise from the surface
        curvature (normal derivatives, inverse metric, sqrt(det(g))). The
        bending curvature-rotation coupling terms have effective integrand
        complexity of ~6p on curved surfaces, requiring higher quadrature
        than the standard degree + 1 used for flat-plate assembly.

        `max_batch_size` controls memory-efficient batched assembly. When
        set, the assembly processes at most this many knot spans at a time,
        converting each batch to a scipy sparse matrix and accumulating
        incrementally. This bounds peak assembly memory to
        O(max_batch_size * n_dofs_per_span^2) instead of
        O(n_total_spans * n_dofs_per_span^2). A value of 1024 is
        recommended for large meshes (64x64+ knot spans per patch). When
        None (default), the standard assembly path is used.

        :raises ValueError: If the field values of the composite spline do not
            have 6 DOFs per control point, if material fields have wrong shape,
            if patches have different degrees, or if parameters are out of range.
        """
        n_dofs = composite_iga_spline.global_field_values.shape[0]
        dofs_per_cp = composite_iga_spline.global_field_values.shape[1]

        if dofs_per_cp != self.DOFS_PER_CONTROL_POINT:
            raise ValueError(
                "Expected global field values to have shape (n_dofs, 6). "
                f"Got {composite_iga_spline.global_field_values.shape}."
            )

        for field, name in [
            (youngs_modulus, "youngs_modulus"),
            (poisson_ratio, "poisson_ratio"),
            (density, "density"),
            (thickness, "thickness"),
        ]:
            if len(field.shape) != 2 or field.shape[1] != 1 or field.shape[0] != n_dofs:
                raise ValueError(
                    f"`{name}` must have shape ({n_dofs}, 1). Got {field.shape}."
                )

        nu_values = poisson_ratio.numpy()
        if np.any(nu_values >= 0.5) or np.any(nu_values <= -1.0):
            raise ValueError(
                "Poisson ratio values must be in (-1.0, 0.5). "
                f"Found range [{nu_values.min():.6f}, {nu_values.max():.6f}]."
            )

        degree = composite_iga_spline.degree
        for patch in composite_iga_spline:
            if patch.degree != degree:
                raise ValueError(
                    "All patches in the composite spline surface must have the "
                    f"same degree. The global surface has degree {degree}, but "
                    f"patch {patch.name} has degree {patch.degree}."
                )

        if integration_order is None:
            integration_order = 2 * degree + 1
        if integration_order < 1:
            raise ValueError("`integration_order` must be at least 1.")

        if shear_correction_factor <= 0.0:
            raise ValueError("`shear_correction_factor` must be positive.")
        if drilling_stiffness_factor < 0.0:
            raise ValueError("`drilling_stiffness_factor` must be non-negative.")

        if max_batch_size is not None and max_batch_size < 1:
            raise ValueError("`max_batch_size` must be a positive integer.")

        self._composite_iga_spline = composite_iga_spline
        self._youngs_modulus = youngs_modulus
        self._poisson_ratio = poisson_ratio
        self._density = density
        self._thickness = thickness
        self._shear_correction_factor = shear_correction_factor
        self._drilling_stiffness_factor = drilling_stiffness_factor
        self._integration_order = integration_order
        self._max_batch_size = max_batch_size

    # ------------------------------------------------------------------ #
    #  Public assembly interface                                          #
    # ------------------------------------------------------------------ #

    def assemble_local_stiffness_matrices(
        self,
    ) -> tuple["mi.TensorXu", "mi.TensorXd"]:
        """
        Assembles the local stiffness matrices for Reissner-Mindlin shells,
        including membrane, bending, shear, and drilling contributions.

        This method computes the local stiffness matrices for each knot span of
        each patch that constitutes the composite spline surface. All four energy
        contributions are combined:

        - Membrane: in-plane stretching, scales with E * h.
        - Bending: curvature change, scales with E * h^3 / 12.
        - Shear: transverse shear deformation, scales with kappa_s * G * h.
        - Drilling: objective penalty on the non-physical normal rotation,
          scales with drilling_stiffness_factor * G * h.

        The local matrices have a block structure with 6 DOFs per basis function:
        [v_x, v_y, v_z, omega_x, omega_y, omega_z]. Each matrix entry couples
        strain contributions from one DOF to another through the covariant
        elasticity tensor and the inverse metric tensor.

        All assembly is performed in double precision (`mi.TensorXd`) to
        preserve bending contributions in thin-shell configurations.

        Time complexity: O(n_patches * n_knot_spans * (6 * n_basis)^2 *
        n_quadrature_points^2). For degree-3 splines with 16 basis functions
        per span, this is 96^2 = 9216 entries per span per quadrature point.

        :returns: A tuple that consists of

            - dofs_per_span: A tensor of shape (sum_p n_knot_spans_p, 6 * n_basis)
              that contains the global DOF indices for each knot span.
            - local_matrices: A double-precision tensor of shape
              (sum_p n_knot_spans_p, 6 * n_basis, 6 * n_basis) that contains the
              local stiffness matrices for each knot span,

            where p is the patch index and sum_p n_knot_spans_p is the total
            number of knot spans across all patches.
        """
        patch_local_indices: list[mi.TensorXu] = []
        patch_local_matrices: list[mi.TensorXd] = []

        for patch, mapping in self._composite_iga_spline.patches_with_mapping:
            knot_span_handler = self.KnotSpanHandler(
                patch=patch, integration_order=self._integration_order
            )
            local_matrices = self._assemble_patch_stiffness(
                patch=patch,
                mapping=mapping,
                quadrature_points=knot_span_handler.gl_coords_and_weights(),
            )

            interp_spline = patch.interpolating_spline
            n_basis = interp_spline.basis_functions_per_span
            basis_indices = interp_spline.active_basis_function_indices(
                parametric_coords=knot_span_handler.midpoint_parametric_coordinates
            )
            dofs_per_span = dr.zeros(
                dtype=mi.TensorXu,
                shape=(basis_indices.shape[0], n_basis * 6),
            )
            for comp in range(self.DOFS_PER_CONTROL_POINT):
                dofs_per_span[:, comp :: self.DOFS_PER_CONTROL_POINT] = dr.ravel(
                    basis_indices * self.DOFS_PER_CONTROL_POINT + comp
                )

            patch_local_matrices.append(local_matrices)
            patch_local_indices.append(dofs_per_span)

        return self._concatenate_patch_local_matrices(
            composite_iga_spline=self._composite_iga_spline,
            local_indices_list=patch_local_indices,
            local_matrices_list=patch_local_matrices,
            dofs_per_cp=self.DOFS_PER_CONTROL_POINT,
            double_precision=True,
        )

    def assemble_local_mass_matrices(
        self,
    ) -> tuple["mi.TensorXu", "mi.TensorXd"]:
        """
        Assembles the local mass matrices for Reissner-Mindlin shells,
        including both translational and projected rotational inertia.

        This method computes the local mass matrices for each knot span of each
        patch that constitutes the composite spline surface. The mass matrix has
        the following structure for nodes I and J:

            M_trans^{IJ}_{ij} = rho h N_I N_J delta_{ij}  (displacement DOFs)
            M_rot^{IJ}_{ij} = rho h^3/12 N_I N_J (delta_{ij} - D_i D_j)  (rotation DOFs)

        where rho is the density, h is the thickness, N_I are basis functions,
        delta_{ij} is the Kronecker delta, and D is the unit surface normal.

        The rotational mass uses the projection (I - D D^T) which gives zero
        inertia to the drilling rotation (omega . D). This is physically correct:
        the drilling rotation does not correspond to any through-thickness motion,
        so it should carry no inertia. The practical effect is that spurious
        drilling modes have infinite eigenvalue and are invisible to shift-invert
        eigensolvers targeting the lowest modes.

        The displacement-rotation cross terms are zero because the kinetic energy
        decomposes as T = T_translational + T_rotational with no coupling.

        All assembly is performed in double precision (`mi.TensorXd`) to
        maintain consistency with the stiffness assembly.

        Time complexity: O(n_patches * n_knot_spans * n_basis^2 *
        n_quadrature_points^2), which is significantly cheaper than stiffness
        assembly because only basis function values (not derivatives) are needed.

        :returns: A tuple that consists of

            - dofs_per_span: A tensor of shape (sum_p n_knot_spans_p, 6 * n_basis)
              that contains the global DOF indices for each knot span.
            - local_matrices: A double-precision tensor of shape
              (sum_p n_knot_spans_p, 6 * n_basis, 6 * n_basis) that contains the
              local mass matrices for each knot span,

            where p is the patch index and sum_p n_knot_spans_p is the total
            number of knot spans across all patches.
        """
        patch_local_indices: list[mi.TensorXu] = []
        patch_local_matrices: list[mi.TensorXd] = []

        for patch, mapping in self._composite_iga_spline.patches_with_mapping:
            knot_span_handler = self.KnotSpanHandler(
                patch=patch, integration_order=self._integration_order
            )
            local_matrices = self._assemble_patch_mass(
                patch=patch,
                mapping=mapping,
                quadrature_points=knot_span_handler.gl_coords_and_weights(),
            )

            interp_spline = patch.interpolating_spline
            n_basis = interp_spline.basis_functions_per_span
            basis_indices = interp_spline.active_basis_function_indices(
                parametric_coords=knot_span_handler.midpoint_parametric_coordinates
            )
            dofs_per_span = dr.zeros(
                dtype=mi.TensorXu,
                shape=(basis_indices.shape[0], n_basis * 6),
            )
            for comp in range(self.DOFS_PER_CONTROL_POINT):
                dofs_per_span[:, comp :: self.DOFS_PER_CONTROL_POINT] = dr.ravel(
                    basis_indices * self.DOFS_PER_CONTROL_POINT + comp
                )

            patch_local_matrices.append(local_matrices)
            patch_local_indices.append(dofs_per_span)

        return self._concatenate_patch_local_matrices(
            composite_iga_spline=self._composite_iga_spline,
            local_indices_list=patch_local_indices,
            local_matrices_list=patch_local_matrices,
            dofs_per_cp=self.DOFS_PER_CONTROL_POINT,
            double_precision=True,
        )

    def assemble_global_stiffness_matrix(self) -> sp.csr_matrix:
        """
        Assembles the symmetrized global stiffness matrix in CSR format.

        When `max_batch_size` is `None`, this delegates to
        `assemble_local_stiffness_matrices` followed by
        `assemble_global_matrix`. When `max_batch_size` is set, knot spans
        are processed in batches and scattered directly into a global CSR
        accumulator, bounding peak memory to O(batch_size * n_dofs^2).

        :returns: The symmetrized global stiffness matrix in CSR format.
        """
        if self._max_batch_size is None:
            indices, matrices = self.assemble_local_stiffness_matrices()
            K = self.assemble_global_matrix(
                local_indices=indices, local_matrices=matrices
            )

            K += K.T
            K *= 0.5
            return K

        n_global_dofs = (
            self._composite_iga_spline.global_field_values.shape[0]
            * self.DOFS_PER_CONTROL_POINT
        )
        K = sp.csr_matrix((n_global_dofs, n_global_dofs), dtype=np.float64)

        for patch, mapping in self._composite_iga_spline.patches_with_mapping:
            knot_span_handler = self.KnotSpanHandler(
                patch=patch, integration_order=self._integration_order
            )
            n_spans = knot_span_handler.num_knot_spans
            n_basis = patch.interpolating_spline.basis_functions_per_span
            n_dofs = n_basis * self.DOFS_PER_CONTROL_POINT

            dof_map = self._batched_global_dof_map(patch, mapping, knot_span_handler)
            all_gl = knot_span_handler.gl_coords_and_weights()

            for b_start in range(0, n_spans, self._max_batch_size):
                b_end = min(b_start + self._max_batch_size, n_spans)
                local_mat = self._assemble_patch_stiffness(
                    patch=patch,
                    mapping=mapping,
                    quadrature_points=all_gl,
                    span_slice=slice(b_start, b_end),
                )
                K = self._scatter_batch_to_global(
                    K, local_mat, dof_map[b_start:b_end], n_dofs, n_global_dofs
                )
                del local_mat
                dr.eval()

        K += K.T
        K *= 0.5
        return K

    def assemble_global_mass_matrix(self) -> sp.csr_matrix:
        """
        Assembles the symmetrized global mass matrix in CSR format.

        When `max_batch_size` is `None`, this delegates to
        `assemble_local_mass_matrices` followed by
        `assemble_global_matrix`. When `max_batch_size` is set, knot spans
        are processed in batches and scattered directly into a global CSR
        accumulator, bounding peak memory to O(batch_size * n_dofs^2).

        :returns: The symmetrized global mass matrix in CSR format.
        """
        if self._max_batch_size is None:
            indices, matrices = self.assemble_local_mass_matrices()
            M = self.assemble_global_matrix(
                local_indices=indices, local_matrices=matrices
            )

            M += M.T
            M *= 0.5
            return M

        n_global_dofs = (
            self._composite_iga_spline.global_field_values.shape[0]
            * self.DOFS_PER_CONTROL_POINT
        )
        M = sp.csr_matrix((n_global_dofs, n_global_dofs), dtype=np.float64)

        for patch, mapping in self._composite_iga_spline.patches_with_mapping:
            knot_span_handler = self.KnotSpanHandler(
                patch=patch, integration_order=self._integration_order
            )
            n_spans = knot_span_handler.num_knot_spans
            n_basis = patch.interpolating_spline.basis_functions_per_span
            n_dofs = n_basis * self.DOFS_PER_CONTROL_POINT

            dof_map = self._batched_global_dof_map(patch, mapping, knot_span_handler)
            all_gl = knot_span_handler.gl_coords_and_weights()

            for b_start in range(0, n_spans, self._max_batch_size):
                b_end = min(b_start + self._max_batch_size, n_spans)
                local_mat = self._assemble_patch_mass(
                    patch=patch,
                    mapping=mapping,
                    quadrature_points=all_gl,
                    span_slice=slice(b_start, b_end),
                )
                M = self._scatter_batch_to_global(
                    M, local_mat, dof_map[b_start:b_end], n_dofs, n_global_dofs
                )
                del local_mat
                dr.eval()

        M += M.T
        M *= 0.5
        return M

    # ------------------------------------------------------------------ #
    #  Constitutive helpers                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _covariant_elastic_contraction(
        row_e11: "mi.TensorXd",
        row_e22: "mi.TensorXd",
        row_e12: "mi.TensorXd",
        col_e11: "mi.TensorXd",
        col_e22: "mi.TensorXd",
        col_e12: "mi.TensorXd",
        g11: "mi.TensorXd",
        g12: "mi.TensorXd",
        g22: "mi.TensorXd",
        lambda_bar: "mi.TensorXd",
        mu: "mi.TensorXd",
        scale: "mi.TensorXd",
    ) -> "mi.TensorXd":
        """
        Contracts two covariant strain tensors through the plane-stress
        isotropic elasticity tensor on a curved surface.

        This method computes the bilinear form that appears in the membrane and
        bending stiffness integrands. For covariant strain tensors e (row) and
        f (col), the contraction evaluates:

            scale * (lambda_bar * tr(e) * tr(f) + 2 * mu * (g . e . g) : f)

        where:
            tr(e) = g^{ab} e_{ab}  (trace using the contravariant metric)
            (g . e . g)^{ab} = g^{ac} e_{cd} g^{db}  (both indices raised)
            (A : B) = A^{ab} B_{ba}  (double contraction)

        On a flat surface with orthonormal coordinates (g^{ab} = delta^{ab}),
        this reduces to the standard plane-stress elastic energy:
            E/(1-nu^2) * (e11*f11 + e22*f22 + nu*(e11*f22 + e22*f11)
                         + (1-nu) * e12*f12)

        `row_e11`, `row_e22`, `row_e12` are the covariant components of the
        row strain for a single DOF, with shape (n_spans,).

        `col_e11`, `col_e22`, `col_e12` are the covariant components of the
        column strains for all DOFs, with shape (n_spans, n_dofs).

        `g11`, `g12`, `g22` are the contravariant (inverse) metric components.

        `lambda_bar` is the plane-stress modified first Lame parameter:
            lambda_bar = 2 * lambda * mu / (lambda + 2 * mu)
        This modification converts the 3D elasticity tensor to plane stress,
        equivalent to the standard D = E/(1-nu^2) formulation.

        `scale` is the thickness factor (h for membrane, h^3/12 for bending),
        of shape (n_spans,).

        :returns: A tensor of shape (n_spans, n_dofs) containing the contracted
            elastic energy contributions from the row DOF to all column DOFs.
        """
        # Trace: tr(e) = g^{11} e_{11} + g^{22} e_{22} + 2 g^{12} e_{12}
        tr_row = g11 * row_e11 + g22 * row_e22 + 2.0 * g12 * row_e12
        tr_col = (
            g11[:, None] * col_e11
            + g22[:, None] * col_e22
            + 2.0 * g12[:, None] * col_e12
        )

        # Double contraction: (g . e_row . g) : e_col
        # First compute g . e_row (raise first index):
        #   (g.e)^1_beta = g^{1 gamma} e_{gamma beta}
        ge_11 = g11 * row_e11 + g12 * row_e12
        ge_12 = g11 * row_e12 + g12 * row_e22
        ge_21 = g12 * row_e11 + g22 * row_e12
        ge_22 = g12 * row_e12 + g22 * row_e22

        # Then (g.e.g)^{alpha beta} = (g.e)^alpha_gamma * g^{gamma beta}
        geg_11 = ge_11 * g11 + ge_12 * g12
        geg_12 = ge_11 * g12 + ge_12 * g22
        geg_21 = ge_21 * g11 + ge_22 * g12
        geg_22 = ge_21 * g12 + ge_22 * g22

        # Contract with col strain: (g.e_row.g)^{ab} * e_col_{ba}
        #   = geg_11 * col_e11 + geg_22 * col_e22 + (geg_12 + geg_21) * col_e12
        double_contraction = (
            geg_11[:, None] * col_e11
            + geg_22[:, None] * col_e22
            + (geg_12 + geg_21)[:, None] * col_e12
        )

        return scale[:, None] * (
            lambda_bar[:, None] * tr_row[:, None] * tr_col
            + 2.0 * mu[:, None] * double_contraction
        )

    @staticmethod
    def _cross_product_component(
        a: "mi.TensorXd",
        b: "mi.TensorXd",
        k: int,
    ) -> "mi.TensorXd":
        """
        Compute the k-th Cartesian component of the cross product (a x b).

        Both a and b have shape (n, 3). Returns shape (n,).
        """
        k1 = (k + 1) % 3
        k2 = (k + 2) % 3
        return a[:, k1] * b[:, k2] - a[:, k2] * b[:, k1]

    # ------------------------------------------------------------------ #
    #  Per-patch assembly                                                 #
    # ------------------------------------------------------------------ #

    def _assemble_patch_stiffness(
        self,
        patch: IGASplinePatch,
        mapping: npt.NDArray[np.uint32],
        quadrature_points: list[tuple["mi.TensorXf", "mi.TensorXf"]],
        span_slice: slice | None = None,
    ) -> "mi.TensorXd":
        """
        Assembles local stiffness matrices for a single patch, or for a subset
        of its knot spans when `span_slice` is provided.

        This method extracts the patch-local material properties, allocates the
        local matrix tensor, and iterates over quadrature points, delegating the
        per-quadrature-point physics to `_assemble_stiffness_contribution`.

        All arithmetic is performed in double precision (`mi.TensorXd`) to
        avoid float32 roundoff in thin-shell configurations where bending
        stiffness entries (proportional to h^3/12) are many orders of magnitude
        below membrane entries (proportional to h).

        :param patch: The IGA spline patch to assemble.
        :param mapping: Index mapping from patch-local to global control points,
            of shape `(n_control_points_patch,)`.
        :param quadrature_points: List of `(param_coords, scaled_weight)`
            tuples from `KnotSpanHandler.gl_coords_and_weights()`, covering
            all knot spans of the patch.
        :param span_slice: If given, only knot spans in this range are
            processed. The quadrature point tensors are sliced accordingly.
            When `None` (default), all spans are assembled.

        :returns: A double-precision tensor of shape `(n_spans, n_dofs, n_dofs)`
            containing the local stiffness matrices, where `n_spans` is either
            the total number of knot spans or the size of the slice.
        """
        interp_spline = patch.interpolating_spline
        n_basis = interp_spline.basis_functions_per_span
        n_dofs = n_basis * self.DOFS_PER_CONTROL_POINT

        mapping_idx = dr.ravel(mi.TensorXu(mapping))
        patch_E = mi.TensorXd(self._youngs_modulus[mapping_idx])
        patch_nu = mi.TensorXd(self._poisson_ratio[mapping_idx])
        patch_h = mi.TensorXd(self._thickness[mapping_idx])

        first_pc = quadrature_points[0][0]
        if span_slice is not None:
            n_spans = span_slice.stop - span_slice.start
        else:
            n_spans = first_pc.shape[0]

        local_matrices = dr.zeros(dtype=mi.TensorXd, shape=(n_spans, n_dofs, n_dofs))

        for pc_full, sw_full in quadrature_points:
            if span_slice is not None:
                pc = mi.TensorXf(pc_full.numpy()[span_slice])
                sw = mi.TensorXf(sw_full.numpy()[span_slice])
            else:
                pc, sw = pc_full, sw_full
            self._assemble_stiffness_contribution(
                patch,
                patch_E,
                patch_nu,
                patch_h,
                pc,
                sw,
                local_matrices,
                n_dofs,
            )

        return local_matrices

    def _assemble_patch_mass(
        self,
        patch: IGASplinePatch,
        mapping: npt.NDArray[np.uint32],
        quadrature_points: list[tuple["mi.TensorXf", "mi.TensorXf"]],
        span_slice: slice | None = None,
    ) -> "mi.TensorXd":
        """
        Assembles local mass matrices for a single patch, or for a subset of
        its knot spans when `span_slice` is provided.

        This method extracts the patch-local material properties, allocates the
        local matrix tensor, and iterates over quadrature points, delegating the
        per-quadrature-point physics to `_assemble_mass_contribution`.

        All arithmetic is performed in double precision (`mi.TensorXd`) to
        maintain consistency with the stiffness assembly and avoid float32
        roundoff in the rotational inertia terms (proportional to h^3/12).

        :param patch: The IGA spline patch to assemble.
        :param mapping: Index mapping from patch-local to global control points,
            of shape `(n_control_points_patch,)`.
        :param quadrature_points: List of `(param_coords, scaled_weight)`
            tuples from `KnotSpanHandler.gl_coords_and_weights()`, covering
            all knot spans of the patch.
        :param span_slice: If given, only knot spans in this range are
            processed. The quadrature point tensors are sliced accordingly.
            When `None` (default), all spans are assembled.

        :returns: A double-precision tensor of shape `(n_spans, n_dofs, n_dofs)`
            containing the local mass matrices, where `n_spans` is either the
            total number of knot spans or the size of the slice.
        """
        interp_spline = patch.interpolating_spline
        n_basis = interp_spline.basis_functions_per_span
        n_dofs = n_basis * self.DOFS_PER_CONTROL_POINT

        mapping_idx = dr.ravel(mi.TensorXu(mapping))
        patch_rho = mi.TensorXd(self._density[mapping_idx])
        patch_h = mi.TensorXd(self._thickness[mapping_idx])

        first_pc = quadrature_points[0][0]
        if span_slice is not None:
            n_spans = span_slice.stop - span_slice.start
        else:
            n_spans = first_pc.shape[0]

        local_matrices = dr.zeros(dtype=mi.TensorXd, shape=(n_spans, n_dofs, n_dofs))

        for pc_full, sw_full in quadrature_points:
            if span_slice is not None:
                pc = mi.TensorXf(pc_full.numpy()[span_slice])
                sw = mi.TensorXf(sw_full.numpy()[span_slice])
            else:
                pc, sw = pc_full, sw_full
            self._assemble_mass_contribution(
                patch,
                patch_rho,
                patch_h,
                pc,
                sw,
                local_matrices,
                n_dofs,
                n_basis,
            )

        return local_matrices

    # ------------------------------------------------------------------ #
    #  Per-quadrature-point physics                                       #
    # ------------------------------------------------------------------ #

    def _assemble_stiffness_contribution(
        self,
        patch: IGASplinePatch,
        patch_E: "mi.TensorXd",
        patch_nu: "mi.TensorXd",
        patch_h: "mi.TensorXd",
        param_coords: "mi.TensorXf",
        scaled_weight: "mi.TensorXf",
        local_matrices: "mi.TensorXd",
        n_dofs: int,
    ) -> None:
        """
        Accumulates the stiffness contribution from a single quadrature point
        into `local_matrices`.

        This is the core physics method for stiffness assembly. It evaluates
        basis functions, geometry, and material properties at the quadrature
        point, builds the 8 covariant strain arrays (membrane, bending, shear),
        contracts them through the elastic bilinear form, and adds the objective
        drilling penalty. The result is scattered row-by-row into
        `local_matrices` via `_scatter_values_to_local_matrices`.

        Basis function evaluations and geometric quantities are computed in
        float32 by the spline methods and immediately cast to `mi.TensorXd`.
        All subsequent arithmetic (strain construction, elastic contraction,
        drilling penalty) operates in double precision to avoid roundoff in
        thin-shell configurations.

        :param patch: The IGA spline patch providing geometry evaluation.
        :param patch_E: Young's modulus at the patch control points, of shape
            `(n_control_points_patch, 1)`. Double precision.
        :param patch_nu: Poisson's ratio at the patch control points, of shape
            `(n_control_points_patch, 1)`. Double precision.
        :param patch_h: Thickness at the patch control points, of shape
            `(n_control_points_patch, 1)`. Double precision.
        :param param_coords: Parametric coordinates of the quadrature point
            for all knot spans being processed, of shape `(n_spans, 2)`.
        :param scaled_weight: Quadrature weights scaled by the knot span area,
            of shape `(n_spans,)`.
        :param local_matrices: Double-precision accumulator tensor of shape
            `(n_spans, n_dofs, n_dofs)` that is modified in-place.
        :param n_dofs: Number of DOFs per knot span (`6 * n_basis`).
        """
        interp_spline = patch.interpolating_spline
        n_spans = param_coords.shape[0]

        # --- Basis functions (float32 → float64) ---
        basis = mi.TensorXd(
            interp_spline.evaluate_basis_functions(parametric_coords=param_coords)
        )
        basis_ders = mi.TensorXd(
            interp_spline.evaluate_basis_functions_first_derivative(
                parametric_coords=param_coords
            )
        )
        dN_du = basis_ders[:, :, 0]
        dN_dv = basis_ders[:, :, 1]

        # --- Geometry (float32 → float64) ---
        jacobian_f = patch.evaluate_jacobian(parametric_coords=param_coords)
        hessian_f = patch.evaluate_hessian(parametric_coords=param_coords)
        normal_f = patch.evaluate_normals(parametric_coords=param_coords)
        normal_ders_f = patch.normal_derivative(
            jacobian=jacobian_f, hessian=hessian_f, normal=normal_f
        )
        metric_f = patch.metric_tensor(jacobian=jacobian_f)
        inv_metric_f, det_metric_f = patch.inv_metric_tensor(metric=metric_f)

        jacobian = mi.TensorXd(jacobian_f)
        normal = mi.TensorXd(normal_f)
        normal_ders = mi.TensorXd(normal_ders_f)
        inv_metric = mi.TensorXd(inv_metric_f)
        det_metric = mi.TensorXd(det_metric_f)

        a1 = jacobian[:, :, 0]
        a2 = jacobian[:, :, 1]
        D = normal
        D_u = normal_ders[:, :, 0]
        D_v = normal_ders[:, :, 1]

        g11 = inv_metric[:, 0, 0]
        g12 = inv_metric[:, 0, 1]
        g22 = inv_metric[:, 1, 1]

        # --- Material properties at quadrature point (float64) ---
        E = mi.TensorXd(
            dr.ravel(
                interp_spline.interpolate(
                    values=patch_E, parametric_coords=param_coords
                )
            )
        )
        nu = mi.TensorXd(
            dr.ravel(
                interp_spline.interpolate(
                    values=patch_nu, parametric_coords=param_coords
                )
            )
        )
        h = mi.TensorXd(
            dr.ravel(
                interp_spline.interpolate(
                    values=patch_h, parametric_coords=param_coords
                )
            )
        )

        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        lambda_bar = 2.0 * lam * mu / (lam + 2.0 * mu)

        min_det = float(dr.min(det_metric).numpy())
        if min_det < 0.0:
            raise ValueError(
                f"Negative metric determinant ({min_det:.4e}) in stiffness "
                "assembly. Check for inverted or degenerate elements."
            )
        area_weight = dr.sqrt(det_metric) * mi.TensorXd(scaled_weight)

        # --- Bending lever vectors: L_alpha = D x a_alpha ---
        L1 = dr.empty(dtype=mi.TensorXd, shape=a1.shape)
        L2 = dr.empty(dtype=mi.TensorXd, shape=a2.shape)
        for k in range(3):
            L1[:, k] = dr.ravel(self._cross_product_component(D, a1, k))
            L2[:, k] = dr.ravel(self._cross_product_component(D, a2, k))

        # --- Build strain arrays: shape (n_spans, n_dofs) ---
        eps11 = dr.zeros(dtype=mi.TensorXd, shape=(n_spans, n_dofs))
        eps22 = dr.zeros(dtype=mi.TensorXd, shape=(n_spans, n_dofs))
        eps12 = dr.zeros(dtype=mi.TensorXd, shape=(n_spans, n_dofs))
        kap11 = dr.zeros(dtype=mi.TensorXd, shape=(n_spans, n_dofs))
        kap22 = dr.zeros(dtype=mi.TensorXd, shape=(n_spans, n_dofs))
        kap12 = dr.zeros(dtype=mi.TensorXd, shape=(n_spans, n_dofs))
        gam1 = dr.zeros(dtype=mi.TensorXd, shape=(n_spans, n_dofs))
        gam2 = dr.zeros(dtype=mi.TensorXd, shape=(n_spans, n_dofs))

        # Displacement DOFs (k = 0, 1, 2 → columns k::6)
        for k in range(3):
            a1k = a1[:, k]
            a2k = a2[:, k]
            Dk = D[:, k]
            Duk = D_u[:, k]
            Dvk = D_v[:, k]

            # Membrane: eps_{ab} = 1/2 (a_{a,k} dN/dxi^b + a_{b,k} dN/dxi^a)
            eps11[:, k::6] = dr.ravel(a1k[:, None] * dN_du)
            eps22[:, k::6] = dr.ravel(a2k[:, None] * dN_dv)
            eps12[:, k::6] = dr.ravel(
                0.5 * (a1k[:, None] * dN_dv + a2k[:, None] * dN_du)
            )

            # Bending from displacement (curvature-displacement coupling):
            # kap_{ab} = 1/2 (D_{a,k} dN/dxi^b + D_{b,k} dN/dxi^a)
            kap11[:, k::6] = dr.ravel(Duk[:, None] * dN_du)
            kap22[:, k::6] = dr.ravel(Dvk[:, None] * dN_dv)
            kap12[:, k::6] = dr.ravel(
                0.5 * (Duk[:, None] * dN_dv + Dvk[:, None] * dN_du)
            )

            # Shear from displacement: gam_a = D_k * dN/dxi^a
            gam1[:, k::6] = dr.ravel(Dk[:, None] * dN_du)
            gam2[:, k::6] = dr.ravel(Dk[:, None] * dN_dv)

        # Rotation DOFs (k = 0, 1, 2 → columns (3+k)::6)
        for k in range(3):
            L1k = L1[:, k]
            L2k = L2[:, k]

            # Curvature-rotation coupling: (D_{,beta} x a_alpha)_k
            DuXa1_k = self._cross_product_component(D_u, a1, k)
            DuXa2_k = self._cross_product_component(D_u, a2, k)
            DvXa1_k = self._cross_product_component(D_v, a1, k)
            DvXa2_k = self._cross_product_component(D_v, a2, k)

            # Bending from rotation gradient: L_{a,k} * dN/dxi^b
            # Plus bending from rotation value: (D_{,b} x a_a)_k * N
            kap11[:, (3 + k) :: 6] = dr.ravel(
                L1k[:, None] * dN_du + DuXa1_k[:, None] * basis
            )
            kap22[:, (3 + k) :: 6] = dr.ravel(
                L2k[:, None] * dN_dv + DvXa2_k[:, None] * basis
            )
            kap12[:, (3 + k) :: 6] = dr.ravel(
                0.5 * (L1k[:, None] * dN_dv + L2k[:, None] * dN_du)
                + 0.5 * (DvXa1_k[:, None] + DuXa2_k[:, None]) * basis
            )

            # Shear from rotation: gam_a = L_{a,k} * N
            gam1[:, (3 + k) :: 6] = dr.ravel(L1k[:, None] * basis)
            gam2[:, (3 + k) :: 6] = dr.ravel(L2k[:, None] * basis)

        # --- Materialize strain arrays before contraction ---
        # DrJit builds a lazy computation graph; evaluating here
        # prevents the graph from growing unboundedly through the
        # row loop below.
        dr.eval(eps11, eps22, eps12, kap11, kap22, kap12, gam1, gam2)

        # --- Contract strains into stiffness matrix ---
        weighted_area = area_weight[:, None]
        h3_12 = h * h * h / 12.0
        shear_scale = self._shear_correction_factor * mu * h

        for row in range(n_dofs):
            values = self._covariant_elastic_contraction(
                eps11[:, row],
                eps22[:, row],
                eps12[:, row],
                eps11,
                eps22,
                eps12,
                g11,
                g12,
                g22,
                lambda_bar,
                mu,
                h,
            )

            values += self._covariant_elastic_contraction(
                kap11[:, row],
                kap22[:, row],
                kap12[:, row],
                kap11,
                kap22,
                kap12,
                g11,
                g12,
                g22,
                lambda_bar,
                mu,
                h3_12,
            )

            row_gam1 = gam1[:, row]
            row_gam2 = gam2[:, row]
            values += shear_scale[:, None] * (
                row_gam1[:, None] * (g11[:, None] * gam1 + g12[:, None] * gam2)
                + row_gam2[:, None] * (g12[:, None] * gam1 + g22[:, None] * gam2)
            )

            values *= weighted_area

            self._scatter_values_to_local_matrices(
                local_matrices=local_matrices,
                values=values,
                num_knot_spans=n_spans,
                num_dofs=n_dofs,
                row=row,
            )

        # Flush the computation graph after the contraction loop
        # to prevent memory accumulation across quadrature points.
        dr.eval(local_matrices)

        # --- Objective drilling penalty ---
        if self._drilling_stiffness_factor > 0.0:
            drill_op = dr.zeros(dtype=mi.TensorXd, shape=(n_spans, n_dofs))
            
            for k in range(3):
                Dxa_up1_k = g11 * L1[:, k] + g12 * L2[:, k]  # (D x a^1)_k
                Dxa_up2_k = g12 * L1[:, k] + g22 * L2[:, k]  # (D x a^2)_k
                
                # Displacement contribution: -omega_phys
                drill_op[:, k :: 6] = dr.ravel(
                    -0.5 * (dN_du * Dxa_up1_k[:, None] + dN_dv * Dxa_up2_k[:, None])
                )
                
                # Rotation contribution: + omega · D
                drill_op[:, (3 + k) :: 6] = dr.ravel(basis * D[:, k][:, None])

            dr.eval(drill_op)

            drill_weight = area_weight * self._drilling_stiffness_factor * mu * h

            for row in range(n_dofs):
                row_values = (
                    drill_op * drill_op[:, row][:, None] * drill_weight[:, None]
                )
                self._scatter_values_to_local_matrices(
                    local_matrices=local_matrices,
                    values=row_values,
                    num_knot_spans=n_spans,
                    num_dofs=n_dofs,
                    row=row,
                )

            dr.eval(local_matrices)

    def _assemble_mass_contribution(
        self,
        patch: IGASplinePatch,
        patch_rho: "mi.TensorXd",
        patch_h: "mi.TensorXd",
        param_coords: "mi.TensorXf",
        scaled_weight: "mi.TensorXf",
        local_matrices: "mi.TensorXd",
        n_dofs: int,
        n_basis: int,
    ) -> None:
        """
        Accumulates the mass contribution from a single quadrature point into
        `local_matrices`.

        This is the core physics method for mass assembly. It evaluates basis
        functions, geometry, and material properties at the quadrature point,
        then assembles both the translational inertia (`rho * h * N_I N_J
        delta_{ij}`) and the projected rotational inertia (`rho * h^3/12 *
        N_I N_J (delta_{ij} - D_i D_j)`). The result is scattered row-by-row
        into `local_matrices` via `_scatter_values_to_local_matrices`.

        Basis function evaluations and geometric quantities are computed in
        float32 by the spline methods and immediately cast to `mi.TensorXd`.
        All subsequent arithmetic operates in double precision.

        :param patch: The IGA spline patch providing geometry evaluation.
        :param patch_rho: Density at the patch control points, of shape
            `(n_control_points_patch, 1)`. Double precision.
        :param patch_h: Thickness at the patch control points, of shape
            `(n_control_points_patch, 1)`. Double precision.
        :param param_coords: Parametric coordinates of the quadrature point
            for all knot spans being processed, of shape `(n_spans, 2)`.
        :param scaled_weight: Quadrature weights scaled by the knot span area,
            of shape `(n_spans,)`.
        :param local_matrices: Double-precision accumulator tensor of shape
            `(n_spans, n_dofs, n_dofs)` that is modified in-place.
        :param n_dofs: Number of DOFs per knot span (`6 * n_basis`).
        :param n_basis: Number of basis functions per knot span.
        """
        interp_spline = patch.interpolating_spline
        n_spans = param_coords.shape[0]

        # --- Basis functions (float32 → float64) ---
        basis = mi.TensorXd(
            interp_spline.evaluate_basis_functions(parametric_coords=param_coords)
        )

        # --- Material properties at quadrature point (float64) ---
        rho = mi.TensorXd(
            dr.ravel(
                interp_spline.interpolate(
                    values=patch_rho, parametric_coords=param_coords
                )
            )
        )
        h = mi.TensorXd(
            dr.ravel(
                interp_spline.interpolate(
                    values=patch_h, parametric_coords=param_coords
                )
            )
        )

        # --- Geometry (float32 → float64) ---
        jacobian_f = patch.evaluate_jacobian(parametric_coords=param_coords)
        normal_f = patch.evaluate_normals(parametric_coords=param_coords)
        metric_f = patch.metric_tensor(jacobian=jacobian_f)
        _, det_metric_f = patch.inv_metric_tensor(metric=metric_f)

        D = mi.TensorXd(normal_f)
        det_metric = mi.TensorXd(det_metric_f)
        min_det = float(dr.min(det_metric).numpy())
        if min_det < 0.0:
            raise ValueError(
                f"Negative metric determinant ({min_det:.4e}) in mass "
                "assembly. Check for inverted or degenerate elements."
            )
        area_weight = dr.sqrt(det_metric) * mi.TensorXd(scaled_weight)

        mass_trans = (rho * h)[:, None]
        mass_rot = (rho * h * h * h / 12.0)[:, None]

        for i in range(n_basis):
            NiNj = basis[:, i][:, None] * basis
            weighted_NiNj = NiNj * area_weight[:, None]

            trans_val = weighted_NiNj * mass_trans
            for k in range(3):
                row = i * 6 + k
                row_values = dr.zeros(dtype=mi.TensorXd, shape=(n_spans, n_dofs))
                row_values[:, k::6] = dr.ravel(trans_val)
                self._scatter_values_to_local_matrices(
                    local_matrices=local_matrices,
                    values=row_values,
                    num_knot_spans=n_spans,
                    num_dofs=n_dofs,
                    row=row,
                )

            rot_val = weighted_NiNj * mass_rot
            for k in range(3):
                row = i * 6 + 3 + k
                row_values = dr.zeros(dtype=mi.TensorXd, shape=(n_spans, n_dofs))
                Dk = D[:, k]
                for j in range(3):
                    Dj = D[:, j]
                    coeff = -Dk * Dj
                    if k == j:
                        coeff = coeff + 1.0
                    row_values[:, (3 + j) :: 6] = dr.ravel(
                        row_values[:, (3 + j) :: 6] + rot_val * coeff[:, None]
                    )
                self._scatter_values_to_local_matrices(
                    local_matrices=local_matrices,
                    values=row_values,
                    num_knot_spans=n_spans,
                    num_dofs=n_dofs,
                    row=row,
                )

        # Flush computation graph after each quadrature point
        dr.eval(local_matrices)

    # ------------------------------------------------------------------ #
    #  Batched assembly helpers                                           #
    # ------------------------------------------------------------------ #

    def _batched_global_dof_map(
        self,
        patch: IGASplinePatch,
        mapping: npt.NDArray[np.uint32],
        ksh: "IGAManifoldProblem.KnotSpanHandler",
    ) -> npt.NDArray[np.int64]:
        """
        Computes the global DOF indices for all knot spans of a patch, returned
        as a numpy array for use in the batched scatter-to-global path.

        :param patch: The IGA spline patch.
        :param mapping: Index mapping from patch-local to global control points.
        :param ksh: The knot span handler for this patch.

        :returns: An integer array of shape `(n_knot_spans, n_dofs)` mapping
            each local DOF slot to its global DOF index.
        """
        interp = patch.interpolating_spline
        n_basis = interp.basis_functions_per_span
        basis_idx = interp.active_basis_function_indices(
            parametric_coords=ksh.midpoint_parametric_coordinates
        )

        global_cp = mi.TensorXu(mapping[basis_idx])
        dof_map = np.zeros((ksh.num_knot_spans, n_basis * 6), dtype=np.int64)

        for comp in range(self.DOFS_PER_CONTROL_POINT):
            dof_map[:, comp :: self.DOFS_PER_CONTROL_POINT] = (
                global_cp.numpy() * self.DOFS_PER_CONTROL_POINT + comp
            )

        return dof_map

    def _scatter_batch_to_global(
        self,
        K: sp.csr_matrix,
        local_mat: "mi.TensorXd",
        dof_map_batch: npt.NDArray[np.int64],
        n_dofs: int,
        n_global_dofs: int,
    ) -> sp.csr_matrix:
        """
        Converts a batch of local matrices to a sparse COO matrix and adds it
        to the global accumulator.

        :param K: The global sparse matrix accumulator (CSR format).
        :param local_mat: Double-precision local matrices for the batch, of
            shape `(batch_size, n_dofs, n_dofs)`.
        :param dof_map_batch: Global DOF indices for the batch, of shape
            `(batch_size, n_dofs)`.
        :param n_dofs: Number of DOFs per knot span.
        :param n_global_dofs: Total number of global DOFs.

        :returns: The updated global sparse matrix (CSR format).
        """
        local_np = local_mat.numpy()
        rows = np.repeat(dof_map_batch[:, :, np.newaxis], n_dofs, axis=2)
        cols = np.repeat(dof_map_batch[:, np.newaxis, :], n_dofs, axis=1)

        K += sp.coo_matrix(
            (local_np.ravel(), (rows.ravel(), cols.ravel())),
            shape=(n_global_dofs, n_global_dofs),
        ).tocsr()

        return K
