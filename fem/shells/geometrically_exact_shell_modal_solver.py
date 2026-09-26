import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from dolfinx import fem
from scipy.sparse.linalg import eigsh

from fem.petsc_factor_solver import FactorSolver
from fem.shells.geometrically_exact_shell_problem import GeometricallyExactShellProblem
from fem.shells.modal_result import ModalResult
from fem.shells.rigid_body_mode import RigidBodyMode
from fem.shells.rigid_body_mode_energy import RigidBodyModeEnergy
from rebel_shape import logger


class GeometricallyExactShellModalSolver:
    """
    Solves the generalized eigenvalue problem `K phi = omega^2 M phi` of a
    :class:`fem.shells.geometrically_exact_shell_problem.GeometricallyExactShellProblem`,
    by either of two paths.

    `solve_modes` assembles the split scipy stiffness blocks and mass and
    solves with `scipy.sparse.linalg.eigsh`, additionally reporting the
    per-mode membrane/bending/shear/drilling energy split; suitable for small
    meshes. `solve_modes_slepc` assembles the combined stiffness and mass as
    PETSc matrices and solves with SLEPc shift-invert and a direct
    factorization, without ever holding the split blocks; suitable for large
    meshes, and the production path.

    Both paths add the same mass-proportional spectral shift `epsilon` (see
    :meth:`_spectral_shift`) so that `K + epsilon*M` is invertible despite
    the six-dimensional rigid-body nullspace of the bare `K`, and subtract it
    from the eigenvalues afterwards, mirroring the IGA
    `rebel_shape.isogeometric_analysis.modal_analysis_solver.ModalAnalysisSolver`.
    `rigid_body_energies` is a diagnostic alongside the two solves: the
    Rayleigh quotient of each of the six analytic rigid-body fields, which
    isolates the formulation's rigid-body invariance from the eigensolver.
    """

    def __init__(self, problem: GeometricallyExactShellProblem) -> None:
        """
        `problem` is the assembled shell problem to solve.
        """
        self._problem = problem

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
        x = self._problem.msh.geometry.x
        length = float((x.max(axis=0) - x.min(axis=0)).max())
        return (1.0e-3 * self._problem.E
                / (self._problem.rho * max(length, 1.0e-30) ** 2))

    def solve_modes(
        self, num_modes: int = 20, spectral_shift: float | None = None
    ) -> ModalResult:
        """
        Runs a small-scale scipy shift-invert solve with a per-mode energy
        split.

        Takes the same mass-proportional shift as `solve_modes_slepc`, and for
        the same reason, so that the two paths differ only in the eigensolver.

        `num_modes` is the number of eigenpairs to return. `spectral_shift` is
        the shift lifting the rigid-body nullspace; `None` uses
        :meth:`_spectral_shift`.

        :returns: The eigenpairs, natural frequencies and per-mode energy
            split, as a :class:`ModalResult`.
        """
        system = self._problem.assemble_scipy_system()
        K, M = system.stiffness.tocsc(), system.mass.tocsc()
        eps = (spectral_shift if spectral_shift is not None
               else self._spectral_shift())
        vals, vecs = eigsh((K + eps * M).tocsc(), k=num_modes, M=M,
                           sigma=0.0, which="LM")
        vals = vals - eps
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
        energies = np.array([
            [float(vecs[:, i] @ (X @ vecs[:, i]))
             for X in (system.membrane, system.bending, system.shear,
                      system.drilling)]
            for i in range(vecs.shape[1])
        ])
        return ModalResult(
            eigenvalues=vals,
            frequencies=np.sqrt(np.abs(vals)) / (2 * np.pi),
            eigenvectors=vecs,
            mode_energies=energies,
            ndofs=self._problem.ndofs,
        )

    def solve_modes_slepc(
        self,
        num_modes: int = 20,
        spectral_shift: float | None = None,
        tol: float = 1.0e-9,
        max_it: int = 2000,
        factor_solver: FactorSolver = FactorSolver.MUMPS,
    ) -> ModalResult:
        """
        Runs a memory-scalable modal solve via SLEPc shift-invert with a
        direct factorization (MUMPS by default).

        Suitable for large meshes where the scipy path would be too
        memory-hungry.

        A mass-proportional shift `epsilon` is added so `K + epsilon*M` is SPD
        (the bare stiffness has a 6-dim rigid-body nullspace, so it is only
        positive semi-definite and cannot be factorized directly). Eigenvalues
        of the pair `(K + epsilon*M, M)` nearest zero are found by shift-invert
        and `epsilon` is subtracted afterwards. This mirrors the IGA
        `ModalAnalysisSolver`. No per-mode energy split is computed here.

        `num_modes` is the number of eigenpairs to return. `spectral_shift` is
        the shift lifting the rigid-body nullspace; `None` uses
        :meth:`_spectral_shift`. `tol` and `max_it` are the SLEPc convergence
        tolerance and maximum iteration count. `factor_solver` is the PETSc
        direct-factorization backend for the shift-invert; it must be
        configured into this PETSc, which `FactorSolver.require` establishes
        before anything is assembled.

        :returns: The eigenpairs and natural frequencies, as a
            :class:`ModalResult` with `mode_energies` left all-zero.
        :raises ImportError: If slepc4py is not available.
        :raises RuntimeError: If run on more than one rank, or if PETSc was not
            configured with `factor_solver`.
        """
        # Deferred: slepc4py is the one heavy dependency this module can do
        # without (`solve_modes` needs only scipy), so importing it lazily is
        # what lets the message below stand in for a bare ModuleNotFoundError.
        try:
            from slepc4py import SLEPc
        except ImportError as exc:
            raise ImportError(
                "SLEPc (slepc4py) is required for the shift-invert modal solve. "
                "Install with `conda install -c conda-forge slepc4py`, or use "
                "`solve_modes` (scipy) on a small enough mesh."
            ) from exc

        if self._problem.msh.comm.size > 1:
            raise RuntimeError(
                "solve_modes_slepc is implemented for serial runs only "
                "(MPI.COMM_SELF); the eigenvector gather assumes local == global.")

        FactorSolver.require(factor_solver.value, "the shift-invert modal solve")

        system = self._problem.assemble_petsc_system()
        K, M = system.stiffness, system.mass

        epsilon = (spectral_shift if spectral_shift is not None
                   else self._spectral_shift())
        K.axpy(epsilon, M)  # K <- K + epsilon*M, in place: SPD, and bare K is spent
        logger.info(
            f"SLEPc shift-invert ({factor_solver.value}), spectral shift "
            f"epsilon={epsilon:.4e}, num_modes={num_modes}, "
            f"ndofs={self._problem.ndofs}.")

        eps_solver = SLEPc.EPS().create(self._problem.msh.comm)
        eps_solver.setOperators(K, M)
        eps_solver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        # Generous Krylov subspace: the 6 rigid-body modes form a near-degenerate
        # cluster at the target, which a small ncv can fail to resolve fully.
        # `2 * num_modes` is SLEPc's own guidance; the `num_modes + 30` floor is
        # what covers a small request, where doubling still leaves less headroom
        # than the rigid-body cluster plus the first degenerate elastic cluster
        # (5-fold on a sphere) need.
        ncv = min(max(2 * num_modes, num_modes + 30), self._problem.ndofs)
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
        pc.setFactorSolverType(factor_solver.value)
        eps_solver.setFromOptions()
        eps_solver.solve()

        nconv = eps_solver.getConverged()
        if nconv < num_modes:
            logger.warning(f"SLEPc converged only {nconv}/{num_modes} modes.")
        n_extract = min(nconv, num_modes)
        vals = np.zeros(n_extract)
        vecs = np.zeros((self._problem.ndofs, n_extract))
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
            ndofs=self._problem.ndofs,
        )

    def rigid_body_energies(self) -> tuple[RigidBodyModeEnergy, ...]:
        """
        Computes the Rayleigh-quotient "frequency" of each analytic rigid-body
        field (should be ~0).

        :returns: One `RigidBodyModeEnergy` per
            :meth:`RigidBodyMode.standard_basis`, in that order, each
            quotient in Hz, approximately zero, or `nan` for a field whose
            modal mass is zero.
        """
        system = self._problem.assemble_scipy_system()
        Vu, u_map = self._problem.V.sub(0).collapse()
        Vt, t_map = self._problem.V.sub(1).collapse()

        return tuple(
            RigidBodyModeEnergy(
                mode=mode,
                frequency=self._rayleigh_quotient(
                    mode, Vu, u_map, Vt, t_map, system.stiffness, system.mass
                ),
            )
            for mode in RigidBodyMode.standard_basis()
        )

    def _rayleigh_quotient(
        self,
        mode: RigidBodyMode,
        Vu: fem.FunctionSpace,
        u_map: npt.NDArray[np.int32],
        Vt: fem.FunctionSpace,
        t_map: npt.NDArray[np.int32],
        K: sp.csr_matrix,
        M: sp.csr_matrix,
    ) -> float:
        """
        Rayleigh-quotient frequency [Hz] of the analytic rigid-body field `mode`.

        `Vu`/`u_map` and `Vt`/`t_map` are the collapsed displacement and
        rotation subspaces of `self._problem.V`, with their DOF maps back into
        the mixed space. `K` and `M` are the assembled combined stiffness and
        mass, in scipy CSR.

        :returns: The quotient in Hz, or `nan` if the field carries no
            modal mass.
        """
        phi = np.zeros(self._problem.ndofs)
        uf = fem.Function(Vu)
        uf.interpolate(mode.displacement)
        tf = fem.Function(Vt)
        tf.interpolate(lambda x, w=mode.rotation: np.tile(w[:, None], (1, x.shape[1])))
        phi[u_map] = uf.x.array
        phi[t_map] = tf.x.array
        tot = float(phi @ (K @ phi))
        mass = float(phi @ (M @ phi))
        if mass <= 0:
            return float("nan")
        return float(np.sqrt(max(tot, 0.0) / mass) / (2 * np.pi))
