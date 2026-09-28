from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class MeshMetricsReport:
    """
    Every recomputed validation metric for one recorded end-to-end mesh.

    Assembled from the artifacts a finished run already wrote (nothing is
    re-solved) and serialised as one entry of `metrics/recompute_metrics.json`.
    The groups are the metrics the directory README names: M1 (mode-shape
    agreement), M2 (modal parity), H1 (first heat eigenvalue) and geometry error;
    `e_reference` repeats the E numbers the pipeline had already reported, so the
    geometry-against-simulation correlation can be taken over a single record.
    """

    @dataclass(frozen=True, kw_only=True)
    class Cluster:
        """
        One near-degenerate frequency cluster of the FEM elastic spectrum.

        Within a cluster the individual eigenvectors are arbitrary and only the
        spanned subspace is comparable, so `subspace_overlap` rather than the
        per-mode diagonal MAC is the honest agreement measure (metric M4).
        """

        fem_modes: list[int]
        """One-based indices of the elastic FEM modes forming this cluster."""
        size: int
        """How many modes the cluster holds; 1 is a simple, non-degenerate mode."""
        fem_hz: list[float]
        """Each clustered mode's FEM frequency [Hz]."""
        subspace_overlap: float
        """
        Mean cos^2 of the principal angles between the FEM and IGA cluster
        subspaces: 1 when the two span the same subspace, however arbitrarily the
        individual within-cluster vectors are mixed.
        """
        diag_mac: list[float]
        """Index-paired MAC of each clustered mode, to read beside the overlap."""

    @dataclass(frozen=True, kw_only=True)
    class ModalMac:
        """
        M1: area-weighted MAC between the FEM-GT and IGA-spline mode shapes.

        A high `mean_best_mac` beside a low `mean_diagonal_mac` says the shapes
        are recovered but reorder between the solvers, which makes the per-mode
        frequency maximum a pairing artifact rather than a modelling error.
        """

        n_elastic: int
        """Elastic modes compared, after the rigid-body modes are dropped."""
        mac_matrix: list[list[float]]
        """Full `(n_elastic, n_elastic)` MAC matrix; rows FEM, columns IGA."""
        diagonal_mac: list[float]
        """Index-paired MAC of each elastic mode (FEM mode i against IGA mode i)."""
        mean_diagonal_mac: float
        """Mean of `diagonal_mac`."""
        best_mac: list[float]
        """Each FEM mode's largest MAC over all IGA modes."""
        best_partner_iga_mode: list[int]
        """One-based IGA mode index attaining each entry of `best_mac`."""
        mean_best_mac: float
        """Mean of `best_mac`: "is the shape recovered somewhere?"."""
        n_best_mac_ge_0p5: int
        """How many FEM modes have a best MAC of at least 0.5."""
        n_reordered: int
        """How many FEM modes' best partner is not the mode of the same index."""
        n_clusters: int
        """Near-degenerate clusters found in the FEM elastic spectrum."""
        n_multi_mode_clusters: int
        """Clusters holding more than one mode: the genuinely degenerate ones."""
        clusters: list["MeshMetricsReport.Cluster"]
        """Every cluster, in frequency order, singletons included."""
        mean_cluster_subspace_overlap: float
        """Mean `Cluster.subspace_overlap` over all clusters."""

    @dataclass(frozen=True, kw_only=True)
    class ModalParity:
        """
        M2: least-squares parity fit `f_IGA = slope * f_FEM + intercept_hz`.

        A systematic stiffness, mass or scale bias shows up as a slope away from
        1 while `r_squared` stays near 1; scattered shape error shows up the other
        way round.
        """

        slope: float
        """Slope of the affine fit; 1 means no systematic frequency bias."""
        intercept_hz: float
        """Intercept of the affine fit [Hz]."""
        r_squared: float
        """Coefficient of determination of the affine fit."""
        scale_through_origin: float
        """Least-squares slope of the fit constrained through the origin."""
        n_modes: int
        """Elastic modes the fit was taken over."""

    @dataclass(frozen=True, kw_only=True)
    class HeatEigenvalue:
        """
        H1(k=1): the first heat-operator eigenvalue on each geometry.

        `lambda_1 = 1 / T_relax` is the geometry-intrinsic Laplace-Beltrami
        scalar both runs already logged, so comparing it needs no re-assembly.
        """

        T_relax_fem_gt_s: float
        """Diffusive relaxation time of the FEM ground-truth geometry [s]."""
        T_relax_iga_spline_s: float
        """Diffusive relaxation time of the recovered IGA spline [s]."""
        lambda1_fem_gt: float
        """First heat eigenvalue of the ground-truth geometry [1/s]."""
        lambda1_iga_spline: float
        """First heat eigenvalue of the recovered geometry [1/s]."""
        rel_error: float
        """Relative difference of the two eigenvalues, referred to the FEM one."""

    @dataclass(frozen=True, kw_only=True)
    class Geometry:
        """
        Sec. 4: surface distance between the co-registered GT and recovered
        analysis meshes.

        Every distance is also given as a percentage of the ground-truth
        bounding-box diagonal, which is what makes the four meshes comparable
        despite their different physical sizes.
        """

        bbox_diag: float
        """Ground-truth bounding-box diagonal, in model units."""
        chamfer: float
        """Symmetric Chamfer distance, in model units."""
        hausdorff: float
        """Symmetric Hausdorff distance, in model units."""
        chamfer_pct_diag: float
        """`chamfer` as a percentage of `bbox_diag`."""
        hausdorff_pct_diag: float
        """`hausdorff` as a percentage of `bbox_diag`."""
        mean_surf_dist_pct_diag: float
        """Mean recovered-to-GT surface distance, as a percentage of `bbox_diag`."""
        rms_surf_dist_pct_diag: float
        """RMS recovered-to-GT surface distance, as a percentage of `bbox_diag`."""
        area_ratio_rec_over_gt: float
        """Recovered surface area divided by the ground-truth surface area."""

    @dataclass(frozen=True, kw_only=True)
    class ReferenceErrors:
        """
        The E numbers the pipeline's own comparison stages already reported.

        Carried along unchanged so the geometry metrics above can be correlated
        against them without re-reading the comparison artifacts.
        """

        heat_H_t0_rel_error: float
        """Relative error of the total thermal energy at the initial time."""
        heat_gen_rate_rel_error: float
        """Relative error of the heat-generation rate."""
        modal_mean_rel_error: float
        """Mean relative elastic-frequency error."""
        modal_median_rel_error: float
        """Median relative elastic-frequency error."""
        modal_max_rel_error: float
        """Largest relative elastic-frequency error."""

    mesh: str
    """Name of the recorded mesh these metrics belong to."""
    mac: ModalMac
    """M1: mode-shape agreement."""
    parity: ModalParity
    """M2: modal parity fit."""
    heat_lambda1: HeatEigenvalue
    """H1(k=1): first heat-operator eigenvalue on both geometries."""
    geometry: Geometry
    """Sec. 4: geometry error between the two analysis surfaces."""
    e_reference: ReferenceErrors
    """The pre-existing E numbers, for the geometry-against-simulation correlation."""
