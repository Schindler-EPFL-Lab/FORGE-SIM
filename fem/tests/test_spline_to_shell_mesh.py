import tempfile
import unittest
from pathlib import Path

# Imported for its module-level side effect: `_fem_fixtures` puts the
# repository root on `sys.path`, so `fem` and `forge_sim` resolve.
import _fem_fixtures  # noqa: F401
import basix.ufl
import drjit as dr
import jsonpickle
import mitsuba as mi
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx import mesh
from mpi4py import MPI
from scipy.spatial import cKDTree

from fem.dolfinx_mesh import create_manifold_mesh
from fem.shells.spline_to_shell_mesh import build_shell_mesh, load_and_rescale_spline
from forge_sim.meshing.tessellated_spline_surface import TessellatedSplineSurface
from forge_sim.splines.cube_initialization import cubed_sphere


class TestSplineToShellMesh(unittest.TestCase):
    """
    Geometry and topology of the built mesh, the rescaling helper, and the
    DOLFINx handover the mesh builder shares with `create_manifold_mesh`.

    `build_shell_mesh` is the geometric bridge between the reconstruction
    side (a multi-patch B-spline surface) and the shell solver. Its
    correctness claim is specific and checkable: every geometry node
    (vertices and the order-2 mid-edge nodes) sits on the spline, not
    merely near it, and the result is conforming across patch seams, so a
    mid-edge node shared by two patches is one node, not two coincident
    ones. Both are properties of the surface, not of a golden file, so they
    are asserted directly against the spline evaluation.
    `load_and_rescale_spline`, the other function of the module, is pinned
    next to it: modal frequencies scale as 1/length, so a mismatch there is
    indistinguishable from a physics discrepancy.

    The fixture is a six-patch cubed sphere, whose exact surface is known:
    `cubed_sphere` interpolates control points that lie on a sphere, so the
    patch parametrisation is not analytic, but its own `evaluate_positions`
    is the oracle the mesh must agree with. The DOLFINx handover tests at
    the end pin `fem.dolfinx_mesh.create_manifold_mesh`'s order-2 branch
    against its input; `build_shell_mesh` is that branch's only caller.

    Mitsuba objects are built per test in `setUp` rather than held on the
    class: Dr.Jit takes exception to objects outliving into an unrelated
    later test.
    """

    SPLINE_SUBDIVISIONS = 4
    """Coarsest cubed sphere that still exercises six patches, their seams and
    the local->global scatter."""

    TESSELLATION_SUBDIVISIONS = 8
    """Tessellation density of the spline underlying `make_sphere_surface`."""

    SUBDIVISIONS = 2
    """
    Subdivisions per patch for the built shell mesh. Two gives 2*2^2 = 8
    triangles per patch, 48 in total: small enough that the conformity
    assertions can be brute-forced, large enough to have interior edges as
    well as seam edges.
    """

    @staticmethod
    def make_sphere_surface(
        subdivisions: int = SPLINE_SUBDIVISIONS,
    ) -> TessellatedSplineSurface:
        """A six-patch cubed-sphere `TessellatedSplineSurface` (radius 1)."""
        return cubed_sphere(
            name="shell_mesh_test_sphere",
            degree=3,
            spline_subdivisions=subdivisions,
            tessellation_subdivisions=TestSplineToShellMesh.TESSELLATION_SUBDIVISIONS,
            optimize_g1=False,
        )

    @classmethod
    def setUpClass(cls) -> None:
        mi.set_variant("llvm_ad_mono")

    def setUp(self) -> None:
        self.surface = self.make_sphere_surface()
        self._temporary = tempfile.TemporaryDirectory()
        self.path = Path(self._temporary.name) / "sphere.json"
        self.path.write_text(jsonpickle.encode(self.surface))

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_order_one_mesh_matches_the_tessellation(self) -> None:
        """
        Order 1 is a pass-through of `composite_mesh_tesselation`: it must add no
        nodes and reuse the conforming vertex topology as-is. DOLFINx permutes
        the vertices, so the assertion is that the two vertex sets coincide, not
        that the two vertex arrays match element-wise; each mesh node matches
        a distinct tessellation vertex.

        Tolerance 1e-6: the tessellation is evaluated in Mitsuba's float32, so
        this is round-off on coordinates of order 1. Requiring the nearest-vertex
        map to be a bijection is what makes it a set comparison rather than a
        containment one.
        """
        msh = build_shell_mesh(self.surface, self.SUBDIVISIONS, order=1)
        tessellation = self.surface.composite_spline.composite_mesh_tesselation(
            subdivisions_per_patch=self.SUBDIVISIONS
        )
        expected = np.asarray(tessellation.global_vertices.numpy(), dtype=np.float64)

        self.assertEqual(msh.geometry.dofmap.shape[1], 3)
        self.assertEqual(msh.geometry.x.shape[0], expected.shape[0])

        distances, nearest = cKDTree(expected).query(msh.geometry.x, k=1)
        self.assertLess(float(distances.max()), 1.0e-6)
        self.assertEqual(len(set(nearest.tolist())), expected.shape[0])

    def test_order_two_adds_one_node_per_edge(self) -> None:
        """
        A conforming P2 triangulation has six nodes per cell and V + E of them in
        total. Getting E right is the whole point of keying mid-edge nodes by the
        global vertex pair: keying per patch would duplicate every seam edge, and
        the node count would come out too high by exactly that many.

        Closed triangulated surface: every edge is shared by two faces, so
        E = 3F/2, and Euler's formula fixes V = 2 - F + E. Both are properties of
        the topology alone, so the expected node count is exact.
        """
        msh = build_shell_mesh(self.surface, self.SUBDIVISIONS, order=2)
        n_faces = msh.topology.index_map(msh.topology.dim).size_global
        n_edges = 3 * n_faces // 2
        n_vertices = 2 - n_faces + n_edges

        self.assertEqual(msh.geometry.dofmap.shape[1], 6)
        self.assertEqual(msh.geometry.x.shape[0], n_vertices + n_edges)

    def test_all_geometry_nodes_lie_on_the_spline(self) -> None:
        """
        The module's central claim: "All P2 geometry nodes therefore sit on the
        spline surface". For the cubed sphere that surface is very nearly the
        unit sphere, but "very nearly" is not an oracle, so each node is instead
        required to be reproducible by evaluating some patch, checked here in
        the sharpest available form, that the node radii match those of a densely
        sampled spline evaluation.

        Tolerance 2e-6 relative: the spline is evaluated in Mitsuba's float32
        (~1e-7 relative) and the comparison is against a sampled min/max envelope,
        so this is round-off with an order of margin. A mesh whose mid-edge nodes
        were interpolated instead of evaluated would sit ~h^2/8 inside the
        surface (3e-3 here, three orders above the bound).
        """
        msh = build_shell_mesh(self.surface, self.SUBDIVISIONS, order=2)
        node_radii = np.linalg.norm(msh.geometry.x, axis=1)

        dense = self.surface.composite_spline.composite_mesh_tesselation(
            subdivisions_per_patch=16
        )
        surface_radii = np.linalg.norm(
            np.asarray(dense.global_vertices.numpy(), dtype=np.float64), axis=1
        )
        self.assertGreaterEqual(
            node_radii.min(), surface_radii.min() * (1.0 - 2.0e-6)
        )
        self.assertLessEqual(node_radii.max(), surface_radii.max() * (1.0 + 2.0e-6))

    def test_midside_nodes_are_shared_across_patch_seams(self) -> None:
        """
        Conformity, stated as "no two geometry nodes coincide". A mid-edge node
        evaluated independently on each side of a seam would produce a duplicate
        pair at (nearly) the same point, a mesh DOLFINx accepts happily and
        which then behaves as a slit, halving the connectivity across every seam.

        The separation threshold is a hundredth of the smallest edge, far below
        any legitimate node spacing and far above float32 evaluation noise.
        """
        msh = build_shell_mesh(self.surface, self.SUBDIVISIONS, order=2)
        points = msh.geometry.x
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        smallest_gap = float(distances.min())

        vertices = points[np.asarray(msh.geometry.dofmap)[:, :3]]
        shortest_edge = min(
            float(np.linalg.norm(vertices[:, a] - vertices[:, b], axis=1).min())
            for a, b in ((0, 1), (1, 2), (0, 2))
        )
        self.assertGreater(smallest_gap, 0.01 * shortest_edge)

    def test_every_edge_receives_a_curved_node(self) -> None:
        """The builder raises rather than shipping a mesh with a node left at the
        origin; assert the happy path really does fill them all (the guard is the
        only thing standing between a mapping bug and a silently folded mesh)."""
        msh = build_shell_mesh(self.surface, self.SUBDIVISIONS, order=2)
        self.assertGreater(np.linalg.norm(msh.geometry.x, axis=1).min(), 0.1)

    def test_rejects_unsupported_geometry_order(self) -> None:
        for order in (0, 3):
            with self.subTest(order=order):
                with self.assertRaises(ValueError):
                    build_shell_mesh(self.surface, self.SUBDIVISIONS, order=order)

    def test_finer_subdivision_gives_the_expected_cell_count(self) -> None:
        """`2 * S^2` triangles per patch, six patches: the documented contract,
        and what the entry point's DOF estimates are based on."""
        for subdivisions in (1, 3):
            with self.subTest(subdivisions=subdivisions):
                msh = build_shell_mesh(self.surface, subdivisions, order=2)
                n_cells = msh.topology.index_map(msh.topology.dim).size_global
                self.assertEqual(n_cells, 6 * 2 * subdivisions**2)

    @staticmethod
    def longest_extent(surface: TessellatedSplineSurface) -> float:
        control_points = surface.composite_spline.global_control_points
        extent = dr.max(control_points, axis=0) - dr.min(control_points, axis=0)
        return float(dr.max(extent).numpy())

    def test_target_length_scale_sets_the_longest_control_point_extent(self) -> None:
        """The definition of the parameter, and it is on the control points, not
        on the surface: the surface lies inside the control hull, so the two
        differ. Tolerance 1e-5 relative: control points are float32."""
        for target in (3.0, 0.25):
            with self.subTest(target_length_scale=target):
                surface = load_and_rescale_spline(self.path, target_length_scale=target)
                self.assertAlmostEqual(
                    self.longest_extent(surface) / target, 1.0, delta=1.0e-5
                )

    def test_none_leaves_the_geometry_untouched(self) -> None:
        """Rescaling is opt-in; `None` must be a true no-op, not a normalisation
        to 1."""
        unscaled = load_and_rescale_spline(self.path, target_length_scale=None)
        original = self.make_sphere_surface()
        np.testing.assert_allclose(
            unscaled.composite_spline.global_control_points.numpy(),
            original.composite_spline.global_control_points.numpy(),
            rtol=0.0,
        )

    def test_rescaling_is_a_uniform_scale(self) -> None:
        """Not a per-axis normalisation: shape must be preserved exactly, or the
        FEM and IGA runs would be comparing different objects. Asserted as a
        constant ratio across every control-point coordinate."""
        original = (
            self.make_sphere_surface().composite_spline.global_control_points.numpy()
        )
        scaled = load_and_rescale_spline(
            self.path, target_length_scale=7.0
        ).composite_spline.global_control_points.numpy()

        significant = np.abs(original) > 1.0e-3
        ratios = scaled[significant] / original[significant]
        self.assertAlmostEqual(float(ratios.max() / ratios.min()), 1.0, delta=1.0e-5)

    def test_rescaled_mesh_scales_with_it(self) -> None:
        """The property that actually matters downstream: the scale reaches the
        built mesh, so the solver sees the rescaled geometry."""
        small = build_shell_mesh(
            load_and_rescale_spline(self.path, target_length_scale=1.0),
            self.SUBDIVISIONS,
            order=2,
        )
        large = build_shell_mesh(
            load_and_rescale_spline(self.path, target_length_scale=4.0),
            self.SUBDIVISIONS,
            order=2,
        )
        ratio = np.linalg.norm(large.geometry.x, axis=1).max() / np.linalg.norm(
            small.geometry.x, axis=1
        ).max()
        self.assertAlmostEqual(ratio, 4.0, delta=1.0e-4)

    def test_degenerate_geometry_is_rejected(self) -> None:
        """A surface collapsed to a point has no length to normalise; the builder
        must say so rather than dividing by ~zero and returning inf coordinates."""
        surface = self.make_sphere_surface()
        composite = surface.composite_spline
        composite.global_control_points = dr.zeros(
            mi.TensorXf, shape=composite.global_control_points.shape
        )
        collapsed = Path(self._temporary.name) / "collapsed.json"
        collapsed.write_text(jsonpickle.encode(surface))

        with self.assertRaises(ValueError):
            load_and_rescale_spline(collapsed, target_length_scale=3.0)

    def test_serial_rank_zero_supplies_the_full_mesh(self) -> None:
        """The serial path must be unaffected by the rank-0 gating that protects
        the (unused) multi-rank one."""
        msh = build_shell_mesh(
            self.surface, self.SUBDIVISIONS, order=2, comm=MPI.COMM_SELF
        )
        self.assertEqual(
            msh.topology.index_map(msh.topology.dim).size_global,
            6 * 2 * self.SUBDIVISIONS**2,
        )

    @staticmethod
    def curved_patch() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
        """
        Two triangles sharing an edge, as a curved P2 patch.

        Four vertices and five mid-edge nodes, the latter lifted off the
        straight midpoints so this is a genuinely curved patch and not a flat
        one an order-1 handover could equally have produced. The cells are
        int32, what a caller that does not cast them supplies.

        :returns: The nine node coordinates and the two six-node cells, laid
            out as `basix_edge_table` defines: `[v0, v1, v2, e12, e02, e01]`.
        """
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.3],
             [0.5, 0.5, 0.1], [0.0, 0.5, 0.05], [0.5, 0.0, 0.05],
             [0.5, 1.0, 0.15], [1.0, 0.5, 0.15]]
        )
        # cell 0 = (v0, v1, v2) with edges (e12, e02, e01) = nodes 4, 5, 6;
        # cell 1 = (v1, v3, v2) with edges (e32, e12, e13) = nodes 7, 4, 8.
        cells = np.array([[0, 1, 2, 4, 5, 6], [1, 3, 2, 7, 4, 8]], dtype=np.int32)
        return points, cells

    def test_the_curved_handover_carries_the_patch_through_unchanged(self) -> None:
        """
        What the order-2 branch must guarantee, stated against its input rather
        than against another call of the same library: the mesh DOLFINx returns
        carries exactly the nodes that were handed in, grouped into exactly the
        cells they were handed in as.

        DOLFINx renumbers both nodes and cells, so the assertions are on sets:
        every node coordinate appears once, and the six-node group of each cell
        survives as a group. A handover that dropped a mid-edge node, silently
        degraded the patch to order 1, or scrambled a cell's node order would
        break one of them; the manual-route comparison below would not, since
        both sides would scramble alike.
        """
        points, cells = self.curved_patch()
        msh = create_manifold_mesh(MPI.COMM_SELF, cells, points, degree=2)

        self.assertEqual(msh.geometry.cmap.degree, 2)
        self.assertEqual(msh.geometry.dofmap.shape, (2, 6))
        np.testing.assert_array_equal(
            np.sort(msh.geometry.x, axis=0), np.sort(points, axis=0)
        )

        built = {
            frozenset(map(tuple, msh.geometry.x[cell]))
            for cell in np.asarray(msh.geometry.dofmap)
        }
        self.assertEqual(built, {frozenset(map(tuple, points[cell]))
                                 for cell in cells})

    def test_curved_handover_matches_the_manual_route(self) -> None:
        """
        `create_manifold_mesh(degree=2)` is the shared implementation of this
        handover. Its body and the three lines below build the identical mesh,
        which is the point: this pins them together, so that a later change to
        the shared helper (a reordering, a different coordinate element)
        shows up here rather than as a moved spectrum.

        It is a mirror by construction and cannot fail on its own; the test
        above is the one that checks the handover against its input. (The
        internal int64 cast in particular is not what this pins: DOLFINx
        accepts the int32 cells either way, verified by mutation.)
        """
        points, cells = self.curved_patch()
        shared = create_manifold_mesh(MPI.COMM_SELF, cells, points, degree=2)
        coordinate_element = basix.ufl.element(
            "Lagrange", "triangle", 2, shape=(3,)
        )
        manual = mesh.create_mesh(
            MPI.COMM_SELF,
            cells.astype(np.int64),
            points,
            ufl.Mesh(coordinate_element),
        )

        np.testing.assert_array_equal(shared.geometry.x, manual.geometry.x)
        np.testing.assert_array_equal(
            np.asarray(shared.geometry.dofmap), np.asarray(manual.geometry.dofmap)
        )
