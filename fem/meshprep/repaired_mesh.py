from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from fem.meshprep.mesh_repair import (
    _orient_outward,
    cap_holes,
    directed_boundary_loops,
    largest_component,
)


@dataclass(frozen=True, kw_only=True)
class RepairedMesh:
    """
    A single-body watertight mesh together with the caps the repair had to add.

    `seal_centres` and `seal_radius` are what `smoothed` needs to turn the flat
    caps into a natural lid, and they travel with the mesh because only the
    repair knows where it cut. A mesh that needed no repair carries no seals: an
    empty `seal_centres` and a `seal_radius` of 0.0, which `smoothed` reads as
    "nothing to do".
    """

    SEAL_MARGIN: ClassVar[float] = 1.6
    """
    Multiple of a socket's rim radius that `seal_radius` must reach.

    Smoothing a cap has to reach past the rim into the surface around it, or the lid
    meets the head at the crease the cap introduced. Anything above one reaches
    outside; this is heuristic, chosen to take in a ring about half a rim wide
    without reaching far enough to smooth features the repair never touched.
    """

    vertices: npt.NDArray[np.float64]
    """Vertex positions after welding and capping, shape `(n_vertices, 3)`."""
    faces: npt.NDArray[np.int64]
    """Triangle connectivity, consistently wound outward, shape `(n_faces, 3)`."""
    seal_centres: npt.NDArray[np.float64]
    """Centroid of each hole the dropped sub-bodies left, shape `(n_seals, 3)`."""
    seal_radius: float
    """
    Radius about every entry of `seal_centres` that smoothing should reach, in the
    mesh's own units: the largest rim radius over all seals plus a margin, so the
    smoothed lid blends into a surrounding ring rather than stopping at the cap.
    """

    @classmethod
    def from_raw(
        cls,
        vertices: npt.NDArray[np.float64],
        faces: npt.NDArray[np.int64],
        weld_eps: float = 1e-5,
    ) -> "RepairedMesh":
        """
        Weld UV-seam duplicates and close the model into a single watertight manifold
        by dropping detached interior sub-bodies and sealing the holes they leave.

        Note: this discards every detached component but the largest rather than
        merging them in; call it only where that is the intended repair.

        Suzanne ships as a head shell plus two separate eye caps that sit recessed in
        their sockets and poke back out through the face: each eyeball is a
        detached body whose surface lies (mostly) inside the head hull, terminating
        in empty space. The image reconstruction cannot see them (its area matches a
        socket-capped head to 0.1% vs +10% for the eyeballs), and as geometry they
        make the shell self-intersecting and the modal problem ill-posed. So each
        non-head component is dropped wholesale (it is the "inside" geometry) and
        the socket boundary it leaves on the head is capped. The eyelid/brow detail
        lives on the head, so only the eyeball bulge is removed.

        :returns: The repaired mesh, carrying the socket centroids and a radius
            covering the cap plus a margin, so the caller can smooth just those
            regions (:meth:`smoothed`) into a natural sealed lid, not a flat patch.
            On a closed single body (Spot) the seal centres come back empty and the
            radius 0.
        """
        import open3d as o3d  # deferred: see fem.meshprep.mesh_repair's docstring

        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.ascontiguousarray(vertices, np.float64)),
            o3d.utility.Vector3iVector(np.ascontiguousarray(faces, np.int32)),
        )
        mesh = mesh.merge_close_vertices(weld_eps)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()

        # Drop every detached sub-body (keep the largest connected component, the
        # head).
        head = largest_component(
            np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        )
        head_faces = head.faces.astype(np.int64)
        # Socket holes left by the dropped eyeballs -> centroids + a radius covering
        # each rim plus a margin (so smoothing reaches a surrounding ring to blend
        # into).
        socket_loops = directed_boundary_loops(head_faces)
        seal_centres = np.asarray(
            [head.vertices[loop].mean(axis=0) for loop in socket_loops],
            dtype=np.float64,
        ).reshape(-1, 3)
        seal_radius = 0.0
        for loop, centre in zip(socket_loops, seal_centres):
            rim_radius = float(
                np.linalg.norm(head.vertices[loop] - centre, axis=1).max()
            )
            seal_radius = max(seal_radius, cls.SEAL_MARGIN * rim_radius)
        # Cap the sockets watertight, then make the winding globally consistent +
        # outward.
        capped = cap_holes(head.vertices, head_faces)
        capped_faces = capped.faces.astype(np.int64)
        return cls(
            vertices=capped.vertices,
            faces=_orient_outward(capped.vertices, capped_faces),
            seal_centres=seal_centres,
            seal_radius=seal_radius,
        )

    def smoothed(
        self, iterations: int = 18, lam: float = 0.5, mu: float = -0.53
    ) -> "RepairedMesh":
        """
        Taubin-smooth only the vertices within `seal_radius` of any point in
        `seal_centres`.

        Turns the flat socket caps (after the eyeballs are dropped) into a natural
        sealed-lid dome without disturbing the rest of the ground-truth surface:
        vertices outside every seal region stay pinned, so the brow/nose/etc. are
        bit-for-bit untouched. Taubin's alternating positive (`lam`) / negative
        (`mu`) passes are shrink-free, so the lid blends into the surrounding
        curvature rather than collapsing inward. A mesh with no seals (empty
        `seal_centres`, per this record's own contract) comes back unchanged.

        :returns: This mesh unchanged when there is nothing within `seal_radius`
            of a `seal_centres` point; otherwise a new `RepairedMesh` with the same
            `faces`, `seal_centres` and `seal_radius`, and `vertices` smoothed in
            the seal regions.
        """
        if len(self.seal_centres) == 0:
            return self
        v = self.vertices.astype(np.float64).copy()
        n = v.shape[0]
        mask = np.zeros(n, dtype=bool)
        for c in np.asarray(self.seal_centres, dtype=np.float64):
            mask |= np.linalg.norm(v - c, axis=1) < self.seal_radius
        if not mask.any():
            return self
        # Uniform (graph) Laplacian over the mesh edges.
        f = np.asarray(self.faces, dtype=np.int64)
        rows = np.concatenate([f[:, 0], f[:, 1], f[:, 1], f[:, 2], f[:, 2], f[:, 0]])
        cols = np.concatenate([f[:, 1], f[:, 0], f[:, 2], f[:, 1], f[:, 0], f[:, 2]])
        adj = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()
        adj.data[:] = 1.0
        adj = (adj > 0).astype(np.float64)
        deg = np.asarray(adj.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        inv_deg = sp.diags(1.0 / deg)
        move = np.where(mask)[0]
        for k in range(2 * iterations):
            delta = inv_deg @ (adj @ v) - v  # uniform Laplacian
            step = lam if k % 2 == 0 else mu
            v[move] += step * delta[move]
        return replace(self, vertices=v)
