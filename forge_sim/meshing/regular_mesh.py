import mitsuba as mi
import numpy as np
import numpy.typing as npt

from forge_sim.meshing.rectangular_mesh import RectangularMesh
from forge_sim.plugin_wrappers.diffuse_bsdf import DiffuseBSDF


class RegularMesh(RectangularMesh):
    """
    A rectangular mesh that maintains a regular grid topology and can be refined.

    Vertices, faces, and texture coordinates are arranged as a regular `(height + 1) x
    (width + 1)` grid, identical to the layout produced by
    `RectangularMesh.unit_square`. The class records the grid resolution so that
    `h_refinement` can produce a finer regular grid by linearly interpolating the
    existing vertex positions in `(u, v)` space. Texture coordinates are regenerated
    as the canonical `(j / W, i / H)` grid, and faces are regenerated using the same
    triangulation pattern.

    Vertex positions on the refined grid are bilinearly interpolated from the four
    surrounding old vertices, so the current shape is preserved piecewise-bilinearly.
    """

    def __init__(
        self,
        mesh_name: str,
        vertices: npt.NDArray[np.float64],
        faces: npt.NDArray[np.uint32],
        texture_coordinates: npt.NDArray[np.float64],
        width_subdivisions: int,
        height_subdivisions: int,
        normals: npt.NDArray[np.float64] | None = None,
        bsdf: DiffuseBSDF | None = None,
    ) -> None:
        """
        Creates a new regular mesh of the given grid resolution.

        `width_subdivisions` and `height_subdivisions` are the number of grid cells
        along each direction. The number of `vertices`, `faces`, and
        `texture_coordinates` must match what a `(width_subdivisions + 1) x
        (height_subdivisions + 1)` grid would produce.

        :raises ValueError: If either subdivision count is less than 1, if
            `width_subdivisions` and `height_subdivisions` differ, or if the number
            of `vertices`, `faces`, or `texture_coordinates` is inconsistent with a
            regular grid of the given resolution.
        """
        if width_subdivisions < 1 or height_subdivisions < 1:
            raise ValueError(
                "`width_subdivisions` and `height_subdivisions` must be at least 1.",
            )
        if width_subdivisions != height_subdivisions:
            # `_build_grid_faces` indexes vertices as if the grid were square (it
            # mixes up the row/column stride between the two dimensions); passing
            # unequal subdivisions silently produces out-of-range face indices.
            # This is the real defect to fix one day - until then, refuse the call.
            raise ValueError(
                "Non-square grids are not supported: `width_subdivisions` "
                f"({width_subdivisions}) must equal `height_subdivisions` "
                f"({height_subdivisions}) because the face-index construction "
                "assumes a square grid.",
            )
        expected_vertex_count = (width_subdivisions + 1) * (height_subdivisions + 1)
        expected_face_count = width_subdivisions * height_subdivisions * 2
        if vertices.shape[0] != expected_vertex_count:
            raise ValueError(
                f"Expected {expected_vertex_count} vertices for a "
                f"{width_subdivisions}x{height_subdivisions} regular grid, "
                f"got {vertices.shape[0]}.",
            )
        if faces.shape[0] != expected_face_count:
            raise ValueError(
                f"Expected {expected_face_count} faces for a "
                f"{width_subdivisions}x{height_subdivisions} regular grid, "
                f"got {faces.shape[0]}.",
            )
        if texture_coordinates.shape[0] != expected_vertex_count:
            raise ValueError(
                f"Expected {expected_vertex_count} texture coordinates, "
                f"got {texture_coordinates.shape[0]}.",
            )

        super().__init__(
            mesh_name=mesh_name,
            vertices=vertices,
            faces=faces,
            texture_coordinates=texture_coordinates,
            normals=normals,
            bsdf=bsdf,
        )

        self._width_subdivisions = width_subdivisions
        self._height_subdivisions = height_subdivisions

    @property
    def width_subdivisions(self) -> int:
        """Number of grid cells along the u (width) direction."""
        return self._width_subdivisions

    @property
    def height_subdivisions(self) -> int:
        """Number of grid cells along the v (height) direction."""
        return self._height_subdivisions

    @staticmethod
    def _build_grid_vertices(
        width_subdivisions: int,
        height_subdivisions: int,
    ) -> npt.NDArray[np.float64]:
        """
        Returns vertex positions of a unit square mesh in the XY plane with the
        specified grid resolution, flattened in row-major order matching
        `RectangularMesh.unit_square`.
        """
        corner_vertices = np.array(
            [[-0.5, -0.5, 0], [0.5, -0.5, 0], [-0.5, 0.5, 0], [0.5, 0.5, 0]],
            dtype=np.float64,
        )
        u_coords_1d = (
            np.arange(width_subdivisions + 1, dtype=np.float64) / width_subdivisions
        )
        v_coords_1d = (
            np.arange(height_subdivisions + 1, dtype=np.float64) / height_subdivisions
        )
        u_coords, v_coords = np.meshgrid(u_coords_1d, v_coords_1d, indexing="xy")
        u_coords = u_coords.reshape(-1, 1)
        v_coords = v_coords.reshape(-1, 1)

        return np.array(
            corner_vertices[0][np.newaxis, :]
            + u_coords * (corner_vertices[1] - corner_vertices[0])[np.newaxis, :]
            + v_coords * (corner_vertices[2] - corner_vertices[0])[np.newaxis, :],
            dtype=np.float64,
        )

    @staticmethod
    def _build_grid_faces(
        width_subdivisions: int,
        height_subdivisions: int,
    ) -> npt.NDArray[np.uint32]:
        """
        Returns triangle face indices for a regular `(height_subdivisions + 1) x
        (width_subdivisions + 1)` grid of vertices, ordered identically to
        `RectangularMesh.unit_square`.
        """
        i_indices = np.arange(height_subdivisions)
        j_indices = np.arange(width_subdivisions)
        i_grid, j_grid = np.meshgrid(i_indices, j_indices, indexing="xy")

        idx00 = j_grid * (width_subdivisions + 1) + i_grid
        idx10 = j_grid * (width_subdivisions + 1) + (i_grid + 1)
        idx01 = (j_grid + 1) * (width_subdivisions + 1) + i_grid
        idx11 = (j_grid + 1) * (width_subdivisions + 1) + (i_grid + 1)

        faces = np.zeros(
            shape=(height_subdivisions * width_subdivisions * 2, 3),
            dtype=np.uint32,
        )
        faces[0::2, 0] = idx00.flatten()
        faces[0::2, 1] = idx10.flatten()
        faces[0::2, 2] = idx11.flatten()
        faces[1::2, 0] = idx00.flatten()
        faces[1::2, 1] = idx11.flatten()
        faces[1::2, 2] = idx01.flatten()

        return faces

    @staticmethod
    def _build_grid_texcoords(
        width_subdivisions: int,
        height_subdivisions: int,
    ) -> npt.NDArray[np.float64]:
        """
        Returns canonical `(u, v)` texture coordinates for a regular grid where
        `u = j / width_subdivisions` and `v = i / height_subdivisions`, flattened in
        row-major order matching `RectangularMesh.unit_square`.
        """
        u_coords_1d = (
            np.arange(width_subdivisions + 1, dtype=np.float64) / width_subdivisions
        )
        v_coords_1d = (
            np.arange(height_subdivisions + 1, dtype=np.float64) / height_subdivisions
        )
        u_coords, v_coords = np.meshgrid(u_coords_1d, v_coords_1d, indexing="xy")

        return np.hstack([u_coords.reshape(-1, 1), v_coords.reshape(-1, 1)])

    @classmethod
    def unit_square(
        cls,
        mesh_name: str,
        width_subdivisions: int = 1,
        height_subdivisions: int = 1,
        bsdf: DiffuseBSDF | None = None,
    ) -> "RegularMesh":
        """
        Creates a unit-square `RegularMesh` in the XY plane with the specified grid
        resolution. Output vertices, faces, and texcoords are identical to
        `RectangularMesh.unit_square`, but the result also carries the grid resolution
        needed for refinement.
        """
        if width_subdivisions < 1 or height_subdivisions < 1:
            raise ValueError(
                "`width_subdivisions` and `height_subdivisions` must be at least 1.",
            )

        vertices = cls._build_grid_vertices(width_subdivisions, height_subdivisions)
        faces = cls._build_grid_faces(width_subdivisions, height_subdivisions)
        texcoords = cls._build_grid_texcoords(width_subdivisions, height_subdivisions)

        return cls(
            mesh_name=mesh_name,
            vertices=vertices,
            faces=faces,
            texture_coordinates=texcoords,
            width_subdivisions=width_subdivisions,
            height_subdivisions=height_subdivisions,
            bsdf=bsdf,
        )

    def h_refinement(self, factor: int = 2) -> None:
        """
        Refines the grid by `factor` along both directions, multiplying the vertex
        count by approximately `factor ** 2`.

        New vertex positions are obtained by bilinearly interpolating the four
        surrounding old vertices in (u, v) space, so the current shape is preserved
        piecewise-bilinearly. Texture coordinates are regenerated as the canonical
        regular `(j / W_new, i / H_new)` grid, and faces are regenerated with the
        same triangulation pattern as the original grid.

        The internal Mitsuba mesh object is replaced because Mitsuba meshes have a
        fixed buffer size set at construction time. Callers that hold a reference to
        the underlying mesh in a Mitsuba scene must reload or update the scene to
        observe the new resolution.

        :raises ValueError: If `factor` is not an integer greater than 1.
        """
        if factor <= 1:
            raise ValueError("`factor` must be an integer greater than 1.")

        old_W = self._width_subdivisions
        old_H = self._height_subdivisions
        new_W = old_W * factor
        new_H = old_H * factor

        old_vertices_flat = self.vertices.numpy().astype(np.float64)
        old_grid = old_vertices_flat.reshape(old_H + 1, old_W + 1, 3)

        new_i = np.arange(new_H + 1)
        new_j = np.arange(new_W + 1)
        new_i_grid, new_j_grid = np.meshgrid(new_i, new_j, indexing="ij")

        old_i0 = np.minimum(new_i_grid // factor, old_H)
        old_j0 = np.minimum(new_j_grid // factor, old_W)
        old_i1 = np.minimum(old_i0 + 1, old_H)
        old_j1 = np.minimum(old_j0 + 1, old_W)
        t_v = (new_i_grid % factor).astype(np.float64) / float(factor)
        t_u = (new_j_grid % factor).astype(np.float64) / float(factor)

        w00 = ((1.0 - t_u) * (1.0 - t_v))[..., None]
        w01 = (t_u * (1.0 - t_v))[..., None]
        w10 = ((1.0 - t_u) * t_v)[..., None]
        w11 = (t_u * t_v)[..., None]

        new_grid = (
            w00 * old_grid[old_i0, old_j0]
            + w01 * old_grid[old_i0, old_j1]
            + w10 * old_grid[old_i1, old_j0]
            + w11 * old_grid[old_i1, old_j1]
        )
        new_vertices = new_grid.reshape(-1, 3).astype(np.float64)

        new_faces = RegularMesh._build_grid_faces(new_W, new_H)
        new_texcoords = RegularMesh._build_grid_texcoords(new_W, new_H)

        self._width_subdivisions = new_W
        self._height_subdivisions = new_H

        # Replace the internal Mitsuba mesh at the new resolution. The Mesh wrapper's
        # `_set_mesh_bsdf` (dump-to-PLY + reload) is reused so the new mesh carries
        # the same BSDF.
        self._mesh = self._create_mitsuba_mesh(
            mesh_name=self._mesh_name,
            vertices=new_vertices,
            faces=new_faces,
            normals=None,
            texture_coordinates=new_texcoords,
        )
        self._set_mesh_bsdf()
        self._mesh_params = mi.traverse(self._mesh)

        self._update_side_indices()
