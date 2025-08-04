import drjit as dr
import mitsuba as mi
import numpy as np
import numpy.typing as npt

from rebel_shape.meshing.mesh import Mesh
from rebel_shape.meshing.patch import Patch
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF


class RectangularMesh(Mesh, Patch):
    """
    Represents a mesh in 3D space that can be obtained by transforming a unit square.

    Inherits most of the functionality for the `Mesh` class. Its main importance is
    representing a tessellated patch that is outputted by `SplineSurface`.
    Additionally, it is the building block of simple patch meshes, such as the
    `CompositeMesh` class.

    Conceptually, a rectangular mesh is distinguished by having clearly defined
    sides and natural texture coordinates. Sides are identified by these coordinates:
    - LEFT:     0 <= u < 1, v = 0
    - RIGHT:    0 <= u < 1, v = 1
    - TOP:      u = 1,      0 <= v < 1
    - BOTTOM:   u = 0,      0 <= v < 1
    This parametrization is directly tied to the position of the vertices of the
    unit square mesh - bottom left has texcoords (0, 0) while top right has
    texcoords (1, 1).
    """

    SIDE_TOLERANCE = 1e-6
    """
    How close must a texture coordinate be to the edge of the mesh to be considered
    part of that side. This is used to determine which side of the mesh a vertex
    belongs to.
    """

    def __init__(
        self,
        mesh_name: str,
        vertices: npt.NDArray[np.float64],
        faces: npt.NDArray[np.uint32],
        texture_coordinates: npt.NDArray[np.float64],
        normals: npt.NDArray[np.float64] | None = None,
        bsdf: DiffuseBSDF | None = None,
    ) -> None:
        """
        Creates a new rectangular mesh object.

        `vertices` is a numpy array of shape (V, 3) containing the vertex positions,
        `faces` is a numpy array of shape (F, 3) containing the face indices.
        `texture_coordinates` is a numpy array of shape (V, 2) containing the texture
        coordinates for each vertex and must be provided, unlike the base class.
        `normals` is a numpy array of shape (V, 3) containing the vertex normals.

        Empty meshes are not allowed at the time of writing because Mitsuba does not
        support them.

        The `bsdf` is used to control the appearance of the mesh in the scene.

        :raises ValueError: If the shape of `vertices`, `faces`, `vertex_normals`, or
            `texture_coordinates` is not as expected, if the number of `vertices`,
            `texture_coordinates`, and `vertex_normals` does not match, if
            texture coordinates are not in the range [0, 1], if the normals are not
            normalized, or if the `faces` or `vertices` are empty.
        """
        super().__init__(
            mesh_name=mesh_name,
            vertices=vertices,
            faces=faces,
            vertex_normals=normals,
            texture_coordinates=texture_coordinates,
            bsdf=bsdf,
        )

        self._left_side_indices: "mi.ArrayXu" = None
        self._right_side_indices: "mi.ArrayXu" = None
        self._top_side_indices: "mi.ArrayXu" = None
        self._bottom_side_indices: "mi.ArrayXu" = None

        self._update_side_indices()

    def _update_side_indices(self) -> None:
        """
        To avoid recomputing the side indices every time they are requested, associated
        indices are computed once during initialization (recomputed if needed at
        some point in the future). Then, they can be accessed via the
        `get_node_indices_for` method.

        A private variant of this method is used to initialize the side indices
        based on the texture coordinates of the mesh vertices.
        """
        self._left_side_indices = self._get_node_indices_for(Patch.Side.LEFT)
        self._right_side_indices = self._get_node_indices_for(Patch.Side.RIGHT)
        self._top_side_indices = self._get_node_indices_for(Patch.Side.TOP)
        self._bottom_side_indices = self._get_node_indices_for(Patch.Side.BOTTOM)

    def _get_node_indices_for(self, side: Patch.Side) -> "mi.ArrayXu":
        """
        Returns the vertices of the specified side of the rectangular mesh.

        Determines which vertices belong to the specified side based on their texture
        coordinates. `side` is an enum that represents one of the four sides.

        :returns: A Mitsuba array of vertex indices that belong to the specified side.
        """
        # Use the texture coordinates to locate the vertices of the specified side
        if side == Patch.Side.LEFT:
            mask = self.texcoords[:, 0].numpy() < self.SIDE_TOLERANCE
        if side == Patch.Side.RIGHT:
            mask = self.texcoords[:, 0].numpy() > 1 - self.SIDE_TOLERANCE
        if side == Patch.Side.TOP:
            mask = self.texcoords[:, 1].numpy() > 1 - self.SIDE_TOLERANCE
        if side == Patch.Side.BOTTOM:
            mask = self.texcoords[:, 1].numpy() < self.SIDE_TOLERANCE

        indices = np.array(np.arange(self.vertices.shape[0])[mask], dtype=np.uint32)

        return dr.ravel(mi.TensorXu(indices))

    def get_node_indices_for(self, side: Patch.Side) -> "mi.ArrayXu":
        """
        Returns the node indices at the specified side of the rectangular mesh.

        Determines which vertices belong to the specified side based on their texture
        coordinates. `side` is an enum that represents one of the four sides.

        :returns: A Mitsuba array of vertex indices that belong to the specified side.
        """
        if side == Patch.Side.LEFT:
            return self._left_side_indices
        if side == Patch.Side.RIGHT:
            return self._right_side_indices
        if side == Patch.Side.TOP:
            return self._top_side_indices
        if side == Patch.Side.BOTTOM:
            return self._bottom_side_indices

    @property
    def nodes(self) -> "mi.TensorXf":
        """
        Nodes of the rectangular mesh, which are the vertices of the mesh.
        """
        return self.vertices

    @classmethod
    def unit_square(
        cls,
        mesh_name: str,
        width_subdivisions: int = 1,
        height_subdivisions: int = 1,
        bsdf: DiffuseBSDF | None = None,
    ) -> "RectangularMesh":
        """
        Creates a new unit square in the XY plane mesh with the specified number of
        subdivisions.

        The unit square mesh is defined by the four corner vertices:

                            y ^
                              |
            (-0.5, 0.5, 0) ---|--- (0.5, 0.5, 0)
                      |       |       |
                      |       |       |
                ----------------------------> x
                      |       |       |
                      |       |       |
            (-0.5,-0.5, 0) ---|--- (0.5,-0.5, 0)
                              |

        `width_subdivisions` and `height_subdivisions` control the number of
        quadrilaterals in the mesh, and consequently the number of vertices and faces.

        The mesh is subdivided into a grid of quadrilaterals, where each quadrilateral
        where each quadrilateral is defined by four vertices:

            idx01 --- idx11
              |  1    / |
              |    /    |
              | /    0  |
            idx00 --- idx10

        Thus, one can operate on a grid of quadrilaterals all at once - the faces are
        independent and which vertex belongs to which face is determined from the
        scheme above. Each quad has 2 triangles, denoted by 0 and 1 which are really
        offsets into the `faces` array - bottom triangles are denoted by 0 and top
        ones by 1. `idx00`, `idx10`, `idx01`, and `idx11` are lists of all vertex
        indices at the corners of the quadrilaterals.

        `bsdf` is used to control the appearance of the mesh in the scene.

        :raises ValueError: If `length_subdivisions` or `width_subdivisions` is less
            than 1.
        """
        if bsdf is None:
            bsdf = DiffuseBSDF()

        if height_subdivisions < 1 or width_subdivisions < 1:
            raise ValueError(
                "`length_subdivisions` and `width_subdivisions` must be at least 1."
            )

        corner_vertices = np.array(
            [[-0.5, -0.5, 0], [0.5, -0.5, 0], [-0.5, 0.5, 0], [0.5, 0.5, 0]],
            dtype=np.float64,
        )

        u_coords = (
            np.arange(width_subdivisions + 1, dtype=np.float64) / width_subdivisions
        )
        v_coords = (
            np.arange(height_subdivisions + 1, dtype=np.float64) / height_subdivisions
        )
        u_coords, v_coords = np.meshgrid(u_coords, v_coords, indexing="xy")

        u_coords = u_coords.reshape(-1, 1)
        v_coords = v_coords.reshape(-1, 1)

        vertices = np.array(
            corner_vertices[0][np.newaxis, :]
            + u_coords * (corner_vertices[1] - corner_vertices[0])[np.newaxis, :]
            + v_coords * (corner_vertices[2] - corner_vertices[0])[np.newaxis, :],
            dtype=np.float64,
        )

        i_indices = np.arange(height_subdivisions)
        j_indices = np.arange(width_subdivisions)
        i_grid, j_grid = np.meshgrid(i_indices, j_indices, indexing="xy")

        idx00 = j_grid * (width_subdivisions + 1) + i_grid
        idx10 = j_grid * (width_subdivisions + 1) + (i_grid + 1)
        idx01 = (j_grid + 1) * (width_subdivisions + 1) + i_grid
        idx11 = (j_grid + 1) * (width_subdivisions + 1) + (i_grid + 1)

        faces = np.zeros(
            shape=(height_subdivisions * width_subdivisions * 2, 3), dtype=np.uint32
        )
        faces[0::2, 0] = idx00.flatten()
        faces[0::2, 1] = idx10.flatten()
        faces[0::2, 2] = idx11.flatten()
        faces[1::2, 0] = idx00.flatten()
        faces[1::2, 1] = idx11.flatten()
        faces[1::2, 2] = idx01.flatten()

        texture_coordinates = np.hstack([u_coords, v_coords])

        return cls(
            mesh_name=mesh_name,
            vertices=vertices,
            faces=faces,
            texture_coordinates=texture_coordinates,
            bsdf=bsdf,
        )
