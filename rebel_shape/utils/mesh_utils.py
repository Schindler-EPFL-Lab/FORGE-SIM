from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np
import numpy.typing as npt

from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF


def create_cubedsphere(radius: float = 1.0, subdivisions: int = 0) -> "mi.Mesh":
    """
    Create a cubed sphere mesh by projecting a cube onto a sphere.

    The cubed sphere is of `radius` and is subdivided `subdivisions` times.
    0 subdivisions correspond to a cube (8 vertices, 12 triangular faces).
    Each subdivision adds 4 times the number of faces.

    This approach creates a more uniform mesh than a UV sphere while avoiding the
    pole pinching problem. It is especially useful for spline-based methods.

    :returns: A Mitsuba mesh object representing the cubed sphere

    :raises ValueError: if `radius` is a non-positive number or if `subdivisions` is a
        negative integer.
    """
    if radius <= 0.0:
        raise ValueError("`radius` must be a positive real number.")
    if subdivisions < 0:
        raise ValueError("`subdivisions` must be a non-negative integer.")

    vertices, faces = _create_cube(radius)

    for _ in range(subdivisions):
        vertices, faces, _ = subdivide_mesh(
            vertices=vertices, faces=faces, radius=radius
        )

    # Create texture coordinates (simple spherical mapping)
    texcoords = _cube_map_parametrization(vertices=vertices)
    mesh = create_mitsuba_mesh(
        mesh_name=f"cubedsphere_{subdivisions}",
        faces=faces,
        vertices=vertices,
        texture_coordinates=texcoords,
    )

    return mesh


def create_mitsuba_mesh(
    mesh_name: str,
    faces: npt.NDArray[np.float64],
    vertices: npt.NDArray[np.float64],
    texture_coordinates: npt.NDArray[np.float64] | None = None,
) -> "mi.Mesh":
    """
    Create a Mitsuba mesh named 'mesh_name`  from `vertices` and `faces`.

    Optionally, `texture_coordinates` can be provided to specify how a texture should
    be mapped to the mesh. Note that Mitsuba does not generate them independently
    so if they are not provided, the mesh cannot be used for texture optimization.

    :returns: A Mitsuba mesh object.

    :raises ValueError: if `faces` or `vertices` are empty or if `texture_coordinates`
        is not None and does not match the number of vertices.
    """
    if len(faces) < 1:
        raise ValueError("`faces` must contain at least one face.")
    if len(vertices) < 1:
        raise ValueError("`vertices` must contain at least one vertex.")
    if (
        texture_coordinates is not None
        and vertices.shape[0] != texture_coordinates.shape[0]
    ):
        raise ValueError(
            "`texture_coordinates` must have the same length as `vertices`."
        )

    mesh = mi.Mesh(
        mesh_name,
        vertex_count=len(vertices),
        face_count=len(faces),
        has_vertex_normals=False,
        has_vertex_texcoords=True if texture_coordinates is not None else False,
    )
    mesh_params = mi.traverse(mesh)

    # The data has to be flattened to be passed to Mitsuba
    vertex_positions = mi.TensorXf(vertices.ravel())
    mesh_params["vertex_positions"] = dr.ravel(vertex_positions)

    face_indices = mi.TensorXu(faces.ravel())
    mesh_params["faces"] = dr.ravel(face_indices)

    if texture_coordinates is not None:
        vertex_texcoords = mi.TensorXf(texture_coordinates.ravel())
        mesh_params["vertex_texcoords"] = dr.ravel(vertex_texcoords)

    mesh_params.update()

    return mesh


def _create_cube(
    radius: float = 1.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Generate vertices and faces for a cube projected onto a sphere of `radius`.

    This cube is used as the base for creating a cubed sphere mesh.
    """
    # Create 8 vertices of a unit cube
    vertices = np.array(
        [
            [-1, -1, -1],  # 0: bottom-left-back
            [1, -1, -1],  # 1: bottom-right-back
            [1, 1, -1],  # 2: top-right-back
            [-1, 1, -1],  # 3: top-left-back
            [-1, -1, 1],  # 4: bottom-left-front
            [1, -1, 1],  # 5: bottom-right-front
            [1, 1, 1],  # 6: top-right-front
            [-1, 1, 1],  # 7: top-left-front
        ],
        dtype=np.float32,
    )
    vertices = radius * vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    # Define 12 triangular faces (6 square faces, each made of 2 triangles)
    faces = np.array(
        [
            [4, 5, 7],
            [5, 6, 7],  # 0 and 1: Front face
            [1, 0, 2],
            [0, 3, 2],  # 2 and 3: Back face
            [0, 4, 3],
            [4, 7, 3],  # 4 and 5: Left face
            [5, 1, 6],
            [1, 2, 6],  # 6 and 7: Right face
            [7, 6, 3],
            [6, 2, 3],  # Top face
            [0, 1, 4],
            [1, 5, 4],  # Bottom face
        ],
        dtype=np.int32,
    )

    return vertices, faces


def subdivide_mesh(
    vertices: npt.NDArray[np.float64],
    faces: npt.NDArray[np.float64],
    texture_coordinates: npt.NDArray[np.float64] | None = None,
    radius: float | None = None,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.int32], npt.NDArray[np.float64] | None
]:
    """
    Subdivide each face of the mesh into 4 smaller triangles.

    Each face is subdivided by introducing midpoints on each edge and connecting them
    to form 4 new triangles.

    if `radius` is provided, the new vertices are projected onto a sphere of that
    radius. It is assumed that the mesh is centered at the origin.

    If `texture_coordinates` are provided, they are also interpolated for the new
    vertices, otherwise a `None` is returned for new texture coordinates.

    :returns: A tuple of (new_vertices, new_faces, new_texcoords)

    :raises ValueError: if `faces` or `vertices` are empty, if `texture_coordinates`
        is not None and does not match the number of vertices, or if `radius` is not
        `None` and is not a positive real number.
    """
    if len(faces) < 1:
        raise ValueError("`faces` must contain at least one face.")
    if len(vertices) < 1:
        raise ValueError("`vertices` must contain at least one vertex.")
    if (
        texture_coordinates is not None
        and vertices.shape[0] != texture_coordinates.shape[0]
    ):
        raise ValueError(
            "`texture_coordinates` must have the same length as `vertices`."
        )
    if radius is not None and radius <= 0.0:
        raise ValueError("`radius` must be a positive real number.")

    new_faces: list[list[float]] = []
    new_vertices = vertices.copy()
    new_texcoords = (
        texture_coordinates.copy() if texture_coordinates is not None else None
    )
    # Caches edge midpoints to avoid repetition
    edge_midpoints: dict[tuple[int, int], int] = {}

    for face in faces:
        v1, v2, v3 = face

        # Get or create midpoints for each edge
        edges = [(v1, v2), (v2, v3), (v3, v1)]
        midpoints = []

        for p1, p2 in edges:
            # Check cache to avoid duplicating vertices
            edge = tuple(sorted([p1, p2]))
            if edge in edge_midpoints:
                midpoints.append(edge_midpoints[edge])
                continue

            # Create new midpoint
            mid = 0.5 * (new_vertices[p1] + new_vertices[p2])

            # Project to sphere if requested
            if radius is not None:
                # If radius is None, project to same length as original vertex
                mid = radius * mid / np.linalg.norm(mid)

            # Add to vertices list and cache
            mid_idx = len(new_vertices)
            new_vertices = np.vstack((new_vertices, mid))
            edge_midpoints[edge] = mid_idx
            midpoints.append(mid_idx)

            if texture_coordinates is not None:
                assert new_texcoords is not None
                mid_texcoord = 0.5 * (new_texcoords[p1] + new_texcoords[p2])
                new_texcoords = np.vstack((new_texcoords, mid_texcoord))

        m1, m2, m3 = midpoints

        # Create 4 new triangles
        new_faces.append([v1, m1, m3])
        new_faces.append([v2, m2, m1])
        new_faces.append([v3, m3, m2])
        new_faces.append([m1, m2, m3])

    return new_vertices, np.array(new_faces, dtype=np.int32), new_texcoords


def visualize_mesh_parametrization(mesh_path: Path) -> "mi.Mesh":
    """
    Load a mesh from `mesh_path` and visualize how the texture coordinates are mapped.

    Intended to be used for debugging purposes and is included in the code for future
    visualization of the parametrization process.
    """
    mesh = mi.load_dict(
        {
            "type": "ply",
            "filename": str(mesh_path),
            "bsdf": DiffuseBSDF(
                reflectance=DiffuseBSDF.MeshAttributeTexture(
                    attribute_key="vertex_color"
                )
            ).mitsuba_dict,
        }
    )
    # Color vertices based on texcoords
    tex_coords = mesh.vertex_texcoords_buffer()
    color_list = []
    tex_coords_list = tex_coords.array.numpy().tolist()
    for i, coord in enumerate(tex_coords_list):
        color_list.append(coord)
        if i % 2 == 0:
            color_list.append(0.0)

    mesh.add_attribute("vertex_color", 3, color_list)

    return mesh


def _cube_map_parametrization(
    vertices: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Computes a cube map parametrization for vertices of a sphere.

    This maps each point of the sphere to one of six faces of a cube, which are then
    laid out in a square domain [0, 1] x [0, 1].

    The layout follows this pattern:
    v        +---+
    ^        | To|
    |    +---+---+---+---+
    |    | L | F | R | B |
    |    +---+---+---+---+
    |        | Bo|
    |        +---+
    ---------------------> u
    Where:
    - Bo (Bottom, -Y): [0.25, 0.5] x [0.0, 0.25]
    - L (Left, -X): [0.0, 0.25] x [0.25, 0.5]
    - F (Front, +Z): [0.25, 0.5] x [0.25, 0.5]
    - R (Right, +X): [0.5, 0.75] x [0.25, 0.5]
    - B (Back, -Z): [0.75, 1.0] x [0.25, 0.5]
    - To (Top, +Y): [0.25, 0.5] x [0.5, 0.75]

    To preserve the equal area properties of the cube map, an arctan mapping is used
    for each of the six faces.

    :returns: A 2D array of shape (N, 2) where N is the number of vertices.
    """
    if len(vertices) < 1:
        raise ValueError("`vertices` must contain at least one vertex.")

    uv = np.zeros((len(vertices), 2), dtype=np.float32)

    for i, vertex in enumerate(vertices):
        x, y, z = vertex
        abs_x, abs_y, abs_z = abs(x), abs(y), abs(z)

        # Determine which face this vertex projects onto based on dominant axis
        if abs_x >= abs_y and abs_x >= abs_z:
            sc = np.arctan(z / abs_x) / (0.25 * np.pi)
            tc = np.arctan(y / abs_x) / (0.25 * np.pi)

            if x >= 0:  # Right face (+X)
                face = "R"
                s = -sc
                t = tc
            else:  # Left face (-X)
                face = "L"
                s = sc
                t = tc

        elif abs_y >= abs_x and abs_y >= abs_z:
            sc = np.arctan(x / abs_y) / (0.25 * np.pi)
            tc = np.arctan(z / abs_y) / (0.25 * np.pi)

            if y >= 0:  # Top face (+Y)
                face = "To"
                s = sc
                t = -tc
            else:  # Bottom face (-Y)
                face = "Bo"
                s = sc
                t = tc

        else:
            sc = np.arctan(x / abs_z) / (0.25 * np.pi)
            tc = np.arctan(y / abs_z) / (0.25 * np.pi)

            if z >= 0:  # Front face (+Z)
                face = "F"
                s = sc
                t = tc
            else:  # Back face (-Z)
                face = "B"
                s = -sc
                t = tc

        # Convert local coordinates from [-1, 1] to [0, 0.25]
        s = 0.25 * 0.5 * (s + 1.0)
        t = 0.25 * 0.5 * (t + 1.0)

        # Map to appropriate region in the texture space
        u, v = None, None
        if face == "Bo":
            u = s + 0.25
            v = t
        if face == "L":
            u = s
            v = t + 0.25
        if face == "F":
            u = s + 0.25
            v = t + 0.25
        if face == "R":
            u = s + 0.5
            v = t + 0.25
        if face == "B":
            u = s + 0.75
            v = t + 0.25
        if face == "To":
            u = s + 0.25
            v = t + 0.5

        assert u is not None
        assert v is not None
        uv[i] = [u, v]

    return uv
