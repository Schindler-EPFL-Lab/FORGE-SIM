import tempfile
from pathlib import Path

import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import numpy.typing as npt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from rebel_shape.plugin_wrappers.mitsuba_plugin import MitsubaPlugin


class Mesh(MitsubaPlugin):
    """
    A class that wraps `mitsuba.Mesh` and makes it more convenient to use.

    Mitsuba mesh uses flat buffers for all values and special care needs to be taken
    when updating certain properties; e.g., updating vertices requires updating mesh
    parameters but these also recompute the vertex normals and mesh bounding boxes.

    This class aims to bridge these gaps by making mesh handling simpler and less
    error-prone, as well supporting some functionality needed for patch handling.
    To maintain compatibility with Mitsuba scenes it inherits from `MitsubaPlugin`, buy
    it returns the `mi.Mesh` itself rather than a `dict`. For more details see
    https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/mesh_io_and_manipulation.html
    """

    NORMAL_TOLERANCE = 1e-6
    """How much the length of normals can deviate from 1.0"""

    def __init__(
        self,
        mesh_name: str,
        vertices: npt.NDArray[np.float64],
        faces: npt.NDArray[np.uint32],
        vertex_normals: npt.NDArray[np.float64] | None = None,
        texture_coordinates: npt.NDArray[np.float64] | None = None,
        bsdf: DiffuseBSDF | None = None,
    ) -> None:
        """
        Creates a new `Mesh` object with a unique `mesh_name`.

        `vertices` is a numpy array of shape (V, 3) containing the vertex positions,
        `faces` is a numpy array of shape (F, 3) containing the face indices.
        Optionally, `vertex_normals` and `texture_coordinates` can be provided as numpy
        arrays of shape (V, 3) and (V, 2), respectively. Texture coordinates must be
        in the range [0, 1] and normals must be normalized (length 1).

        Empty meshes are not allowed at the time of writing because Mitsuba does not
        support them.

        The `bsdf` is used to control the appearance of the mesh in the scene.

        :raises ValueError: If the shape of `vertices`, `faces`, `vertex_normals`, or
            `texture_coordinates` is not as expected, if the number of `vertices`,
            `texture_coordinates`, and `vertex_normals` does not match, if
            texture coordinates are not in the range [0, 1], if the normals are not
            normalized, or if the `faces` or `vertices` are empty.
        """
        if bsdf is None:
            bsdf = DiffuseBSDF()

        if len(faces.shape) != 2 or faces.shape[1] != 3:
            raise ValueError("`faces` must be a 2D array with shape (F, 3).")
        if len(vertices.shape) != 2 or vertices.shape[1] != 3:
            raise ValueError("`vertices` must be a 2D array with shape (V, 3).")

        # Mitsuba behavior is weird if the mesh is empty - empty meshes have
        # undefined normal behavior so they are not allowed
        if faces.shape[0] == 0:
            raise ValueError("`faces` must not be empty.")
        if vertices.shape[0] == 0:
            raise ValueError("`vertices` must not be empty.")

        if texture_coordinates is not None:
            if len(texture_coordinates.shape) != 2 or texture_coordinates.shape[1] != 2:
                raise ValueError(
                    "`texture_coordinates` must be a 2D array with shape (V, 2)."
                )
            if texture_coordinates.shape[0] != vertices.shape[0]:
                raise ValueError(
                    "`texture_coordinates` must have the same length as `vertices`."
                )
            if np.min(texture_coordinates) < 0.0 or np.max(texture_coordinates) > 1.0:
                raise ValueError("`texture_coordinates` must be in the range [0, 1].")

        if vertex_normals is not None:
            if len(vertex_normals.shape) != 2 or vertex_normals.shape[1] != 3:
                raise ValueError("`normals` must be a 2D array with shape (V, 3).")
            if vertex_normals.shape[0] != vertices.shape[0]:
                raise ValueError("`normals` must have the same length as `vertices`.")
            if (
                np.max(np.abs(np.sum(vertex_normals * vertex_normals, axis=1) - 1.0))
                > self.NORMAL_TOLERANCE
            ):
                raise ValueError("`normals` are not normalized!")

        self._mesh_name = mesh_name
        self._bsdf = bsdf

        self._mesh = self._create_mitsuba_mesh(
            mesh_name=mesh_name,
            vertices=vertices,
            faces=faces,
            normals=vertex_normals,
            texture_coordinates=texture_coordinates,
        )
        self._set_mesh_bsdf()
        self._mesh_params = mi.traverse(self._mesh)

    @classmethod
    def from_ply_file(cls, file_path: Path, bsdf: DiffuseBSDF | None = None) -> "Mesh":
        """
        Creates a new `Mesh` object from a PLY file.

        The PLY file at `file_path` must contain vertex positions and face indices. It
        uses the `mitsuba.Mesh` backend to load the mesh and extract the vertex
        positions and face indices, as well as optional vertex normals and texture
        coordinates.

        These are then passed to the constructor of the `Mesh` class with the
        appropriate `bsdf`.

        :raises FileNotFoundError: If `file_path` does not exist.
        :raises ValueError: If the provided `file_path` is not a file or not a PLY file.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        if not file_path.is_file():
            raise ValueError(f"File {file_path} is not a file.")
        if file_path.suffix != ".ply":
            raise ValueError(f"File {file_path} is not a PLY file.")

        mesh: mi.Mesh = mi.load_dict(
            {
                "type": "ply",
                "filename": str(file_path),
            }
        )

        vertices = mesh.vertex_positions_buffer().numpy().reshape(-1, 3)
        faces = mesh.faces_buffer().numpy().reshape(-1, 3)

        texture_coordinates = (
            np.array(
                mesh.vertex_texcoords_buffer().numpy().reshape(-1, 2), dtype=np.float64
            )
            if mesh.has_vertex_texcoords()
            else None
        )
        vertex_normals = (
            np.array(
                mesh.vertex_normals_buffer().numpy().reshape(-1, 3), dtype=np.float64
            )
            if mesh.has_vertex_normals()
            else None
        )

        return cls(
            mesh_name=mesh.id(),
            vertices=np.array(vertices, dtype=np.float64),
            faces=np.array(faces, dtype=np.uint32),
            vertex_normals=vertex_normals,
            texture_coordinates=texture_coordinates,
            bsdf=bsdf,
        )

    @staticmethod
    def _create_mitsuba_mesh(
        mesh_name: str,
        vertices: npt.NDArray[np.float64],
        faces: npt.NDArray[np.uint32],
        normals: npt.NDArray[np.float64] | None = None,
        texture_coordinates: npt.NDArray[np.float64] | None = None,
    ) -> "mi.Mesh":
        """
        Returns a Mitsuba mesh object from the given `vertices`, `faces`, `normals`,
        and `texture_coordinates`.

        `mesh_name` is the unique name of the mesh, `vertices` is a numpy array of shape
        (V, 3) containing the vertex positions, `faces` is a numpy array of shape
        (F, 3) containing the face indices, `normals` is a numpy array of shape (V, 3)
        containing the vertex normals, and `texture_coordinates` is a numpy array of
        shape (V, 2) containing the texture coordinates.

        Does not check these invariants - callers are responsible for maintaining them.
        """
        mesh = mi.Mesh(
            name=mesh_name,
            vertex_count=len(vertices),
            face_count=len(faces),
            has_vertex_normals=True,
            has_vertex_texcoords=texture_coordinates is not None,
        )
        mesh_params = mi.traverse(mesh)

        mesh_params["vertex_positions"] = dr.ravel(mi.TensorXf(vertices))
        mesh_params["faces"] = dr.ravel(mi.TensorXu(faces))

        if texture_coordinates is not None:
            mesh_params["vertex_texcoords"] = dr.ravel(mi.TensorXf(texture_coordinates))

        # This computes the vertex normals always and overrides the provided ones
        mesh_params.update()
        # So, do this to apply the correct, provided normals
        if normals is not None:
            mesh_params["vertex_normals"] = dr.ravel(mi.TensorXf(normals))

        return mesh

    def _set_mesh_bsdf(self) -> None:
        """
        Sets the BSDF of the mesh to the internal `_bsdf` object.

        This is a workaround; Mitsuba does not support setting the BSDF after the
        mesh has also been created but also the bsdf seemingly cannot be set in the
        constructor. Digging into the C++ source code did not reveal a mechanism
        to set it so this workaround is used.

        Effectively, the mesh is dumped into a temporary directory and then loaded
        again with a BSDF that is not the internal Mitsuba default, but rather the
        one provided by the user.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            self.save_to_ply(temp_dir)

            mesh_dict = {
                "type": "ply",
                "filename": str(temp_dir / f"{self._mesh_name}.ply"),
                "bsdf": self._bsdf.mitsuba_dict,
            }

            self._mesh = mi.load_dict(mesh_dict)

    @staticmethod
    def _is_v_by_3_2d_tensor(tensor: "mi.TensorXf") -> None:
        """
        Validates the shape of the vertices or normals.

        `vertices_or_normals` is a 2D tensor of shape (V, 3) where V is the number of
        vertices or normals.

        :raises ValueError: If the shape is not as expected, or if the number of
            vertices or normals is zero,
        """
        if len(tensor.shape) != 2:
            raise ValueError("Must be a 2D tensor!")
        if tensor.shape[0] == 0:
            raise ValueError("Must not be empty!")
        if tensor.shape[1] != 3:
            raise ValueError(
                f"Three coordinates are expected, but got {tensor.shape[1]}."
            )

    def _validate_indices_shape(self, indices: "mi.ArrayXu") -> None:
        """
        Validates the shape of the indices.

        `indices` is a 1D array of shape (N,) where N is the number of indices.

        :raises ValueError: If the shape is not as expected, or if the number of
            indices is zero.
        """
        if len(indices.shape) != 1:
            raise ValueError("`indices` must be a 1D array.")
        if indices.shape[0] == 0:
            raise ValueError("`indices` must not be empty.")
        if dr.max(indices) >= self._mesh.vertex_count():
            raise IndexError(
                f"Indices must be less than {self._mesh.vertex_count()}, "
                f"but got {dr.max(indices)}."
            )

    @property
    def vertices(self) -> "mi.TensorXf":
        """
        Vertex positions of the Mitsuba mesh, shaped as (V, 3).
        """
        return dr.reshape(
            dtype=mi.TensorXf,
            value=self._mesh.vertex_positions_buffer(),
            shape=(self._mesh.vertex_count(), 3),
        )

    @vertices.setter
    def vertices(self, vertices: "mi.TensorXf") -> None:
        """
        Sets the vertex positions of the Mitsuba mesh.

        `vertices` is a 2D tensor of shape (V, 3) where V is the number of vertices.
        Note that updating the vertices will automatically trigger an update of
        `vertex_normals`, so if specific normals are needed, they should be set
        after setting the vertices.

        :raises ValueError: If the number of vertices does not match the mesh, or
            if the shape of `vertices` is not as expected.
        """
        Mesh._is_v_by_3_2d_tensor(vertices)

        if vertices.shape[0] != self._mesh.vertex_count():
            raise ValueError(
                f"Expected {self._mesh.vertex_count()} vertices, "
                f"but got {vertices.shape[0]}."
            )

        self._mesh_params["vertex_positions"] = dr.ravel(vertices)
        self._mesh_params.update()

    def set_vertices_at(self, indices: "mi.ArrayXu", vertices: "mi.TensorXf") -> None:
        """
        Updates the vertex positions of the Mitsuba mesh at the specified `indices`.

        `indices` are a flat array of shape (V,) while `vertices` are a 2D tensor
        of shape (V, 3) where V is the number of vertices to update.

        :raises ValueError: If the `vertices` and `indices` do not follow the same shape
            requirements, if `vertices` and `indices` do not have the same length.
        :raises IndexError: If the `indices` array contains out-of-bounds indices.
        """
        self._validate_indices_shape(indices)
        Mesh._is_v_by_3_2d_tensor(vertices)

        if vertices.shape[0] != len(indices):
            raise ValueError(
                f"`vertices` and `indices` must have the same length, "
                f"but got {vertices.shape[0]} and {len(indices)}."
            )

        new_vertices = self.vertices

        # Mitsuba tensors can only be updated by linear arrays, so either everything
        # has to be flattened and then reshaped or the individual coordinates have to be
        # updated separately. The latter is done here.
        for coord in range(3):
            new_vertices[dr.ravel(indices), coord] = dr.ravel(vertices[:, coord])
        self.vertices = new_vertices

    @property
    def faces(self) -> "mi.TensorXu":
        """
        Face indices of the Mitsuba mesh, shaped as (F, 3).
        """
        return dr.reshape(
            dtype=mi.TensorXu,
            value=self._mesh.faces_buffer(),
            shape=(self._mesh.face_count(), 3),
        )

    @property
    def vertex_normals(self) -> "mi.TensorXf":
        """
        Vertex normals of the Mitsuba mesh, shaped as (V, 3).
        """
        return dr.reshape(
            dtype=mi.TensorXf,
            value=self._mesh.vertex_normals_buffer(),
            shape=(self._mesh.vertex_count(), 3),
        )

    @vertex_normals.setter
    def vertex_normals(self, normals: "mi.TensorXf") -> None:
        """
        Sets the vertex normals of the Mitsuba mesh.

        `normals` is a 2D tensor of shape (V, 3) where V is the number of vertices.

        :raises ValueError: If the number of normals does not match the mesh,
            if the shape of `normals` is not as expected, or if the normals are not
            normalized.
        """
        self._is_v_by_3_2d_tensor(normals)

        if normals.shape[0] != self._mesh.vertex_count():
            raise ValueError(
                f"Expected {self._mesh.vertex_count()} vertices, "
                f"but got {normals.shape[0]}."
            )

        max_deviation = dr.max(dr.abs(dr.sum(normals * normals, axis=1) - 1.0))
        if max_deviation > self.NORMAL_TOLERANCE:
            raise ValueError(
                "`normals` are not normalized! Maximum deviation "
                f"from 1.0 is {max_deviation.numpy():.6f}."
            )

        self._mesh_params["vertex_normals"] = dr.ravel(normals)

    def set_vertex_normals_at(
        self, indices: "mi.ArrayXu", normals: "mi.TensorXf"
    ) -> None:
        """
        Updates the vertex normals of the Mitsuba mesh at the specified `indices`.

        `indices` are a flat array of shape (N,) while `normals` are a 2D tensor
        of shape (N, 3), where N is the number of vertex normals to update.

        :raises ValueError: If the `normals` and `indices` do not follow the same shape
            requirements, if `normals` and `indices` do not have the same length,
            or if `normals` are not normalized.
        :raises IndexError: If the `indices` array contains out-of-bounds indices.
        """
        self._validate_indices_shape(indices)
        self._is_v_by_3_2d_tensor(normals)

        if normals.shape[0] != len(indices):
            raise ValueError(
                f"`normals` and `indices` must have the same length, "
                f"but got {normals.shape[0]} and {len(indices)}."
            )

        new_normals = self.vertex_normals

        # Same as in `set_vertices_at_indices`, update is done coordinate-wise
        for d in range(3):
            new_normals[dr.ravel(indices), d] = dr.ravel(normals[:, d])

        self.vertex_normals = new_normals

    @property
    def texcoords(self) -> "mi.TensorXf":
        """
        Vertex texture coordinates of the Mitsuba mesh.

        `texcoords` is a 2D tensor of shape (V, 2) where V is the number of vertices.

        :raises ValueError: If the mesh does not have texture coordinates.
        """
        if not self._mesh.has_vertex_texcoords():
            raise ValueError("Mesh does not have texture coordinates!")

        return dr.reshape(
            dtype=mi.TensorXf,
            value=self._mesh.vertex_texcoords_buffer(),
            shape=(self._mesh.vertex_count(), 2),
        )

    @property
    def bsdf(self) -> DiffuseBSDF:
        """
        Returns the BSDF of the mesh.
        """
        return self._bsdf

    @bsdf.setter
    def bsdf(self, bsdf: DiffuseBSDF) -> None:
        """
        Sets the BSDF of the mesh to the given `bsdf`.

        The BSDF is used to control the appearance of the mesh in the scene.
        """
        self._bsdf = bsdf
        self._set_mesh_bsdf()

    @property
    def name(self) -> str:
        """
        Name of the mesh.
        """
        return self._mesh_name

    def save_to_ply(self, output_dir: Path, save_name: str | None = None) -> None:
        """
        Saves the mesh to `output_dir` with the given `save_name` in the PLY format.

        If `save_name` is not provided, the `mesh_name` is used. The backend for
        saving the file is managed by Mitsuba. The resulting path is
        ``output_dir/save_name.ply``.

        :raises FileNotFoundError: If `output_dir` does not exist.
        :raises ValueError: If `output_dir` is not a directory.
        """
        if save_name is None:
            save_name = self._mesh_name

        if not output_dir.exists():
            raise FileNotFoundError(f"Output directory {output_dir} does not exist.")
        if not output_dir.is_dir():
            raise ValueError(f"Output path {output_dir} is not a directory.")

        # Dump the mesh to local storage
        mesh_output_path_str = str(output_dir / f"{save_name}.ply")
        self._mesh.write_ply(mesh_output_path_str)

    @property
    def mitsuba_dict(self) -> "mi.Mesh":
        """
        Returns the `mitsuba.Mesh` object that is wrapped by this class.

        This is one of the unique cases where one does not return a dict but the
        actual mesh object. Alternatively, one could dump the mesh to a file and
        define the path in the dict, but this would complicate the implementation.

        The mesh is inserted into the dict as ``{"mesh.name": mesh.mitsuba_dict}.``

        :returns: The Mitsuba mesh object to be put in the scene dict.
        """
        return self._mesh

    def visualize(
        self,
        show_wireframe: bool = True,
        show_normals: bool = True,
        show_vertices: bool = False,
    ) -> None:
        """
        Plot the tessellated spline surface using matplotlib.

        This method generates a 3D plot of the tessellated spline surface,
        displaying the surface, optionally one can `show_wireframe`, `show_normals`,
        or `show_vertices`. The surface is colored based on the texture coordinates.

        Tesselation is generated using the `tessellate` method with `u_subdivisions` and
        `v_subdivisions` subdivisions in the u and v directions, respectively.
        """
        # Convert the mesh data to numpy arrays for plotting
        vertices = self.vertices.numpy()
        normals = self.vertex_normals.numpy()
        faces = self.faces.numpy()

        texcoords = (
            self.texcoords.numpy()
            if self._mesh.has_vertex_texcoords()
            else np.zeros((vertices.shape[0], 2))
        )

        # Create figure with appropriate style
        fig = plt.figure(figsize=(10, 8), dpi=150)
        ax = fig.add_subplot(111, projection="3d")

        # Plot the mesh as a collection of triangles
        mesh_collection = Poly3DCollection([vertices[face] for face in faces])
        if show_wireframe:
            mesh_collection.set_edgecolor("black")
            mesh_collection.set_linewidth(0.2)
        else:
            mesh_collection.set_edgecolor("none")

        # Use viridis colormap for consistency with plot method
        cmap = plt.get_cmap("viridis")
        texcoords_color = cmap(0.5 * texcoords[:, 0] + 0.5 * texcoords[:, 1])

        # Extract face colors based on the average texture coordinate of each face
        face_colors = np.array([texcoords_color[face].mean(axis=0) for face in faces])

        mesh_collection.set_facecolor(face_colors)
        mesh_collection.set_alpha(0.7)
        ax.add_collection3d(mesh_collection)

        vector_length = 0.2

        # Plot vertices if requested
        if show_vertices:
            ax.scatter(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                color="red",
                s=20,
                alpha=0.8,
                label="Vertices",
            )

        # Plot normals if requested
        if show_normals:
            for start_point, normal in zip(vertices, normals):
                end_point = start_point + vector_length * normal

                ax.quiver(
                    start_point[0],
                    start_point[1],
                    start_point[2],
                    end_point[0] - start_point[0],
                    end_point[1] - start_point[1],
                    end_point[2] - start_point[2],
                    color="green",
                    arrow_length_ratio=vector_length,
                )

        # Set axis limits with proper padding
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
        z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()

        padding = max(
            vector_length, 0.1 * max(x_max - x_min, y_max - y_min, z_max - z_min)
        )

        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_zlim(z_min - padding, z_max + padding)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_title(f"Mesh '{self.name}' (V = {len(vertices)}, F = {len(faces)})")

        # Add a legend
        handles: list[plt.Artist] = []
        labels: list[str] = []
        # Create a proxy for the surface color in the legend
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        # Create a custom colormap for the legend
        proxy = plt.imshow(gradient, aspect="auto", cmap=plt.get_cmap("viridis"))
        proxy.set_visible(False)

        proxy = plt.Rectangle((0, 0), 1, 1, color="lightblue", alpha=0.7)
        handles.append(proxy)
        labels.append("Surface")

        if show_vertices:
            handles.append(
                plt.Line2D([0], [0], marker="o", color="r", linestyle="None")
            )
            labels.append("Vertices")

        if show_normals:
            handles.append(plt.Line2D([0], [0], color="g", linestyle="-"))
            labels.append("Normal Vectors")

        ax.legend(handles, labels, loc="upper right")

        plt.tight_layout()
        plt.show()
