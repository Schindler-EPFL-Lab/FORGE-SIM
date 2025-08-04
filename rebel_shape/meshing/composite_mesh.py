from collections.abc import Iterator, Sequence

import drjit as dr
import mitsuba as mi
import numpy as np
import numpy.typing as npt

from rebel_shape.meshing.mesh import Mesh
from rebel_shape.meshing.patch_manager import PatchManager
from rebel_shape.meshing.rectangular_mesh import RectangularMesh
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF


class CompositeMesh(PatchManager):
    """
    A class that handles multiple rectangular patches and updates the global mesh
    that represents the entire composite mesh.

    Typically, it is difficult to obtain a texture parametrization that is
    free of seams and discontinuities. However, naively combining multiple meshes
    or introducing additional vertices to cure this behavior (Blender behavior) can
    lead to tears during shape optimization. Instead, each patch is maintained as
    an independent entity, but the global mesh that contains all the patches is
    updated to reflect the changes in the individual patches.

    Apart from issues during optimization, another motivation to introduce this
    "patch manager" is to support splines - they are defined as a collection of
    patches, and the composite mesh is a natural way to represent them during
    optimization.

    `CompositeMesh` should be built using the `CompositeMeshFactory` class, which is
    responsible for adding patches and establishing connections between them. So, the
    user should not call the constructor directly and the patch structure is static.

    If the vertex positions or normals of the patches change, the global mesh has to be
    updated. This is done by calling `update_global_mesh`. It updates the vertex
    information of all patches, averaging them at the connections sides, to avoid
    discontinuities. Normals are also updated from the initial patch normals, and they
    are averaged at the connections.
    """

    _patch_list: Sequence[RectangularMesh]
    """Array of `Rectangular` instances that constitute the composite mesh"""

    def __init__(
        self,
        composite_mesh_name: str,
        patch_list: Sequence[RectangularMesh],
        connection_list: Sequence[PatchManager.PatchConnections],
        bsdf: DiffuseBSDF | None = None,
    ) -> None:
        """
        Creates a new composite mesh object with a unique name `composite_mesh_name`.

        This method is not intended to be called directly, but rather through the
        `CompositeMeshFactory` class. The factory class is responsible for adding
        patches and establishing connections between them. It calls this method
        with appropriate `patch_list`, connection_list, and `bsdf` parameters.
        """
        if bsdf is None:
            bsdf = DiffuseBSDF()
        self._bsdf = bsdf

        super().__init__(
            name=composite_mesh_name,
            patch_list=patch_list,
            connection_list=connection_list,
        )

        self._build_global_mesh()
        self.update_global_representation()

    def _build_global_mesh(self) -> None:
        """
        Builds the global mesh representation from the patches and builds a Mitsuba
        mesh object that can be used in the Mitsuba scene.

        Unlike building the mapping, this method is less expensive as it can be
        vectorized, but it still requires to go through all the patches to
        construct the global mesh.

        The position of the vertices on the edges is not averaged but rather the
        last patch that was inserted is used to determine the position of the
        vertex. This choice is not propagated to the patches so the representation is
        not fully consistent yet. Calling `update_global_representation` will
        average the vertex positions and normals across all patches and propagate
        the changes to all connected patches.
        """
        num_global_vertices = len(self._global_to_patches_map)

        global_vertices: npt.NDArray[np.float32] = np.zeros(
            shape=(num_global_vertices, 3), dtype=np.float32
        )
        global_faces: list[npt.NDArray[np.uint32]] = []

        for patch_idx, patch in enumerate(self._patch_list):
            local_to_global = self._patch_to_global_map[patch_idx]
            global_vertices[local_to_global] = patch.vertices.numpy()
            global_faces.extend(local_to_global[patch.faces.numpy()])

        # The vertices here might not be averaged properly, call `update_global_mesh`
        self._global_mesh = Mesh(
            mesh_name=self.name,
            vertices=np.array(global_vertices, dtype=np.float64),
            faces=np.array(global_faces, dtype=np.uint32),
            vertex_normals=None,
            texture_coordinates=None,
            bsdf=self._bsdf,
        )

    def __getitem__(self, item: int) -> RectangularMesh:
        """
        Returns the mesh patch with the specified index.

        :raises IndexError: If the index is out of range.
        """
        if item < 0 or item >= len(self._patch_list):
            raise IndexError("Mesh index out of range.")

        return self._patch_list[item]

    def __len__(self) -> int:
        """
        Returns the number of patches in the composite mesh.
        """
        return len(self._patch_list)

    def __iter__(self) -> Iterator[RectangularMesh]:
        """
        Returns an iterator over the patches in the composite mesh.
        """
        return iter(self._patch_list)

    def update_global_representation(self) -> None:
        """
        Updates the global mesh representation propagates changes in vertex positions
        and normals to all connected patches.

        This method is called to ensure that the global mesh is up-to-date with the
        current state of the patches. To avoid discontinuities between the patches,
        the average vertex positions and normals are computed for each shared vertex.
        This updated information is then propagated to all connected patches.
        """
        new_vertices = dr.zeros(dtype=mi.TensorXf, shape=self.global_vertices.shape)
        new_normals = dr.zeros(
            dtype=mi.TensorXf, shape=self.global_vertex_normals.shape
        )

        assert new_vertices.shape[0] == len(self._global_to_patches_map)
        assert new_normals.shape[0] == len(self._global_to_patches_map)

        for patch_idx, patch in enumerate(self._patch_list):
            local_to_global = mi.TensorXu(self._patch_to_global_map[patch_idx])

            self._scatter_values(
                target=new_vertices, values=patch.vertices, indices=local_to_global
            )
            self._scatter_values(
                target=new_normals, values=patch.vertex_normals, indices=local_to_global
            )

        self.global_vertices = self._averaging_weights * new_vertices

        new_normals = self._averaging_weights * new_normals
        magnitudes = dr.sqrt(dr.sum(new_normals * new_normals, axis=1))
        magnitudes = dr.reshape(dtype=mi.TensorXf, value=magnitudes, shape=(-1, 1))

        self.global_vertex_normals = new_normals / magnitudes

    @property
    def global_vertices(self) -> "mi.TensorXf":
        """
        Vertex positions of the Mitsuba mesh, shaped as (V, 3).

        Global vertices are the union of all vertices from the patches,
        where each vertex is represented only once. Updates to patch vertices are
        not reflected in the global mesh until `update_global_representation` is called.
        """
        return self._global_mesh.vertices

    @global_vertices.setter
    def global_vertices(self, vertices: "mi.TensorXf") -> None:
        """
        Updates the vertices of the mesh and propagates the changes to all
        connected meshes.

        `vertices` is a 2D tensor of shape (V, 3) where V is the number of vertices
        of the global mesh. Note that updating the vertices will automatically
        trigger an update of `vertex_normals`, so if specific normals are needed,
        they should be set after setting the vertices.

        :raises ValueError: If the number of vertices does not match the mesh, or
            if the shape of `vertices` is not as expected.
        """
        self._global_mesh.vertices = vertices

        for patch_idx, patch in enumerate(self._patch_list):
            local_to_global = dr.ravel(
                mi.TensorXf(self._patch_to_global_map[patch_idx])
            )
            patch.vertices = vertices[local_to_global]

    @property
    def global_faces(self) -> "mi.TensorXi":
        """
        Face indices of the global mesh, shaped as (F, 3).
        """
        return self._global_mesh.faces

    @property
    def global_vertex_normals(self) -> "mi.TensorXf":
        """
        Vertex normals of the global mesh, shaped as (V, 3).
        """
        return self._global_mesh.vertex_normals

    @global_vertex_normals.setter
    def global_vertex_normals(self, normals: "mi.TensorXf") -> None:
        """
        Sets the vertex normals of the global mesh and propagates the changes to all
        connected meshes.

        `normals` is a 2D tensor of shape (V, 3) where V is the number of vertices
        of the global mesh.

        :raises ValueError: If the number of normals does not match the mesh,
            if the shape of `normals` is not as expected, or if the normals are not
            normalized.
        """
        self._global_mesh.vertex_normals = normals

        for patch_idx, patch in enumerate(self._patch_list):
            local_to_global = dr.ravel(
                mi.TensorXf(self._patch_to_global_map[patch_idx])
            )
            patch.vertex_normals = normals[local_to_global]

    def visualize(
        self,
        show_wireframe: bool = True,
        show_normals: bool = True,
        show_vertices: bool = False,
    ) -> None:
        """
        Visualizes the global mesh using Matplotlib.

        This method is useful for debugging and visualizing the global mesh
        structure. It shows the vertices, faces, and optionally the normals and
        wireframe of the mesh.
        """
        self._global_mesh.visualize(
            show_wireframe=show_wireframe,
            show_normals=show_normals,
            show_vertices=show_vertices,
        )
