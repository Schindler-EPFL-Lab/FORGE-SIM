from collections.abc import Iterator, Sequence
from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np
import numpy.typing as npt

from forge_sim.meshing.mesh import Mesh
from forge_sim.meshing.patch_manager import PatchManager
from forge_sim.meshing.rectangular_mesh import RectangularMesh
from forge_sim.plugin_wrappers.diffuse_bsdf import DiffuseBSDF


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
        global_bsdf: DiffuseBSDF | None = None,
    ) -> None:
        """
        Creates a new composite mesh object with a unique name `composite_mesh_name`.

        This method is not intended to be called directly, but rather through the
        `CompositeMeshFactory` class. The factory class is responsible for adding
        patches and establishing connections between them. It calls this method
        with appropriate `patch_list`, and connection_list parameters.

        Additionally, the `global_bsdf` parameter can be used to set the BSDF of the
        global mesh that represents the entire composite mesh. If not provided, a
        default `DiffuseBSDF` is used.

        :raises ValueError:  If the number of patches and connection lists do not match,
            or if they are empty.
        """
        if global_bsdf is None:
            global_bsdf = DiffuseBSDF()

        super().__init__(
            name=composite_mesh_name,
            patch_list=patch_list,
            connection_list=connection_list,
        )

        self._build_global_mesh(global_bsdf)
        self.update_global_representation()

    def __getstate__(self) -> dict[str, Any]:
        """
        Exports the composite mesh state for serialization with `jsonpickle`.

        `_averaging_weights` is a Mitsuba tensor and is stored as a numpy array; the
        global mesh `_global_mesh` is dropped and rebuilt from the patches on load.
        Each patch serializes itself (see `Mesh.__getstate__`).

        :returns: A dictionary representing the composite mesh state.
        """
        state = self.__dict__.copy()
        state["_averaging_weights"] = self._averaging_weights.numpy()
        del state["_global_mesh"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restores the composite mesh, rebuilding the global mesh from the patches.

        The patches and patch-to-global mappings are restored from `state`, then the
        welded global mesh is rebuilt with a default BSDF and re-averaged. A Mitsuba
        variant must already be set by the caller.
        """
        averaging_weights = state.pop("_averaging_weights")
        self.__dict__.update(state)

        self._averaging_weights = mi.TensorXf(averaging_weights)
        self._build_global_mesh(DiffuseBSDF())
        self.update_global_representation()

    def _build_global_mesh(self, global_bsdf: DiffuseBSDF) -> None:
        """
        Builds the global mesh representation from the patches and builds a Mitsuba
        mesh object that can be used in the Mitsuba scene.

        Unlike building the mapping, this method is less expensive as it can be
        vectorized, but it still requires to go through all the patches to
        construct the global mesh.

        `global_bsdf` is the BSDF that will be assigned to the global mesh.
        It is used to define the material properties of the entire composite mesh.
        This state might not be consistent with the patches, as the BSDF is not
        propagated to the patches during updates.

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

        for patch, mapping in self.patches_with_mapping:
            global_vertices[mapping] = patch.vertices.numpy()
            global_faces.extend(mapping[patch.faces.numpy()])

        # The vertices here might not be averaged properly, call `update_global_mesh`
        self._global_mesh = Mesh(
            mesh_name=self.name,
            vertices=np.array(global_vertices, dtype=np.float64),
            faces=np.array(global_faces, dtype=np.uint32),
            vertex_normals=None,
            texture_coordinates=None,
            bsdf=global_bsdf,
        )

    def __getitem__(self, item: int) -> RectangularMesh:
        """
        Returns the mesh patch with the specified index.

        :raises IndexError: If the index is out of range.
        """
        if item < 0 or item >= len(self._patch_list):
            raise IndexError("Mesh index out of range.")

        return self._patch_list[item]

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

        for patch, mapping in self.patches_with_mapping:
            local_to_global = mi.TensorXu(mapping)

            self._scatter_values(
                target=new_vertices, values=patch.vertices, indices=local_to_global
            )
            self._scatter_values(
                target=new_normals, values=patch.vertex_normals, indices=local_to_global
            )

        self.global_vertices = self._averaging_weights * new_vertices

        new_normals = self._averaging_weights * new_normals
        magnitudes = dr.safe_sqrt(dr.sum(new_normals * new_normals, axis=1))

        self.global_vertex_normals = new_normals / magnitudes[:, None]

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

        for patch, mapping in self.patches_with_mapping:
            patch.vertices = vertices[dr.ravel(mi.TensorXf(mapping))]

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

        for patch, mapping in self.patches_with_mapping:
            patch.vertex_normals = normals[dr.ravel(mi.TensorXf(mapping))]

    @property
    def global_mesh_bsdf(self) -> DiffuseBSDF:
        """
        Returns the BSDF of the global mesh
        """
        return self._global_mesh.bsdf

    @global_mesh_bsdf.setter
    def global_mesh_bsdf(self, global_bsdf: DiffuseBSDF) -> None:
        """
        Sets the BSDF of the global mesh to the new `bsdf` value.
        """
        self._global_mesh.bsdf = global_bsdf

    @property
    def global_mesh(self) -> Mesh:
        """
        The welded global `Mesh` that merges all patches, with seam vertices
        shared across patch boundaries.

        Kept in sync with the patches by `update_global_representation`; saving it
        (e.g. via `global_mesh.save_to_ply`) yields the full surface as a single
        watertight mesh instead of one PLY per patch.
        """
        return self._global_mesh

    @property
    def patches_with_mapping(self) -> Iterator[tuple[Mesh, npt.NDArray[np.uint32]]]:
        """
        Returns an iterator of tuples containing the mesh patches with the corresponding
        local-to-global mapping.
        """
        return zip(self._patch_list, self._patch_to_global_map)

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
