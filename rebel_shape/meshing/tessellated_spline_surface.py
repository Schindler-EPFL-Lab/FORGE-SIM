from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import drjit as dr
import jsonpickle
import mitsuba as mi

from rebel_shape.meshing.composite_spline_surface import CompositeSplineSurface
from rebel_shape.meshing.rectangular_mesh import RectangularMesh
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF


class TessellatedSplineSurface:
    """
    Manages the composite spline surface and its tessellated mesh representation.

    During optimization, at the time of writing, Mitsuba does not support working
    directly with spline surfaces. Instead, the spline surface is tessellated and then
    the tessellated mesh is optimized. One has to be careful about state updates of:
    - global control points
    - per spline patch control points
    - global tessellated mesh
    - per patch tessellated meshes
    They are all interlinked. Incorrect updates could lead to discontinuities,
    incorrect normals, or stale states in one of the components, resulting in some
    patches not being updated correctly or the global mesh not reflecting the changes.

    Most of the handling is already done in the `CompositeSplineSurface` and
    `CompositeMesh` classes, so this class ensure the final step: synchronizing the
    tessellated mesh with the composite spline surface.
    """

    def __init__(
        self,
        composite_spline_surface: CompositeSplineSurface,
        tessellation_subdivisions_per_patch: int,
        per_patch_bsdf: Sequence[DiffuseBSDF] | None = None,
        global_mesh_bsdf: DiffuseBSDF | None = None,
    ) -> None:
        """
        Initializes the `TessellatedSplineSurface` with a composite spline surface.

        The composite spline surface has to be properly initialized. Internally, the
        `composite_mesh_tesselation` method of the `CompositeSplineSurface` is called
        to create the tessellated mesh representation of the spline surface.

        The `tessellation_subdivisions_per_patch` parameter specifies how many
        subdivisions along u and v directions will be used for each patch in the
        composite spline surface. This is used to determine the resolution of the
        tessellated mesh. There will be a total of 2 *
        `tessellation_subdivisions_per_patch` ** 2 triangles per patch.

        `per_patch_bsdf` is an optional sequence of BSDFs that are assigned to each
        patch mesh in the tessellated surface. It must match the number of patches
        in the composite spline surface and be ordered in the same way as the
        patches. If not provided, the patches will receive a default DiffuseBSDF.

        The `global_mesh_bsdf` is an optional BSDF that is assigned to the
        global tessellated mesh (union of all patch meshes). If not provided,
        the global mesh will receive a default DiffuseBSDF.

        :raises ValueError: If `tessellation_subdivisions_per_patch` is less than 1.
        """
        if tessellation_subdivisions_per_patch < 1:
            raise ValueError("`subdivisions_per_patch` must be at least 1.")
        self._triangle_subdivisions = tessellation_subdivisions_per_patch

        self._composite_spline_surface = composite_spline_surface
        self._tessellated_mesh = composite_spline_surface.composite_mesh_tesselation(
            subdivisions_per_patch=tessellation_subdivisions_per_patch,
            per_patch_bsdf=per_patch_bsdf,
            global_mesh_bsdf=global_mesh_bsdf,
        )

    @classmethod
    def from_file(
        cls,
        composite_spline_surface_path: Path,
    ) -> "TessellatedSplineSurface":
        """
        Loads a tessellated spline surface from file and sets the bsdf for each patch
        as a default DiffuseBSDF.

        This class method deserializes a previously saved `TessellatedSplineSurface`
        from a JSON file. Since BSDF information is not preserved during serialization,
        the result spline has the default textures.

        :returns: A decoded `TessellatedSplineSurface` instance with default textures

        :raises FileNotFoundError: If the surface file does not exist.
            JSONDecodeError: If the surface file contains invalid JSON.
        """
        if not composite_spline_surface_path.exists():
            raise FileNotFoundError(
                f"File {composite_spline_surface_path} does not exist."
            )
        if not composite_spline_surface_path.is_file():
            raise ValueError(f"Path {composite_spline_surface_path} is not a file.")

        with open(composite_spline_surface_path, "r") as tessellated_spline_file:
            tessellated_spline: TessellatedSplineSurface = jsonpickle.decode(
                tessellated_spline_file.read()
            )
        return tessellated_spline

    @classmethod
    def from_file_with_textures(
        cls, composite_spline_surface_path: Path, texture_paths: Sequence[Path]
    ) -> "TessellatedSplineSurface":
        """
        Load a tessellated spline surface from file and apply textures to each patch.

        This class method deserializes a previously saved `TessellatedSplineSurface`
        from a JSON file. Since BSDF information is not preserved during serialization,
        if the BSDF had an attached texture it must be saved separately.

        `texture_paths` are a sequence of paths to texture files that will be ordered
        to match the patches in the spline surface and the textures will be applied
        to each patch, restoring its full state. The method assumes that the texture
        files are named in such a way that they can be uniquely matched to patches
        based on their names. For example, if a patch is named "patch1", the texture
        file should contain "patch1" in its name or path. The method then

        :returns: A decoded `TessellatedSplineSurface` instance with textures applied to
            each patch, restoring its full state.

        :raises FileNotFoundError: If the surface file or any texture file does not
            exist.
            ValueError: If the number of texture files doesn't match the number of
                patches, or if patch names cannot be uniquely matched to texture files.
            JSONDecodeError: If the surface file contains invalid JSON.

        Note:
            The texture files will be reordered internally to match the patch order
            in the spline surface, regardless of the order they are provided in.
        """
        tessellated_spline = cls.from_file(
            composite_spline_surface_path=composite_spline_surface_path
        )

        # Checks the existence of texture files internally
        ordered_files = tessellated_spline.composite_spline.match_files_to_patches(
            patch_paths=texture_paths,
        )

        for texture_file, patch_mesh in zip(ordered_files, tessellated_spline):
            patch_mesh.bsdf = DiffuseBSDF(
                DiffuseBSDF.Texture(texture_path=texture_file)
            )

        return tessellated_spline

    def __getstate__(self) -> dict[str, Any]:
        """
        Returns the serialized state of the tessellated spline surface.

        Instead of keeping the entire state, we only keep the spline surface and the
        triangle subdivisions. The tessellated mesh is not stored as it is too delicate
        to serialize and store. It is only a representation and can be rebuilt
        from the composite spline surface at any time.

        :returns: A dictionary containing the state of the tessellated spline surface.
        """
        st = self.__dict__.copy()
        # Remove the tessellated mesh as it is not serializable and will be rebuilt
        st.pop("_tessellated_mesh")

        return st

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restore the tessellated spline surface from its serialized state.

        Mitsuba meshes are too delicate to store, so instead they are rebuilt from the
        composite spline surface the same way it was constructed in the
        `__init__` method.

        However, note that the BSDFs are not stored in the serialized state,
        so the `per_patch_bsdf` and `global_mesh_bsdf` parameters are not restored.
        They are initialized to the default values and the user is expected to
        set them after restoring the state if needed. This is because paths to
        textures can change, making BSDFs invalid if they are stored in the state.
        """
        self.__dict__.update(state)

        self._tessellated_mesh = (
            self._composite_spline_surface.composite_mesh_tesselation(
                subdivisions_per_patch=self._triangle_subdivisions
            )
        )

    @property
    def spline_name(self) -> str:
        """
        Name of the composite spline surface.
        """
        return self._composite_spline_surface.name

    @property
    def tessellation_subdivisions(self) -> int:
        """
        Number of subdivisions per patch in the tessellated spline surface.
        This is the number of subdivisions along u and v directions for each patch.
        """
        return self._triangle_subdivisions

    @property
    def mesh_name(self) -> str:
        """
        Name of the tessellated mesh.
        """
        return self._tessellated_mesh.name

    @property
    def global_vertices(self) -> "mi.TensorXf":
        """
        Returns the vertices of the tessellated spline surface (global mesh), shaped
        as (V, 3).
        """
        return self._tessellated_mesh.global_vertices

    @property
    def global_faces(self) -> "mi.TensorXf":
        """
        Returns the faces of the tessellated spline surface (global mesh), shaped as
        (F, 3).
        """
        return self._tessellated_mesh.global_faces

    @property
    def global_vertex_normals(self) -> "mi.TensorXf":
        """
        Returns the vertex normals of the tessellated spline surface (global mesh),
        shaped as (V, 3).
        """
        return self._tessellated_mesh.global_vertex_normals

    @property
    def global_control_points(self) -> "mi.TensorXf":
        """
        Returns the control points of the composite spline surface, shaped as (C, 3).
        """
        return self._composite_spline_surface.global_control_points

    @global_control_points.setter
    def global_control_points(self, value: "mi.TensorXf") -> None:
        """
        Updates global spline surface control points.

        Sets the control points of the composite spline surface, which triggers an
        update to local control points of each spline patch. Each spline patch then
        updates its tessellated patch based on the new control points. Finally, the
        global tessellated mesh is updated to reflect the changes.

        This method assumes that the texture coordinates of the tessellated mesh
        correspond to the parametric coordinates of the spline surface, which is the
        default assumption in the `tessellate` method of the `SplinePatch` class.
        """
        self._composite_spline_surface.global_control_points = value

        for spline, tesselation in zip(
            self._composite_spline_surface, self._tessellated_mesh
        ):
            # Update tessellated mesh based on texcoords
            parametric_coords = spline.texcoords_to_parametric_coords(
                tesselation.texcoords
            )

            vertex_positions = spline.evaluate_coordinates(parametric_coords)
            vertex_normals = spline.evaluate_normals(parametric_coords)

            tesselation.vertices = vertex_positions
            tesselation.vertex_normals = vertex_normals

        self._tessellated_mesh.update_global_representation()

    def transform_global_control_points(self, transform: "mi.Transform4f") -> None:
        """
        Applies a 4x4 transformation matrix `transform` to all global control points
        of the spline surface
        """

        original_control_points = dr.unravel(
            dtype=mi.Point3f,
            array=self.global_control_points.array,
        )

        new_control_points = transform @ original_control_points
        new_control_points = dr.ravel(new_control_points)
        new_control_points = dr.reshape(
            dtype=mi.TensorXf,
            value=new_control_points,
            shape=self.global_control_points.shape,
        )

        self.global_control_points = new_control_points

    @property
    def global_quads(self) -> "mi.TensorXu":
        """
        Global quads which contain the indices of the quads that make up the
        composite spline surface.

        The quads are stored in a Mitsuba tensor of shape (n_quads, 4), where n_quads
        is the number of quads in the composite spline surface.
        """
        return self._composite_spline_surface.global_quads

    @property
    def composite_spline(self) -> CompositeSplineSurface:
        """
        Returns the composite spline surface associated with this tessellated surface.
        """
        return self._composite_spline_surface

    def __getitem__(self, index: int) -> RectangularMesh:
        """
        Returns the tessellated mesh at the specified index.

        :raises IndexError: If the index is out of range.
        """
        if index < 0 or index >= len(self._tessellated_mesh):
            raise IndexError("Mesh index out of range.")

        return self._tessellated_mesh[index]

    def __len__(self) -> int:
        """
        Returns the number of tessellated meshes in the spline surface.
        """
        return len(self._tessellated_mesh)

    def __iter__(self) -> Iterator[RectangularMesh]:
        """
        Returns an iterator over the tessellated meshes in the spline surface.
        """
        return iter(self._tessellated_mesh)
