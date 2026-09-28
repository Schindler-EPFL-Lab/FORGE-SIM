from collections.abc import Sequence
from pathlib import Path

import mitsuba as mi

from forge_sim.meshing.composite_iga_spline_surface import CompositeIGASplineSurface
from forge_sim.meshing.tessellated_spline_surface import TessellatedSplineSurface
from forge_sim.plugin_wrappers.diffuse_bsdf import DiffuseBSDF


class TessellatedIGASplineSurface(TessellatedSplineSurface):
    """
    A specialization of TessellatedSplineSurface for isogeometric analysis (IGA).

    While `TessellatedSplineSurface` takes as input a `CompositeSplineSurface`,
    this class takes in a `CompositeIGASplineSurface`. Since the latter also inherits
    from `CompositeSplineSurface`, it can be used interchangeably with the former.

    All the logic of the base class is preserved while exposing the global field
    values required for isogeometric analysis.
    """

    _composite_spline_surface: CompositeIGASplineSurface

    def __init__(
        self,
        iga_spline_surface: CompositeIGASplineSurface,
        tessellation_subdivisions_per_patch: int,
        per_patch_bsdf: Sequence[DiffuseBSDF] | None = None,
        global_mesh_bsdf: DiffuseBSDF | None = None,
    ) -> None:
        """
        Initializes the `TessellatedIGASplineSurface` with a composite spline surface.

        The composite IGA spline surface must be properly initialized. Internally, the
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
        super().__init__(
            composite_spline_surface=iga_spline_surface,
            tessellation_subdivisions_per_patch=tessellation_subdivisions_per_patch,
            per_patch_bsdf=per_patch_bsdf,
            global_mesh_bsdf=global_mesh_bsdf,
        )

    @classmethod
    def from_file(
        cls,
        composite_spline_surface_path: Path,
    ) -> "TessellatedIGASplineSurface":
        """
        Loads a tessellated spline surface from file and sets the bsdf for each patch
        as a default DiffuseBSDF.

        This class method deserializes a previously saved `TessellatedSplineSurface`
        from a JSON file. Since BSDF information is not preserved during serialization,
        the result spline has the default textures.

        :returns: A decoded `TessellatedSplineSurface` instance with default textures

        :raises FileNotFoundError: If the surface file does not exist.
            RuntimeError: If the file is not valid JSON, or decodes to
                something other than a `TessellatedSplineSurface`.
        """
        tessellated_iga_spline_surface = super().from_file(
            composite_spline_surface_path=composite_spline_surface_path
        )
        if not isinstance(tessellated_iga_spline_surface, TessellatedIGASplineSurface):
            raise TypeError(
                f"Expected file to contain a TessellatedIGASplineSurface, "
                f"but got {type(tessellated_iga_spline_surface).__name__}. "
                f"Ensure the file at {composite_spline_surface_path} was saved "
                "from an IGA spline surface."
            )

        return tessellated_iga_spline_surface

    @classmethod
    def from_tessellated_spline_surface(
        cls,
        tessellated_spline_surface: TessellatedSplineSurface,
        per_patch_field_values: Sequence["mi.TensorXf"],
    ) -> "TessellatedIGASplineSurface":
        """
        Creates a `TessellatedIGASplineSurface` from an existing
        `TessellatedSplineSurface` and a sequence of field values for each patch.

        This method is useful when the optimized spline surface is already
        represented as a `TessellatedSplineSurface`, but one wants to convert it
        to an IGA-compatible representation.

        :returns: An instance of `TessellatedIGASplineSurface`.
        """
        if len(tessellated_spline_surface) != len(per_patch_field_values):
            raise ValueError(
                "The number of patches in the spline surface must match the number "
                "of provided field values."
            )

        composite_iga_spline = CompositeIGASplineSurface.from_spline_surface(
            spline_surface=tessellated_spline_surface.composite_spline,
            per_patch_field_values=per_patch_field_values,
        )

        return cls(
            iga_spline_surface=composite_iga_spline,
            tessellation_subdivisions_per_patch=tessellated_spline_surface.tessellation_subdivisions,
            per_patch_bsdf=[mesh.bsdf for mesh in tessellated_spline_surface],
        )

    @property
    def global_field_values(self) -> "mi.TensorXf":
        """
        Returns the global field values of the composite spline surface of shape
        (n_control_points, dofs_per_cp).
        """
        return self._composite_spline_surface.global_field_values

    @global_field_values.setter
    def global_field_values(self, field_values: "mi.TensorXf") -> None:
        """
        Updates the global field values of the composite spline surface.

        Unlike the control point update, which has to propagate to the mesh
        representation, the global field values are simply forwarded to the
        wrapped `CompositeIGASplineSurface`.
        """
        self._composite_spline_surface.global_field_values = field_values

    @property
    def composite_spline(self) -> CompositeIGASplineSurface:
        """
        Returns the composite IGA spline surface associated with this tessellated
        surface.
        """
        return self._composite_spline_surface
