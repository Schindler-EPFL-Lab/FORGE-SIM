import mitsuba as mi

from rebel_shape.meshing.composite_iga_spline_surface import CompositeIGASplineSurface
from rebel_shape.meshing.tessellated_spline_surface import TessellatedSplineSurface


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
        composite_spline_surface: CompositeIGASplineSurface,
        tessellation_subdivisions_per_patch: int,
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

        :raises ValueError: If `tessellation_subdivisions_per_patch` is less than 1.
        """
        super().__init__(
            composite_spline_surface=composite_spline_surface,
            tessellation_subdivisions_per_patch=tessellation_subdivisions_per_patch,
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
