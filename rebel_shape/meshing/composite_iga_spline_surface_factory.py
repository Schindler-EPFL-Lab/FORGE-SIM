from rebel_shape.meshing.composite_iga_spline_surface import CompositeIGASplineSurface
from rebel_shape.meshing.patch_manager_factory import PatchManagerFactory
from rebel_shape.splines.iga_spline_patch import IGASplinePatch


class CompositeIGASplineSurfaceFactory(PatchManagerFactory):
    """
    A factory class for creating composite IGA spline surfaces from a list of IGA
    compatible spline patches.

    This class allows the user to add multiple IGA spline patches and link their
    sides together to create a composite spline surface. In the end, the build
    method returns a `CompositeIGASplineSurface` object that is ready for
    downstream tasks such as optimization and isogeometric analysis.
    """

    _patch_list: list[IGASplinePatch]

    def __init__(self, manager_name: str) -> None:
        """
        Initialize the `CompositeIGAMeshFactory`.

        `manager_name` is a unique name associated with the composite mesh. It is
        used to identify the mesh in the Mitsuba scene.
        """
        super().__init__(manager_name)

    def build(self) -> CompositeIGASplineSurface:
        """
        Builds the composite mesh from the list of patches and their connections.

        :returns: A CompositeMesh object representing the composite mesh.
        """
        return CompositeIGASplineSurface(
            composite_spline_name=self._manager_name,
            patch_list=self._patch_list,
            connection_list=self._connection_list,
        )
