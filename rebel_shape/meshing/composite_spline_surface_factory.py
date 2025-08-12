from rebel_shape.meshing.composite_spline_surface import CompositeSplineSurface
from rebel_shape.meshing.patch_manager_factory import PatchManagerFactory
from rebel_shape.splines.spline_patch import SplinePatch


class CompositeSplineSurfaceFactory(PatchManagerFactory):
    """
    A factory class for creating composite spline surfaces from a list of spline
    patches.

    This class allows the user to add multiple spline patches and link their
    sides together to create a composite spline surface. In the end, the build
    method returns a `CompositeSplineSurface` object that is ready for
    downstream tasks such as rendering or optimization.
    """

    _patch_list: list[SplinePatch]

    def __init__(self, manager_name: str) -> None:
        """
        Initialize the CompositeMeshFactory.

        `manager_name` is a unique name associated with the composite mesh. It is
        used to identify the mesh in the Mitsuba scene.
        """
        super().__init__(manager_name)

    def build(self) -> CompositeSplineSurface:
        """
        Builds the composite mesh from the list of patches and their connections.

        :returns: A CompositeMesh object representing the composite mesh.
        """
        return CompositeSplineSurface(
            composite_spline_name=self._manager_name,
            patch_list=self._patch_list,
            connection_list=self._connection_list,
        )
