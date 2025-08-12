from rebel_shape.meshing.composite_mesh import CompositeMesh
from rebel_shape.meshing.patch_manager_factory import PatchManagerFactory
from rebel_shape.meshing.rectangular_mesh import RectangularMesh
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF


class CompositeMeshFactory(PatchManagerFactory):
    """
    A factory class for creating composite meshes from a list of rectangular patches.

    This class allows the user to add multiple rectangular meshes and link their
    sides together to create a composite mesh. In the end, the build
    method returns a `CompositeMesh` object that is ready for downstream tasks
    such as rendering or optimization.
    """

    _patch_list: list[RectangularMesh]

    def __init__(self, manager_name: str, bsdf: DiffuseBSDF | None = None) -> None:
        """
        Initialize the `CompositeMeshFactory` with defaults from the parent class.

        `manager_name` is a unique name associated with the composite mesh. It is
        used to identify the mesh in the Mitsuba scene. The `bsdf` parameter is
        optional and can be used to specify a BSDF for the global mesh created by
        `CompositeMesh`.
        """
        super().__init__(manager_name)

        if bsdf is None:
            bsdf = DiffuseBSDF()
        self._bsdf = bsdf

    def build(self) -> CompositeMesh:
        """
        Builds the composite mesh from the list of patches and their connections.

        The constructor of composite mesh is required to correctly build the global
        mesh representation and ensure it is ready for downstream tasks.

        :returns: A CompositeMesh object representing the composite mesh.
        """
        return CompositeMesh(
            composite_mesh_name=self._manager_name,
            patch_list=self._patch_list,
            connection_list=self._connection_list,
            global_bsdf=self._bsdf,
        )
