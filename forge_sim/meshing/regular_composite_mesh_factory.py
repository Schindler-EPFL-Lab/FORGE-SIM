from forge_sim.meshing.composite_mesh_factory import CompositeMeshFactory
from forge_sim.meshing.regular_composite_mesh import RegularCompositeMesh


class RegularCompositeMeshFactory(CompositeMeshFactory):
    """
    Factory that builds a `RegularCompositeMesh` from a list of `RegularMesh` patches.

    Inherits `add_patch` and `link_sides` from `CompositeMeshFactory` /
    `PatchManagerFactory`. The only difference is the type produced by `build()` so
    the resulting composite supports `h_refinement`.
    """

    def build(self) -> RegularCompositeMesh:
        """
        Builds the regular composite mesh from the collected patches and connections.

        :returns: A `RegularCompositeMesh` ready for downstream optimization or
            rendering.
        """
        return RegularCompositeMesh(
            composite_mesh_name=self._manager_name,
            patch_list=self._patch_list,
            connection_list=self._connection_list,
            global_bsdf=self._bsdf,
        )
