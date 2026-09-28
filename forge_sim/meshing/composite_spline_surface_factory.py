import drjit as dr

from forge_sim.meshing.composite_spline_surface import CompositeSplineSurface
from forge_sim.meshing.patch import Patch
from forge_sim.meshing.patch_manager_factory import PatchManagerFactory
from forge_sim.splines.spline_patch import SplinePatch


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

    def link_sides(
        self,
        first_patch_name: str,
        first_patch_side: Patch.Side,
        second_patch_name: str,
        second_patch_side: Patch.Side,
    ) -> None:
        """
        Links the specified sides of two selected patches.

        The `first_patch_index` and `second_patch_index` specify the indices of the
        patches to be connected, while `first_patch_side` and `second_patch_side`
        specify the sides of the spline patches that will be connected.

        The connection is established in both directions and the number of control
        points on the two sides must match. If either patch is already connected to
        another patch on that side, an error is raised.

        Furthermore, spline knot spans along the specified sides need to match in
        order to ensure surface continuity (C^0). This check, together with the control
        point count check, effectively enforces knot vector compatibility along the
        connected sides.

        The patches are not updated with the new connection; it is up to the
        concrete implementations of `PatchManager` to implement the desired behavior.

        :raises ValueError: If the provided patch names are incorrect, if the number of
            control points on the specified sides do not match, if either side is
            already connected to another spline patch, or if the knot spans along the
            specified sides do not match
        """
        super().link_sides(
            first_patch_name,
            first_patch_side,
            second_patch_name,
            second_patch_side,
        )

        first_patch_index = self._name_to_index_map[first_patch_name]
        second_patch_index = self._name_to_index_map[second_patch_name]

        first_patch = self._patch_list[first_patch_index]
        second_patch = self._patch_list[second_patch_index]

        first_knot_spans = first_patch.side_knot_spans(side=first_patch_side)
        second_knot_spans = second_patch.side_knot_spans(side=second_patch_side)

        if first_knot_spans.shape != second_knot_spans.shape:
            raise ValueError(
                f"Cannot link sides {first_patch_side} of patch {first_patch_name} "
                + f"and {second_patch_side} of patch {second_patch_name} due to "
                + "mismatched number of knot spans."
            )

        if not dr.allclose(first_knot_spans, second_knot_spans, rtol=1e-5, atol=1e-7):
            raise ValueError(
                f"Cannot link sides {first_patch_side} of patch {first_patch_name} "
                + f"and {second_patch_side} of patch {second_patch_name} due to "
                + "mismatched knot spans."
            )