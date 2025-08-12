import abc

from rebel_shape.meshing.patch import Patch
from rebel_shape.meshing.patch_manager import PatchManager


class PatchManagerFactory(abc.ABC):
    """
    An Abstract factory class for creating patch managers.

    This class defines the interface for creating patch managers, which are responsible
    for managing patches and their connections in a global representation.

    The factory allows adding patches, linking their sides, and building a
    `PatchManager` instance (required to be implemented by concrete implementations).
    """

    def __init__(self, manager_name: str) -> None:
        """
        Initialize the CompositeMeshFactory.

        `manager_name` is a unique name associated with the concrete patch manager.
        It is used to identify the mesh in the Mitsuba scene. The `bsdf` parameter is
        optional and can be used to specify a BSDF for the downstream mesh created by
        the `PatchManager`.
        """
        self._manager_name: str = str(manager_name)
        self._patch_list: list[Patch] = []
        self._connection_list: list[PatchManager.PatchConnections] = []

        self._name_to_index_map: dict[str, int] = {}
        """A mapping from unique patch names to their indices in the patch list."""

    def add_patch(self, patch: Patch) -> None:
        """
        Adds a new patch to the list of patches.

        The patch is added is not linked to any other patches. `link_sides` must be
        called to connect the new patch to existing patches .

        :raises ValueError: If a patch with the same name already exists in the
            composite mesh.
        """
        if patch.name in self._name_to_index_map:
            raise ValueError(
                f"Mesh with name {patch.name} already exists in the composite mesh."
            )

        self._patch_list.append(patch)
        self._connection_list.append(PatchManager.PatchConnections())

        self._name_to_index_map[patch.name] = len(self._patch_list) - 1

    def link_sides(
        self,
        first_patch_name: str,
        first_patch_side: Patch.Side,
        second_patch_name: str,
        second_patch_side: Patch.Side,
    ) -> None:
        """
        Links the specified sides of two patches.

        The `first_patch_index` and `second_patch_index` specify the indices of the
        patches to be connected, while `first_patch_side` and `second_mesh_side`
        specify the sides that will be connected.

        The connection is established in both directions and the number of vertices on
        the two sides must match. If either side of the connection is already
        connected to another patch, an error is raised.

        The patches are not updated with the new connection; it is up to the
        concrete implementations of `PatchManager` to implement the desired behavior.

        :raises IndexError: If the specified patch indices are out of range.
        :raises ValueError: If the number of vertices on the specified sides do not
            match, or if either side is already connected to another patch.
        """
        if first_patch_name not in self._name_to_index_map:
            raise ValueError(f"Patch with name {first_patch_name} was not added.")
        if second_patch_name not in self._name_to_index_map:
            raise ValueError(f"Patch with name {second_patch_name} was not added.")

        first_patch_index = self._name_to_index_map[first_patch_name]
        second_patch_index = self._name_to_index_map[second_patch_name]

        if self._connection_list[first_patch_index][first_patch_side] is not None:
            raise ValueError(
                f"Mesh {first_patch_index} is already connected to mesh "
                f"{self._connection_list[first_patch_index][first_patch_side]} on side "
                f"{first_patch_side}."
            )
        if self._connection_list[second_patch_index][second_patch_side] is not None:
            raise ValueError(
                f"Mesh {second_patch_index} is already connected to mesh "
                f"{self._connection_list[second_patch_index][second_patch_side]} on "
                f"side {second_patch_side}."
            )

        first_mesh = self._patch_list[first_patch_index]
        second_mesh = self._patch_list[second_patch_index]

        first_side_idxes = first_mesh.get_node_indices_for(first_patch_side)
        second_side_idxes = second_mesh.get_node_indices_for(second_patch_side)

        if len(first_side_idxes) != len(second_side_idxes):
            raise ValueError(
                "The number of vertices on the specified sides must match."
            )

        self._connection_list[first_patch_index][first_patch_side] = second_patch_index
        self._connection_list[second_patch_index][second_patch_side] = first_patch_index

    @abc.abstractmethod
    def build(self) -> PatchManager:
        """
        Builds a new patch manager with the current patches and connections.

        This method should be implemented by subclasses to create a concrete
        `PatchManager` instance that manages the patches and their connections.

        It is up to the concrete implementation to ensure that the global representation
        is consistent and continuous. This can be done in the build method or in the
        constructor of the `PatchManager`. In any case, the user expects that the
        `PatchManager` instance returned by this method is ready to use.

        :returns: A new instance of `PatchManager` or a subclass thereof.
        :raises NotImplementedError: If a subclass does not implement this method.
        """
        raise NotImplementedError(
            "Classes that inherit from `PatchManagerFactory` must implement the "
            "`build` method."
        )
