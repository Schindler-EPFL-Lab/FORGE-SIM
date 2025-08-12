import abc
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np
import numpy.typing as npt

from rebel_shape.meshing.patch import Patch


class PatchManager(abc.ABC):
    """
    An interface that handles the connectivity and management of multiple patches.
    It provides mappings between local patch nodes and global nodes but does not
    specify how the global mesh is represented or updated.

    Internally, it stores a graph of rectangular surface patches, where each patch is
    connected to up to four other patches. If all connections are established,
    the resulting representation will be watertight.

    Classes that inherit from `PatchManager` should also create an appropriate factory
    that implements the `PatchManagerFactory` interface to create instances of that
    class. The factory is responsible for creating patches, linking them together, and
    calling the `build` method to create the final instance.

    Patch nodes can be updated, and since a global representation is built from the
    local patches, the global representation is not updated automatically when a
    patch is modified. How this is performed is up to the specific implementation of the
    `PatchManager` through the `update_global_representation` method.
    """

    @dataclass
    class PatchConnections:
        """
        A graph node representing patch connections.

        Each patch can be connected to at most four other patches, one on each side.
        To make accessing easier, patch indices can also be
        object and
        exploit the one-to-one mapping between the sides and indices of the sides
        prescribed in the `Patch.Side` enum.

        An iterator that yields the indices of the connected meshes is also provided,
        and the order is left, right, top, bottom.
        """

        left_mesh_index: int | None = None
        right_mesh_index: int | None = None
        top_mesh_index: int | None = None
        bottom_mesh_index: int | None = None

        def __getitem__(self, item: Patch.Side) -> int | None:
            if item == Patch.Side.LEFT:
                return self.left_mesh_index
            if item == Patch.Side.RIGHT:
                return self.right_mesh_index
            if item == Patch.Side.TOP:
                return self.top_mesh_index
            if item == Patch.Side.BOTTOM:
                return self.bottom_mesh_index

        def __setitem__(self, item: Patch.Side, value: int) -> None:
            if item == Patch.Side.LEFT:
                self.left_mesh_index = value
            if item == Patch.Side.RIGHT:
                self.right_mesh_index = value
            if item == Patch.Side.TOP:
                self.top_mesh_index = value
            if item == Patch.Side.BOTTOM:
                self.bottom_mesh_index = value

        def __iter__(self) -> Iterator[tuple[Patch.Side, int | None]]:
            yield Patch.Side.LEFT, self.left_mesh_index
            yield Patch.Side.RIGHT, self.right_mesh_index
            yield Patch.Side.TOP, self.top_mesh_index
            yield Patch.Side.BOTTOM, self.bottom_mesh_index

    @dataclass
    class SharedEdge:
        """
        Represents a shared edge between two patches.

        A shared edge is an edge that is connected to two patches on opposite sides.
        The `first_index` and `second_index` are the indices of the patches in the
        `patch_list`, and `first_side` and `second_side` are the sides that are
        connected to each other.

        The graph is undirected, so the order of the patches is not important, but for
        consistency, the `first_index` should always less than the `second_index`.
        This is also done to prevent duplicate edges in the list of shared edges.
        """

        first_index: int
        second_index: int
        first_side: Patch.Side
        second_side: Patch.Side

        def __post_init__(self) -> None:
            if self.first_index >= self.second_index:
                raise ValueError("`first_index` must be less than `second_index`.")

    def __init__(
        self,
        name: str,
        patch_list: Sequence[Patch],
        connection_list: Sequence[PatchConnections],
    ) -> None:
        """
        Creates a new composite mesh object with a unique name `composite_mesh_name`.

        This method is not intended to be called directly, but rather through the
        `CompositeMeshFactory` class. The factory class is responsible for adding
        patches and establishing connections between them. It calls this method
        with appropriate `patch_list`, connection_list, and `bsdf` parameters.

        :raises ValueError: If the number of patches and connection lists do not match,
            or if they are empty.
        """
        if len(patch_list) == 0:
            raise ValueError("At least one patch must be provided.")
        if len(patch_list) != len(connection_list):
            raise ValueError("The number of patches and connection lists must match.")

        self._name = name
        self._patch_list = patch_list
        self._connection_list = connection_list

        self._patch_to_global_map: list[npt.NDArray[np.uint32]] = []
        """
        List of mappings that take a local vertex index of a patch and return the
        global vertex index of the composite mesh.
        """
        self._global_to_patches_map: dict[int, dict[int, int]] = {}
        """
        Dictionary that keeps track of which global vertices appear in which patches 
        and their local index.
        """

        self._build_mappings()

        self._averaging_weights = self._build_averaging_weights()
        """
        A (V, 1) tensor that contains the inverse number of patches that share each 
        vertex in the global mesh. This is used to efficiently compute the average 
        vertex positions and normals when updating the global mesh.
        """

    @property
    def shared_edges(self) -> list[SharedEdge]:
        """
        Returns a list of shared edges between patches.

        Shared edges are edges that are connected to two patches on opposite sides.
        The method returns a list of `SharedEdge` objects that contain the indices of
        the patches and the sides they are connected to.

        The order of the edges is such that the first patch index is always less than
        the second patch index, ensuring that each edge is only represented once.

        :returns: List of `SharedEdge` objects.
        """
        shared_edges: list[PatchManager.SharedEdge] = []

        for patch_idx, connections in enumerate(self._connection_list):
            for side, neighbor_idx in connections:
                if neighbor_idx is None or neighbor_idx <= patch_idx:
                    continue

                neighbor_side = self._get_neighbor_side(
                    from_patch=neighbor_idx, to_patch=patch_idx
                )

                shared_edges.append(
                    PatchManager.SharedEdge(
                        first_index=patch_idx,
                        first_side=side,
                        second_index=neighbor_idx,
                        second_side=neighbor_side,
                    )
                )

        return shared_edges

    def _get_neighbor_side(
        self,
        from_patch: int,
        to_patch: int,
    ) -> Patch.Side:
        """
        Returns the index of the neighboring patch on the specified side.

        This method is used to determine to which side of the `from_patch` the patch
        `to_patch` is connected.

        The connection is established in both directions, but it is not as simple as
        "left connects to right" because either patch could simply be rotated or
        flipped in space, so the order of `from_patch` and `to_patch` is important.

        :returns: Neighboring patch side connected to the current patch.
        :raises ValueError: If the neighbor is not connected to the current patch.
            (This should not happen if the connection is established correctly.)
        """
        for side, connection in self._connection_list[from_patch]:
            if connection == to_patch:
                return side

        raise ValueError(f"Patch {to_patch} is not connected to patch {from_patch}.")

    def _build_mappings(self) -> None:
        """
        Builds the mapping between the global representations indexes and the local
        patch indexes.

        Building the global representation is expensive as it requires a serial
        iteration over all patches and all their vertices - there can easily
        be 10 000+ of them. Because of this, the operation is not performed eagerly,
        but rather only once in the constructor. Thus, the impact of this operation
        should be acceptable (~10 seconds for 50 000 vertices).

        Conceptually, the global representation is a collection of all the patches.
        The global representation has its own nodes and a local to global mapping
        provides a way to update the local mesh vertices with global vertex values.
        Similarly, a mapping from global vertices to patches is created to allow for
        easy access to the local indexing used in patches.

        To apply proper averaging of positions and normals, call
        `update_global_representation` after the patches are added and connected.
        """
        global_node_index = 0
        self._patch_to_global_map: list[npt.NDArray[np.uint32]] = []
        self._global_to_patches_map: dict[int, dict[int, int]] = {}

        for patch_idx, patch in enumerate(self._patch_list):
            num_nodes = patch.nodes.shape[0]
            local_to_global = np.full(shape=num_nodes, fill_value=-1, dtype=np.int32)

            for side, neighbor_idx in self._connection_list[patch_idx]:
                if neighbor_idx is None or neighbor_idx >= patch_idx:
                    continue

                neighbor = self._patch_list[neighbor_idx]
                neighbor_map = self._patch_to_global_map[neighbor_idx]

                side_idxes = dr.ravel(patch.get_node_indices_for(side))

                neighbor_side = self._get_neighbor_side(
                    from_patch=neighbor_idx, to_patch=patch_idx
                )
                neighbor_side_idxes = dr.ravel(
                    neighbor.get_node_indices_for(neighbor_side)
                )

                side_vertices = patch.nodes[side_idxes]
                neighbor_side_vertices = neighbor.nodes[neighbor_side_idxes]

                # determine the local orientation from the vector of two sides
                side_vector = (side_vertices[-1] - side_vertices[0]).numpy()
                neighbor_vector = (
                    neighbor_side_vertices[-1] - neighbor_side_vertices[0]
                ).numpy()

                dot_product = np.dot(side_vector, neighbor_vector)

                if np.abs(dot_product) < 0.7 * np.linalg.norm(
                    side_vector
                ) * np.linalg.norm(neighbor_vector):
                    logging.warning(
                        f"Side {side} and neighbor side {neighbor_side} are not "
                        f"aligned well. This can lead to serious averaging issues. "
                    )

                if dot_product < 0:
                    temp_neighbor_side_idxes = mi.TensorXu(neighbor_side_idxes)
                    neighbor_side_idxes = dr.ravel(temp_neighbor_side_idxes[::-1])

                for vi, vj in zip(side_idxes, neighbor_side_idxes):
                    local_to_global[vi] = neighbor_map[vj]

            # Add vertices
            for vi in range(num_nodes):
                if local_to_global[vi] == -1:
                    local_to_global[vi] = global_node_index
                    global_node_index += 1

                global_idx = int(local_to_global[vi])
                if global_idx not in self._global_to_patches_map:
                    self._global_to_patches_map[global_idx] = {}

                self._global_to_patches_map[global_idx][patch_idx] = vi

            assert not np.any(local_to_global < 0), "Invalid local to global mapping."
            self._patch_to_global_map.append(np.array(local_to_global, dtype=np.uint32))

    def _build_averaging_weights(self) -> "mi.TensorXf":
        """
        Builds the averaging structure for the global mesh.

        The averaging structure is a tensor that contains the number of patches
        that share each vertex in the global mesh. This is used to compute the
        average vertex positions and normals when updating the global mesh.

        :returns: A tensor of shape (num_global_vertices, 1) containing the
                  inverse number of patches that share each vertex in the global mesh.
        """
        averaging_weights = np.zeros(
            shape=(len(self._global_to_patches_map), 1), dtype=np.float32
        )
        for global_idx, mappings in self._global_to_patches_map.items():
            averaging_weights[global_idx] = len(mappings)

        return mi.TensorXf(1.0 / averaging_weights)

    @staticmethod
    def _scatter_values(
        target: "mi.TensorXf", values: "mi.TensorXf", indices: "mi.TensorXu"
    ) -> None:
        """
        Scatters and adds `values` to `target` at the specified `indices`.

        This operation is commonly used for averaging node values across patches and
        doing it incorrectly can lead to hard-to-debug issues. Under the hood, it
        views the `target` tensor as a flat array and expands the `indices` to
        account for flattened components.

        The changes are applied in-place to the `target` tensor.
        """
        if len(target.shape) != 2 or len(values.shape) != 2 or len(indices.shape) != 1:
            raise ValueError(
                "`target` and `values` must be 2D tensors, and `indices` must be a "
                f"1D tensor, got {target.shape}, {values.shape}, and {indices.shape}."
            )
        if values.shape[0] != indices.shape[0]:
            raise ValueError(
                "`target_tensor` and `values` must have the same number of rows."
                f"got {values.shape[0]} and {target.shape[0]} for `target`."
            )
        if values.shape[1] != target.shape[1]:
            raise ValueError(
                "`target_tensor` and `values` must have the same number of columns, "
                f"got {values.shape[1]} and {target.shape[1]} for `target`."
            )

        num_components = target.shape[1]

        indices_extended = dr.zeros(
            dtype=mi.TensorXu, shape=(indices.shape[0] * num_components,)
        )
        for comp in range(num_components):
            indices_extended[comp::num_components] = dr.ravel(
                num_components * indices + comp
            )

        # Change mode to NoConflicts if at some point we mandate that the indices are
        # unique. The default Auto mode, does not work, at least in Mitsuba 3.6.4.
        # it leads to some target values being skipped, suggesting atomicity violations.
        # Direct mode forces an atomic RMW operation, which is slower but safe.
        dr.scatter_add(
            target=target.array,
            value=dr.ravel(values),
            index=dr.ravel(indices_extended),
            mode=dr.ReduceMode.Direct,
        )

    def match_files_to_patches(self, patch_paths: Sequence[Path]) -> list[Path]:
        """
        Matches file paths to patches based on patch names contained in file names.

        For each patch in the patch list, this method finds the first file path whose
        name contains the patch name as a substring. The returned list maintains the
        same order as the patches in the patch list.

        This method is used for matching texture files to patches which can be
        mixed up when reloading the patch manager. Any other file that might be
        matched with a patch should also contain the patch name in its file name.

        :returns: List of file paths ordered to match the order of the patch list,
            with one path per patch.

        :raises FileNotFoundError: if any path in `patch_paths` does not exist.

        :raises ValueError: if the number of elements in `patch_paths` does not match
            the number of patches, if any path is invalid (does not exist or is not a
            file), if a patch name is not found in any file path,
            or if a patch name is found in multiple file paths.
        """
        if len(patch_paths) != len(self._patch_list):
            raise ValueError(
                "The number of elements in `patch_paths` must match the number of "
                "patches that the `PatchManager` manages."
            )

        for patch_path in patch_paths:
            if not patch_path.exists():
                raise FileNotFoundError(
                    f"Patch path {patch_path} does not exist. "
                    "Ensure all patch paths are valid."
                )
            if not patch_path.is_file():
                raise ValueError(
                    f"Patch path {patch_path} is not a file. "
                    "Ensure all patch paths point to files."
                )

        sorted_paths: list[Path] = []

        for patch in self._patch_list:
            found_patch = False

            for texture_path in patch_paths:
                # Ensure the path is unique
                if patch.name in texture_path.name and not found_patch:
                    sorted_paths.append(texture_path)
                    found_patch = True
                    break
                elif patch.name in texture_path.name and found_patch:
                    raise ValueError(
                        f"Patch name '{patch.name}' is found in multiple file paths. "
                        "Ensure that each patch name is unique in the provided paths."
                    )

            if not found_patch:
                raise ValueError(
                    f"No file path contains patch name '{patch.name}' as a substring"
                )

        return sorted_paths

    def __getitem__(self, item: int) -> Patch:
        """
        Returns the mesh patch with the specified index.

        :raises IndexError: If the index is out of range.
        """
        if item < 0 or item >= len(self._patch_list):
            raise IndexError("Mesh index out of range.")

        return self._patch_list[item]

    def __len__(self) -> int:
        """
        Returns the number of patches.
        """
        return len(self._patch_list)

    def __iter__(self) -> Iterator[Patch]:
        """
        Returns an iterator over the patches managed by this patch manager.
        """
        return iter(self._patch_list)

    @property
    def patches_with_mapping(self) -> Iterator[tuple[Patch, npt.NDArray[np.uint32]]]:
        """
        Returns an iterator of tuples containing the patches with the corresponding
        local-to-global mapping
        """
        return zip(self._patch_list, self._patch_to_global_map)

    @property
    def name(self) -> str:
        """
        Unique name of the patch manager.
        """
        return self._name

    @abc.abstractmethod
    def update_global_representation(self) -> None:
        """
        Updates the global node representation in a way that is appropriate for the
        specific patch manager implementation.

        :raises NotImplementedError: If a subclass does not implement this method.
        """
        raise NotImplementedError(
            "Classes that inherit from `PatchManager` must implement the "
            "`update_global_representation` method."
        )
