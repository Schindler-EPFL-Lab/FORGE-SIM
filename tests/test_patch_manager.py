import tempfile
import unittest
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np

from rebel_shape.meshing.patch import Patch
from rebel_shape.meshing.patch_manager import PatchManager


class TestPatchManager(unittest.TestCase):
    class TestPatchManager(PatchManager):
        def update_global_representation(self):
            # Simple implementation for testing
            pass

    class TestPatch(Patch):
        def __init__(self, name: str, nodes: "mi.TensorXf"):
            assert len(nodes) == 4, "Patch must have exactly 4 nodes"
            self._name = name
            self._nodes = nodes

        def get_node_indices_for(self, side: Patch.Side) -> "mi.ArrayXu":
            if side == Patch.Side.LEFT:
                return dr.ravel(mi.TensorXu([0, 2]))
            if side == Patch.Side.RIGHT:
                return dr.ravel(mi.TensorXu([1, 3]))
            if side == Patch.Side.TOP:
                return dr.ravel(mi.TensorXu([2, 3]))
            if side == Patch.Side.BOTTOM:
                return dr.ravel(mi.TensorXu([0, 1]))

        @property
        def nodes(self) -> "mi.TensorXf":
            return self._nodes

        @property
        def name(self) -> str:
            return self._name

    @classmethod
    def setUpClass(cls):
        """Set up test patches and connections for the PatchManager."""
        mi.set_variant("llvm_ad_rgb")

        cls.patch1 = cls.TestPatch(
            name="patch1",
            nodes=mi.TensorXf(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ]
            ),
        )
        cls.patch2 = cls.TestPatch(
            name="patch2",
            nodes=mi.TensorXf(
                [
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [2.0, 1.0, 0.0],
                ]
            ),
        )

        cls.patch3 = cls.TestPatch(
            name="patch3",
            nodes=mi.TensorXf(
                [
                    [1.0, 1.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [1.0, 2.0, 0.0],
                    [2.0, 2.0, 0.0],
                ]
            ),
        )

        # Connections: patch1 connected to patch2 on the right side
        connections1 = PatchManager.PatchConnections(right_mesh_index=1)
        connections2 = PatchManager.PatchConnections(
            left_mesh_index=0, top_mesh_index=2
        )
        # patch2 connected to patch3 on the top side
        connections3 = PatchManager.PatchConnections(
            bottom_mesh_index=1,
        )

        cls.patch_list = [cls.patch1, cls.patch2, cls.patch3]
        cls.connection_list = [connections1, connections2, connections3]

        cls.default_manager = cls.TestPatchManager(
            name="test_patch_manager",
            patch_list=cls.patch_list,
            connection_list=cls.connection_list,
        )

    def test_init(self):
        """
        Test whether the basic properties have been set correctly during initialization.
        """
        self.assertEqual(len(self.default_manager._patch_list), 3)
        self.assertEqual(len(self.default_manager._connection_list), 3)
        self.assertEqual(self.default_manager.name, "test_patch_manager")

    def test_invalid_init(self):
        """Test invalid initialization scenarios for PatchManager."""
        with self.assertRaises(ValueError):
            self.TestPatchManager(name="invalid", patch_list=[], connection_list=[])

        # Test initialization with mismatched lists
        with self.assertRaises(ValueError):
            self.TestPatchManager(
                name="invalid",
                patch_list=self.patch_list,
                connection_list=[self.connection_list[0]],
            )

    def test_mesh_connections_getitem(self):
        """Test whether MeshConnections.__getitem__ returns the correct values"""
        connections = PatchManager.PatchConnections(
            left_mesh_index=1, right_mesh_index=2, top_mesh_index=3, bottom_mesh_index=4
        )

        self.assertEqual(connections[Patch.Side.LEFT], 1)
        self.assertEqual(connections[Patch.Side.RIGHT], 2)
        self.assertEqual(connections[Patch.Side.TOP], 3)
        self.assertEqual(connections[Patch.Side.BOTTOM], 4)

    def test_mesh_connections_setitem(self):
        """Test whether MeshConnections.__setitem__ sets the values correctly"""
        connections = PatchManager.PatchConnections()

        connections[Patch.Side.LEFT] = 5
        connections[Patch.Side.RIGHT] = 6
        connections[Patch.Side.TOP] = 7
        connections[Patch.Side.BOTTOM] = 8

        self.assertEqual(connections.left_mesh_index, 5)
        self.assertEqual(connections.right_mesh_index, 6)
        self.assertEqual(connections.top_mesh_index, 7)
        self.assertEqual(connections.bottom_mesh_index, 8)

    def test_mesh_connections_iter(self):
        """Test whether MeshConnections.__iter__ returns the correct items"""
        connections = PatchManager.PatchConnections(
            left_mesh_index=1, right_mesh_index=2, top_mesh_index=3, bottom_mesh_index=4
        )

        expected = [
            (Patch.Side.LEFT, 1),
            (Patch.Side.RIGHT, 2),
            (Patch.Side.TOP, 3),
            (Patch.Side.BOTTOM, 4),
        ]

        items = list(connections.__iter__())
        self.assertEqual(items, expected)

    def test_get_neighbor_side(self):
        """Test _get_neighbor_side method for valid and invalid inputs."""
        side = self.default_manager._get_neighbor_side(0, 1)
        self.assertEqual(side, Patch.Side.RIGHT)
        side = self.default_manager._get_neighbor_side(1, 0)
        self.assertEqual(side, Patch.Side.LEFT)

        side = self.default_manager._get_neighbor_side(1, 2)
        self.assertEqual(side, Patch.Side.TOP)
        side = self.default_manager._get_neighbor_side(2, 1)
        self.assertEqual(side, Patch.Side.BOTTOM)

        with self.assertRaises(ValueError):
            self.default_manager._get_neighbor_side(0, 2)

    def test_iterator_support(self):
        """Test __len__, __iter__, and __getitem__ methods of PatchManager."""
        # Length of the patch manager should match the number of patches
        self.assertEqual(len(self.default_manager), 3)

        # Iterating through the patch manager should yield patches
        patches = list(self.default_manager)
        self.assertEqual(patches, self.patch_list)

        # Test accessing patches by index
        for i in range(len(self.default_manager)):
            self.assertEqual(self.default_manager[i], self.patch_list[i])

        with self.assertRaises(IndexError):
            _ = self.default_manager[3]

    def test_shared_edges(self):
        """Test whether shared edges are correctly identified."""
        # Check shared edges between patch1 and patch2
        shared_edges = self.default_manager.shared_edges
        self.assertEqual(len(shared_edges), 2)

        self.assertEqual(
            shared_edges[0],
            PatchManager.SharedEdge(
                first_index=0,
                first_side=Patch.Side.RIGHT,
                second_index=1,
                second_side=Patch.Side.LEFT,
            ),
        )

        self.assertEqual(
            shared_edges[1],
            PatchManager.SharedEdge(
                first_index=1,
                first_side=Patch.Side.TOP,
                second_index=2,
                second_side=Patch.Side.BOTTOM,
            ),
        )

    def test_build_mappings(self):
        # Mappings should already be built in the constructor
        # Check that patch_to_global_map has correct size
        self.assertEqual(len(self.default_manager._patch_to_global_map), 3)
        for patch_idx, _ in enumerate(self.default_manager._patch_to_global_map):
            self.assertEqual(
                len(self.default_manager._patch_to_global_map[patch_idx]), 4
            )

        # Check shared vertices between patch1 and patch2
        self.assertEqual(
            self.default_manager._patch_to_global_map[0][1],
            self.default_manager._patch_to_global_map[1][0],
        )
        self.assertEqual(
            self.default_manager._patch_to_global_map[0][3],
            self.default_manager._patch_to_global_map[1][2],
        )

        # Check shared vertices between patch2 and patch3
        self.assertEqual(
            self.default_manager._patch_to_global_map[1][2],
            self.default_manager._patch_to_global_map[2][0],
        )
        self.assertEqual(
            self.default_manager._patch_to_global_map[1][3],
            self.default_manager._patch_to_global_map[2][1],
        )

        # Check global_to_patches_map contains entries for each global vertex
        # Should have 8 unique vertices in total (4 + 2 + 2)
        self.assertEqual(len(self.default_manager._global_to_patches_map), 8)

        expected_global_to_patches_map = {
            0: {0: 0},  # patch1 bottom left
            1: {0: 1, 1: 0},  # shared patch1 bottom right
            2: {0: 2},  # patch1 top left
            3: {0: 3, 1: 2, 2: 0},  # triple shared patch1 top right
            4: {1: 1},  # patch2 bottom left
            5: {1: 3, 2: 1},  # shared patch2 top right
            6: {2: 2},  # patch3 top left
            7: {2: 3},  # patch3 top right
        }

        for vertex_idx, vertex_patches in expected_global_to_patches_map.items():
            self.assertIn(vertex_idx, self.default_manager._global_to_patches_map)
            for patch_idx, local_idx in vertex_patches.items():
                self.assertIn(
                    patch_idx, self.default_manager._global_to_patches_map[vertex_idx]
                )
                self.assertEqual(
                    self.default_manager._global_to_patches_map[vertex_idx][patch_idx],
                    local_idx,
                )

    def test_build_averaging_weights(self):
        # Check that averaging weights are built correctly
        self.default_manager._build_averaging_weights()

        # Each vertex should have a weight of 1.0 for itself and 0.5 for shared vertices
        expected_weights = [1.0, 0.5, 1.0, 1.0 / 3.0, 1.0, 0.5, 1.0, 1.0]

        np.testing.assert_array_almost_equal(
            self.default_manager._averaging_weights.numpy(),
            np.array(expected_weights, dtype=np.float32).reshape(-1, 1),
        )

    def test_match_files_to_patches_success(self):
        """Test successful matching of file paths to patches."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files with patch names
            file1 = temp_path / "texture_patch1_data.txt"
            file2 = temp_path / "mesh_patch2_config.json"
            file3 = temp_path / "patch3_vertices.obj"

            # Create the files
            file1.touch()
            file2.touch()
            file3.touch()

            patch_paths = [file3, file2, file1]

            # Test matching
            sorted_paths = self.default_manager.match_files_to_patches(
                patch_paths=patch_paths
            )

            # Verify correct order matching patch list order
            self.assertEqual(len(sorted_paths), 3)
            self.assertEqual(sorted_paths[0], file1)  # patch1
            self.assertEqual(sorted_paths[1], file2)  # patch2
            self.assertEqual(sorted_paths[2], file3)  # patch3

    def test_match_files_to_patches_matching_errors(self):
        """Test patch name matching errors in match_files_to_patches method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test missing patch name
            file1 = temp_path / "random_file1.txt"
            file2 = temp_path / "random_file2.txt"
            file3 = temp_path / "random_file3.txt"

            file1.touch()
            file2.touch()
            file3.touch()

            with self.assertRaises(ValueError):
                self.default_manager.match_files_to_patches(
                    patch_paths=[file1, file2, file3]
                )

            # Test duplicate patch name
            file1_dup = temp_path / "texture_patch1_data.txt"
            file2_dup = temp_path / "backup_patch1_config.json"
            file3_ok = temp_path / "patch3_vertices.obj"

            file1_dup.touch()
            file2_dup.touch()
            file3_ok.touch()

            with self.assertRaises(ValueError):
                self.default_manager.match_files_to_patches(
                    patch_paths=[file1_dup, file2_dup, file3_ok]
                )

    def test_match_files_to_patches_validation_errors(self):
        """Test various validation errors in match_files_to_patches method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test wrong count
            file1 = temp_path / "patch1.txt"
            file1.touch()

            with self.assertRaises(ValueError):
                self.default_manager.match_files_to_patches(patch_paths=[file1])

            # Test nonexistent file
            nonexistent_file = Path("/nonexistent/path/patch1.txt")
            with self.assertRaises(FileNotFoundError):
                self.default_manager.match_files_to_patches(
                    patch_paths=[nonexistent_file, nonexistent_file, nonexistent_file]
                )

            # Test directory instead of file
            dir_path = temp_path / "patch1_dir"
            dir_path.mkdir()
            with self.assertRaises(ValueError):
                self.default_manager.match_files_to_patches(
                    patch_paths=[dir_path, dir_path, dir_path]
                )

    def test_scatter_values_single_component(self):
        target = mi.TensorXf([[0.0], [0.0], [0.0], [0.0]])
        values = mi.TensorXf([[1.0], [2.0]])
        indices = mi.TensorXu([1, 3])

        self.default_manager._scatter_values(
            target=target, values=values, indices=indices
        )
        np.testing.assert_array_equal(
            np.array(target), np.array([[0.0], [1.0], [0.0], [2.0]])
        )

    def test_scatter_values_multi_component(self):
        target = mi.TensorXf(np.zeros((4, 2), dtype=np.float32))
        values = mi.TensorXf([[1.0, 10.0], [2.0, 20.0]])
        indices = mi.TensorXu([1, 3])

        self.default_manager._scatter_values(
            target=target, values=values, indices=indices
        )
        np.testing.assert_array_equal(
            np.array(target),
            np.array([[0.0, 0.0], [1.0, 10.0], [0.0, 0.0], [2.0, 20.0]]),
        )

    def test_scatter_values_accumulation(self):
        target = mi.TensorXf([[0.0], [0.0], [0.0]])
        values = mi.TensorXf([[1.0], [2.0], [3.0]])
        indices = mi.TensorXu([1, 1, 2])

        self.default_manager._scatter_values(
            target=target, values=values, indices=indices
        )
        np.testing.assert_array_equal(np.array(target), np.array([[0.0], [3.0], [3.0]]))

    def test_scatter_values_invalid(self):
        """
        Test invalid inputs for _scatter_values method,
        including incorrect shapes and out-of-bounds indices.
        """
        # Wrong shape: target not 2D
        with self.assertRaises(ValueError):
            self.default_manager._scatter_values(
                target=mi.TensorXf([0.0, 0.0]),
                values=mi.TensorXf([[1.0]]),
                indices=mi.TensorXu([0]),
            )
        # Wrong shape: values not 2D
        with self.assertRaises(ValueError):
            self.default_manager._scatter_values(
                target=mi.TensorXf([[0.0], [0.0]]),
                values=mi.TensorXf([1.0, 2.0]),
                indices=mi.TensorXu([0, 1]),
            )
        # Wrong shape: indices not 1D
        with self.assertRaises(ValueError):
            self.default_manager._scatter_values(
                target=mi.TensorXf([[0.0], [0.0]]),
                values=mi.TensorXf([[1.0], [2.0]]),
                indices=mi.TensorXu([[0, 1]]),
            )
        # Mismatched rows
        with self.assertRaises(ValueError):
            self.default_manager._scatter_values(
                target=mi.TensorXf([[0.0], [0.0]]),
                values=mi.TensorXf([[1.0]]),
                indices=mi.TensorXu([0, 1]),
            )
        # Mismatched columns
        with self.assertRaises(ValueError):
            self.default_manager._scatter_values(
                target=mi.TensorXf([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                values=mi.TensorXf([[1.0, 2.0], [3.0, 4.0]]),
                indices=mi.TensorXu([0, 1]),
            )
        # Out-of-bounds index
        with self.assertRaises(IndexError):
            self.default_manager._scatter_values(
                target=mi.TensorXf([[0.0], [0.0]]),
                values=mi.TensorXf([[1.0], [2.0]]),
                indices=mi.TensorXu([0, 2]),
            )
