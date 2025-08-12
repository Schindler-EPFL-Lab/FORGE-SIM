import unittest
import tempfile
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

    def setUp(self):
        """Set up test patches and connections for the PatchManager."""
        mi.set_variant("llvm_ad_rgb")

        self.patch1 = self.TestPatch(
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
        self.patch2 = self.TestPatch(
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

        self.patch3 = self.TestPatch(
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

        self.patch_list = [self.patch1, self.patch2, self.patch3]
        self.connection_list = [connections1, connections2, connections3]

    def test_init(self):
        """
        Test successful initialization of PatchManager with valid patches and
        connections.
        """
        patch_manager = self.TestPatchManager(
            name="test_patch_manager",
            patch_list=self.patch_list,
            connection_list=self.connection_list,
        )
        self.assertEqual(len(patch_manager._patch_list), 3)
        self.assertEqual(len(patch_manager._connection_list), 3)
        self.assertEqual(patch_manager.name, "test_patch_manager")

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
        """Test _get_neighbor_side method for valid connections."""
        patch_manager = self.TestPatchManager(
            name="test_patch_manager",
            patch_list=self.patch_list,
            connection_list=self.connection_list,
        )

        side = patch_manager._get_neighbor_side(0, 1)
        self.assertEqual(side, Patch.Side.RIGHT)
        side = patch_manager._get_neighbor_side(1, 0)
        self.assertEqual(side, Patch.Side.LEFT)

        side = patch_manager._get_neighbor_side(1, 2)
        self.assertEqual(side, Patch.Side.TOP)
        side = patch_manager._get_neighbor_side(2, 1)
        self.assertEqual(side, Patch.Side.BOTTOM)

    def test_iterator_support(self):
        """Test __len__, __iter__, and __getitem__ methods of PatchManager."""

        patch_manager = self.TestPatchManager(
            name="test_patch_manager",
            patch_list=self.patch_list,
            connection_list=self.connection_list,
        )

        # Length of the patch manager should match the number of patches
        self.assertEqual(len(patch_manager), 3)

        # Iterating through the patch manager should yield patches
        patches = list(patch_manager)
        self.assertEqual(patches, self.patch_list)

        # Test accessing patches by index
        for i in range(len(patch_manager)):
            self.assertEqual(patch_manager[i], self.patch_list[i])

        with self.assertRaises(IndexError):
            _ = patch_manager[3]

    def test_get_neighbor_side_error(self):
        """Test _get_neighbor_side method if the connection is not defined."""
        patch_manager = self.TestPatchManager(
            name="test_patch_manager",
            patch_list=self.patch_list,
            connection_list=[PatchManager.PatchConnections() for _ in range(3)],
        )
        with self.assertRaises(ValueError):
            patch_manager._get_neighbor_side(0, 1)

    def test_shared_edges(self):
        """Test whether shared edges are correctly identified."""
        patch_manager = self.TestPatchManager(
            name="test_patch_manager",
            patch_list=self.patch_list,
            connection_list=self.connection_list,
        )

        # Check shared edges between patch1 and patch2
        shared_edges = patch_manager.shared_edges
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
        # Test _build_mappings method
        patch_manager = self.TestPatchManager(
            name="test_patch_manager",
            patch_list=self.patch_list,
            connection_list=self.connection_list,
        )

        # Mappings should already be built in the constructor

        # Check that patch_to_global_map has correct size
        self.assertEqual(len(patch_manager._patch_to_global_map), 3)
        for patch_idx, _ in enumerate(patch_manager._patch_to_global_map):
            self.assertEqual(len(patch_manager._patch_to_global_map[patch_idx]), 4)

        # Check shared vertices between patch1 and patch2
        self.assertEqual(
            patch_manager._patch_to_global_map[0][1],
            patch_manager._patch_to_global_map[1][0],
        )
        self.assertEqual(
            patch_manager._patch_to_global_map[0][3],
            patch_manager._patch_to_global_map[1][2],
        )

        # Check shared vertices between patch2 and patch3
        self.assertEqual(
            patch_manager._patch_to_global_map[1][2],
            patch_manager._patch_to_global_map[2][0],
        )
        self.assertEqual(
            patch_manager._patch_to_global_map[1][3],
            patch_manager._patch_to_global_map[2][1],
        )

        # Check global_to_patches_map contains entries for each global vertex
        # Should have 8 unique vertices in total (4 + 2 + 2)
        self.assertEqual(len(patch_manager._global_to_patches_map), 8)

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
            self.assertIn(vertex_idx, patch_manager._global_to_patches_map)
            for patch_idx, local_idx in vertex_patches.items():
                self.assertIn(
                    patch_idx, patch_manager._global_to_patches_map[vertex_idx]
                )
                self.assertEqual(
                    patch_manager._global_to_patches_map[vertex_idx][patch_idx],
                    local_idx,
                )

    def test_build_averaging_weights(self):
        # Test _build_averaging_weights method
        patch_manager = self.TestPatchManager(
            name="test_patch_manager",
            patch_list=self.patch_list,
            connection_list=self.connection_list,
        )

        # Check that averaging weights are built correctly
        patch_manager._build_averaging_weights()

        # Each vertex should have a weight of 1.0 for itself and 0.5 for shared vertices
        expected_weights = [1.0, 0.5, 1.0, 1.0 / 3.0, 1.0, 0.5, 1.0, 1.0]

        np.testing.assert_array_almost_equal(
            patch_manager._averaging_weights.numpy(),
            np.array(expected_weights, dtype=np.float32).reshape(-1, 1),
        )

    def test_match_files_to_patches_success(self):
        """Test successful matching of file paths to patches."""

        patch_manager = self.TestPatchManager(
            name="test_patch_manager",
            patch_list=self.patch_list,
            connection_list=self.connection_list,
        )

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
            sorted_paths = patch_manager.match_files_to_patches(patch_paths)

            # Verify correct order matching patch list order
            self.assertEqual(len(sorted_paths), 3)
            self.assertEqual(sorted_paths[0], file1)  # patch1
            self.assertEqual(sorted_paths[1], file2)  # patch2
            self.assertEqual(sorted_paths[2], file3)  # patch3

    def test_match_files_to_patches_matching_errors(self):
        """Test patch name matching errors in match_files_to_patches method."""
        patch_manager = self.TestPatchManager(
            name="test_patch_manager",
            patch_list=self.patch_list,
            connection_list=self.connection_list,
        )

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
                patch_manager.match_files_to_patches([file1, file2, file3])

            # Test duplicate patch name
            file1_dup = temp_path / "texture_patch1_data.txt"
            file2_dup = temp_path / "backup_patch1_config.json"
            file3_ok = temp_path / "patch3_vertices.obj"

            file1_dup.touch()
            file2_dup.touch()
            file3_ok.touch()

            with self.assertRaises(ValueError):
                patch_manager.match_files_to_patches([file1_dup, file2_dup, file3_ok])

    def test_match_files_to_patches_validation_errors(self):
        """Test various validation errors in match_files_to_patches method."""
        patch_manager = self.TestPatchManager(
            name="test_patch_manager",
            patch_list=self.patch_list,
            connection_list=self.connection_list,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test wrong count
            file1 = temp_path / "patch1.txt"
            file1.touch()

            with self.assertRaises(ValueError):
                patch_manager.match_files_to_patches([file1])

            # Test nonexistent file
            nonexistent_file = Path("/nonexistent/path/patch1.txt")
            with self.assertRaises(FileNotFoundError) :
                patch_manager.match_files_to_patches(
                    [nonexistent_file, nonexistent_file, nonexistent_file]
                )

            # Test directory instead of file
            dir_path = temp_path / "patch1_dir"
            dir_path.mkdir()
            with self.assertRaises(ValueError):
                patch_manager.match_files_to_patches([dir_path, dir_path, dir_path])

