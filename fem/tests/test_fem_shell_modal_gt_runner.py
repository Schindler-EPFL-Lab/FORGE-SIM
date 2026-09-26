import tempfile
import unittest
from pathlib import Path

import numpy as np
from _fem_fixtures import TETRAHEDRON_FACES, TETRAHEDRON_VERTICES

from fem.end_to_end.modal_gt.fem_shell_modal_gt_mesh import (
    FEMShellModalGTMeshParameters,
    run_modal_analysis,
)
from fem.end_to_end.modal_gt.fem_shell_modal_gt_runner import FEMShellModalGTRunner
from fem.meshprep.p2_geometry import basix_edge_table


class TestFEMShellModalGTRunner(unittest.TestCase):
    """
    The handover of the raw arrays, at both geometry orders, and the
    `modal_analysis_results.npz` archive a finished run leaves behind: the
    only thing downstream stages ever see of it.

    The order-2 branch is fed by mid-edge nodes a separate process baked
    into the `.npz`, so the only thing carrying the meaning across is the
    column order of `edge_index`.

    The solve's accuracy is not re-validated here: it is the same
    `GeometricallyExactShellProblem` / `GeometricallyExactShellModalSolver`
    pair that `test_geometrically_exact_shell_modal_solver.py` pins against
    the steel sphere. The one test that does run it uses the closed
    tetrahedron fixture, whose four cells solve instantly, and looks only
    at what was written.

    Its four saved arrays are the schema `shell_modal_analysis_spline` writes,
    and this driver's module docstring promises the two match; nothing else
    here runs the driver end to end, so without the one test that does, the
    save call is the one line of it no test reaches.
    """

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self._temporary.name)
        self.mesh_npz = self.folder / "modal_gt.npz"
        np.savez(
            self.mesh_npz,
            vertices=TETRAHEDRON_VERTICES,
            faces=TETRAHEDRON_FACES,
            scale=1.0,
        )

    def tearDown(self) -> None:
        self._temporary.cleanup()

    @staticmethod
    def curved_arrays() -> tuple[np.ndarray, np.ndarray]:
        """Mid-edge nodes pushed onto the unit sphere, plus their per-face
        indices, exactly as the baking stage writes them into the `.npz`."""
        unique_edges, edge_index = basix_edge_table(TETRAHEDRON_FACES)
        midpoints = 0.5 * (
            TETRAHEDRON_VERTICES[unique_edges[:, 0]]
            + TETRAHEDRON_VERTICES[unique_edges[:, 1]]
        )
        lifted = midpoints + 0.05 * np.array([0.0, 0.0, 1.0])
        return lifted, edge_index

    def test_flat_mesh_without_edge_data(self) -> None:
        """Order 1 is the default path: three nodes per cell, and the nodes are
        the vertices that were handed in."""
        msh = FEMShellModalGTRunner.build_gt_shell_mesh(
            TETRAHEDRON_VERTICES, TETRAHEDRON_FACES
        )

        self.assertEqual(msh.geometry.cmap.degree, 1)
        self.assertEqual(msh.geometry.dofmap.shape, (4, 3))
        np.testing.assert_array_equal(
            np.sort(msh.geometry.x, axis=0), np.sort(TETRAHEDRON_VERTICES, axis=0)
        )

    def test_curved_mesh_keeps_every_baked_node(self) -> None:
        """
        Order 2 must place all six baked mid-edge nodes and group them with the
        right cell. A driver that mis-wired `edge_index` would still produce a
        valid mesh and a plausible spectrum; the geometry would just be
        scrambled, so the check is that each cell's six nodes are exactly the
        six the arrays name for it.
        """
        edge_nodes, edge_index = self.curved_arrays()
        msh = FEMShellModalGTRunner.build_gt_shell_mesh(
            TETRAHEDRON_VERTICES,
            TETRAHEDRON_FACES,
            edge_nodes=edge_nodes,
            edge_index=edge_index,
        )

        self.assertEqual(msh.geometry.cmap.degree, 2)
        self.assertEqual(msh.geometry.dofmap.shape, (4, 6))

        points = np.vstack([TETRAHEDRON_VERTICES, edge_nodes])
        expected_cells = np.column_stack(
            [TETRAHEDRON_FACES, len(TETRAHEDRON_VERTICES) + edge_index]
        )
        self.assertEqual(
            {frozenset(map(tuple, msh.geometry.x[cell]))
             for cell in np.asarray(msh.geometry.dofmap)},
            {frozenset(map(tuple, points[cell])) for cell in expected_cells},
        )

    def test_stale_edge_data_is_rejected(self) -> None:
        """The baked arrays travel through an `.npz` written by another process,
        so the one inconsistency that is detectable here (a row count that no
        longer matches the faces) must raise rather than mis-wire the mesh."""
        edge_nodes, edge_index = self.curved_arrays()
        with self.assertRaises(ValueError):
            FEMShellModalGTRunner.build_gt_shell_mesh(
                TETRAHEDRON_VERTICES,
                TETRAHEDRON_FACES,
                edge_nodes=edge_nodes,
                edge_index=edge_index[:-1],
            )

    def test_the_run_writes_the_four_arrays_the_spline_driver_writes(self) -> None:
        """
        The scipy path is used so `mode_energies` carries a real split rather than
        the zeros the SLEPc path documents: an array saved as zeros either way
        would pass a shape check while never having been wired to the result.
        """
        parameters = FEMShellModalGTMeshParameters(
            mesh_npz=self.mesh_npz,
            output_folder=self.folder / "results",
            num_modes=8,
            use_scipy_solver=True,
            export_xdmf=False,
        )

        result = run_modal_analysis(params=parameters)

        saved = np.load(parameters.output_folder / "modal_analysis_results.npz")
        self.assertEqual(
            set(saved.keys()),
            {"eigenvalues", "eigenvectors", "frequencies", "mode_energies"},
        )
        np.testing.assert_allclose(saved["eigenvalues"], result.eigenvalues, rtol=0.0)
        np.testing.assert_allclose(saved["frequencies"], result.frequencies, rtol=0.0)
        self.assertGreater(float(np.abs(result.eigenvectors).max()), 0.0)
        np.testing.assert_allclose(saved["eigenvectors"], result.eigenvectors, rtol=0.0)
        self.assertEqual(saved["mode_energies"].shape, (8, 4))
        self.assertGreater(
            saved["mode_energies"][6].sum(),
            0.0,
            msg="The scipy path must report a non-zero split for the elastic mode.",
        )
