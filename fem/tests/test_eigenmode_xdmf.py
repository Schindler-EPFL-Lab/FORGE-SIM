import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree

# Imported for its module-level side effect: `_fem_fixtures` puts the
# repository root on `sys.path`, so `fem` and `forge_sim` resolve.
import _fem_fixtures  # noqa: F401
import h5py
import numpy as np

from fem.shells.curved_sphere_mesh import make_curved_sphere_mesh
from fem.shells.eigenmode_xdmf import export_mode_shapes_xdmf
from fem.shells.geometrically_exact_shell_modal_solver import (
    GeometricallyExactShellModalSolver,
)
from fem.shells.geometrically_exact_shell_problem import GeometricallyExactShellProblem
from fem.shells.shell_element import ShellElement

# Recorded end-to-end modal runs, if this machine has them. They live outside the
# repository, so the tests against them skip by default: the explicit layout
# assertions above are the guard, these are confirmation. Same convention (and
# environment variable) as `test_format_compatibility.py`: an unset or empty
# `FORGE_SIM_RECORDED_RUNS` means there are none.
_RECORDED_RUNS_VALUE = os.environ.get("FORGE_SIM_RECORDED_RUNS", "")
_RECORDED_RUNS = Path(_RECORDED_RUNS_VALUE)
_RECORDED_MODAL = _RECORDED_RUNS / "fem_modal_simulation" / "eigenmodes.xdmf"
_HAVE_RECORDED = _RECORDED_RUNS_VALUE != "" and _RECORDED_MODAL.exists()
_SKIP_REASON = (
    f"No recorded modal run at {_RECORDED_MODAL} (set FORGE_SIM_RECORDED_RUNS)."
    if _RECORDED_RUNS_VALUE
    else "FORGE_SIM_RECORDED_RUNS is not set."
)


class TestEigenmodeExport(unittest.TestCase):
    """
    The layout on disk, and the deformation and sign conventions inside it.

    This is an on-disk contract, and the expensive kind to get wrong:
    recorded modal runs live outside this repository under
    `<archive>/*/fem_modal_simulation/eigenmodes.{xdmf,h5}`, were written
    by this layout, and must stay readable. The dataset paths, shapes,
    dtypes and the temporal-collection structure are therefore asserted
    explicitly rather than round-tripped, and confirmed against a recorded
    archive when this machine has one. It is the FEM writer's own
    contract; the IGA exporter is a deliberately independent
    implementation of the same format and is not imported here.

    The solver is only a source of eigenvectors here, so the fixture is
    the smallest gmsh sphere that has a spectrum: no Mitsuba and no
    spline anywhere in this file.
    """

    # Datasets every mode carries, and the trailing dimension of each.
    _MODE_DATASETS = {
        "displacement": 3,
        "rotation": 3,
    }

    @classmethod
    def setUpClass(cls) -> None:
        # No Mitsuba anywhere in this class, so a shared problem/solver is safe.
        cls.problem, cls.solver = cls._tiny_problem_and_solver()
        cls.result = cls.solver.solve_modes_slepc(num_modes=8)

    @staticmethod
    def _tiny_problem_and_solver() -> tuple[
        GeometricallyExactShellProblem, GeometricallyExactShellModalSolver
    ]:
        """The smallest sphere shell problem that still has a spectrum, for
        exercising the export path. Concrete-like material; the numbers are
        irrelevant here."""
        problem = GeometricallyExactShellProblem(
            make_curved_sphere_mesh(1.0, 0.9, order=2),
            youngs_modulus=30.0e9,
            poisson_ratio=0.2,
            density=2250.0,
            thickness=0.2,
            element=ShellElement(locking_cure=ShellElement.LockingCure.BBAR),
        )
        return problem, GeometricallyExactShellModalSolver(problem)

    @staticmethod
    def _visit_datasets(h5_file) -> list[str]:
        paths: list[str] = []
        h5_file.visit(paths.append)
        return [path for path in paths if isinstance(h5_file[path], h5py.Dataset)]

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def export(self, indices=(0, 1, 2)) -> Path:
        export_mode_shapes_xdmf(
            self.problem.V, self.result, list(indices), self.folder
        )
        return self.folder / "eigenmodes.xdmf"

    def test_hdf5_layout(self) -> None:
        """Dataset paths, shapes and dtypes exactly as the archive carries them:
        an int32 `/Mesh/topology` of triangles, a float64 `/Mesh/vertices`, and
        `/Modes/mode_<k>/<field>` per exported mode, indexed by export position
        `k` (not by the original mode index, which lives in the XDMF)."""
        self.export(indices=(0, 3, 7))
        with h5py.File(self.folder / "eigenmodes.h5", "r") as h5_file:
            topology = h5_file["/Mesh/topology"]
            vertices = h5_file["/Mesh/vertices"]
            self.assertEqual(topology.dtype, np.int32)
            self.assertEqual(topology.shape[1], 3)
            self.assertEqual(vertices.dtype, np.float64)
            self.assertEqual(vertices.shape[1], 3)
            self.assertEqual(
                vertices.shape[0], self.problem.msh.geometry.x.shape[0]
            )
            for k in range(3):
                for name, columns in self._MODE_DATASETS.items():
                    dataset = h5_file[f"/Modes/mode_{k}/{name}"]
                    self.assertEqual(
                        dataset.shape, (vertices.shape[0], columns), msg=name
                    )
                    self.assertEqual(dataset.dtype, np.float64, msg=name)
            self.assertNotIn("mode_3", h5_file["/Modes"])

    def test_undeformed_vertices_are_stored_for_alignment(self) -> None:
        """`/Mesh/vertices` is the undeformed geometry; the post-hoc MAC alignment
        step reads it, and every mode's XDMF grid references this same array
        rather than a per-mode deformed copy."""
        self.export()
        with h5py.File(self.folder / "eigenmodes.h5", "r") as h5_file:
            np.testing.assert_allclose(
                np.asarray(h5_file["/Mesh/vertices"]),
                self.problem.msh.geometry.x[:, :3],
                rtol=0.0,
            )

    def test_the_mesh_comes_from_the_function_space(self) -> None:
        """The writer takes the mixed space alone and reads its mesh, rather than
        being handed both and trusting them to agree. `/Mesh/topology` must
        therefore be the space's own cells."""
        self.export(indices=(0,))
        geom_dofmap = np.asarray(self.problem.V.mesh.geometry.dofmap)
        with h5py.File(self.folder / "eigenmodes.h5", "r") as h5_file:
            np.testing.assert_array_equal(
                np.asarray(h5_file["/Mesh/topology"]),
                geom_dofmap.reshape(geom_dofmap.shape[0], -1)[:, :3],
            )

    def test_mode_sign_is_canonical(self) -> None:
        """An eigenvector is defined up to sign, so the exporter fixes one: the
        largest-magnitude displacement component is positive. Without it, a FEM
        and an IGA run of the same mode can render as mirror images."""
        self.export(indices=(0, 1, 2))
        with h5py.File(self.folder / "eigenmodes.h5", "r") as h5_file:
            for k in range(3):
                flat = np.asarray(h5_file[f"/Modes/mode_{k}/displacement"]).ravel()
                self.assertGreater(flat[int(np.argmax(np.abs(flat)))], 0.0)

    def test_rotation_follows_the_displacement_sign_flip(self) -> None:
        """The flip is applied to the whole mode. Flipping only the displacement
        would leave a physically inconsistent mode shape on disk."""
        result = replace(
            self.result, eigenvectors=-self.result.eigenvectors.copy()
        )
        export_mode_shapes_xdmf(self.problem.V, result, [0], self.folder)
        with h5py.File(self.folder / "eigenmodes.h5", "r") as h5_file:
            flipped = {
                field: np.asarray(h5_file[f"/Modes/mode_0/{field}"])
                for field in ("displacement", "rotation")
            }
        self.export(indices=(0,))
        with h5py.File(self.folder / "eigenmodes.h5", "r") as h5_file:
            for field, values in flipped.items():
                # Negating the input then re-canonicalising restores the same mode.
                # IEEE negation is exact and the interpolation is linear, so this
                # should be bit-identical; 1e-12 leaves room for that not to hold.
                np.testing.assert_allclose(
                    values,
                    np.asarray(h5_file[f"/Modes/mode_0/{field}"]),
                    rtol=1.0e-12,
                )

    def test_xdmf_is_a_temporal_collection_of_uniform_grids(self) -> None:
        """The structure ParaView needs to animate the modes, and the structure a
        recorded file has. Time value is the export index, so modes advance one
        per frame however they were selected."""
        path = self.export(indices=(0, 4, 7))
        root = ElementTree.parse(path).getroot()
        collection = root.find("Domain/Grid")
        self.assertEqual(collection.get("GridType"), "Collection")
        self.assertEqual(collection.get("CollectionType"), "Temporal")

        grids = collection.findall("Grid")
        self.assertEqual([grid.get("Name") for grid in grids],
                         ["mode_0", "mode_4", "mode_7"])
        for k, grid in enumerate(grids):
            self.assertEqual(grid.find("Time").get("Value"), str(k))
            self.assertEqual(
                grid.find("Topology").get("TopologyType"), "Triangle"
            )
            self.assertEqual(grid.find("Geometry").get("GeometryType"), "XYZ")
            self.assertEqual(
                grid.find("Geometry").find("DataItem").text,
                "eigenmodes.h5:/Mesh/vertices",
            )
            attributes = {a.get("Name"): a.get("AttributeType")
                          for a in grid.findall("Attribute")}
            self.assertEqual(
                attributes,
                {"displacement": "Vector", "rotation": "Vector"},
            )

    def test_xdmf_references_only_the_companion_hdf5_by_name(self) -> None:
        """Relative reference, so the pair stays readable after being moved or
        copied out of `outputs/`, which is how the archive was assembled."""
        path = self.export()
        for item in ElementTree.parse(path).getroot().iter("DataItem"):
            self.assertTrue(item.text.strip().startswith("eigenmodes.h5:"))

    def test_out_of_range_indices_are_skipped_with_a_warning(self) -> None:
        """The default index list is a fixed range, so asking for more modes than
        were computed is routine, but it must be reported, not silent."""
        with self.assertLogs("forge_sim", level="WARNING"):
            path = self.export(indices=(0, 99))
        grids = ElementTree.parse(path).getroot().findall("Domain/Grid/Grid")
        self.assertEqual([grid.get("Name") for grid in grids], ["mode_0"])

    def test_no_valid_indices_writes_nothing(self) -> None:
        """A requested index outside the computed modes also leaves nothing
        written, which gets its own warning distinct from the skip-warning
        above."""
        with self.assertLogs("forge_sim", level="WARNING") as logs:
            export_mode_shapes_xdmf(self.problem.V, self.result, [99], self.folder)
        self.assertTrue(
            any("writing nothing" in message for message in logs.output)
        )
        self.assertFalse((self.folder / "eigenmodes.xdmf").exists())

    def test_empty_index_list_writes_nothing(self) -> None:
        """An empty request has no length mismatch to trigger the skip-warning,
        so the nothing-written warning is the only signal that nothing ran."""
        with self.assertLogs("forge_sim", level="WARNING") as logs:
            export_mode_shapes_xdmf(self.problem.V, self.result, [], self.folder)
        self.assertTrue(
            any("writing nothing" in message for message in logs.output)
        )
        self.assertFalse((self.folder / "eigenmodes.xdmf").exists())

    @unittest.skipUnless(_HAVE_RECORDED, _SKIP_REASON)
    def test_recorded_archive_has_the_same_layout(self) -> None:
        """
        Confirmation against a file this code did not write. A writer change
        mirrored by a reader change would pass every round trip here while making
        the recorded runs unreadable; only a recorded file catches that.
        """
        self.export(indices=(0, 1))
        with h5py.File(self.folder / "eigenmodes.h5", "r") as written, h5py.File(
            _RECORDED_MODAL.with_suffix(".h5"), "r"
        ) as recorded:
            written_paths = {
                path for path in self._visit_datasets(written) if "mode_0/" in path
                or path.startswith("Mesh/")
            }
            recorded_paths = {
                path for path in self._visit_datasets(recorded) if "mode_0/" in path
                or path.startswith("Mesh/")
            }
            self.assertEqual(written_paths, recorded_paths)
            for path in written_paths:
                self.assertEqual(
                    written[path].dtype, recorded[path].dtype, msg=path
                )
                self.assertEqual(
                    written[path].shape[1:], recorded[path].shape[1:], msg=path
                )

        def structure(xdmf_path: Path) -> list[str]:
            root = ElementTree.parse(xdmf_path).getroot()
            grid = root.find("Domain/Grid/Grid")
            return [element.tag for element in grid] + sorted(
                attribute.get("Name") for attribute in grid.findall("Attribute")
            )

        self.assertEqual(
            structure(self.folder / "eigenmodes.xdmf"), structure(_RECORDED_MODAL)
        )
