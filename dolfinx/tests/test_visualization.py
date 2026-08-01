"""
Smoke test for the PyVista temperature viewer.

`visualize_temperature_field` is a human-facing debugging aid, so there is
nothing numerical to assert -- pixels are not a contract. What it is worth
guarding is that the call still *works*: it runs at the very end of a simulation,
after all the expensive work, against a fast-moving third-party API (PyVista's
plotter, slider widget and key events, plus DOLFINx's `vtk_mesh`). One rendered
screenshot is enough to catch a signature change in any of them.

The test skips when off-screen rendering is unavailable (no GL, headless CI
without OSMesa), since that says nothing about this repository.

Known gap: the frame-stepping callback `show_frame` -- the only user of PyVista's
deprecated `Plotter.update_scalars` -- is reachable only through the interactive
slider and key events, so it stays uncovered. Verified deprecated but functional
in PyVista 0.46.5; it will break when PyVista removes it, and only interactively.

Run from the repository root::

    conda run -n dolfinx-rebel-shape python -m unittest discover dolfinx/tests
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from dolfinx import fem
from mpi4py import MPI

from _fem_fixtures import function_from_dof_values, make_sphere_mesh
from visualization import visualize_temperature_field


class TestVisualizeTemperatureField(unittest.TestCase):
    """
    Off-screen rendering of a multi-frame temperature series.
    """

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import pyvista as pv

            probe = pv.Plotter(off_screen=True)
            probe.add_mesh(pv.Sphere())
            probe.show(auto_close=True)
        except Exception as error:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"Off-screen rendering unavailable: {error}")

        cls.comm = MPI.COMM_SELF
        cls.msh = make_sphere_mesh(comm=cls.comm)
        cls.V = fem.functionspace(cls.msh, ("Lagrange", 1))

    def test_multi_frame_screenshot_is_written(self):
        """
        A three-frame series renders to a non-empty PNG.

        Three frames rather than one is deliberate: it takes the branch that adds
        the time-slider widget (`nF > 1`) and the automatic colour-limit
        computation over all frames, which is most of the module's surface
        against the PyVista API.
        """
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        screenshot = directory / "temperature.png"

        z = self.V.tabulate_dof_coordinates()[:, 2]
        frames = [300.0 + factor * z for factor in (1.0, 5.0, 20.0)]

        visualize_temperature_field(
            V=self.V,
            u=function_from_dof_values(self.V, frames[-1]),
            frames=frames,
            times=[0.0, 1.0, 2.0],
            screenshot_path=str(screenshot),
            show=False,
        )

        self.assertTrue(screenshot.is_file())
        # A rendered scene is tens of kilobytes; the floor only distinguishes it
        # from an empty or truncated file, which is all a smoke test should claim.
        self.assertGreater(screenshot.stat().st_size, 1024)


if __name__ == "__main__":
    unittest.main()
