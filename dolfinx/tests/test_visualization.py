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
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dolfinx import fem
from mpi4py import MPI

from _fem_fixtures import function_from_dof_values, make_sphere_mesh
from visualization import visualize_temperature_field

# Renders a throwaway sphere. Whether this succeeds is the precondition for the
# whole class, and it is deliberately run as a *separate process*; see
# `_off_screen_rendering_works`.
_RENDER_PROBE = (
    "import pyvista as pv; "
    "plotter = pv.Plotter(off_screen=True); "
    "plotter.add_mesh(pv.Sphere()); "
    "plotter.show(auto_close=True)"
)

_PROBE_TIMEOUT_SECONDS = 120


def _off_screen_rendering_works() -> tuple[bool, str]:
    """
    Reports whether off-screen rendering is available, without risking the run.

    The probe runs in a child interpreter rather than in-process, because VTK
    does not raise when it cannot find an OpenGL implementation: it tries GLX,
    then EGL, then OSMesa, warns about each, and then calls through a null
    function pointer. The process dies with SIGSEGV, which no `except` clause can
    intercept. On a headless runner that took the entire test session down with
    it -- and since MPI was initialised, PETSc's signal handler escalated the
    crash to `MPI_ABORT`, discarding the pending tracebacks of every other test.

    A child process turns that crash into an exit status, which is catchable.

    :returns: A pair of the availability flag and a reason, empty when available.
    """
    try:
        probe = subprocess.run(
            [sys.executable, "-c", _RENDER_PROBE],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"the render probe hung for {_PROBE_TIMEOUT_SECONDS} s"

    if probe.returncode == 0:
        return True, ""

    # A negative code is death by signal -- the null-dereference above. Report the
    # tail of stderr either way, since VTK names the backend it failed to reach.
    detail = probe.stderr.strip().splitlines()[-3:]
    return False, f"exit code {probe.returncode}: {' | '.join(detail)}"


class TestVisualizeTemperatureField(unittest.TestCase):
    """
    Off-screen rendering of a multi-frame temperature series.
    """

    @classmethod
    def setUpClass(cls) -> None:
        available, reason = _off_screen_rendering_works()
        if not available:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"Off-screen rendering unavailable: {reason}")

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
