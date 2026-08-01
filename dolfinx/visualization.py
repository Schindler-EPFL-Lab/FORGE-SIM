"""
PyVista visualization utilities for DOLFINx heat simulations.

This module provides an interactive viewer for temperature fields on DOLFINx
meshes, with a time slider and keyboard navigation between frames, plus
screenshot export. It is a human-facing debugging aid; the archival output of a
run is the XDMF time series, which ParaView reads.

The visualization style is designed to be consistent with the Mitsuba-based
rendering used in the IGA heat simulations, enabling visual comparison between
IGA and FEM results.
"""

import numpy as np
import pyvista as pv
from dolfinx import fem
from dolfinx.plot import vtk_mesh

from rebel_shape import logger


def visualize_temperature_field(
    V: fem.FunctionSpace,
    u: fem.Function,
    frames: list[np.ndarray] | None = None,
    times: list[float] | None = None,
    clim: tuple[float, float] | None = None,
    screenshot_path: str | None = None,
    show: bool = True,
    cmap: str = "coolwarm",
    title: str = "Temperature [K]",
) -> None:
    """
    Visualizes the temperature field using PyVista.

    Provides interactive visualization with time slider for frame navigation and
    screenshot capability. The visualization includes keyboard controls for
    stepping through frames.

    Keyboard controls (when interactive):
        - Right Arrow / k: Next frame
        - Left Arrow / j: Previous frame
        - Home: First frame
        - End: Last frame
        - r: Reset color scale to current frame range

    :param V: Function space for temperature
    :param u: Final temperature function (used if frames is None)
    :param frames: Optional list of temperature arrays for animation.
        Each array should have shape (n_dofs,).
    :param times: Time stamps corresponding to frames. If None, frame
        indices are used.
    :param clim: Color limits (min, max) for the temperature scale.
        If None, computed automatically from all frames.
    :param screenshot_path: Path to save a screenshot. If None, no screenshot.
    :param show: Whether to show interactive visualization window.
    :param cmap: Matplotlib/PyVista colormap name.
    :param title: Title for the colorbar.
    """
    # Build PyVista grid from DOLFINx mesh
    topo, cell_types, x = vtk_mesh(V)
    grid = pv.UnstructuredGrid(topo, cell_types, x)

    # Ensure we have frames to display
    final_vals = u.x.array.copy()
    if frames is None or len(frames) == 0:
        frames = [final_vals]
        times = [0.0] if times is None else times

    nF = len(frames)
    if times is None or len(times) != nF:
        times = list(range(nF))

    # Auto-compute color limits from all frames
    if clim is None:
        vmin = min(np.min(f) for f in frames)
        vmax = max(np.max(f) for f in frames)
        clim = (float(vmin), float(vmax))

    # Determine if running off-screen
    off_screen = bool(screenshot_path) or not show
    pl = pv.Plotter(off_screen=off_screen)

    # Add initial mesh with temperature data
    grid.point_data["Temperature"] = frames[0]
    actor = pl.add_mesh(
        grid,
        scalars="Temperature",
        clim=clim,
        scalar_bar_args={"title": title},
        cmap=cmap,
    )
    pl.add_axes()
    pl.view_isometric()

    # Add HUD text showing current time and frame
    hud = pl.add_text(f"t = {times[0]:.2f}s (frame 1/{nF})", font_size=12)

    # Frame navigation state
    idx = {"i": 0}

    def show_frame(i: int) -> None:
        """Updates the visualization to show frame i."""
        i = max(0, min(nF - 1, int(i)))
        idx["i"] = i
        # Assigning into `point_data` is what actually updates the scalars;
        # `pl.render()` below draws the result. `Plotter.update_scalars` is
        # deprecated in PyVista and slated for removal, and was redundant with
        # both of those.
        grid.point_data["Temperature"] = frames[i]
        hud.SetText(0, f"t = {times[i]:.2f}s (frame {i+1}/{nF})")
        pl.render()

    # Add time slider widget
    def slider_callback(val: float) -> None:
        show_frame(round(val))

    if nF > 1:
        pl.add_slider_widget(
            slider_callback,
            rng=[0, nF - 1],
            value=0,
            title="Time step",
            pointa=(0.025, 0.08),
            pointb=(0.35, 0.08),
            style="modern",
            interaction_event="always",
        )

    # Add keyboard navigation
    def step_forward():
        show_frame(idx["i"] + 1)

    def step_backward():
        show_frame(idx["i"] - 1)

    def go_first():
        show_frame(0)

    def go_last():
        show_frame(nF - 1)

    def rescale_colors():
        """Rescale colors to current frame's data range."""
        current_data = frames[idx["i"]]
        actor.mapper.scalar_range = (float(np.min(current_data)), 
                                      float(np.max(current_data)))
        pl.render()

    pl.add_key_event("Right", step_forward)
    pl.add_key_event("Left", step_backward)
    pl.add_key_event("k", step_forward)
    pl.add_key_event("j", step_backward)
    pl.add_key_event("Home", go_first)
    pl.add_key_event("End", go_last)
    pl.add_key_event("r", rescale_colors)

    # Export screenshot if requested
    if screenshot_path:
        pl.show(screenshot=screenshot_path, auto_close=True)
        logger.info(f"Saved screenshot to {screenshot_path}")
        return

    # Interactive visualization
    if show:
        pl.show()
