from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class PipelineSteps:
    """Which of the runner's five steps a `PipelineStage` selection turns on."""

    mask_fix: bool
    """Repair the thermal image paths in the scene's `transforms.json`."""
    thermal_inpaint: bool
    """Inpaint the thermal images onto the reconstruction."""
    windows_inpaint: bool
    """Inpaint the facade mask onto the reconstruction."""
    heat_sim: bool
    """Simulate heat over the inpainted fields."""
    modal: bool
    """Solve the Reissner-Mindlin modal problem on the reconstruction."""
