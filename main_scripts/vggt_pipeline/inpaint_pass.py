from enum import Enum


class InpaintPass(Enum):
    """
    One inpainting pass over a scene: which images it reads, and what it writes.

    The member name is the choice `inpaint_data_spline`'s `--inpainting-images-type`
    accepts, since tyro renders an enum argument as its member names rather than its
    values. The value is the label the pass's outputs are filed under, which the heat
    simulation reads back as `inpainted_<label>_data.json`. The two spellings differ for
    the windows pass, so both live here rather than one being derived from the other at
    every use.
    """

    THERMAL = "thermal"
    """Thermal images; the recovered field is the initial temperature."""
    WINDOWS = "facade"
    """Windows-mask images; the recovered field is the facade material mask."""
