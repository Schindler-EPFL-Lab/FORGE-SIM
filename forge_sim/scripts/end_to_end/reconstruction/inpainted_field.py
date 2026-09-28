from enum import Enum


class InpaintedField(Enum):
    """
    One of the two scalar fields recovered onto the spline by image-space inpainting.

    Each member's value is the name the field is known by on the wire: it selects the
    single-field worker through the `E2E_INPAINT_ONLY` environment variable and
    names the stage's log lines.
    """

    TEMPERATURE = "temperature"
    """Initial temperature, recovered from the dataset's thermal images."""
    MATERIAL = "material"
    """Binary material mask, recovered from the dataset's windows-mask images."""
