from enum import Enum


class DatasetModality(Enum):
    """
    One rendered channel of the synthetic dataset.

    Each member's value is the sub-directory of the dataset root that holds that
    channel's PNGs, which is also the name the ThermoScenes loader convention uses in
    `transforms.json`. RGB is the only sRGB-encoded channel: the other two are
    linear scalar fields stored raw, exactly like real FLIR thermal data.
    """

    RGB = "images"
    """Colour render under the environment map, with the silhouette in alpha."""
    THERMAL = "thermal"
    """Emissive render of the per-vertex temperature field."""
    MATERIAL = "windows_masks"
    """Emissive render of the binary material field (steel = 1)."""
