from dataclasses import dataclass
from pathlib import Path
from typing import Any

import drjit as dr
import mitsuba as mi

from rebel_shape.plugin_wrappers.mitsuba_plugin import MitsubaPlugin


class DiffuseBSDF(MitsubaPlugin):
    """
    A wrapper for the Mitsuba diffuse BSDF plugin.
    https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html

    This class encapsulates a diffuse (Lambertian) bidirectional scattering
    distribution function (BSDF), which can be configured with various
    reflectance types: uniform albedo, solid color, texture, or mesh attribute.
    """

    @dataclass
    class Albedo(MitsubaPlugin):
        """
        A uniform reflectance with the same value for all RGB channels, resulting
        in a neutral gray appearance. Typical values for everyday objects are in
        interval [0.3, 0.7].
        """

        albedo: float = 0.5
        """Value between 0.0 and 1.0 representing reflectance intensity"""

        def __post_init__(self) -> None:
            if self.albedo < 0.0 or self.albedo > 1.0:
                raise ValueError("`albedo` must be a value in interval [0.0, 1.0].")

        @property
        def mitsuba_dict(self) -> dict[str, Any]:
            return {"type": "rgb", "value": [self.albedo, self.albedo, self.albedo]}

    @dataclass
    class SolidColor(MitsubaPlugin):
        """
        A reflectance with separate RGB channel albedo.

        Represents a uniformly colored material where each RGB channel can be
        set individually to create a specific color.
        """

        red: float
        """Red albedo channel value between 0.0 and 1.0"""
        green: float
        """Green albedo channel value between 0.0 and 1.0"""
        blue: float
        """Blue albedo channel value between 0.0 and 1.0"""

        def __post_init__(self) -> None:
            if self.red < 0.0 or self.red > 1.0:
                raise ValueError("`red` must be a value in interval [0.0, 1.0].")
            if self.green < 0.0 or self.green > 1.0:
                raise ValueError("`green` must be a value in interval [0.0, 1.0].")
            if self.blue < 0.0 or self.blue > 1.0:
                raise ValueError("`blue` must be a value in interval [0.0, 1.0].")

        @property
        def mitsuba_dict(self) -> dict[str, Any]:
            return {"type": "rgb", "value": [self.red, self.green, self.blue]}

    @dataclass
    class Texture(MitsubaPlugin):
        """
        A texture-based reflectance loaded from a bitmap file.

        Uses an image file to create a spatially varying reflectance pattern that
        is mapped to the surface of an object.
        """

        texture_path: Path
        """Path to the texture image file"""

        def __post_init__(self) -> None:
            if not self.texture_path.exists():
                raise ValueError(
                    f"Provided `texture_path` {self.texture_path} does not exist."
                )

        @property
        def mitsuba_dict(self) -> dict[str, Any]:
            return {
                "type": "bitmap",
                "filename": str(self.texture_path),
            }

    @dataclass
    class TextureFromTensor(MitsubaPlugin):
        """
        A texture-based reflectance loaded from TensorXf data.

        Uses a TensorXf tensor to create a spatially varying reflectance pattern that
        is mapped to the surface of an object.
        """

        tensor: "mi.TensorXf"
        """Texture tensor"""

        def __post_init__(self) -> None:
            if self.tensor.ndim != 3:
                raise ValueError("`tensor` must be a 3D tensor for RGB texture.")
            if self.tensor.shape[-1] != 3:
                raise ValueError("`tensor` must be have 3 channels for RGB texture.")
            if dr.min(self.tensor) < 0.0 or dr.max(self.tensor) > 1.0:
                raise ValueError(
                    "All tensor values must be in the interval [0.0, 1.0]."
                )
            self.tensor = dr.detach(self.tensor)

        @property
        def mitsuba_dict(self) -> dict[str, Any]:
            return {
                "type": "bitmap",
                "data": dr.copy(self.tensor),
            }

    @dataclass
    class MeshAttributeTexture(MitsubaPlugin):
        """
        Uses a mesh attribute `attribute_key` to define the color on the mesh.

        Mitsuba meshes can be assigned additional properties, allowing for per vertex
        or per face properties to determine the color of the mesh. The selected
        property must start with either `vertex_` or `face_`.
        """

        attribute_key: str
        """
        Name/key of the attribute that is used for color information. Must start with
        either `vertex_` or `face_`
        """

        def __post_init__(self) -> None:
            if not self.attribute_key.startswith(
                "vertex_"
            ) and not self.attribute_key.startswith("face_"):
                raise ValueError(
                    "The selected property must start with either `vertex_` or `face_`."
                )

        @property
        def mitsuba_dict(self) -> dict[str, Any]:
            return {
                "type": "mesh_attribute",
                "name": self.attribute_key,
            }

    def __init__(
        self,
        reflectance: Albedo
        | SolidColor
        | Texture
        | TextureFromTensor
        | MeshAttributeTexture = Albedo(),
    ) -> None:
        """
        Initialize a diffuse (Lambertian) BSDF with the specified reflectance.

        The reflectance can be one of the 3 following types:
        - Albedo: A uniform grayscale reflectance
        - SolidColor: A uniform RGB color
        - Texture: A bitmap texture from an image file
        - TextureFromData: A bitmap texture from a TensorXf tensor
        - MeshAttributeTexture: A mesh attribute defining vertex or face colors

        The default is a uniform albedo of 0.5 (mid-gray), matches Mitsuba default.
        """
        self._reflectance = reflectance

    @property
    def mitsuba_dict(self) -> dict[str, Any]:
        """
        Get the Mitsuba compatible dictionary representation of this diffuse BSDF.

        :returns: A dictionary that is passed to the Mitsuba scene description dict.
        """
        return {"type": "diffuse", "reflectance": self._reflectance.mitsuba_dict}
