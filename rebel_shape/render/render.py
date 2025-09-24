from pathlib import Path

import drjit as dr
import mitsuba as mi

from rebel_shape.plugin_wrappers.perspective_camera import PerspectiveCamera


class Render:
    """
    A renderer class for generating images from a Mitsuba scene.
    """

    def __init__(self, scene: "mi.Scene") -> None:
        """
        Initializes the renderer with the given Mitsuba Scene `scene`.
        """

        self._scene = scene

    def render(
        self,
        camera: PerspectiveCamera,
        samples_per_pixel: int,
        save_path: Path | None = None,
        output_type: mi.Struct.Type = mi.Struct.Type.UInt8,
    ) -> mi.Bitmap:
        """
        Renders the current scene from the given camera and returns the resulting image
        as a TensorXf tensor. The output image is in sRGB space. The output image stored
        in `save_path` if provided. The result encoded in `output_type` format which
        is uint8 by default.
        """

        rendered_image = mi.render(
            scene=self._scene,
            sensor=camera.mitsuba_sensor,
            spp=samples_per_pixel,
        )

        rendered_image = dr.clip(rendered_image, min=0.0, max=1.0)
        rendered_image = mi.util.convert_to_bitmap(rendered_image, uint8_srgb=False)

        rendered_image = rendered_image.convert(
            mi.Bitmap.PixelFormat.RGB, output_type, True
        )

        if save_path is not None:
            rendered_image.write(str(save_path))

        return rendered_image
