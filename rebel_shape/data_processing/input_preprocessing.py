from pathlib import Path

import mitsuba as mi

from rebel_shape.data_processing.thermoscenes_loader import ThermoScenesLoader
from rebel_shape.plugin_wrappers.sampler import Sampler


def preprocess_rgb_images(
    scene_path: Path,
    environment_map_path: Path,
    output_dir: Path | None = None,
    spp: int = 16,
    train: bool = True,
    sampler: Sampler = Sampler(),
) -> tuple[list["mi.Sensor"], list["mi.TensorXf"]]:
    """
    Blends the environment map used in optimization with foreground object that is
    reconstructed from RGBA images.

    Loads a scene from `scene_path` assuming it is compatible with `ThermoScenesLoader`.
    Boolean `train` controls whether the images are taken from the training set (`True`)
    or evaluation set (`False`).

    Then, the background from `environment_map_path` is rendered for appropriate camera
    poses. `spp` samples per pixel are used in this step. Choosing a higher value
    than the default (`16`) has limited benefit.

    The `sampler` determines the sampling strategy used for rendering and is passed
    to all cameras loaded from the ThermoScenes scene.

    Finally, the transparent regions of the input RGBA images are replaced
    by the rendered background. Optionally, the generated images are saved to the
    `output_dir`.

    :returns: Two equal length lists that contain instantiated Mitsuba
        sensors and composited images in the same order.

    :raises FileNotFoundError: If the environment map file cannot be found

    :raises RuntimeError: If the images in scene_path are not RGBA.
    """

    if not environment_map_path.exists():
        raise FileNotFoundError(f"Environment map not found at {environment_map_path}")

    loader = ThermoScenesLoader(scene_path=scene_path, sensor_sampler=sampler)

    cameras = loader.train_cameras if train else loader.eval_cameras
    rgb_images = loader.train_images_rgb if train else loader.eval_images_rgb

    # Create a simple scene with just the environment map
    scene_dict = {
        "type": "scene",
        "integrator": {"type": "path"},
        "environment": {"type": "envmap", "filename": str(environment_map_path)},
    }
    scene = mi.load_dict(scene_dict)

    composited_images: list["mi.TensorXf"] = []
    sensors: list["mi.Sensor"] = []

    if output_dir is not None:
        output_dir.mkdir(exist_ok=True)

    # Process each image and camera
    for idx, (camera, rgba_image) in enumerate(zip(cameras, rgb_images)):
        sensor = mi.load_dict(camera.mitsuba_dict)
        background = mi.render(scene=scene, sensor=sensor, spp=spp)

        if rgba_image.shape[-1] == 4:
            alpha = rgba_image[:, :, 3:4]
            image = rgba_image[:, :, :3]

            composited_image = image * alpha + background * (1.0 - alpha)
        else:
            raise RuntimeError("Image does not have an alpha channel")

        if output_dir is not None:
            bitmap = mi.Bitmap(composited_image)
            filename = f"composited_frame_{'train' if train else 'eval'}{idx:05d}.exr"
            bitmap.write(str(output_dir / filename))

        composited_images.append(composited_image)
        sensors.append(sensor)

    return sensors, composited_images
