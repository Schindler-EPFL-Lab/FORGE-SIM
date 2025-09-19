from enum import Enum
from pathlib import Path

import mitsuba as mi

from rebel_shape.data_processing.masked_pose_image import MaskedPoseImage
from rebel_shape.data_processing.thermoscenes_loader import ThermoScenesLoader
from rebel_shape.plugin_wrappers.sampler import Sampler


class MaskStrategy(Enum):
    OBJECT = 1
    FOREGROUND = 2


def preprocess_rgb_images(
    scene_path: Path,
    environment_map_path: Path | None = None,
    output_dir: Path | None = None,
    spp: int = 16,
    train: bool = True,
    mask_strategy: MaskStrategy = MaskStrategy.OBJECT,
    sampler: Sampler = Sampler(),
) -> tuple[list[MaskedPoseImage], "mi.ScalarTransform4f"]:
    """
    Blends the environment map used in optimization with foreground object that is
    reconstructed from RGBA images.

    Loads a scene from `scene_path` assuming it is compatible with `ThermoScenesLoader`.
    Boolean `train` controls whether the images are taken from the training set (`True`)
    or evaluation set (`False`).

    Then, the background from `environment_map_path` is rendered for appropriate camera
    poses. `spp` samples per pixel are used in this step. Choosing a higher value than
    the default (`16`) has limited benefit. If the path is not provided, a solid white
    background is used.

    The `sampler` determines the sampling strategy used for rendering and is passed to
    all cameras loaded from the ThermoScenes scene.

    Finally, the transparent regions of the input RGBA images are replaced by the
    rendered background. Optionally, the generated images are saved to the `output_dir`.
    The output MaskedPoseImages mask foreground and backgrounds if `mask_strategy` is
    MaskStrategy.object. They mask foreground if `mask_strategy` is
    MaskStrategy.foreground.

    :returns: A tuple containing a list of masked pose images train/eval images and
    dataloder transform which was applied to camera poses.

    :raises FileNotFoundError: If the environment map file cannot be found

    :raises ValueError: if `mask_foreground_and_background` and `mask_foreground` are
        True.

    :raises RuntimeError: If the images in scene_path are not RGBA.
    """

    if environment_map_path is not None and not environment_map_path.exists():
        raise FileNotFoundError(f"Environment map not found at {environment_map_path}")

    loader = ThermoScenesLoader(scene_path=scene_path, sensor_sampler=sampler)

    cameras = loader.train_cameras if train else loader.eval_cameras
    rgb_images = loader.train_images_rgb if train else loader.eval_images_rgb
    foreground_masks = (
        loader.train_foreground_masks if train else loader.eval_foreground_masks
    )

    environment_map = (
        {"type": "envmap", "filename": str(environment_map_path)}
        if environment_map_path is not None
        else {
            "type": "constant",
            "radiance": {
                "type": "rgb",
                "value": 1.0,
            },
        }
    )

    # Create a simple scene with just the environment map
    scene_dict = {
        "type": "scene",
        "integrator": {"type": "path"},
        "environment": environment_map,
    }
    scene = mi.load_dict(scene_dict)

    composited_images: list[MaskedPoseImage] = []

    if output_dir is not None:
        output_dir.mkdir(exist_ok=True)

    # Process each image and camera
    for idx, (camera, rgba_image, foreground_mask) in enumerate(
        zip(cameras, rgb_images, foreground_masks)
    ):
        sensor = mi.load_dict(camera.mitsuba_dict)
        background = mi.render(scene=scene, sensor=sensor, spp=spp)

        if rgba_image.shape[-1] == 4:
            alpha = rgba_image[:, :, 3:4]
            image = rgba_image[:, :, :3]

            composited_image = image * alpha + background * (1.0 - alpha)

            mask = None
            if mask_strategy is MaskStrategy.OBJECT:
                mask = alpha
            if mask_strategy is MaskStrategy.FOREGROUND:
                mask = 1.0 - foreground_mask

            composited_image_masked_pose = MaskedPoseImage(
                image=composited_image,
                camera=camera,
                mask=mask,
            )
        else:
            raise RuntimeError("Image does not have an alpha channel")

        if output_dir is not None:
            bitmap = mi.Bitmap(composited_image)
            filename = f"composited_frame_{'train' if train else 'eval'}{idx:05d}.exr"
            bitmap.write(str(output_dir / filename))

        composited_images.append(composited_image_masked_pose)

    return composited_images, loader.tf_cameras
