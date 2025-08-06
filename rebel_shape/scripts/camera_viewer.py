"""
Utility script to visualize the camera layout in a ThermoScenes scene.
Meant as a debugging tool to verify the position and orientation of cameras
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import mitsuba as mi
import tyro

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append("./")

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape import logger
from rebel_shape.data_processing.image_scene_loader import ImageSceneLoader
from rebel_shape.scripts.configs.camera_ring_config import CameraRingConfig
from rebel_shape.utils.image_utils import show_rendered_views


@dataclass
class CameraViewerParams(ReverseCli):
    thermoscenes_scene_path: Path
    """
    Path to the ThermoScenes scene that is optimized. Assumes the foreground
    has been extracted
    """
    environment_map_path: Path

    """Path to the image background"""
    camera_ring_config: CameraRingConfig = field(
        default_factory=lambda: CameraRingConfig(
            samples_per_pixel=16,
            radial_distance=10.0,
            sensor_count=6,
            sensor_resolution_height=640,
            sensor_resolution_width=640,
            sensor_fov=45.0,
        )
    )
    """Camera configuration for rendering the scene during the simulation."""

    image_output_folder: Path = Path("./outputs/")
    """Where to save the rendered cameras (created if it does not exist)"""

    use_gpu: bool = False
    """
    Whether to use GPU for rendering or not. Note that the compute instance
    must have GPU support for this to work. As the script is typically used for
    local testing, the default is False.
    """

    def __post_init__(self) -> None:
        if not self.thermoscenes_scene_path.exists():
            raise ValueError(
                f"ThermoScenes scene {self.thermoscenes_scene_path} does not exist."
            )
        if not self.environment_map_path.exists():
            raise ValueError(
                f"Environment map {self.environment_map_path} does not exist."
            )


def render_cameras(params: CameraViewerParams) -> None:
    """
    Displays views of cameras in a ThermoScenes compatible scene for the purpose
    of verifying their position and orientation.

    Renders `params.sensor_count` images from radially symmetric views around the
    barycenter (average position) of all cameras. They look down at the barycenter
    with angle `params.viewing_angle` in degrees (positive angle = down,
    negative angle = up).

    The distance of the cameras to the barycenter is given by `params.radial_distance`.
    Resolution is `params.sensor_resolution_width` x `params.sensor_resolution_height`
    and the width of the shot is controlled by `params.sensor_fov`.

    Cameras are represented by pyramids with rectangular base and the images are
    rendered with `params.samples_per_pixel` samples. The rendered views are saved to
    directory `params.image_output_folder` (created if it does not exist).
    """
    if params.use_gpu:
        mi.set_variant("cuda_ad_rgb")
        logger.info("Using the GPU for rendering.")
    else:
        mi.set_variant("llvm_ad_rgb")
        logger.info("Using the CPU for rendering.")

    loader = ImageSceneLoader(scene_path=params.thermoscenes_scene_path)

    center_coords = mi.ScalarPoint3f(0.0, 0.0, 0.0)
    for camera in loader.train_cameras:
        center_coords += camera.tf_pose_matrix.matrix.numpy()[0:3, 3]

    center_coords /= len(loader.train_cameras)

    scene_dict = {
        "type": "scene",
        "integrator": {"type": "path"},
        "environment": {"type": "envmap", "filename": str(params.environment_map_path)},
    }

    sensors = params.camera_ring_config.sensors

    script_dir = Path(__file__).parent.resolve()

    for i, camera in enumerate(loader.train_cameras):
        scene_dict[f"camera_{i}"] = {
            "type": "ply",
            "filename": str(
                script_dir / "../../data/test_scenes/meshes/camera_pyramid.ply"
            ),
            "bsdf": {"type": "diffuse"},
            "to_world": camera.tf_pose_matrix,
        }

    scene: "mi.Scene" = mi.load_dict(scene_dict)

    rendered_images = [mi.render(scene=scene, sensor=sensor) for sensor in sensors]

    show_rendered_views(
        views=[mi.util.convert_to_bitmap(image) for image in rendered_images],
        image_dpi=200,
        save_path=params.image_output_folder / "Camera_layout.png",
    )


if __name__ == "__main__":
    parameters = tyro.cli(CameraViewerParams)

    render_cameras(params=parameters)
