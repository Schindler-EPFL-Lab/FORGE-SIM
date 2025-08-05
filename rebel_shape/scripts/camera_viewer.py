"""
Utility script to visualize the camera layout in a ThermoScenes scene.
Meant as a debugging tool to verify the position and orientation of cameras
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import mitsuba as mi
import tyro

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append("./")

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape.data_processing.thermoscenes_loader import ThermoScenesLoader
from rebel_shape.plugin_wrappers.sampler import Sampler
from rebel_shape.utils.image_utils import show_rendered_views
from rebel_shape.utils.sensor_utils import create_sensors_radial


@dataclass
class CameraViewerParams(ReverseCli):
    thermoscenes_scene_path: Path
    """
    Path to the ThermoScenes scene that is optimized. Assumes the foreground
    has been extracted
    """
    environment_map_path: Path
    """Path to the image background"""
    samples_per_pixel: int = 16
    """Number of samples per pixel used for rendering"""
    sensor_count: int = 6
    """Number of sensors placed in radial pattern around the origin"""
    radial_distance: float = 12.0
    """Distance of cameras from the origin"""
    viewing_angle: float = 30.0
    """Camera viewing angle in degrees (negative looks down, positive looks up)"""
    sensor_resolution_width: int = 640
    """Resolution of the camera sensor along width"""
    sensor_resolution_height: int = 640
    """Resolution of the camera sensor along height"""
    sensor_fov: float = 45
    """Field of view of the camera sensors in degrees"""
    image_output_folder: Path = Path("./outputs/")
    """Where to save the rendered cameras (created if it does not exist)"""

    def __post_init__(self) -> None:
        if self.samples_per_pixel < 1:
            raise ValueError("`samples_per_pixel` must be a positive integer.")
        if self.sensor_count < 1:
            raise ValueError("`sensor_count` must be a positive integer.")
        if self.radial_distance <= 0.0:
            raise ValueError("`radial_distance` must be a positive real number.")
        if self.sensor_resolution_width < 1:
            raise ValueError("`sensor_resolution_width` must be a positive integer.")
        if self.sensor_resolution_height < 1:
            raise ValueError("`sensor_resolution_height` must be a positive integer.")

        if self.viewing_angle <= -90.0 or self.viewing_angle >= 90.0:
            raise ValueError(
                "`viewing_angle` must be between -90 and 90 degrees, exclusive"
            )
        if self.sensor_fov <= 0.0 or self.sensor_fov >= 180.0:
            raise ValueError(
                "`sensor_fov` must be between 0 and 180 degrees, exclusive."
            )

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
    mi.set_variant("llvm_ad_rgb")

    loader = ThermoScenesLoader(scene_path=params.thermoscenes_scene_path)

    center_coords = mi.ScalarPoint3f(0.0, 0.0, 0.0)
    for camera in loader.train_cameras:
        center_coords += camera.tf_pose_matrix.matrix.numpy()[0:3, 3]

    center_coords /= len(loader.train_cameras)

    scene_dict = {
        "type": "scene",
        "integrator": {"type": "path"},
        "environment": {"type": "envmap", "filename": str(params.environment_map_path)},
    }

    sensors = create_sensors_radial(
        sensor_count=params.sensor_count,
        radial_distance=params.radial_distance,
        target_coords=(center_coords.x, center_coords.y, center_coords.z),
        viewing_angle=params.viewing_angle,
        sensor_resolution=(
            params.sensor_resolution_width,
            params.sensor_resolution_height,
        ),
        sensor_fov=params.sensor_fov,
        sampler=Sampler(
            sample_count=params.samples_per_pixel, type=Sampler.Type.MULTIJITTER
        ),
    )

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

    rendered_images = [
        mi.render(scene=scene, sensor=sensor, spp=params.samples_per_pixel)
        for sensor in sensors
    ]

    show_rendered_views(
        views=[mi.util.convert_to_bitmap(image) for image in rendered_images],
        image_dpi=200,
        save_path=params.image_output_folder / "Camera_layout.png",
    )


if __name__ == "__main__":
    parameters = tyro.cli(CameraViewerParams)

    render_cameras(params=parameters)
