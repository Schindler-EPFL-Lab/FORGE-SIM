"""
Utility script to visualize the camera layout in a ThermoScenes scene.
Meant as a debugging tool to verify the position and orientation of cameras
"""

from dataclasses import dataclass, field
from pathlib import Path

import mitsuba as mi
import numpy as np
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

from forge_sim import logger
from forge_sim.data_processing.image_scene_loader import ImageSceneLoader
from forge_sim.scripts.configs.camera_ring_config import CameraRingConfig
from forge_sim.utils.image_utils import show_rendered_views


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

    camera_size: float = 1.0
    """
    Scale factor applied to the camera pyramid meshes. Use values <1 to shrink
    the markers and >1 to enlarge them.
    """

    auto_radial_distance: bool = True
    """
    If True, overrides `camera_ring_config.radial_distance` and
    `camera_ring_config.origin_position` in place: `radial_distance` is set to
    the max distance from the scene cameras' barycenter to any scene camera,
    scaled by `auto_radial_distance_factor`, and `origin_position` is set to that
    barycenter, so the viewing camera ring is centered on and sized to fit the
    whole scene-camera layout. When False, both values from `camera_ring_config`
    are used as-is.
    """

    auto_radial_distance_factor: float = 2.5
    """
    Multiplier on the max distance from the barycenter to any scene camera when
    `auto_radial_distance` is True. Larger values pull the viewing cameras back.
    """

    use_gpu: bool = False
    """
    Whether to use GPU for rendering or not. Note that the compute instance
    must have GPU support for this to work. As the script is typically used for
    local testing, the default is False.
    """
    from_thermoscene: bool = True
    """
    USe the dataloader for a thermoscene dataset.
    Reverts to the VGGT format when False
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
        if self.camera_size <= 0.0:
            raise ValueError("`camera_size` must be a positive real number.")
        if self.auto_radial_distance_factor <= 0.0:
            raise ValueError(
                "`auto_radial_distance_factor` must be a positive real number."
            )


def render_cameras(params: CameraViewerParams) -> None:
    """
    Displays views of cameras in a ThermoScenes compatible scene for the purpose
    of verifying their position and orientation.

    Renders `params.sensor_count` images from radially symmetric views around the
    barycenter (average position) of all cameras. They look down at the barycenter
    with angle `params.viewing_angle` in degrees (positive angle = down,
    negative angle = up).

    The distance of the cameras to the barycenter is given by `params.radial_distance`,
    or, when `params.auto_radial_distance` is True, computed from the spread of the
    scene cameras as `max_offset * params.auto_radial_distance_factor`.
    Resolution is `params.sensor_resolution_width` x `params.sensor_resolution_height`
    and the width of the shot is controlled by `params.sensor_fov`.

    Cameras are represented by pyramids with rectangular base, scaled by
    `params.camera_size`, and the images are rendered with `params.samples_per_pixel`
    samples. The rendered views are saved to directory `params.image_output_folder`
    (created if it does not exist).
    """
    if params.use_gpu:
        mi.set_variant("cuda_ad_rgb")
        logger.info("Using the GPU for rendering.")
    else:
        mi.set_variant("llvm_ad_rgb")
        logger.info("Using the CPU for rendering.")

    if params.from_thermoscene:
        loader = ImageSceneLoader.from_thermoscene(
            scene_path=params.thermoscenes_scene_path
        )
    else:
        loader = ImageSceneLoader.from_vggt(scene_path=params.thermoscenes_scene_path)

    camera_positions = np.stack(
        [
            camera.tf_pose_matrix.matrix.numpy()[0:3, 3]
            for camera in loader.train_cameras
        ]
    )
    center_coords = camera_positions.mean(axis=0)

    if params.auto_radial_distance:
        max_offset = float(
            np.linalg.norm(camera_positions - center_coords, axis=1).max()
        )
        params.camera_ring_config.radial_distance = (
            max_offset * params.auto_radial_distance_factor
        )
        params.camera_ring_config.origin_position = (
            float(center_coords[0]),
            float(center_coords[1]),
            float(center_coords[2]),
        )
        logger.info(
            f"Auto radial distance: {params.camera_ring_config.radial_distance:.3f} "
            f"(max offset {max_offset:.3f} x factor "
            f"{params.auto_radial_distance_factor})"
        )

    scene_dict = {
        "type": "scene",
        "integrator": {"type": "path"},
        "environment": {"type": "envmap", "filename": str(params.environment_map_path)},
    }

    sensors = params.camera_ring_config.sensors

    script_dir = Path(__file__).parent.resolve()

    camera_scale = mi.ScalarTransform4f().scale(params.camera_size)

    for i, camera in enumerate(loader.train_cameras):
        scene_dict[f"camera_{i}"] = {
            "type": "ply",
            "filename": str(script_dir / "../../data/meshes/camera_pyramid.ply"),
            "bsdf": {"type": "diffuse"},
            "to_world": camera.tf_pose_matrix @ camera_scale,
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
