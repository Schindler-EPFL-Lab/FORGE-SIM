from dataclasses import dataclass

import drjit as dr
import mitsuba as mi
from dataclasses_reverse_cli.reverse_cli import ReverseCli

from forge_sim.plugin_wrappers.perspective_camera import PerspectiveCamera
from forge_sim.plugin_wrappers.sampler import Sampler


@dataclass(kw_only=True)
class CameraRingConfig(ReverseCli):
    """
    Configuration for a ring of cameras placed around the specified origin.

    The cameras are placed in a radial pattern around the origin at a specified
    distance, with a specified viewing angle and field of view. The number of cameras
    """

    samples_per_pixel: int = 16
    """Number of samples per pixel used for rendering."""
    sampler: Sampler.Type = Sampler.Type.MULTIJITTER
    """Type of sampler to use for rendering."""
    sensor_count: int = 6
    """Number of sensors placed in radial pattern around the origin."""

    origin_position_x: float = 0.0
    """X coordinate of the origin position."""
    origin_position_y: float = 0.0
    """Y coordinate of the origin position."""
    origin_position_z: float = 0.0
    """Z coordinate of the origin position."""
    radial_distance: float = 10.0
    """Distance of cameras from the origin."""

    sensor_resolution_width: int = 480
    """Resolution of the camera sensor along width"""
    sensor_resolution_height: int = 480
    """Resolution of the camera sensor along height"""
    sensor_fov: float = 45.0
    """Field of view of the camera sensors in degrees"""
    viewing_angle: float = -10.0
    """Camera viewing angle in degrees (negative looks down, positive looks up)"""

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

    @property
    def camera_resolution(self) -> tuple[int, int]:
        """Returns the camera resolution as a tuple (width, height)."""
        return self.sensor_resolution_width, self.sensor_resolution_height

    @property
    def origin_position(self) -> tuple[float, float, float]:
        """Returns the origin position as a tuple (x, y, z)."""
        return self.origin_position_x, self.origin_position_y, self.origin_position_z

    @origin_position.setter
    def origin_position(self, position: tuple[float, float, float]) -> None:
        """Sets the origin position from a tuple (x, y, z)."""
        self.origin_position_x, self.origin_position_y, self.origin_position_z = (
            position
        )

    @property
    def sensors(self) -> list["mi.Sensor"]:
        """
        Creates a list of `sensor_count` sensors placed in a radially symmetric pattern
        around an origin point.

        Cameras are placed in a circle around `targets_coords` at a distance of
        `radial_distance` (symmetric about the y-axis).
        The cameras are placed and oriented looking down at `target_coords` with
        angle `viewing_angle`.

        The resolution of the produced images is specified by `sensor_resolution`
        and the width of the shot by `sensor_fov`. The sampler information is
        provided by `sampler`.

        :returns: A list of Mitsuba sensor object instances
        """
        phis = dr.linspace(
            dtype=mi.Float,
            start=0.0,
            stop=2.0 * dr.pi,
            num=self.sensor_count,
            endpoint=False,
        )
        sensors: list["mi.Sensor"] = []

        for phi in phis:
            camera_x = self.origin_position_x + self.radial_distance * dr.cos(phi)
            camera_z = self.origin_position_z + self.radial_distance * dr.sin(phi)
            camera_y = self.origin_position_y + self.radial_distance * dr.tan(
                self.viewing_angle * dr.pi / 180.0
            )

            camera_position = (camera_x, camera_y, camera_z)

            perspective_camera = PerspectiveCamera.from_origin_and_target(
                origin=camera_position,
                target=(
                    self.origin_position_x,
                    self.origin_position_y,
                    self.origin_position_z,
                ),
                camera_resolution=self.camera_resolution,
                fov=self.sensor_fov,
                fov_axis=PerspectiveCamera.FovAxis.LARGER,
                sampler=Sampler(type=self.sampler, sample_count=self.samples_per_pixel),
            )

            sensors.append(mi.load_dict(perspective_camera.mitsuba_dict))

        return sensors
