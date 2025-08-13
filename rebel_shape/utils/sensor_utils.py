import drjit as dr
import mitsuba as mi

from rebel_shape.plugin_wrappers.perspective_camera import PerspectiveCamera
from rebel_shape.plugin_wrappers.sampler import Sampler


def create_cameras_fibonacci(
    sensor_count: int = 8,
    radial_distance: float = 5.0,
    target_coords: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sensor_resolution: tuple[int, int] = (256, 256),
    sensor_fov: float = 45,
    sampler: Sampler = Sampler(),
) -> list[PerspectiveCamera]:
    """
    Creates a list of `sensor_count` sensors placed on the surface of a sphere
    following the Fibonacci lattice.

    Where the camera is pointing is specified by `target_coords` while
    `radial_distance` denotes the radius of the sphere. `target_coords` are
    given as as a tuple with 3 components, first is the x-component, second y, and
    third z.

    The resolution of the
    produced images is specified by `sensor_resolution` and the width of the shot
    by `sensor_fov`. Finally, the sampler information is provided by `sampler`
    which returns a compatible dict.

    :returns: A list of PerspectiveCamera object instances

    :raises ValueError: If `radial_distance` or `sensor_count` is non-positive
    """

    if sensor_count < 1:
        raise ValueError("`sensor_count` must be a positive integer.")

    if radial_distance <= 0.0:
        raise ValueError("The 'radial_distance' of the sphere has to be positive.")

    cameras: list[PerspectiveCamera] = []
    golden_ratio = (1 + 5**0.5) / 2

    for i in range(sensor_count):
        theta = 2 * dr.pi * i / golden_ratio
        phi = dr.acos(1 - 2 * (i + 0.5) / sensor_count)

        origin = (
            target_coords[0] + radial_distance * dr.cos(theta) * dr.sin(phi),
            target_coords[1] + radial_distance * dr.sin(theta) * dr.sin(phi),
            target_coords[2] + radial_distance * dr.cos(phi),
        )

        perspective_camera = PerspectiveCamera.from_origin_and_target(
            origin=origin,
            target=target_coords,
            camera_resolution=sensor_resolution,
            fov=sensor_fov,
            sampler=sampler,
        )

        cameras.append(perspective_camera)

    return cameras


def create_sensors_radial(
    sensor_count: int = 8,
    radial_distance: float = 10.0,
    target_coords: tuple[float, float, float] = (0.0, 0.0, 0.0),
    viewing_angle: float = 45.0,
    sensor_resolution: tuple[int, int] = (640, 640),
    sensor_fov: float = 60,
    sampler: Sampler = Sampler(),
) -> list["mi.Sensor"]:
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

    :raises ValueError: If `radial_distance` or `sensor_count` is non-positive
    """

    if sensor_count < 1:
        raise ValueError("`sensor_count` must be a positive integer.")

    if radial_distance <= 0.0:
        raise ValueError("The 'radial_distance' of the circle has to be positive.")

    sensors: list["mi.Sensor"] = []
    phi_step = 2.0 * dr.pi / sensor_count
    viewing_angle_rad = viewing_angle * dr.pi / 180.0

    for i in range(sensor_count):
        phi = i * phi_step

        camera_x = target_coords[0] + radial_distance * dr.cos(phi)
        camera_z = target_coords[2] + radial_distance * dr.sin(phi)
        camera_y = target_coords[1] + radial_distance * dr.tan(viewing_angle_rad)

        camera_position = (camera_x, camera_y, camera_z)

        perspective_camera = PerspectiveCamera.from_origin_and_target(
            origin=camera_position,
            target=target_coords,
            camera_resolution=sensor_resolution,
            fov=sensor_fov,
            fov_axis=PerspectiveCamera.FovAxis.LARGER,
            sampler=sampler,
        )

        sensors.append(mi.load_dict(perspective_camera.mitsuba_dict))

    return sensors
