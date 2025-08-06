import drjit as dr

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
