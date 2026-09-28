from enum import Enum
from typing import Any

import mitsuba as mi

from forge_sim.plugin_wrappers.mitsuba_plugin import MitsubaPlugin
from forge_sim.plugin_wrappers.sampler import Sampler


class PerspectiveCamera(MitsubaPlugin):
    """
    Pyhon wrapper around Perspective pinhole camera Sensor
    https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_sensors.html

    """

    class FovAxis(Enum):
        """Specifies which axis/direction the field of view setting applies to"""

        X = "x"
        """x-axis in screen space; along width. Mitsuba default"""
        Y = "y"
        """y-axis in screen space; along width"""
        DIAGONAL = "diagonal"
        """along screen diagonal"""
        SMALLER = "smaller"
        """along the smaller dimension (x when width < height)"""
        LARGER = "larger"
        """along the larger dimension (y when width < height)"""

    class PixelFormat(Enum):
        """
        Specifies how many channels are used to represent a pixel

        When dealing with thermal data, the pixel format should be set to
        `luminance` whereas otherwise it should be set to `rgb`. `rgba` is
        adn other mitsuba pixel format that is not supported by this wrapper.
        """

        RGB = "rgb"
        """RGB pixel format - 3 channels, standard color image"""
        LUMINANCE = "luminance"
        """Luminance pixel format - 1 channel, grayscale image"""

    def __init__(
        self,
        tf_camera_pose: "mi.ScalarTransform4f",
        camera_resolution: tuple[int, int] = (256, 256),
        fov: float = 45,
        fov_axis: FovAxis = FovAxis.X,
        pixel_format: PixelFormat = PixelFormat.RGB,
        sampler: Sampler = Sampler(),
    ) -> None:
        """
        `pose_matrix` is a native Mitsuba type that encapsulates the 4x4 matrix
        (homogenous coordinate transform) representing the pose of the camera.

        The width of the shot is determined by `fov` and the level of detail by
        `camera_resolution` (first element of the tuple is pixel width and
        the second pixel height). The `fov` value is applied along `fov_axis`.

        Make sure that the total number of samples (resolution * `sampler.sample_count`)
        is less than 2**32 (unsigned 32-bit integer size).

        :raises ValueError: if values provided in `camera_resolution` are not positive,
            or if the field of view `fov` is not in (0.0, 180.0), or the product of
            the number of pixels and `sample_count` from sampler is higher than
            Mitsuba supports.
        """

        if camera_resolution[0] < 1 or camera_resolution[1] < 1:
            raise ValueError("`sensor_resolution` cannot be smaller than (1, 1).")

        if fov <= 0.0 or fov >= 180.0:
            raise ValueError(
                "`sensor_fov` must be between 0 and 180 degrees, exclusive."
            )

        if camera_resolution[0] * camera_resolution[1] * sampler.sample_count >= 2**32:
            raise ValueError(
                "Mitsuba 3 uses excessive memory when the total number of samples "
                "in the image to render exceeds 2**32 - 1 samples. "
                "Reduce the resolution or the `sample_count` of the `sampler`"
            )

        self._tf_camera_pose = tf_camera_pose
        self._camera_resolution = camera_resolution
        self._fov = fov
        self._fov_axis = fov_axis
        self._pixel_format = pixel_format
        self._sampler = sampler

    @classmethod
    def from_origin_and_target(
        cls,
        origin: tuple[float, float, float],
        target: tuple[float, float, float],
        camera_resolution: tuple[int, int] = (256, 256),
        fov: float = 45,
        fov_axis: FovAxis = FovAxis.X,
        sampler: Sampler = Sampler(),
    ) -> "PerspectiveCamera":
        """
        Creates a camera base on where the camera is located and where it is pointing.

        The camera pose is determined by the 3D coordinates of the camera `origin`
        and the point the camera is looking at `target`. The coordinates are passed
        as a tuple, with first element being the x-component, second y, and third
        the z component.

        The width of the shot is determined by `fov` and the level of detail by
        `camera_resolution` (first element of the tuple is pixel width and
        the second pixel height). The `fov` value is applied along `fov_axis`.

        Make sure that the total number of samples (resolution * `sampler.sample_count`)
        is less than 2**32 (unsigned 32-bit integer size).

        :returns: Initialized PerspectiveCamera object.

        :raises ValueError: if the `origin` or `target` list do not have exactly 3
            elements.
        """
        if len(origin) != 3:
            raise ValueError("The `origin` list has to have 3 elements.")

        if len(target) != 3:
            raise ValueError("The `target` list has to have 3 elements.")

        tf_camera_pose = mi.ScalarTransform4f().look_at(
            target=mi.ScalarPoint3f(target),
            origin=mi.ScalarPoint3f(origin),
            up=mi.ScalarPoint3f([0, 1, 0]),
        )

        return cls(
            tf_camera_pose=tf_camera_pose,
            camera_resolution=camera_resolution,
            fov=fov,
            fov_axis=fov_axis,
            sampler=sampler,
        )

    @property
    def tf_pose_matrix(self) -> "mi.ScalarTransform4f":
        """
        Camera pose matrix of the camera.

        The camera pose matrix is a 4x4 matrix that represents the
        transformation from camera space to world space.
        """
        return self._tf_camera_pose

    @tf_pose_matrix.setter
    def tf_pose_matrix(self, value: "mi.ScalarTransform4f") -> None:
        self._tf_camera_pose = value

    @property
    def pixel_format(self) -> PixelFormat:
        """
        Pixel format of the camera.

        When doing thermal simulation it is important to set it to `luminance` and
        not `rgb`.
        """
        return self._pixel_format

    @pixel_format.setter
    def pixel_format(self, value: PixelFormat) -> None:
        self._pixel_format = value

    @property
    def mitsuba_dict(self) -> dict[str, Any]:
        """
        Get the Mitsuba compatible dictionary representation of this sensor.

        :returns: A dictionary that is passed to the Mitsuba scene description dict.
        """
        return {
            "type": "perspective",
            "fov": self._fov,
            "fov_axis": self._fov_axis.value,
            "to_world": self._tf_camera_pose,
            "film": {
                "type": "hdrfilm",
                "width": self._camera_resolution[0],
                "height": self._camera_resolution[1],
                "pixel_format": self._pixel_format.value,
                "filter": {"type": "gaussian"},
                "sample_border": True,
            },
            "sampler": self._sampler.mitsuba_dict,
        }

    @property
    def mitsuba_sensor(self) -> "mi.Sensor":
        """
        Returns a mitsuba sensor corresponding to `self`
        """
        return mi.load_dict(self.mitsuba_dict)
