from pathlib import Path

import mitsuba as mi

from rebel_shape.data_processing.image_scene_loader import ImageSceneLoader
from rebel_shape.data_processing.masked_pose_image import MaskedPoseImage
from rebel_shape.data_processing.pose_scene_loader import PoseSceneLoader
from rebel_shape.data_processing.vggt_pose_scene_loader import VGGTPoseSceneLoader
from rebel_shape.plugin_wrappers.perspective_camera import PerspectiveCamera
from rebel_shape.plugin_wrappers.sampler import Sampler


class SilhouetteSceneLoader(ImageSceneLoader):
    """
    An `ImageSceneLoader` that skips background rendering and compositing entirely.
    Instead of a composited RGB image, each loaded image is the silhouette
    `1 - object_mask`, a single-channel image (shape `(H, W, 1)`, black on the
    object, white elsewhere), with no foreground mask. This is intended for
    pose-only optimization where the loss is a pure silhouette match.
    """

    def __init__(
        self,
        scene_path: Path,
        pose_scene_loader: PoseSceneLoader,
        output_dir: Path | None = None,
    ) -> None:
        """
        Loads the images and object masks from the directory `scene_path`, using the
        camera poses already loaded by `pose_scene_loader`, and stores the silhouette
        `1 - object_mask` as a single-channel image (shape `(H, W, 1)`, black on the
        object, white elsewhere). No background rendering or compositing is
        performed, and no foreground mask is used.

        :raises RuntimeError: If a presented image is not a colour image carrying its
            object mask in an alpha channel.
        """
        super().__init__(
            scene_path=scene_path,
            pose_scene_loader=pose_scene_loader,
            use_foreground_masks=False,
            environment_map_path=None,
            output_dir=output_dir,
        )

    def _generate_image(
        self,
        background_scene: "mi.Scene",
        camera: PerspectiveCamera,
        mitsuba_image_rgba: "mi.TensorXf",
        object_mask: "mi.TensorXf",
        mitsuba_foreground_mask: "mi.TensorXf",
    ) -> MaskedPoseImage:
        # Silhouette: black where the object is, white elsewhere.
        # Single-channel (H, W, 1) tensor, no background render, no mask.
        composited_image = 1.0 - object_mask

        return MaskedPoseImage(
            image=composited_image,
            camera=camera,
            mask=None,
        )

    @classmethod
    def from_thermoscene(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        scene_path: Path,
        sensor_sampler: Sampler = Sampler(),
        align_cameras_xz: bool = True,
        output_dir: Path | None = None,
        pixel_format: PerspectiveCamera.PixelFormat = PerspectiveCamera.PixelFormat.RGB,
    ) -> "ImageSceneLoader":
        """
        Loads the ThermoScenes images, object masks, and camera poses from the
        directory `scene_path` via `PoseSceneLoader`. Optionally aligns camera poses
        so they lie on the xz-plane if `align_cameras_xz` is `True`. Each camera
        samples rays through `sensor_sampler`. Rather than compositing against a
        rendered background, each image is stored as the silhouette `1 - object_mask`
        (see `SilhouetteSceneLoader`). Optionally stores the silhouette images in
        `output_dir` if provided.

        :raises FileNotFoundError: If the scene path does not exist.
        :raises RuntimeError: If a presented image is not a colour image carrying its
            object mask in an alpha channel.
        """
        pose_loader = PoseSceneLoader(
            scene_path=scene_path,
            sensor_sampler=sensor_sampler,
            align_cameras_xz=align_cameras_xz,
            pixel_format=pixel_format,
        )

        return cls(
            scene_path,
            pose_loader,
            output_dir,
        )

    @classmethod
    def from_vggt(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        scene_path: Path,
        sensor_sampler: Sampler = Sampler(),
        align_cameras_xz: bool = True,
        output_dir: Path | None = None,
        pixel_format: PerspectiveCamera.PixelFormat = PerspectiveCamera.PixelFormat.RGB,
    ) -> "ImageSceneLoader":
        """
        Loads the VGGT images, object masks, and camera poses from the directory
        `scene_path` via `VGGTPoseSceneLoader`. Optionally aligns camera poses so
        they lie on the xz-plane if `align_cameras_xz` is `True`. Each camera
        samples rays through `sensor_sampler`. Rather than compositing against a
        rendered background, each image is stored as the silhouette `1 - object_mask`
        (see `SilhouetteSceneLoader`). Optionally stores the silhouette images in
        `output_dir` if provided.

        :raises FileNotFoundError: If the scene path does not exist.
        :raises RuntimeError: If a presented image is not a colour image carrying its
            object mask in an alpha channel.
        """
        pose_loader = VGGTPoseSceneLoader(
            scene_path=scene_path,
            sensor_sampler=sensor_sampler,
            align_cameras_xz=align_cameras_xz,
            pixel_format=pixel_format,
        )

        return cls(
            scene_path,
            pose_loader,
            output_dir,
        )
