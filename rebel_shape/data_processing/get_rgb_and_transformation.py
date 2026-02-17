import json
import math
from enum import Enum
from pathlib import Path

import drjit as dr
import drjit.auto.ad as drad
import mitsuba as mi
import numpy as np
import numpy.typing as npt

from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from rebel_shape.plugin_wrappers.perspective_camera import PerspectiveCamera
from rebel_shape.plugin_wrappers.prb_integrator import GuidingOptions, PRBIntegrator
from rebel_shape.plugin_wrappers.sampler import Sampler
from rebel_shape.utils.sensor_utils import create_cameras_fibonacci


class SaveFormats(Enum):
    EXR = "exr"
    PNG = "png"


def _infer_intrinsics_from_mitsuba_sensor(
    sensor: "mi.Sensor",
) -> tuple[int, int, float, float, float, float]:
    """
    Infer OPENCV intrinsics from a Mitsuba perspective `sensor`.

    Steps:
    1) Read x_fov (degrees) and film.size (w, h) via mi.traverse(sensor)
    2) Convert fov -> focal length:
         fl_x = w / (2 * tan(x_fov/2))
       and assume square pixels -> fl_y = fl_x
    3) Set principal point using Nerfstudio convention:
         cx = (w - 1)/2, cy = (h - 1)/2
       then apply optional principal_point_offset_* if present.

    :returns: horizontal resolution, vertical resolution,focal length on x axis , focal
    lenght on y axis, principal point x, and y.
    """

    def to_numpy(data: drad.TensorXf) -> npt.NDArray[np.float64]:
        return dr.detach(data).numpy()

    params = mi.traverse(sensor)

    x_fov_deg = float(to_numpy(params["x_fov"]))

    film_size_np = to_numpy(params["film.size"])

    w = int(film_size_np[0])
    h = int(film_size_np[1])

    x_fov_rad = math.radians(x_fov_deg)
    fl_x = w / (2.0 * math.tan(0.5 * x_fov_rad))
    fl_y = fl_x

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    if "principal_point_offset_x" in params:
        cx += float(to_numpy(params["principal_point_offset_x"])) * w
    if "principal_point_offset_y" in params:
        cy += float(to_numpy(params["principal_point_offset_y"])) * h

    return w, h, fl_x, fl_y, cx, cy


def export_transforms_from_cameras_opencv(
    cameras: list[PerspectiveCamera],
    out_path: Path,
    image_prefix: str,
    save_format: SaveFormats = SaveFormats.EXR,
) -> None:
    """
    Export a Nerfstudio-style transforms.json using camera_model="OPENCV".

    Reads intrinsics from the first Mitsuba sensor in the `cameras` list (x_fov +
    film.size) and converts them into (fl_x, fl_y, cx, cy) using the Nerfstudio
    convention.

    The images and transform.json file are writtent to `out_path`. Images can optionally
    have a `image_prefix` in front of their filenames, and are saved in the format
    specified by `save_format`.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sensor0 = cameras[0].mitsuba_sensor
    w, h, fl_x, fl_y, cx, cy = _infer_intrinsics_from_mitsuba_sensor(sensor0)

    out = {
        "w": int(w),
        "h": int(h),
        "fl_x": float(fl_x),
        "fl_y": float(fl_y),
        "cx": float(cx),
        "cy": float(cy),
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "camera_model": "OPENCV",
        "frames": [],
    }

    for i, cam in enumerate(cameras):
        T_mitsuba = cam.tf_pose_matrix.matrix.numpy()

        mitsuba_to_opencv = np.diag([-1.0, 1.0, -1.0, 1.0])
        T_export = (T_mitsuba @ mitsuba_to_opencv).tolist()

        out["frames"].append(
            {
                "file_path": f"images/{image_prefix}_{i:05d}.{save_format.value}",
                "transform_matrix": T_export,
            }
        )

    out_path.write_text(json.dumps(out, indent=2))


def render_and_save_reference_set(
    sensor_count: int,
    spp_continuous: int,
    spp_projective: int,
    spp: int,
    image_resolution: int,
    save_dir: Path,
    mesh_path: Path,
    image_prefix: str,
    save_format: SaveFormats = SaveFormats.EXR,
) -> None:
    """
    Render a reference set for a given mesh (found at `mesh_path`) and export
    `sensor_count` images together with a Nerfstudio-style `transforms.json`.

    The scene is built with:
    - a diffuse checkerboard texture,
    - a set environment map for illumination,

    Cameras are distributed on a Fibonacci sphere around the origin and one
    image is rendered per camera. Rendered images are saved to `save_dir` and camera
    intrinsics/extrinsics are exported in OPENCV format.

    `spp` stands for  Samples Per Pixel and is the number of samples per pixel for the
    primal rendering step. `spp_continuous` and `spp_projective` are parameters of the
    `PRBIntegrator`. `image_resolution` defines the cameras resolution along x and y.
    Each image can be saved with an `image_prefix` before it's name and is saved in the
    format defined by `save_format`.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "images").mkdir(parents=True, exist_ok=True)

    integrator = PRBIntegrator(
        max_bounces=4,
        bounces_without_russian_roulette=4,
        spp_continuous=spp_continuous,
        spp_projective=spp_projective,
        spp_indirect=spp_projective,
        guiding=GuidingOptions.NONE,
    )

    repo_root = Path(__file__).resolve().parents[2]
    target_texture_path = repo_root / "data" / "textures" / "checkerboard_texture.jpg"
    envmap_path = repo_root / "data" / "textures" / "envmap2.exr"

    scene_dict = {
        "type": "scene",
        "integrator": integrator.mitsuba_dict,
        "emitter": {"type": "envmap", "filename": str(envmap_path)},
        "shape": {
            "type": "ply",
            "filename": str(mesh_path),
            "bsdf": DiffuseBSDF(
                reflectance=DiffuseBSDF.Texture(texture_path=target_texture_path)
            ).mitsuba_dict,
        },
    }

    scene_target: mi.Scene = mi.load_dict(scene_dict)

    cameras = create_cameras_fibonacci(
        sensor_count=sensor_count,
        target_coords=(0.0, 0.0, 0.0),
        sensor_resolution=(image_resolution, image_resolution),
        sampler=Sampler(type=Sampler.Type.MULTIJITTER, sample_count=spp_continuous),
    )

    for i, cam in enumerate(cameras):
        img = mi.render(scene_target, sensor=cam.mitsuba_sensor, spp=spp)
        out_path = save_dir / "images" / f"{image_prefix}_{i:05d}.{save_format.value}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mi.Bitmap(img).write(str(out_path))

    export_transforms_from_cameras_opencv(
        cameras=cameras,
        out_path=save_dir / "transforms.json",
        image_prefix=image_prefix,
        save_format=save_format,
    )
