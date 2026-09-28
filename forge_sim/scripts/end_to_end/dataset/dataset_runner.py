import json
import math

import mitsuba as mi
import numpy as np
import numpy.typing as npt

from forge_sim import logger
from forge_sim.data_processing.masked_pose_image import MaskedPoseImage
from forge_sim.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from forge_sim.plugin_wrappers.perspective_camera import PerspectiveCamera
from forge_sim.plugin_wrappers.sampler import Sampler
from forge_sim.scripts.end_to_end.dataset.dataset_modality import DatasetModality
from forge_sim.scripts.end_to_end.dataset.dataset_summary import DatasetSummary
from forge_sim.scripts.end_to_end.dataset.mesh_arrays import MeshArrays
from forge_sim.scripts.end_to_end.dataset.synthetic_fields import (
    material_labels,
    synthetic_temperature,
)
from forge_sim.scripts.end_to_end.dataset.synthetic_reference_images import (
    SyntheticReferenceImages,
)
from forge_sim.scripts.end_to_end.end_to_end_config import EndToEndConfig
from forge_sim.utils.sensor_utils import create_cameras_fibonacci


class DatasetGenerationRunner:
    """
    Renders and writes the synthetic multi-view dataset (RGB + thermal + material),
    and reloads it back through the same deterministic cameras for downstream stages.

    The thermal and material renders are emissive (black BSDF + area emitter reading
    a per-vertex attribute) so they encode the field directly, with no shading. The
    same `transforms.json` later drives both reconstruction (RGB) and inpainting
    (thermal/material), guaranteeing identical camera frames, which is also why
    `load_synthetic_references` bypasses the loaders' re-framing and rebuilds the
    exact cameras `run` used.
    """


    def __init__(self, config: EndToEndConfig) -> None:
        """Stores the `config` this run's dataset is generated and loaded from."""
        self._config = config

    def run(self) -> DatasetSummary:
        """
        Render and write the synthetic dataset; return where it went and how it split.

        A Mitsuba variant must already be set by the caller.

        :raises RuntimeError: If the source mesh has no UVs while the RGB renders are
            configured to use the checkerboard texture (`solid_albedo` is None).
        """
        out = self._config.dataset_dir
        for modality in DatasetModality:
            (out / modality.value).mkdir(exist_ok=True)

        source = MeshArrays.from_ply(self._config.mesh_path)
        if source.uvs is None and self._config.dataset.solid_albedo is None:
            raise RuntimeError(
                f"{self._config.mesh_path} has no UVs but solid_albedo is None; "
                "the RGB checkerboard texture needs UVs. Set a solid_albedo or "
                "provide a UV'd mesh."
            )
        temperature = synthetic_temperature(
            source.vertices, source.uvs, self._config.thermal
        )
        material = material_labels(
            source.vertices,
            self._config.material_axis,
            self._config.material_threshold,
        )

        rgb_scene = self._rgb_scene()
        sil_scene = self._emissive_scene(None)
        thermal_scene = self._emissive_scene(temperature)
        material_scene = self._emissive_scene(material)

        cameras = self.build_cameras(self._config)
        res = self._config.dataset.image_resolution
        half_fov = 0.5 * math.radians(self._config.dataset.camera_fov_degrees)
        fl = res / (2.0 * math.tan(half_fov))
        spp_rgb = self._config.dataset.reference_samples
        spp_field = self._config.dataset.field_samples
        mitsuba_to_opencv = np.diag([-1.0, 1.0, -1.0, 1.0])

        num_train = self._config.dataset.num_views
        frames = []
        for i, cam in enumerate(cameras):
            name = self._frame_name(i, num_train)
            sensor = cam.mitsuba_sensor

            rgb = np.array(mi.render(rgb_scene, sensor=sensor, spp=spp_rgb))[:, :, :3]
            alpha = np.array(mi.render(sil_scene, sensor=sensor, spp=spp_field))[
                :, :, 0
            ]
            therm = np.array(mi.render(thermal_scene, sensor=sensor, spp=spp_field))[
                :, :, 0
            ]
            mat = np.array(mi.render(material_scene, sensor=sensor, spp=spp_field))[
                :, :, 0
            ]

            rgba = np.dstack(
                [self._to_u8(self._linear_to_srgb(rgb)), self._to_u8(alpha)]
            )
            mi.Bitmap(rgba).write(str(out / DatasetModality.RGB.value / f"{name}.png"))
            # Thermal + material are linear scalar fields, stored raw (value =
            # field*255) exactly like real FLIR thermal data. The loader marks the
            # 8-bit source linear (set_srgb_gamma(False)) so convert applies no sRGB
            # decode, so save+load is a verbatim round-trip. (RGB above is
            # sRGB-encoded; real photos are sRGB.)
            mi.Bitmap(self._to_u8(therm)).write(
                str(out / DatasetModality.THERMAL.value / f"{name}.png")
            )
            mi.Bitmap(self._to_u8(mat)).write(
                str(out / DatasetModality.MATERIAL.value / f"{name}.png")
            )

            t_mi = np.array(cam.tf_pose_matrix.matrix)
            frames.append(
                {
                    "file_path": f"{DatasetModality.RGB.value}/{name}.png",
                    "thermal_file_path": f"{DatasetModality.THERMAL.value}/{name}.png",
                    "windows_mask_file_path": (
                        f"{DatasetModality.MATERIAL.value}/{name}.png"
                    ),
                    "transform_matrix": (t_mi @ mitsuba_to_opencv).tolist(),
                }
            )

        transforms = {
            "w": res,
            "h": res,
            "fl_x": fl,
            "fl_y": fl,
            "cx": (res - 1) / 2.0,
            "cy": (res - 1) / 2.0,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "camera_model": "OPENCV",
            "frames": frames,
        }
        (out / "transforms.json").write_text(json.dumps(transforms, indent=2))

        n_eval = self._config.dataset.num_eval_views
        logger.info(
            f"[step 2a] Wrote {len(frames)} views ({num_train} train + {n_eval} eval) "
            f"to {out} | steel frac {material.mean():.3f}, "
            f"hot frac {(temperature > 0.5).mean():.3f}"
        )
        return DatasetSummary(dataset_dir=out, num_train=num_train, num_eval=n_eval)

    @classmethod
    def load_synthetic_references(
        cls, config: EndToEndConfig, modality: DatasetModality
    ) -> SyntheticReferenceImages:
        """
        Load the dataset images for `modality` paired with the same deterministic
        cameras that rendered them, bypassing the loaders' re-framing so the object
        stays at the origin and the reconstruction/inpainting init overlaps it.

        A Mitsuba variant must already be set by the caller.

        :returns: The dataset split by the same rule `run` wrote it with.
        """
        is_rgb = modality is DatasetModality.RGB
        pixel_format = (
            PerspectiveCamera.PixelFormat.RGB
            if is_rgb
            else PerspectiveCamera.PixelFormat.LUMINANCE
        )
        cameras = cls.build_cameras(config)
        num_train = config.dataset.num_views
        train: list[MaskedPoseImage] = []
        eval_: list[MaskedPoseImage] = []
        for i, cam in enumerate(cameras):
            name = cls._frame_name(i, num_train)
            path = config.dataset_dir / modality.value / f"{name}.png"
            raw = mi.Bitmap(str(path))
            if not is_rgb:
                # Linear scalar field (thermal/material) stored raw like real FLIR
                # data: mark the 8-bit source linear so convert applies no sRGB decode.
                raw.set_srgb_gamma(False)
            bitmap = raw.convert(
                component_format=mi.Struct.Type.Float32, srgb_gamma=is_rgb
            )
            image = mi.TensorXf(bitmap)
            if is_rgb:
                image = image[:, :, :3]
            else:
                if image.ndim == 2:
                    image = image[:, :, None]
                image = image[:, :, :1]
            cam.pixel_format = pixel_format
            masked = MaskedPoseImage(image=image, camera=cam, mask=None)
            (train if i < num_train else eval_).append(masked)
        return SyntheticReferenceImages(train_images=train, eval_images=eval_)

    @classmethod
    def build_cameras(cls, config: EndToEndConfig) -> list[PerspectiveCamera]:
        """
        Fibonacci-sphere cameras looking at the origin (RGB sampler). The first
        `num_views` are training views; `num_eval_views` more are appended as eval.
        """
        res = config.dataset.image_resolution
        total = config.dataset.num_views + config.dataset.num_eval_views
        return create_cameras_fibonacci(
            sensor_count=total,
            radial_distance=config.dataset.camera_distance,
            target_coords=(0.0, 0.0, 0.0),
            sensor_resolution=(res, res),
            sensor_fov=config.dataset.camera_fov_degrees,
            sampler=Sampler(
                type=Sampler.Type.MULTIJITTER,
                sample_count=config.dataset.camera_sample_count,
            ),
        )

    @staticmethod
    def _frame_name(index: int, num_train: int) -> str:
        """Frame stem shared by all three modalities, carrying the train/eval split."""
        split = "train" if index < num_train else "eval"
        return f"frame_{split}_{index:05d}"

    def _rgb_scene(self) -> "mi.Scene":
        """The lit scene behind `images/`: the source mesh under the env map."""
        albedo = self._config.dataset.solid_albedo
        if albedo is not None:
            reflectance = DiffuseBSDF.Albedo(albedo=albedo)  # uniform grey
        else:
            reflectance = DiffuseBSDF.Texture(
                texture_path=self._config.dataset.checkerboard_texture
            )
        return mi.load_dict(
            {
                "type": "scene",
                "integrator": {"type": "path", "max_depth": 4},
                "emitter": {
                    "type": "envmap",
                    "filename": str(self._config.dataset.envmap),
                },
                "shape": {
                    "type": "ply",
                    "filename": str(self._config.mesh_path),
                    "bsdf": DiffuseBSDF(reflectance=reflectance).mitsuba_dict,
                },
            }
        )

    def _emissive_scene(
        self, attr_values: npt.NDArray[np.float64] | None
    ) -> "mi.Scene":
        """
        Black-BSDF mesh with an area emitter. If `attr_values` is given the emitter
        reads a per-vertex attribute (field render); otherwise it emits a constant 1
        (silhouette render).

        :raises RuntimeError: If `attr_values` does not carry one value per vertex.
        """
        if attr_values is None:
            radiance = {"type": "rgb", "value": 1.0}
        else:
            radiance = {"type": "mesh_attribute", "name": "vertex_field"}

        scene = mi.load_dict(
            {
                "type": "scene",
                "integrator": {"type": "path", "max_depth": 2},
                "shape": {
                    "type": "ply",
                    "filename": str(self._config.mesh_path),
                    "bsdf": {
                        "type": "diffuse",
                        "reflectance": {"type": "rgb", "value": 0.0},
                    },
                    "emitter": {"type": "area", "radiance": radiance},
                },
            }
        )
        if attr_values is not None:
            shape = scene.shapes()[0]
            if shape.vertex_count() != attr_values.shape[0]:
                raise RuntimeError(
                    f"attribute length {attr_values.shape[0]} != vertex count "
                    f"{shape.vertex_count()}"
                )
            buf = np.repeat(attr_values[:, None], 3, axis=1).astype(np.float32).ravel()
            shape.add_attribute("vertex_field", 3, buf)
        return scene

    @staticmethod
    def _linear_to_srgb(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Encode linear radiance, clipped to [0, 1], with the sRGB transfer curve."""
        x = np.clip(x, 0.0, 1.0)
        return np.where(
            x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055
        )

    @staticmethod
    def _to_u8(x: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        """Quantise values, clipped to [0, 1], onto the 8-bit range."""
        return np.clip(np.round(np.clip(x, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
