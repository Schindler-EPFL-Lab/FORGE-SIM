"""
This script calculates the metrics of novel view synthesis and shape reconstruction on a
buildnet3d dataset.
"""

import os
from abc import abstractmethod

import torchvision

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import json
from dataclasses import dataclass
from pathlib import Path

import mitsuba as mi
import open3d as o3d
import torch
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

from forge_sim.data_processing.image_scene_loader import ImageSceneLoader
from forge_sim.meshing.tessellated_spline_surface import TessellatedSplineSurface
from forge_sim.metrics.novel_view_synthesis import (
    calculate_metrics_novel_view_synthesis,
    mitsuba_to_torch,
)
from forge_sim.metrics.shapes import (
    calculate_metrics_shape_mesh,
    tessellated_spline_to_open3d_mesh,
)


@dataclass(kw_only=True)
class EvalNVSParametersBase(ReverseCli):
    """
    Parameters used for evaluation of novel view synthesis.
    """

    thermoscenes_scene_path: Path
    """Path to the ThermoScenes scene on what the spline was optimized."""
    rendered_images_folder: Path
    """Path to the rendered images folder of the optimized spline surface."""
    environment_map_path: Path | None = None
    """
    Path to the envionment map that was used as background in optimization. If not
    provided, a solid white background is used instead.
    """
    samples_per_pixel: int = 169
    """Number of samples per pixel used for rendering"""

    reference_shape_path: Path
    """Path to the target shape on what the spline was optimized"""
    n_sampled_points: int = 100000
    """How many points to sample on both meshes to compute shapes similarity"""

    output_folder: Path = Path("./outputs/")
    """Where to save the metrics (created if does not exist)"""

    def __post_init__(self) -> None:
        if not self.thermoscenes_scene_path.exists():
            raise FileNotFoundError(f"'{self.thermoscenes_scene_path}' does not exist.")
        if not self.rendered_images_folder.exists():
            raise FileNotFoundError(f"'{self.rendered_images_folder}' does not exist.")

    @abstractmethod
    def get_open3d_mesh(self) -> o3d.t.geometry.TriangleMesh:
        """Returns a shape mesh converted to Open3D"""
        raise NotImplementedError("The method `get_open3d_mesh` is not implemented.")


@dataclass(kw_only=True)
class EvalNVSParametersSpline(EvalNVSParametersBase):
    """A config for spline evaluation."""

    spline_path: Path
    """Path to the reconstructed spline surface"""
    tessellation_subdivisions: int = 256
    """
    How well the tessellated triangle mesh approximates the original B-spline surface.
    The higher the number, the more triangles there are, and the better the
    approximation.
    """

    def __post_init__(self) -> None:
        if not self.spline_path.exists():
            raise FileNotFoundError(f"'{self.spline_path}' does not exist.")

    def get_open3d_mesh(self) -> o3d.t.geometry.TriangleMesh:
        """Returns a spline tessellated mesh converted to Open3D"""
        optimized_spline_surface = TessellatedSplineSurface.from_file(self.spline_path)
        optimized_spline_surface._composite_spline_surface.composite_mesh_tesselation(
            subdivisions_per_patch=self.tessellation_subdivisions
        )
        return tessellated_spline_to_open3d_mesh(optimized_spline_surface)


def eval_buildnet3d(params: EvalNVSParametersBase) -> None:
    """
    Calculates the metrics for novel view synthesis between the ground truth images and
    the optimized spline surface rendered images.

    It loads the ThermoScenes scene from `params.thermoscenes_scene_path` along with
    rendered images from `params.rendered_images_folder`. Then it calculates novel view
    synthesis metrics between ground truth (GT) and rendered images.
    It assumes that stems of rendered images and grouth truth images
    are the same for the corresponding images. It saves the result metrics in json file
    at `params.output_file_path`

    :raises ValueError: If `params.thermoscenes_scene_path` is not a valid path, or
        it is not a directory.
    """

    mi.set_variant("llvm_ad_rgb")
    params.output_folder.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    loader = ImageSceneLoader.from_thermoscene(
        scene_path=params.thermoscenes_scene_path,
        environment_map_path=params.environment_map_path,
        use_foreground_masks=True,
    )

    rendered_images_paths = sorted(list(params.rendered_images_folder.glob("*")))
    rendered_images_paths = {
        image_path.stem: image_path for image_path in rendered_images_paths
    }

    eval_images_torch: list[torch.Tensor] = []
    rendered_images_torch: list[torch.Tensor] = []

    for filename, eval_image in loader.eval_data:
        eval_image_torch = mitsuba_to_torch(eval_image.image, device=device)
        eval_images_torch.append(eval_image_torch)

        rendered_image_path = rendered_images_paths[Path(filename).stem]
        mitsuba_rendered_image = mi.TensorXf(
            mi.Bitmap(str(rendered_image_path)).convert(
                component_format=mi.Struct.Type.Float32, srgb_gamma=False
            )
        )
        rendered_image = eval_image.apply_mask(mitsuba_rendered_image)
        rendered_image_torch = mitsuba_to_torch(rendered_image, device=device)
        rendered_images_torch.append(rendered_image_torch)

    torchvision.utils.save_image(
        eval_images_torch, params.output_folder / "eval_images.png"
    )
    torchvision.utils.save_image(
        rendered_images_torch, params.output_folder / "rendered_images.png"
    )

    novel_view_synthes_metrics = calculate_metrics_novel_view_synthesis(
        rendered_images=rendered_images_torch,
        ground_truth_images=eval_images_torch,
        device=device,
    )

    output_file_novel_view_synthesis = params.output_folder / "eval_nvs.json"
    with open(output_file_novel_view_synthesis, "w") as f:
        json.dump(novel_view_synthes_metrics, f, indent=4)

    # calculate the shapes metrics
    ref_shape = o3d.t.io.read_triangle_mesh(str(params.reference_shape_path))
    tf_initial_pose = mi.Transform4f(loader.tf_cameras)
    tf_initial_pose_np = tf_initial_pose.matrix.numpy()[:, :, 0]

    ref_shape.transform(tf_initial_pose_np)

    optimized_shape = params.get_open3d_mesh()

    # calculate the metrics
    metrics = calculate_metrics_shape_mesh(
        predicted_mesh=optimized_shape,
        ground_truth_mesh=ref_shape,
        n_sampled_points=params.n_sampled_points,
    )

    output_file_shape = params.output_folder / "eval_shape.json"
    with open(output_file_shape, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    parameters = tyro.cli(EvalNVSParametersSpline)
    eval_buildnet3d(params=parameters)
