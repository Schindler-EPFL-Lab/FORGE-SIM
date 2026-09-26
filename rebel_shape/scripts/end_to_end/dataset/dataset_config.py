from dataclasses import dataclass
from pathlib import Path


@dataclass(kw_only=True)
class DatasetConfig:
    """Synthetic multi-view dataset generation (RGB + thermal + material)."""

    num_views: int = 19
    """
    Number of training camera views (Fibonacci sphere). 19 matches the view count
    that built the supplied pre-reconstructed spline; these are frames 0..18.
    """
    num_eval_views: int = 4
    """
    Number of additional held-out eval views, appended after the training views
    (frames num_views..num_views+num_eval_views-1).
    """
    image_resolution: int = 512
    """Square render resolution along height and width."""
    camera_distance: float = 5.0
    """Radius of the Fibonacci camera sphere, in the source mesh's own units."""
    camera_fov_degrees: float = 45.0
    """Field of view of every dataset camera; also fixes the reported focal length."""
    camera_sample_count: int = 64
    """Sampler count carried by the cameras; each render overrides it with its spp."""
    reference_samples: int = 4096
    """Samples per pixel for the reference RGB renders."""
    field_samples: int = 256
    """Samples per pixel for the (emissive, low-variance) thermal/material renders."""
    solid_albedo: float | None = 0.5
    """
    If set, RGB renders use a uniform grey diffuse albedo of this value instead of
    the checkerboard texture. The reconstruction is supplied pre-built, so the RGB
    dataset is only illustrative and a solid albedo reads cleaner in the paper. Set
    to None to texture the surface with the checkerboard.
    """
    checkerboard_texture: Path = Path("data/textures/checkerboard_texture.jpg")
    """
    RGB texture (used only if solid_albedo is None); also the source of the
    thermal field (sampled at UVs to set per-vertex temperature).
    """
    envmap: Path = Path("data/textures/checkerboard.exr")
    """
    Environment map (background + illumination) for the RGB renders, shared by dataset
    generation and reconstruction so the two stay consistent. Its ~uniform mean
    irradiance keeps the recovered albedo faithful while giving a patterned background.
    """

    def __post_init__(self) -> None:
        if self.num_views < 1:
            raise ValueError(
                f"`num_views` must be a positive integer, got {self.num_views}."
            )
        if self.num_eval_views < 0:
            raise ValueError(
                f"`num_eval_views` must be non-negative, got {self.num_eval_views}."
            )
        if self.image_resolution < 1:
            raise ValueError(
                f"`image_resolution` must be a positive integer, got "
                f"{self.image_resolution}."
            )
        if self.reference_samples < 1:
            raise ValueError(
                f"`reference_samples` must be a positive integer, got "
                f"{self.reference_samples}."
            )
        if self.field_samples < 1:
            raise ValueError(
                f"`field_samples` must be a positive integer, got {self.field_samples}."
            )
        if self.solid_albedo is not None and not 0.0 <= self.solid_albedo <= 1.0:
            raise ValueError(
                f"`solid_albedo` must be a reflectance in [0, 1], got "
                f"{self.solid_albedo}."
            )
