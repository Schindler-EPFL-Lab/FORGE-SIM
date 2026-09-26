"""
The ground-truth physical fields the end-to-end experiment defines on a mesh.

Two fields drive the experiment: an initial temperature (a checkerboard, taken from
the mesh's UV texture where it has one and from world position where it does not)
and a binary material label (a geometric half-split). They are the experiment's
controlled inputs, so both sides of the comparison must agree on them exactly:
the dataset stage renders them into the images the reconstruction sees, and the
ground-truth baking stage evaluates the same rule on the FEM analysis mesh. That is
why they are defined here once rather than per stage.

For targets with no known ground-truth texture (real facades) the same two slots are
filled by a measured field instead, inpainted onto the surface from the thermal and
mask images; :meth:`mesh_arrays.MeshArrays.from_inpainted_ply` reads those back.

Everything operates on plain NumPy arrays, so the module is usable from the FEM side
of the comparison as well as this one. The meshes these fields are evaluated on are
loaded by :class:`mesh_arrays.MeshArrays`, whose Mitsuba-backed PLY reader recovers
the per-vertex UVs that Suzanne carries; a caller must set a Mitsuba variant first.

Mesh repair, remeshing and the reference-surface queries that serve the FEM analysis
mesh live in `fem/meshprep/` instead, not this library's business.
"""

from pathlib import Path

import mitsuba as mi
import numpy as np
import numpy.typing as npt

from rebel_shape import logger
from rebel_shape.scripts.end_to_end.heat.thermal_sim_config import ThermalSimConfig


def sample_texture_intensity(
    uvs: npt.NDArray[np.float64], texture_path: Path
) -> npt.NDArray[np.float64]:
    """
    Sample the pattern intensity of an image at UV coordinates (nearest, v flipped to
    image-row convention). Returns values in [0, 1], one per UV.

    Intensity is the largest of the three channels, not a perceptual luminance. The
    checkerboards this reads are warm-tinted rather than neutral: the brightest
    pixel of the tracked one is `(1.000, 0.947, 0.815)`, and luminance weights
    green at 0.715 and blue at 0.072, so it scores a saturated hot cell at 0.949
    instead of 1.0 and biases the whole hot half of the field down by ~5 %. The cells
    are the signal here, not how bright they look, so the reduction that preserves a
    saturated cell in any hue is the right one.
    """
    bitmap = np.array(
        mi.Bitmap(str(texture_path)).convert(
            mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=False
        )
    )
    bitmap = np.squeeze(bitmap)
    if bitmap.ndim == 3:
        bitmap = bitmap.max(axis=2)  # (H, W)
    height, width = bitmap.shape[:2]

    u = np.clip(uvs[:, 0], 0.0, 1.0)
    v = np.clip(uvs[:, 1], 0.0, 1.0)
    col = np.clip((u * (width - 1)).round().astype(int), 0, width - 1)
    row = np.clip(((1.0 - v) * (height - 1)).round().astype(int), 0, height - 1)
    return bitmap[row, col].astype(np.float64)


def vertex_temperature(
    uvs: npt.NDArray[np.float64],
    texture_path: Path,
    temp_cold: float,
    temp_hot: float,
    intensity_threshold: float = 0.5,
    binary: bool = True,
) -> npt.NDArray[np.float64]:
    """
    Temperature field from a checkerboard texture, sampled at the mesh UVs.

    `binary` (default): threshold the intensity at `intensity_threshold` -> a hard
    `temp_cold`/`temp_hot` step, which Gibbs-rings on a smooth spline.
    `binary=False`: map the intensity linearly onto `[temp_cold, temp_hot]` using
    the smoothed (band-limited) texture, giving a smooth field with the same area mean
    that the spline can represent without ringing.
    """
    intensity = sample_texture_intensity(uvs, texture_path)
    if binary:
        return np.where(intensity >= intensity_threshold, temp_hot, temp_cold).astype(
            np.float64
        )
    intensity = np.clip(intensity, 0.0, 1.0)
    return (temp_cold + (temp_hot - temp_cold) * intensity).astype(np.float64)


def world_checker_cell_size(
    source_vertices: npt.NDArray[np.float64], num_cells: int
) -> float:
    """
    Edge length of one 3D checker cell so the longest model axis spans `num_cells`.
    Deriving it from the source mesh extent makes the dataset render and the GT field
    agree on one cell size regardless of which (remeshed) vertices each evaluates.
    """
    longest = float((source_vertices.max(axis=0) - source_vertices.min(axis=0)).max())
    return longest / max(int(num_cells), 1)


def world_checker_temperature(
    vertices: npt.NDArray[np.float64],
    cell_size: float,
    temp_cold: float,
    temp_hot: float,
    binary: bool = True,
    smoothness: float = 0.06,
) -> npt.NDArray[np.float64]:
    """
    Checkerboard temperature from world position (UV-free), for meshes without UVs
    (Stanford bunny/armadillo): the analogue of the UV checkerboard for raw assets, so
    the GT field and the rendered thermal images share one position-defined pattern.

    `binary` (default): parity of `floor(x/cell)+floor(y/cell)+floor(z/cell)` -> a
    hard `temp_cold`/`temp_hot` 3D checker (Gibbs-rings on a smooth spline).
    `binary=False`: a smooth checker, `prod_i tanh(sin(pi*x_i/cell)/smoothness)`
    mapped to `[temp_cold, temp_hot]`: band-limited (edge ramp width ~`smoothness`
    of a cell, matching the blurred UV texture) with the same 0.5 area mean, recoverable
    without ringing.
    """
    v = np.asarray(vertices, dtype=np.float64) / float(cell_size)
    if binary:
        idx = np.floor(v)
        hot = (idx.sum(axis=1).astype(np.int64) % 2) == 0
        return np.where(hot, temp_hot, temp_cold).astype(np.float64)
    s = max(float(smoothness), 1e-6)
    checker = np.prod(np.tanh(np.sin(np.pi * v) / s), axis=1)  # in [-1, 1], smooth
    return (temp_cold + (temp_hot - temp_cold) * (0.5 + 0.5 * checker)).astype(
        np.float64
    )


def synthetic_temperature(
    vertices: npt.NDArray[np.float64],
    uvs: npt.NDArray[np.float64] | None,
    thermal: ThermalSimConfig,
) -> npt.NDArray[np.float64]:
    """
    The experiment's ground-truth temperature field, evaluated at `vertices`.

    A UV checkerboard or a world-space checkerboard of the same character, chosen by
    `thermal.temperature_field_source` rather than by what the asset happens to
    carry. The world cell size comes from `vertices`' own extent, so an asset
    without UVs still gets one pattern rather than a per-caller one.

    Single-sourced deliberately, unlike the two XDMF writers, which this repository
    duplicates just as deliberately. Those sit downstream of the IGA/FEM fork, where two
    independent implementations agreeing is exactly the evidence the cross-validation
    exists to produce. This sits upstream, on the trunk both branches share: it is the
    experiment's controlled input, not a measurement of anything, so a second copy
    would be an uncontrolled variable that no comparison downstream can catch, and a
    duplicate that has drifted out of step looks exactly like a real result.
    """
    source = thermal.temperature_field_source
    logger.info(f"Ground-truth temperature field: {source.value}.")

    if source is ThermalSimConfig.TemperatureFieldSource.UV_TEXTURE:
        if uvs is None:
            raise ValueError(
                "`temperature_field_source` is UV_TEXTURE but the mesh carries no "
                "UVs, so the texture cannot be sampled. Supply a UV-mapped mesh, or "
                "set the source to WORLD_CHECKER."
            )
        return vertex_temperature(
            uvs,
            thermal.temperature_texture,
            thermal.temp_cold,
            thermal.temp_hot,
            binary=thermal.binary_temperature,
        )
    return world_checker_temperature(
        vertices,
        world_checker_cell_size(vertices, thermal.world_checker_cells),
        thermal.temp_cold,
        thermal.temp_hot,
        binary=thermal.binary_temperature,
    )


def material_labels(
    vertices: npt.NDArray[np.float64], axis: int, threshold: float = 0.0
) -> npt.NDArray[np.float64]:
    """
    Binary material field splitting the mesh into two halves along `axis`.

    Steel (1.0) is the half with `vertices[:, axis] < centre + threshold`, wood is
    the other half; `centre` is the bounding-box midpoint along the axis, so the
    default `threshold=0` yields two symmetric halves regardless of where the mesh
    sits. With `axis=0` (x) this is the left/right split through the sagittal plane.
    Steel is 1.0 so it is the `> 0.5` ("active") region the two-material heat
    consumers select for the source.
    """
    centre = 0.5 * (vertices[:, axis].min() + vertices[:, axis].max())
    return (vertices[:, axis] < centre + threshold).astype(np.float64)
