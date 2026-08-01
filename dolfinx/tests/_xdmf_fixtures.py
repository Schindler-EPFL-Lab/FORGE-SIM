"""
Shared fixtures for the XDMF writer / reader tests.

Everything here is pure numpy: no DOLFINx, no Mitsuba, no gmsh. It builds
synthetic triangulated grids and writes IGA/FEM XDMF pairs through the real
writer (`xdmf_time_series.write_time_series_xdmf`), which is what the comparison
tests need in order to exercise `compare_iga_fem` on meshes of *independent*
resolutions -- the asymmetry under which the difference-output geometry bug hid.

Importing this module makes the `dolfinx/` scripts importable as top-level
modules; see `_fem_fixtures` for why that is safe.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

# Imported for its module-level side effect: `_fem_fixtures` puts `dolfinx/` and
# the repository root on `sys.path`, which is what makes the modules under test
# importable as top-level modules.
import _fem_fixtures  # noqa: F401

from xdmf_time_series import XdmfLayout, write_time_series_xdmf

# A field of a vertex position and a time. Used to give the two sides of a
# comparison pair fields that agree pointwise in space, so the difference between
# them is a known quantity independent of the two resolutions.
FieldFunction = Callable[[np.ndarray, float], np.ndarray]


def grid_mesh(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds an `n x n` planar triangulated grid over the unit square in `z = 0`.

    A flat grid is deliberate: the comparison code only ever treats vertices as
    points in R^3 (nearest-neighbour matching) and faces as indices, so curvature
    would add nothing while making the expected values harder to state.

    :param n: Number of vertices per side (`n >= 2`).
    :returns: `(vertices, faces)` with shapes `(n^2, 3)` and `(2 (n-1)^2, 3)`.
    """
    if n < 2:
        raise ValueError("`n` must be at least 2.")

    axis = np.linspace(0.0, 1.0, n)
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="ij")
    vertices = np.column_stack(
        [x_grid.ravel(), y_grid.ravel(), np.zeros(n * n, dtype=np.float64)]
    )

    corner = np.arange(n * n).reshape(n, n)[:-1, :-1].ravel()
    faces = np.concatenate(
        [
            np.column_stack([corner, corner + 1, corner + n + 1]),
            np.column_stack([corner, corner + n + 1, corner + n]),
        ]
    ).astype(np.int64)

    return vertices, faces


def write_time_series(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    times: Sequence[float],
    field_fn: FieldFunction,
    *,
    layout: XdmfLayout,
    attribute_name: str = "temperature",
) -> None:
    """
    Writes one XDMF/HDF5 time series whose field is `field_fn` sampled at the
    mesh's own vertices, one snapshot per entry of `times`.

    `declare_float_precision` follows `layout` so that each dialect comes out
    exactly as its real producer writes it (the FEM runner passes COMPACT with
    the precision declared, the difference writer INDENTED without).
    """
    write_time_series_xdmf(
        path,
        vertices,
        faces,
        times,
        [field_fn(vertices, t) for t in times],
        attribute_name=attribute_name,
        layout=layout,
        declare_float_precision=layout is XdmfLayout.COMPACT,
        create_parent_dirs=True,
    )


def write_pair(
    directory: Path,
    n_iga: int,
    n_fem: int,
    times: Sequence[float],
    field_fn: FieldFunction,
) -> tuple[Path, Path]:
    """
    Writes an IGA/FEM XDMF pair at two *independent* grid resolutions, each in
    the dialect its real producer uses: the IGA archive files are indented, the
    FEM runner's output is compact.

    Both sides sample the same `field_fn` at their own vertices, so the pair is a
    faithful miniature of a real comparison: same physical field, different
    discretisations, different vertex counts and orderings.

    :param n_iga: Vertices per side of the IGA grid (`n_iga^2` vertices).
    :param n_fem: Vertices per side of the FEM grid (`n_fem^2` vertices).
    :returns: `(iga_xdmf_path, fem_xdmf_path)`.
    """
    iga_path = directory / "iga.xdmf"
    fem_path = directory / "fem.xdmf"

    iga_vertices, iga_faces = grid_mesh(n_iga)
    write_time_series(
        iga_path, iga_vertices, iga_faces, times, field_fn, layout=XdmfLayout.INDENTED
    )

    fem_vertices, fem_faces = grid_mesh(n_fem)
    write_time_series(
        fem_path, fem_vertices, fem_faces, times, field_fn, layout=XdmfLayout.COMPACT
    )

    return iga_path, fem_path
