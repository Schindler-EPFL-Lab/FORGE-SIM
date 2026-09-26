"""
Tiny IGA spline surfaces for the projector / runner tests.

Kept apart from `_fem_fixtures` on purpose. That module stays inside the DOLFINx
ecosystem (gmsh, DOLFINx, PETSc); this one pulls in Mitsuba, Dr.Jit and
jsonpickle. It does not itself select the Mitsuba variant: a consuming test
class calls `mi.set_variant("llvm_ad_mono")` as the first statement of its own
`setUpClass`, before anything here runs, per this repo's fixture convention:
importing this module must not have that process-global side effect as a
surprise. The seventeen solver tests use none of that.

The surface is a six-patch cubed sphere, the same construction the production
splines use, at the coarsest resolution that still exercises the machinery:
`spline_subdivisions=4`, tessellated at 8 subdivisions per patch, giving 218
global control points, 386 vertices and 768 triangles in about a second. Six
patches means patch seams, the local->global scatter and the shared-vertex
averaging are all genuinely covered.

The field is supplied by the caller as a function of the control points' own
positions. That seam is what makes the manufactured-solution oracle possible:
because the geometry and the field share the same basis, and the basis is a
partition of unity, an affine field of the control points evaluates at every mesh
vertex to the same affine function of that vertex's position: a closed form to
test the whole texcoord -> parametric -> evaluate -> scatter chain against.
"""

from collections.abc import Callable
from pathlib import Path

# Imported for its module-level side effect: `_fem_fixtures` puts the
# repository root on `sys.path`, so `fem` and `rebel_shape` resolve.
import _fem_fixtures  # noqa: F401
import jsonpickle
import mitsuba as mi
import numpy as np
import numpy.typing as npt

from rebel_shape.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)
from rebel_shape.splines.cube_initialization import cubed_sphere

# Maps a patch's control-point positions, shape (n_cp, 3), to that patch's field
# values, shape (n_cp, n_dofs).
FieldFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


def affine_field(
    a: float = 0.0, b: float = 0.0, c: float = 1.0, d: float = 0.0
) -> FieldFunction:
    """
    Field `a x + b y + c z + d` of the control-point positions.

    Interpolating it with the spline basis reproduces `a X + b Y + c Z + d` at
    every mesh vertex `(X, Y, Z)` exactly (up to float32 evaluation round-off):
    interpolation is linear in the control values and the basis sums to one, so
    it commutes with any affine map of the control points, including the one
    that produced the geometry itself.
    """

    def field(control_points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return (
            a * control_points[:, 0:1]
            + b * control_points[:, 1:2]
            + c * control_points[:, 2:3]
            + d
        )

    return field


def upper_hemisphere_mask(control_points: npt.NDArray[np.float64]):
    """
    A binary mask control-point field: 1 above the equator, 0 below.

    Thresholding this at 0.5 reproduces the same partition the facade-mask
    splines produce, with a transition zone of the basis's own width around
    `z = 0` and pure material in the two caps.
    """
    return (control_points[:, 2:3] > 0.0).astype(np.float64)


def shell_field(control_points: npt.NDArray[np.float64]):
    """
    A six-DOF (three displacements + three rotations) control-point field, i.e.
    what a Reissner-Mindlin shell spline carries. Non-zero everywhere, so a
    projector that silently read its first column would produce a plausible
    temperature rather than an obviously broken one.
    """
    return np.repeat(control_points[:, 2:3], 6, axis=1) * np.arange(1.0, 7.0)


def make_iga_spline_file(
    path: Path,
    field_fn: FieldFunction,
    spline_subdivisions: int = 4,
    tessellation_subdivisions: int = 8,
) -> Path:
    """
    Writes a jsonpickled `TessellatedIGASplineSurface` (a six-patch cubed
    sphere carrying `field_fn` of its control points) to `path`, which is
    also the destination JSON file. `field_fn` maps each patch's control-point
    positions to field values. `spline_subdivisions` is the knot spans per
    patch per direction, and `tessellation_subdivisions` is the tessellation
    subdivisions per patch.

    G1 optimisation is switched off: it is an iterative smoothing of the control
    points that costs seconds and changes nothing these tests measure (they
    assert on the surface as given, never on its normals).

    :returns: `path`, for convenience.
    """
    surface = cubed_sphere(
        name=path.stem,
        degree=3,
        spline_subdivisions=spline_subdivisions,
        tessellation_subdivisions=tessellation_subdivisions,
        optimize_g1=False,
    )

    per_patch_field_values = [
        # `mi.TensorXf` is float32 in every Mitsuba variant, so the cast is
        # explicit rather than left to an implicit conversion.
        mi.TensorXf(
            np.ascontiguousarray(
                field_fn(patch.control_points.numpy()), dtype=np.float32
            )
        )
        for patch in surface.composite_spline
    ]
    iga_surface = TessellatedIGASplineSurface.from_tessellated_spline_surface(
        tessellated_spline_surface=surface,
        per_patch_field_values=per_patch_field_values,
    )

    path.write_text(jsonpickle.encode(iga_surface))
    return path
