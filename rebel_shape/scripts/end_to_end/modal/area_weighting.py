import numpy as np
import numpy.typing as npt

_NORM_FLOOR: float = 1e-30
"""
Added to a column's norm before dividing, so an all-zero mode normalises to zero
instead of raising. Far below any physical mode's norm, so it never perturbs one.
"""


def vertex_areas(
    vertices: npt.NDArray[np.float64], faces: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
    """
    Lumped (barycentric) vertex areas of a triangle mesh, shape `(n_v,)`.

    Every triangle hands one third of its area to each of its three vertices, so the
    result sums to the total surface area. That is the diagonal of the lumped mass
    matrix, which is what makes it the weight for the area-weighted L2 norms and MACs
    the comparison metrics take over a surface field: without it a densely-tessellated
    region would count for more than the surface it represents.
    """
    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    out = np.zeros(vertices.shape[0])
    for corner in range(3):
        np.add.at(out, faces[:, corner], area / 3.0)
    return out


def mode_column(
    mode: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    sample: npt.NDArray[np.int64] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Flatten one vector mode shape into a unit column under the area-weighted inner
    product, shape `(3 * n_v,)`.

    `weights` are the per-vertex sqrt-areas, so a plain dot product of two columns
    is the mass-normalised modal correlation: the signed cosine the eigenmode
    alignment maximises, and the square root of the MAC the modal metrics report.
    Passing `sample` first gathers the mode at those vertex indices, which is how a
    mode living on one surface is compared against another's vertices.
    """
    v = ((mode if sample is None else mode[sample]) * weights[:, None]).reshape(-1)
    return v / (np.linalg.norm(v) + _NORM_FLOOR)
