from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mitsuba as mi
import numpy as np
import numpy.typing as npt

INPAINTED_FIELD_ATTRIBUTE: str = "vertex_inpainting_data"
"""
Name under which `inpaint_data_mesh` stores its recovered per-vertex scalar.

The PLY writer serialises the Mitsuba mesh attribute as the vertex property
`inpainting_data_0`; Mitsuba's loader regroups it back under this name on read.
Both halves of that round trip are Mitsuba's, so the name is fixed by the writer
rather than chosen here.
"""


@dataclass(frozen=True, kw_only=True)
class MeshArrays:
    """A triangle mesh as NumPy arrays (geometry frame, unscaled)."""

    vertices: npt.NDArray[np.float64]
    """Vertex positions, shape `(n_v, 3)`."""
    faces: npt.NDArray[np.int64]
    """Triangle vertex indices, shape `(n_f, 3)`."""
    uvs: npt.NDArray[np.float64] | None
    """Texture coordinates, shape `(n_v, 2)`, or None if the mesh carries none."""
    normals: npt.NDArray[np.float64] | None
    """Vertex normals, shape `(n_v, 3)`, or None if the mesh carries none."""

    @classmethod
    def from_ply(cls, path: Path) -> "MeshArrays":
        """
        Load a PLY into NumPy arrays (positions, faces, UVs, normals).

        Uses Mitsuba's loader so per-vertex texture coordinates are recovered. A
        Mitsuba variant must already be set by the caller.

        Deliberately not :meth:`forge_sim.meshing.mesh.Mesh.from_ply_file`, which
        performs the same load: that constructor enforces the invariants a
        renderable mesh needs, and one of the experiment's four source meshes
        (`spot.ply`) has four UV components just outside `[0, 1]` and is rejected
        by it. The fields here only sample a texture, which clamps anyway, so the
        looser contract is the right one for this path.
        """
        mesh = mi.load_dict({"type": "ply", "filename": str(path)})
        return cls._from_params(mi.traverse(mesh))

    @classmethod
    def from_inpainted_ply(
        cls, path: Path
    ) -> tuple["MeshArrays", npt.NDArray[np.float64]]:
        """
        Load a mesh written by `inpaint_data_mesh` with its per-vertex scalar.

        This is the real-image counterpart to the synthetic field generators: for
        targets with no known ground-truth texture (e.g. BuildNet3D buildings) the GT
        temperature and material fields are measured (inpainted onto the
        ground-truth surface from the thermal / windows-mask images) rather than
        synthesised from a checkerboard. The scalar is read from
        `INPAINTED_FIELD_ATTRIBUTE`. A Mitsuba variant must already be set by the
        caller.

        :returns: The mesh arrays and the per-vertex field, in vertex order.
        :raises KeyError: If `path` carries no such attribute (not an inpainted
            mesh).
        :raises ValueError: If the field length does not match the vertex count.
        """
        mesh = mi.load_dict({"type": "ply", "filename": str(path)})
        params = mi.traverse(mesh)
        if INPAINTED_FIELD_ATTRIBUTE not in params:
            raise KeyError(
                f"{path} has no '{INPAINTED_FIELD_ATTRIBUTE}' attribute; it was not "
                f"written by inpaint_data_mesh. Available: "
                f"{sorted(str(k) for k in params.keys())}"
            )

        arrays = cls._from_params(params)
        field = (
            np.array(params[INPAINTED_FIELD_ATTRIBUTE]).reshape(-1).astype(np.float64)
        )
        if field.shape[0] != arrays.vertices.shape[0]:
            raise ValueError(
                f"{path}: '{INPAINTED_FIELD_ATTRIBUTE}' has {field.shape[0]} values "
                f"but the mesh has {arrays.vertices.shape[0]} vertices"
            )
        return arrays, field

    @classmethod
    def _from_params(cls, params: Mapping[str, Any]) -> "MeshArrays":
        """Build a `MeshArrays` from an already-traversed Mitsuba mesh."""
        vertices = (
            np.array(params["vertex_positions"]).reshape(-1, 3).astype(np.float64)
        )
        faces = np.array(params["faces"]).reshape(-1, 3).astype(np.int64)

        uvs = None
        if "vertex_texcoords" in params:
            raw = np.array(params["vertex_texcoords"]).reshape(-1, 2)
            if raw.size > 0:
                uvs = raw.astype(np.float64)

        normals = None
        if "vertex_normals" in params:
            raw = np.array(params["vertex_normals"]).reshape(-1, 3)
            if raw.size > 0:
                normals = raw.astype(np.float64)

        return cls(vertices=vertices, faces=faces, uvs=uvs, normals=normals)
