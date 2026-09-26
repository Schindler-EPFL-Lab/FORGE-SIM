"""
Writer for shell eigenmodes in the IGA runner's temporal XDMF/HDF5 format.

The layout is the one
`rebel_shape.scripts.reissner_mindlin_modal_analysis.reissner_mindlin_modal_runner`
writes, so that a FEM run and an IGA run of the same surface open identically in
ParaView and the post-hoc MAC alignment reads either::

    eigenmodes.h5    Mesh/topology            (n_faces, 3)     int32
                     Mesh/vertices            (n_vertices, 3)  float64
                     Modes/mode_<k>/{displacement,rotation}
                                              (n_vertices, 3)  float64
    eigenmodes.xdmf  one temporal grid collection, one Uniform grid per mode

`k` indexes the export position, not the mode: the original mode index appears
in the grid name and in a comment beside it. Every grid shares the same
undeformed `Mesh/vertices` geometry; a deformed view and a magnitude
colouring are both derivable from `displacement`/`rotation` in ParaView (Warp
By Vector, native magnitude colouring) rather than persisted.

The two shell drivers (`shell_modal_analysis_spline` and the ground-truth-mesh
`fem/end_to_end/modal_gt/fem_shell_modal_gt_mesh`) share this writer rather
than a copy each, so their outputs cannot drift apart.
"""

from pathlib import Path

import basix.ufl
import h5py
import numpy as np
import numpy.typing as npt
from dolfinx import fem
from scipy.spatial import cKDTree

from fem.shells.modal_result import ModalResult
from fem.shells.mode_payload import ModePayload
from rebel_shape import logger


def _write_mode_group(h5_file: h5py.File, position: int, mode: ModePayload) -> None:
    """
    Writes one mode's array datasets under
    `/Modes/mode_<position>`, into the open HDF5 file `h5_file`. `position`
    is the export position `k`, which names the group, not
    `mode.mode_index`, which a partial export need not agree with. `mode`
    carries the mode's arrays.
    """
    for name, values in (
        ("displacement", mode.displacement),
        ("rotation", mode.rotation),
    ):
        h5_file.create_dataset(
            f"/Modes/mode_{position}/{name}", data=np.asarray(values, dtype=np.float64)
        )


def export_mode_shapes_xdmf(
    space: fem.FunctionSpace,
    result: ModalResult,
    mode_indices: list[int],
    output_folder: Path,
) -> None:
    """
    Writes the requested eigenmodes as an `eigenmodes.xdmf`/`.h5` pair.

    The values are taken at the mesh geometry vertices: the displacement and
    rotation sub-fields are collapsed out of the mixed space, interpolated to a
    P1 vector space and gathered into geometry-vertex order. Every mode shares
    the same undeformed geometry; a deformed view is a ParaView Warp By Vector
    away rather than persisted.

    An eigenvector is defined only up to sign, so the whole mode (displacement
    and rotation together) is flipped to make its largest-magnitude
    displacement component positive. That is the rule the IGA exporter applies,
    so simple (non-degenerate) modes land on a matching sign for side-by-side
    visualization; degenerate clusters need subspace alignment rather than a
    sign, which is the comparison step's job.

    `space` is the mixed (displacement, rotation) space the eigenvectors are
    defined on; its mesh supplies the geometry that is written. `result` is
    the modal solve result, with eigenvalues/frequencies/eigenvectors.
    `mode_indices` names the mode indices to export; indices outside the
    computed modes are skipped with a warning, and nothing is written when
    none is valid. `output_folder` is the directory the pair is written to
    (created if missing).
    """
    valid = [i for i in mode_indices if 0 <= i < result.eigenvectors.shape[1]]
    if len(valid) != len(mode_indices):
        # Requesting more indices than modes is the normal case (the default list
        # is a fixed range), so this is not an error, but it must not be silent,
        # or a truncated eigensolve would quietly produce a short export.
        logger.warning(
            f"Skipping {len(mode_indices) - len(valid)} requested mode index/indices "
            f"outside the {result.eigenvectors.shape[1]} computed modes."
        )
    if not valid:
        logger.warning(
            f"None of the requested mode index/indices is within the "
            f"{result.eigenvectors.shape[1]} computed modes; writing nothing."
        )
        return

    msh = space.mesh
    vertices = np.ascontiguousarray(msh.geometry.x[:, :3], dtype=np.float64)
    geom_dofmap = np.asarray(msh.geometry.dofmap)
    faces = np.ascontiguousarray(
        geom_dofmap.reshape(geom_dofmap.shape[0], -1)[:, :3], dtype=np.int32
    )
    n_vertices = vertices.shape[0]

    # Map each geometry vertex to its P1 dof index (so per-vertex fields land in the
    # same order as `vertices`/`faces`); a vector P1 dof array reshaped (n, 3) shares
    # this vertex blocking.
    vp1_scalar = fem.functionspace(msh, ("Lagrange", 1))
    p1_coords = np.asarray(vp1_scalar.tabulate_dof_coordinates()[:, :3])
    vtx_to_dof = cKDTree(p1_coords).query(vertices, k=1)[1]
    vp1_vec = fem.functionspace(
        msh, basix.ufl.element("Lagrange", msh.basix_cell(), 1, shape=(3,))
    )

    def _at_vertices(
        sub_index: int, eigvec: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Collapses sub-field `sub_index` of an eigenvector, interpolates it to the
        P1 vector space, and reorders it into geometry-vertex order.

        `sub_index` is the sub-space index into the mixed function space (0
        for displacement, 1 for rotation). `eigvec` is the full eigenvector
        array, indexed by the mixed space's dofs.

        :returns: Per-vertex vector field, shape `(n_vertices, 3)`.
        """
        sub_space, sub_map = space.sub(sub_index).collapse()
        f_sub = fem.Function(sub_space)
        f_sub.x.array[:] = eigvec[sub_map]
        f_p1 = fem.Function(vp1_vec)
        f_p1.interpolate(f_sub)
        return f_p1.x.array.reshape(-1, 3)[vtx_to_dof].astype(np.float64)

    payloads: list[ModePayload] = []
    for i in valid:
        eigvec = result.eigenvectors[:, i]
        disp = _at_vertices(0, eigvec)
        rot = _at_vertices(1, eigvec)
        flat = disp.reshape(-1)
        if flat.size and flat[int(np.argmax(np.abs(flat)))] < 0.0:
            disp = -disp
            rot = -rot
        payloads.append(
            ModePayload(
                mode_index=int(i),
                eigenvalue=float(result.eigenvalues[i]),
                frequency=float(result.frequencies[i]),
                displacement=disp,
                rotation=rot,
            )
        )

    output_folder.mkdir(parents=True, exist_ok=True)
    stem = output_folder / "eigenmodes"
    h5_path = stem.with_suffix(".h5")
    xdmf_path = stem.with_suffix(".xdmf")
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset("/Mesh/topology", data=faces.astype(np.int32))
        # Undeformed vertices, for the post-hoc MAC alignment step (align_modal_xdmf).
        h5_file.create_dataset("/Mesh/vertices", data=vertices.astype(np.float64))
        for k, mode in enumerate(payloads):
            _write_mode_group(h5_file, k, mode)
    write_eigenmodes_xdmf(
        xdmf_path, h5_path.name, n_vertices, faces.shape[0], payloads
    )
    logger.info(
        f"Exported {len(payloads)} mode shapes to {xdmf_path} "
        f"(IGA-compatible eigenmodes format)."
    )


def write_eigenmodes_xdmf(
    xdmf_path: Path,
    h5_name: str,
    num_vertices: int,
    num_triangles: int,
    mode_payloads: list[ModePayload],
) -> None:
    """
    Writes the temporal-collection XDMF descriptor beside an already-written
    HDF5 file.

    `xdmf_path` is the output path for the `.xdmf` descriptor. `h5_name` is
    the name of the companion `.h5` file, referenced by relative path inside
    the XDMF `DataItem` entries. `num_vertices` and `num_triangles` are the
    vertex and triangle-cell counts, shared by every mode grid.
    `mode_payloads` is the per-mode data as built in
    :func:`export_mode_shapes_xdmf`, one entry per exported mode, in export
    order; only the scalars are read here, since the arrays were already
    written to the HDF5 file by :func:`_write_mode_group` and are referenced
    by path.
    """

    def _attr(name: str, attr_type: str, dataset: str, dims: str) -> list[str]:
        """Emits the XDMF markup for one node-centred attribute."""
        return [
            f'        <Attribute Name="{name}" '
            f'AttributeType="{attr_type}" Center="Node">',
            f'          <DataItem Dimensions="{dims}" '
            f'Format="HDF">{h5_name}:{dataset}</DataItem>',
            "        </Attribute>",
        ]

    lines = [
        '<?xml version="1.0"?>',
        '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>',
        '<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">',
        "  <Domain>",
        '    <Grid Name="eigenmodes" GridType="Collection" '
        'CollectionType="Temporal">',
    ]
    for k, mode in enumerate(mode_payloads):
        original = mode.mode_index
        lines.extend(
            [
                f'      <Grid Name="mode_{original}" GridType="Uniform">',
                '        <Topology TopologyType="Triangle" '
                f'NumberOfElements="{num_triangles}" NodesPerElement="3">',
                f'          <DataItem Dimensions="{num_triangles} 3" '
                f'NumberType="Int" Format="HDF">{h5_name}:/Mesh/topology</DataItem>',
                "        </Topology>",
                '        <Geometry GeometryType="XYZ">',
                f'          <DataItem Dimensions="{num_vertices} 3" '
                f'Format="HDF">{h5_name}:/Mesh/vertices</DataItem>',
                "        </Geometry>",
                f'        <Time Value="{k}" />',
                f"        <!-- mode_index={original} "
                f"lambda={mode.eigenvalue:.6e} "
                f"f={mode.frequency:.6f} Hz -->",
            ]
        )
        lines.extend(_attr("displacement", "Vector",
                           f"/Modes/mode_{k}/displacement", f"{num_vertices} 3"))
        lines.extend(_attr("rotation", "Vector",
                           f"/Modes/mode_{k}/rotation", f"{num_vertices} 3"))
        lines.append("      </Grid>")
    lines.extend(["    </Grid>", "  </Domain>", "</Xdmf>"])
    with open(xdmf_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
