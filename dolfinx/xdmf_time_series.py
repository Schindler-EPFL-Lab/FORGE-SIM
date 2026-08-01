"""
Writer for node-based scalar time series in the custom IGA-compatible XDMF format.

The layout is deliberately minimal and shared with the IGA exporter
(`rebel_shape.isogeometric_analysis.iga_xdmf_exporter`), so ParaView and
`compare_iga_fem` can read IGA and FEM output through the same code path::

    <name>.h5    Mesh/Vertices     (n_vertices, 3) float64
                 Mesh/Topology     (n_faces, 3)    int64
                 Function/Step_<i> (n_vertices,)   float64
    <name>.xdmf  temporal grid collection referencing those datasets

It also deliberately replaces DOLFINx's `XDMFFile.write_function`: that routine
silently corrupts the written data when repeatedly writing snapshots of a reused
Function whose vector is driven through the cached PETSc-vector wrapper (the
in-memory solution is correct; only the file is wrong). Writing the raw arrays
ourselves sidesteps it entirely.

Two callers share this module -- the FEM heat runner and the IGA/FEM difference
writer -- and their `.xdmf` sidecars were historically formatted differently.
Those cosmetic differences (`XdmfLayout`, `declare_float_precision`) are
parameters rather than a resolved choice, so that unifying the writers left every
consumer's on-disk output byte-for-byte unchanged. They can be collapsed later,
deliberately, once someone rules on which form is intended.
"""

from collections.abc import Iterable, Sequence
from enum import Enum
from pathlib import Path

import h5py
import numpy as np


class XdmfLayout(Enum):
    """
    Pretty-printing style of the emitted `.xdmf` XML.

    Purely cosmetic: both layouts describe the same grid collection over the same
    HDF5 datasets, and differ only in indentation, in whether a `DataItem` body
    is inline, and in the order of the `DataItem` attributes.
    """

    COMPACT = "compact"
    """One line per element, `DataItem` bodies inline. Used by the FEM heat
    runner's solution output."""

    INDENTED = "indented"
    """Two-space nesting with each `DataItem` body on its own line. Used by the
    IGA/FEM difference output."""


# Indentation of, respectively, the per-snapshot `<Grid>` element and the
# `<Topology>` / `<Geometry>` / `<Attribute>` elements inside it.
_GRID_INDENT = {XdmfLayout.COMPACT: "", XdmfLayout.INDENTED: " " * 6}
_ELEMENT_INDENT = {XdmfLayout.COMPACT: "", XdmfLayout.INDENTED: " " * 8}

_HEADER = {
    XdmfLayout.COMPACT: (
        '<?xml version="1.0"?>',
        '<Xdmf Version="3.0"><Domain>',
        '<Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">',
    ),
    XdmfLayout.INDENTED: (
        '<?xml version="1.0"?>',
        '<Xdmf Version="3.0">',
        "  <Domain>",
        '    <Grid Name="TimeSeries" GridType="Collection" '
        'CollectionType="Temporal">',
    ),
}

_FOOTER = {
    XdmfLayout.COMPACT: ("</Grid></Domain></Xdmf>",),
    XdmfLayout.INDENTED: ("    </Grid>", "  </Domain>", "</Xdmf>"),
}

# XDMF `Precision` is a byte count; the HDF5 datasets are always float64.
_FLOAT_PRECISION = ' Precision="8"'


def _data_element(
    layout: XdmfLayout,
    open_tag: str,
    close_tag: str,
    dimensions: str,
    number_type: str,
    precision: str,
    dataset: str,
) -> list[str]:
    """
    Emits an XDMF element wrapping a single HDF5-backed `DataItem`.

    :param layout: Formatting style; also fixes the `DataItem` attribute order.
    :param open_tag: Opening tag of the wrapping element, e.g. `<Geometry ...>`.
    :param close_tag: Its closing tag.
    :param dimensions: Value of the `Dimensions` attribute, e.g. `"1024 3"`.
    :param number_type: `"Int"` or `"Float"`.
    :param precision: `Precision` attribute markup, or the empty string.
    :param dataset: HDF5 reference of the form `file.h5:/Group/Dataset`.
    :returns: The element's lines, without a trailing newline.
    """
    indent = _ELEMENT_INDENT[layout]

    if layout is XdmfLayout.COMPACT:
        data_item = (
            f'<DataItem Dimensions="{dimensions}" NumberType="{number_type}"'
            f'{precision} Format="HDF">{dataset}</DataItem>'
        )
        return [f"{open_tag}{data_item}{close_tag}"]

    return [
        f"{indent}{open_tag}",
        f'{indent}  <DataItem Format="HDF" Dimensions="{dimensions}" '
        f'NumberType="{number_type}"{precision}>',
        f"{indent}    {dataset}",
        f"{indent}  </DataItem>",
        f"{indent}{close_tag}",
    ]


def write_time_series_xdmf(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    times: Sequence[float],
    fields: Iterable[np.ndarray],
    *,
    attribute_name: str,
    layout: XdmfLayout,
    declare_float_precision: bool,
    create_parent_dirs: bool = False,
) -> None:
    """
    Writes a node-based scalar time series to an HDF5/XDMF pair.

    The mesh is written once and referenced by every snapshot; one
    `Function/Step_<i>` dataset is written per entry of `fields`, in the order
    given, and paired with the matching entry of `times`. Vertices and fields are
    stored as float64 and the connectivity as int64, whatever the input dtypes.

    :param path: Output ``.xdmf`` path (the ``.h5`` is written alongside it,
        under the same stem).
    :param vertices: ``(n_vertices, 3)`` node coordinates.
    :param faces: ``(n_faces, 3)`` triangle connectivity.
    :param times: Snapshot times, one per field.
    :param fields: One ``(n_vertices,)`` array per time, in vertex order.
    :param attribute_name: Name of the XDMF attribute holding the field, as it
        appears in ParaView (e.g. ``"temperature"``).
    :param layout: XML formatting style (cosmetic; see `XdmfLayout`).
    :param declare_float_precision: Whether floating-point `DataItem`s carry
        ``Precision="8"``. The datasets are float64 regardless, so this only
        declares what readers should already infer.
    :param create_parent_dirs: Whether to create `path`'s parent directory;
        when False the caller is responsible for it.
    """
    path = Path(path)
    if create_parent_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)

    h5_path = path.with_suffix(".h5")
    with h5py.File(h5_path, "w") as h5_file:
        mesh_group = h5_file.create_group("Mesh")
        mesh_group.create_dataset("Vertices", data=np.asarray(vertices, np.float64))
        mesh_group.create_dataset("Topology", data=np.asarray(faces, np.int64))
        function_group = h5_file.create_group("Function")
        for index, field in enumerate(fields):
            function_group.create_dataset(
                f"Step_{index}", data=np.asarray(field, np.float64)
            )

    n_vertices, n_faces, h5_name = len(vertices), len(faces), h5_path.name
    grid_indent = _GRID_INDENT[layout]
    precision = _FLOAT_PRECISION if declare_float_precision else ""

    lines = list(_HEADER[layout])
    for index, snapshot_time in enumerate(times):
        lines += [
            f'{grid_indent}<Grid Name="mesh" GridType="Uniform">',
            f'{_ELEMENT_INDENT[layout]}<Time Value="{snapshot_time}"/>',
            *_data_element(
                layout,
                open_tag=f'<Topology TopologyType="Triangle" '
                f'NumberOfElements="{n_faces}">',
                close_tag="</Topology>",
                dimensions=f"{n_faces} 3",
                number_type="Int",
                precision="",
                dataset=f"{h5_name}:/Mesh/Topology",
            ),
            *_data_element(
                layout,
                open_tag='<Geometry GeometryType="XYZ">',
                close_tag="</Geometry>",
                dimensions=f"{n_vertices} 3",
                number_type="Float",
                precision=precision,
                dataset=f"{h5_name}:/Mesh/Vertices",
            ),
            *_data_element(
                layout,
                open_tag=f'<Attribute Name="{attribute_name}" '
                f'AttributeType="Scalar" Center="Node">',
                close_tag="</Attribute>",
                dimensions=f"{n_vertices}",
                number_type="Float",
                precision=precision,
                dataset=f"{h5_name}:/Function/Step_{index}",
            ),
            f"{grid_indent}</Grid>",
        ]
    lines += list(_FOOTER[layout])

    path.write_text("\n".join(lines) + "\n")
