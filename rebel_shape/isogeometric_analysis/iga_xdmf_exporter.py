"""
XDMF export utilities for tessellated IGA spline surfaces.

This module provides functionality to export temperature fields from IGA
simulations to XDMF format, enabling comparison with FEM solutions in DOLFINx.
The exporter creates a tessellated mesh and projects the spline field values
onto mesh vertices for visualization in ParaView or comparison scripts.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from rebel_shape import logger
from rebel_shape.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)


class IGAXDMFExporter:
    """
    Exports IGA spline temperature fields to XDMF format.

    This class creates a tessellated representation of an IGA spline surface
    and exports temperature fields evaluated at mesh vertices. The XDMF format
    is compatible with DOLFINx output, enabling direct comparison between IGA
    and FEM simulations.

    The tessellation is created once during initialization, and temperature
    fields can be written at multiple time steps. The mesh geometry is stored
    in an HDF5 file alongside the XDMF metadata.

    Example usage::

        exporter = IGAXDMFExporter(
            tessellated_iga_spline=spline,
            output_path=Path("./outputs/iga_solution.xdmf"),
            tessellation_subdivisions=512,
        )

        for t, temperature in simulation_results:
            exporter.write_temperature(temperature, t)

        exporter.close()
    """

    def __init__(
        self,
        tessellated_iga_spline: TessellatedIGASplineSurface,
        output_path: Path,
        tessellation_subdivisions: int = 512,
    ) -> None:
        """
        Initializes the XDMF exporter.

        :param tessellated_iga_spline: The IGA spline surface with field values
        :param output_path: Path for the output XDMF file (HDF5 file will have
            same name with .h5 extension)
        :param tessellation_subdivisions: Number of subdivisions per patch for
            the export mesh. Should match FEM tessellation for fair comparison.
        """
        self._spline = tessellated_iga_spline
        self._output_path = output_path
        self._subdivisions = tessellation_subdivisions

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create tessellation for field evaluation
        self._tessellation = (
            tessellated_iga_spline.composite_spline.composite_mesh_tesselation(
                subdivisions_per_patch=tessellation_subdivisions
            )
        )

        # Get mesh geometry. Topology is cast to int32 so it matches the
        # default 4-byte XDMF `NumberType="Int"`; geometry stays float64 and
        # the XDMF DataItems below declare `Precision="8"` to match.
        self._vertices = self._tessellation.global_vertices.numpy().astype(np.float64)
        self._faces = self._tessellation.global_faces.numpy().astype(np.int32)

        logger.info(
            f"Created XDMF exporter with {self._vertices.shape[0]} vertices, "
            f"{self._faces.shape[0]} faces"
        )

        # Initialize HDF5 and XDMF files
        self._h5_path = output_path.with_suffix(".h5")
        self._xdmf_path = output_path.with_suffix(".xdmf")

        self._time_steps: list[float] = []
        self._h5_file = None
        self._initialize_files()

    def _initialize_files(self) -> None:
        """Initializes the HDF5 file and writes mesh geometry."""
        import h5py

        self._h5_file = h5py.File(self._h5_path, "w")

        # Write mesh geometry
        mesh_group = self._h5_file.create_group("Mesh")
        mesh_group.create_dataset("Vertices", data=self._vertices)
        mesh_group.create_dataset("Topology", data=self._faces)

        # Create group for function data
        self._h5_file.create_group("Function")

        logger.debug(f"Initialized HDF5 file: {self._h5_path}")

    def _evaluate_field_at_vertices(self) -> npt.NDArray[np.float64]:
        """
        Evaluates the current spline field at tessellation vertices.

        :returns: Array of field values at mesh vertices, shape (n_vertices,)
        """
        n_vertices = self._vertices.shape[0]
        field_values = np.zeros(n_vertices, dtype=np.float64)

        # Iterate over patches and evaluate field
        # The spline and its tessellation share patch ordering, so the two
        # patch/mapping iterators advance in lockstep.
        for (spline_patch, _), (tessellated_patch, local_to_global) in zip(
            self._spline.composite_spline.patches_with_mapping,
            self._tessellation.patches_with_mapping,
        ):
            # Get texture coordinates and convert to parametric coordinates
            texcoords = tessellated_patch.texcoords
            parametric_coords = spline_patch.texcoords_to_parametric_coords(texcoords)

            # Evaluate field
            patch_field = spline_patch.evaluate_field(parametric_coords)
            patch_field_np = patch_field.numpy()[:, 0].astype(np.float64)

            # Map to global indices
            field_values[local_to_global] = patch_field_np

        return field_values

    def write_temperature(self, time: float) -> None:
        """
        Writes the current temperature field at the specified time.

        The temperature is evaluated from the spline's current global field
        values at the tessellation vertices.

        :param time: Simulation time for this snapshot
        """
        # Evaluate field at vertices
        temperature = self._evaluate_field_at_vertices()

        # Write to HDF5
        step_name = f"Step_{len(self._time_steps)}"
        self._h5_file["Function"].create_dataset(step_name, data=temperature)
        self._h5_file.flush()

        self._time_steps.append(time)
        # Refresh the XDMF descriptor so a mid-simulation crash still leaves a
        # readable file containing all snapshots written so far.
        self._write_xdmf()
        logger.debug(f"Wrote temperature at t={time:.2f}s")

    def close(self) -> None:
        """
        Closes the HDF5 file and writes the XDMF metadata.

        This must be called after all time steps have been written to ensure
        the XDMF file is valid.
        """
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

        # Write XDMF file
        self._write_xdmf()
        logger.info(f"XDMF export complete: {self._xdmf_path}")

    def _write_xdmf(self) -> None:
        """Writes the XDMF metadata file.

        HDF5 dataset references must appear on the same line as the enclosing
        ``<DataItem>`` tags: strict XDMF readers (notably ParaView's
        ``Xdmf3ReaderS``/``Xdmf3ReaderT``) treat any whitespace inside the
        element's text content as part of the path and fail to open the
        dataset. ``Precision="8"`` is set on every floating-point DataItem
        because the HDF5 datasets are written as float64.
        """
        n_vertices = self._vertices.shape[0]
        n_faces = self._faces.shape[0]
        h5_filename = self._h5_path.name

        lines = [
            '<?xml version="1.0"?>',
            '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>',
            '<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">',
            "  <Domain>",
            (
                '    <Grid Name="TimeSeries" GridType="Collection" '
                'CollectionType="Temporal">'
            ),
        ]

        for step_idx, time in enumerate(self._time_steps):
            lines.extend(
                [
                    '      <Grid Name="mesh" GridType="Uniform">',
                    f'        <Time Value="{time}"/>',
                    (
                        '        <Topology TopologyType="Triangle" '
                        f'NumberOfElements="{n_faces}">'
                    ),
                    (
                        f'          <DataItem Dimensions="{n_faces} 3" '
                        f'NumberType="Int" Format="HDF">'
                        f"{h5_filename}:/Mesh/Topology</DataItem>"
                    ),
                    "        </Topology>",
                    '        <Geometry GeometryType="XYZ">',
                    (
                        f'          <DataItem Dimensions="{n_vertices} 3" '
                        f'NumberType="Float" Precision="8" Format="HDF">'
                        f"{h5_filename}:/Mesh/Vertices</DataItem>"
                    ),
                    "        </Geometry>",
                    (
                        '        <Attribute Name="Temperature" '
                        'AttributeType="Scalar" Center="Node">'
                    ),
                    (
                        f'          <DataItem Dimensions="{n_vertices}" '
                        f'NumberType="Float" Precision="8" Format="HDF">'
                        f"{h5_filename}:/Function/Step_{step_idx}</DataItem>"
                    ),
                    "        </Attribute>",
                    "      </Grid>",
                ]
            )

        lines.extend(["    </Grid>", "  </Domain>", "</Xdmf>"])

        with open(self._xdmf_path, "w", encoding="utf-8") as xdmf_file:
            xdmf_file.write("\n".join(lines) + "\n")

    def __enter__(self) -> "IGAXDMFExporter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures files are closed."""
        self.close()
