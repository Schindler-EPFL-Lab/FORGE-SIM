"""
XDMF export of Reissner-Mindlin eigenmodes on tessellated IGA spline surfaces.

This module owns the on-disk format of the modal results: one
``eigenmodes.xdmf`` descriptor plus one ``eigenmodes.h5`` data file, holding the
undeformed mesh and, per exported mode, the deformed geometry together with the
displacement and rotation fields sampled at the tessellation vertices. The
format is read back by the post-processing tools (mode alignment, MAC metrics)
and by ParaView, so it is a durable contract rather than script-local output.
"""

from pathlib import Path

import h5py
import mitsuba as mi
import numpy as np
import numpy.typing as npt

from rebel_shape import logger
from rebel_shape.isogeometric_analysis.reissner_mindlin_modal_problem import (
    ReissnerMindlinModalProblem,
)
from rebel_shape.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)


class EigenmodeXDMFExporter:
    """
    Exports Reissner-Mindlin eigenmodes to a temporal XDMF/HDF5 series.

    Each selected mode becomes one temporal step referencing its own deformed
    geometry and nodal fields, so stepping through "time" in ParaView steps
    through the modes.

    The eigenvector holds control-point degrees of freedom; they are
    interpolated to the tessellation vertices with the patch spline basis and
    the local-to-global mapping of the tessellated mesh.

    Example usage::

        exporter = EigenmodeXDMFExporter(
            tessellated_iga_spline=spline,
            output_folder=Path("./outputs/xdmf"),
        )
        exporter.export_eigenmodes(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            mode_indices=[0, 1, 2],
        )
    """

    def __init__(
        self,
        tessellated_iga_spline: TessellatedIGASplineSurface,
        output_folder: Path,
        scale_factor: float = 0.1,
    ) -> None:
        """
        Initializes the eigenmode exporter.

        :param tessellated_iga_spline: The IGA spline surface the modes were
            computed on, providing both the spline basis and the tessellation
            the fields are sampled at.
        :param output_folder: Directory where ``eigenmodes.xdmf`` and
            ``eigenmodes.h5`` will be written. Created if missing.
        :param scale_factor: Scaling factor for the displacement field when
            deforming the mesh. The displacement is normalized to unit
            magnitude, then scaled by
            ``scale_factor * smallest_bbox_extent``.

        :raises ValueError: If `scale_factor` is negative.
        """
        if scale_factor < 0.0:
            raise ValueError("`scale_factor` must be non-negative.")

        self._spline = tessellated_iga_spline
        self._output_folder = output_folder

        output_folder.mkdir(parents=True, exist_ok=True)

        self._vertices = tessellated_iga_spline.global_vertices.numpy().astype(
            np.float64
        )
        # Topology is cast to int32 so it matches the default 4-byte XDMF
        # `NumberType="Int"`; geometry and fields stay float64.
        self._faces = tessellated_iga_spline.global_faces.numpy().astype(np.int32)

        # The deformation amplitude is tied to the smallest positive bounding
        # box extent, so the warped shape stays legible for any model scale.
        bbox_extents = np.max(self._vertices, axis=0) - np.min(self._vertices, axis=0)
        positive_extents = bbox_extents[bbox_extents > 1.0e-12]
        if positive_extents.size > 0:
            self._deformation_scale = scale_factor * float(np.min(positive_extents))
        else:
            self._deformation_scale = scale_factor

    def export_eigenmodes(
        self,
        eigenvalues: npt.NDArray[np.float64],
        eigenvectors: npt.NDArray[np.float64],
        mode_indices: list[int],
    ) -> None:
        """
        Writes the selected eigenmodes to ``eigenmodes.xdmf`` / ``eigenmodes.h5``.

        :param eigenvalues: Array of eigenvalues from modal analysis.
        :param eigenvectors: Eigenvector matrix with shape
            ``(n_dofs, n_modes)``.
        :param mode_indices: List of mode indices to export (e.g.,
            ``[0, 1, 2]``). The HDF5 groups are numbered by export position,
            while the XDMF grid names keep the original mode index.

        :raises ValueError: If any mode index is out of range.
        """
        h5_path = self._output_folder / "eigenmodes.h5"
        xdmf_path = self._output_folder / "eigenmodes.xdmf"

        logger.info(
            "Exporting %d eigenmodes to temporal XDMF series (%s, %s)...",
            len(mode_indices),
            xdmf_path,
            h5_path,
        )

        num_control_points = self._spline.global_field_values.shape[0]
        mode_payloads: list[dict[str, np.ndarray | float | int]] = []

        for mode_index in mode_indices:
            if mode_index < 0 or mode_index >= eigenvectors.shape[1]:
                raise ValueError(f"Mode index {mode_index} out of range.")

            eigenvalue = eigenvalues[mode_index]
            frequency = np.sqrt(np.abs(eigenvalue)) / (2.0 * np.pi)

            mode_values = eigenvectors[:, mode_index].reshape(
                num_control_points, ReissnerMindlinModalProblem.DOFS_PER_CONTROL_POINT
            )
            displacement_at_vertices, rotation_at_vertices = (
                self._interpolate_at_vertices(control_point_values=mode_values)
            )

            # Canonical sign: an eigenvector is defined only up to sign, so an IGA run
            # and an FEM run can disagree by a global minus, making warped shapes look
            # opposite side-by-side. Flip the whole mode so its largest-magnitude
            # displacement component is positive -- the same rule the FEM exporter
            # applies, so simple (non-degenerate) modes land on a matching sign.
            # (Repeated/degenerate eigenvalues need subspace alignment, not a sign.)
            flat_disp = displacement_at_vertices.reshape(-1)
            if flat_disp.size and flat_disp[int(np.argmax(np.abs(flat_disp)))] < 0.0:
                displacement_at_vertices = -displacement_at_vertices
                rotation_at_vertices = -rotation_at_vertices

            # Normalize displacement for visualization
            disp_max = np.max(np.linalg.norm(displacement_at_vertices, axis=1))
            if disp_max > 1e-12:
                displacement_normalized = displacement_at_vertices / disp_max
            else:
                displacement_normalized = displacement_at_vertices

            displacement_scaled = displacement_normalized * self._deformation_scale

            logger.debug(
                f"Mode {mode_index}: max original disp = {disp_max:.6e}, "
                f"scaling = {self._deformation_scale:.6e}"
            )

            displacement_magnitude = np.linalg.norm(
                displacement_at_vertices, axis=1
            ).astype(np.float64)
            rotation_magnitude = np.linalg.norm(rotation_at_vertices, axis=1).astype(
                np.float64
            )

            mode_payloads.append(
                {
                    "mode_index": mode_index,
                    "eigenvalue": float(eigenvalue),
                    "frequency": float(frequency),
                    "geometry": self._vertices - displacement_scaled,
                    "displacement": displacement_at_vertices,
                    "rotation": rotation_at_vertices,
                    "displacement_magnitude": displacement_magnitude.reshape(-1, 1),
                    "rotation_magnitude": rotation_magnitude.reshape(-1, 1),
                }
            )

        with h5py.File(h5_path, "w") as h5_file:
            h5_file.create_dataset("/Mesh/topology", data=self._faces)
            # Undeformed vertices, for the post-hoc MAC alignment (align_modal_xdmf).
            h5_file.create_dataset("/Mesh/vertices", data=self._vertices)
            for export_idx, mode_data in enumerate(mode_payloads):
                for field_name in (
                    "geometry",
                    "displacement",
                    "rotation",
                    "displacement_magnitude",
                    "rotation_magnitude",
                ):
                    h5_file.create_dataset(
                        f"/Modes/mode_{export_idx}/{field_name}",
                        data=np.asarray(mode_data[field_name], dtype=np.float64),
                    )

        self._write_temporal_xdmf(
            xdmf_path=xdmf_path,
            h5_name=h5_path.name,
            num_vertices=self._vertices.shape[0],
            num_triangles=self._faces.shape[0],
            mode_payloads=mode_payloads,
        )

        logger.info(
            "Exported %d modes to temporal XDMF series at %s",
            len(mode_indices),
            self._output_folder / "eigenmodes",
        )

    def _interpolate_at_vertices(
        self, control_point_values: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Interpolates control-point shell degrees of freedom to mesh vertices.

        All six degrees of freedom are interpolated in a single call per patch:
        the spline basis acts column-wise, so this is identical to interpolating
        each component separately but reuses the basis evaluation.

        :param control_point_values: Array of shape
            ``(n_control_points, 6)`` holding the three translations followed by
            the three rotations of every control point.

        :returns: A tuple of ``(displacement, rotation)`` arrays, each of shape
            ``(n_vertices, 3)`` in global vertex order.
        """
        n_vertices = self._vertices.shape[0]
        displacement_at_vertices = np.zeros((n_vertices, 3), dtype=np.float64)
        rotation_at_vertices = np.zeros((n_vertices, 3), dtype=np.float64)

        # The spline and its tessellation share patch ordering, so the two
        # patch/mapping iterators advance in lockstep.
        for (patch, control_point_mapping), (tessellated_patch, local_to_global) in zip(
            self._spline.composite_spline.patches_with_mapping,
            self._spline.tessellated_mesh.patches_with_mapping,
        ):
            parametric_coords = patch.texcoords_to_parametric_coords(
                tessellated_patch.texcoords
            )
            patch_field = patch.interpolating_spline.interpolate(
                values=mi.TensorXf(control_point_values[control_point_mapping]),
                parametric_coords=parametric_coords,
            )
            patch_field_np = patch_field.numpy().astype(np.float64)

            displacement_at_vertices[local_to_global] = patch_field_np[:, :3]
            rotation_at_vertices[local_to_global] = patch_field_np[:, 3:]

        return displacement_at_vertices, rotation_at_vertices

    @staticmethod
    def _write_temporal_xdmf(
        xdmf_path: Path,
        h5_name: str,
        num_vertices: int,
        num_triangles: int,
        mode_payloads: list[dict[str, np.ndarray | float | int]],
    ) -> None:
        """
        Writes an XDMF temporal collection descriptor pointing to per-mode
        HDF5 datasets.

        :param xdmf_path: Output path for the ``.xdmf`` file.
        :param h5_name: Filename (not path) of the companion HDF5 file.
        :param num_vertices: Number of mesh vertices.
        :param num_triangles: Number of mesh triangles.
        :param mode_payloads: List of per-mode metadata dicts, each containing
            ``mode_index``, ``eigenvalue``, and ``frequency`` keys.
        """

        def _attribute_lines(
            name: str,
            attribute_type: str,
            dataset: str,
            dims: str,
        ) -> list[str]:
            return [
                (
                    f'        <Attribute Name="{name}" '
                    f'AttributeType="{attribute_type}" Center="Node">'
                ),
                (
                    f'          <DataItem Dimensions="{dims}" '
                    f'Format="HDF">{h5_name}:{dataset}</DataItem>'
                ),
                "        </Attribute>",
            ]

        xdmf_lines = [
            '<?xml version="1.0"?>',
            '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>',
            '<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">',
            "  <Domain>",
            (
                '    <Grid Name="eigenmodes" GridType="Collection" '
                'CollectionType="Temporal">'
            ),
        ]

        for export_idx, mode_data in enumerate(mode_payloads):
            eigenvalue = float(mode_data["eigenvalue"])
            frequency = float(mode_data["frequency"])
            original_index = int(mode_data["mode_index"])

            xdmf_lines.extend(
                [
                    f'      <Grid Name="mode_{original_index}" GridType="Uniform">',
                    (
                        '        <Topology TopologyType="Triangle" '
                        f'NumberOfElements="{num_triangles}" NodesPerElement="3">'
                    ),
                    (
                        f'          <DataItem Dimensions="{num_triangles} 3" '
                        f'NumberType="Int" Format="HDF">'
                        f"{h5_name}:/Mesh/topology</DataItem>"
                    ),
                    "        </Topology>",
                    '        <Geometry GeometryType="XYZ">',
                    (
                        f'          <DataItem Dimensions="{num_vertices} 3" '
                        f'Format="HDF">{h5_name}:/Modes/'
                        f"mode_{export_idx}/geometry</DataItem>"
                    ),
                    "        </Geometry>",
                    f'        <Time Value="{export_idx}" />',
                    (
                        f"        <!-- mode_index={original_index} "
                        f"lambda={eigenvalue:.6e} f={frequency:.6f} Hz -->"
                    ),
                ]
            )
            xdmf_lines.extend(
                _attribute_lines(
                    name="displacement",
                    attribute_type="Vector",
                    dataset=f"/Modes/mode_{export_idx}/displacement",
                    dims=f"{num_vertices} 3",
                )
            )
            xdmf_lines.extend(
                _attribute_lines(
                    name="rotation",
                    attribute_type="Vector",
                    dataset=f"/Modes/mode_{export_idx}/rotation",
                    dims=f"{num_vertices} 3",
                )
            )
            xdmf_lines.extend(
                _attribute_lines(
                    name="displacement_magnitude",
                    attribute_type="Scalar",
                    dataset=f"/Modes/mode_{export_idx}/displacement_magnitude",
                    dims=f"{num_vertices} 1",
                )
            )
            xdmf_lines.extend(
                _attribute_lines(
                    name="rotation_magnitude",
                    attribute_type="Scalar",
                    dataset=f"/Modes/mode_{export_idx}/rotation_magnitude",
                    dims=f"{num_vertices} 1",
                )
            )
            xdmf_lines.append("      </Grid>")

        xdmf_lines.extend(["    </Grid>", "  </Domain>", "</Xdmf>"])

        with open(xdmf_path, "w", encoding="utf-8") as xdmf_file:
            xdmf_file.write("\n".join(xdmf_lines) + "\n")
