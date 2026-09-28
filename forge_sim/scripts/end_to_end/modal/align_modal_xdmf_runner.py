from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from forge_sim.scripts.end_to_end.modal.area_weighting import (
    mode_column,
    vertex_areas,
)
from forge_sim.scripts.end_to_end.modal.degenerate_mode_runs import (
    degenerate_mode_runs,
)
from forge_sim.scripts.end_to_end.modal.eigenmodes import Eigenmodes
from forge_sim.scripts.end_to_end.modal.modal_alignment_report import (
    ModalAlignmentReport,
)


class AlignModalXdmfRunner:
    """
    Sign- and subspace-alignment of IGA modes onto their FEM counterparts, and the
    re-export of the aligned modes for ParaView overlay.

    `align` is the entry point `main` reaches for; the rest of the class is its
    supporting numerics: the shared eigenmode XDMF/HDF5 writer.
    """

    @staticmethod
    def _write_eigenmodes(
        out_stem: Path,
        faces: npt.NDArray[np.int64],
        vertices: npt.NDArray[np.float64],
        disp: npt.NDArray[np.float64],
        rot: npt.NDArray[np.float64],
    ) -> Path:
        """Write eigenmodes_aligned.h5/.xdmf in the shared temporal format."""
        n_v, n_tri = vertices.shape[0], faces.shape[0]
        h5_path = out_stem.with_suffix(".h5")
        xdmf_path = out_stem.with_suffix(".xdmf")
        with h5py.File(h5_path, "w") as h5:
            h5.create_dataset("/Mesh/topology", data=faces.astype(np.int32))
            h5.create_dataset("/Mesh/vertices", data=vertices.astype(np.float64))
            for k in range(disp.shape[0]):
                d, r = disp[k], rot[k]
                h5.create_dataset(f"/Modes/mode_{k}/displacement", data=d)
                h5.create_dataset(f"/Modes/mode_{k}/rotation", data=r)
        name = h5_path.name
        lines = [
            '<?xml version="1.0"?>',
            '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>',
            '<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">',
            "  <Domain>",
            '    <Grid Name="eigenmodes" GridType="Collection" '
            'CollectionType="Temporal">',
        ]
        for k in range(disp.shape[0]):
            lines += [
                f'      <Grid Name="mode_{k}" GridType="Uniform">',
                f'        <Topology TopologyType="Triangle" NumberOfElements="{n_tri}" '
                'NodesPerElement="3">',
                f'          <DataItem Dimensions="{n_tri} 3" NumberType="Int" '
                f'Format="HDF">{name}:/Mesh/topology</DataItem>',
                "        </Topology>",
                '        <Geometry GeometryType="XYZ">',
                f'          <DataItem Dimensions="{n_v} 3" Format="HDF">'
                f"{name}:/Mesh/vertices</DataItem>",
                "        </Geometry>",
                f'        <Time Value="{k}" />',
            ]
            for fname, ftype, dim in (
                ("displacement", "Vector", 3),
                ("rotation", "Vector", 3),
            ):
                lines += [
                    f'        <Attribute Name="{fname}" AttributeType="{ftype}" '
                    'Center="Node">',
                    f'          <DataItem Dimensions="{n_v} {dim}" Format="HDF">'
                    f"{name}:/Modes/mode_{k}/{fname}</DataItem>",
                    "        </Attribute>",
                ]
            lines.append("      </Grid>")
        lines += ["    </Grid>", "  </Domain>", "</Xdmf>"]
        xdmf_path.write_text("\n".join(lines) + "\n")
        return xdmf_path

    @classmethod
    def align(
        cls,
        fem_h5: Path,
        iga_h5: Path,
        out_stem: Path,
        fem_freq: npt.NDArray[np.float64],
        iga_freq: npt.NDArray[np.float64],
        degeneracy_rtol: float,
    ) -> ModalAlignmentReport:
        """
        Sign- and subspace-align the IGA modes onto the FEM ones and re-export them.

        `degeneracy_rtol` decides which adjacent modes are treated as one repeated
        eigenvalue (agreeing to within it on both the FEM and the IGA spectrum)
        and so get the Procrustes treatment rather than a plain sign flip.

        :returns: The written XDMF path plus how many modes were aligned, how many
            degenerate runs got the Procrustes treatment, and the signed correlations
            before and after.
        """
        fem = Eigenmodes.from_h5(fem_h5)
        iga = Eigenmodes.from_h5(iga_h5)
        n = min(fem.num_modes, iga.num_modes)
        fem_v = fem.vertices
        nn = cKDTree(iga.vertices).query(fem_v, k=1)[1]  # FEM vtx -> nearest IGA vtx
        w = np.sqrt(np.clip(vertex_areas(fem_v, fem.faces), 0, None))

        FF = np.stack([mode_column(fem.displacement[m], w) for m in range(n)], axis=1)
        II = np.stack(
            [mode_column(iga.displacement[m], w, nn) for m in range(n)], axis=1
        )
        # identity pairing: mode m against mode m
        cos_before = np.array([float(FF[:, m] @ II[:, m]) for m in range(n)])

        aligned_disp = iga.displacement[:n].copy()
        aligned_rot = iga.rotation[:n].copy()
        cos_after = cos_before.copy()
        n_degenerate = 0

        for run in degenerate_mode_runs(degeneracy_rtol, fem_freq[:n], iga_freq[:n]):
            if len(run) == 1:  # simple mode -> sign only
                i = run[0]
                if cos_before[i] < 0.0:
                    aligned_disp[i] = -aligned_disp[i]
                    aligned_rot[i] = -aligned_rot[i]
                    cos_after[i] = -cos_before[i]
            else:  # true degenerate run -> Procrustes
                n_degenerate += 1
                A, B = FF[:, run], II[:, run]
                U, _, Vt = np.linalg.svd(B.T @ A)  # min ||A - B R||, R orthogonal
                R = U @ Vt
                d_block = iga.displacement[run]  # (m, nI, 3) full resolution
                r_block = iga.rotation[run]
                for a, idx in enumerate(run):
                    aligned_disp[idx] = np.tensordot(R[:, a], d_block, axes=(0, 0))
                    aligned_rot[idx] = np.tensordot(R[:, a], r_block, axes=(0, 0))
                    cos_after[idx] = float(
                        FF[:, idx] @ mode_column(aligned_disp[idx], w, nn)
                    )

        return ModalAlignmentReport(
            xdmf_path=cls._write_eigenmodes(
                out_stem,
                iga.faces,
                iga.vertices,
                aligned_disp,
                aligned_rot,
            ),
            n_modes=n,
            n_degenerate_runs=n_degenerate,
            neg_before=int((cos_before < 0).sum()),
            neg_after=int((cos_after < -1e-6).sum()),
            mean_cos_after=float(cos_after.mean()),
        )
