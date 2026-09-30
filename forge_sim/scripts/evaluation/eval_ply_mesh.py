"""
This script calculates the metrics of shape reconstruction between the bspline mesh and
the reference mesh.
"""

from dataclasses import dataclass
from pathlib import Path

import open3d as o3d
import tyro

from forge_sim.scripts.evaluation.eval_ply import (
    EvalShapeMeshParametersBase,
    eval_shape_mesh,
)


@dataclass(kw_only=True)
class EvalShapeMeshParametersMesh(EvalShapeMeshParametersBase):
    """Parameters used for evaluation shape mesh reconstruction using mesh"""

    mesh_path: Path
    """Path to the reconstructed mesh"""

    def get_open3d_mesh(self) -> o3d.t.geometry.TriangleMesh:
        """Returns a mesh converted to Open3D"""
        return o3d.t.io.read_triangle_mesh(str(self.mesh_path))


if __name__ == "__main__":
    parameters = tyro.cli(EvalShapeMeshParametersMesh)
    eval_shape_mesh(params=parameters)
