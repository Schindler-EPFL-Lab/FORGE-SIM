"""
This script calculates the metrics of shape reconstruction between the bspline mesh and
the reference mesh.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import open3d as o3d
import tyro

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append("./")


from rebel_shape.scripts.evaluation.eval_ply import (
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
