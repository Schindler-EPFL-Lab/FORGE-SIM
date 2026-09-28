"""
This script calculates the metrics of novel view synthesis and shape reconstruction on a
buildnet3d dataset for meshes.
"""

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

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


from forge_sim.scripts.evaluation.eval_buildnet3d import (
    EvalNVSParametersBase,
    eval_buildnet3d,
)


@dataclass(kw_only=True)
class EvalNVSParametersMesh(EvalNVSParametersBase):
    """A config for shape reconstruction using meshes."""

    mesh_path: Path
    """Path to the reconstructed mesh"""

    def get_open3d_mesh(self) -> o3d.t.geometry.TriangleMesh:
        """Returns a mesh converted to Open3D"""
        return o3d.t.io.read_triangle_mesh(str(self.mesh_path))


if __name__ == "__main__":
    parameters = tyro.cli(EvalNVSParametersMesh)
    eval_buildnet3d(params=parameters)
