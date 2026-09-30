"""
This script calculates the metrics of novel view synthesis and shape reconstruction on a
buildnet3d dataset for meshes.
"""

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from dataclasses import dataclass
from pathlib import Path

import open3d as o3d
import tyro

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
