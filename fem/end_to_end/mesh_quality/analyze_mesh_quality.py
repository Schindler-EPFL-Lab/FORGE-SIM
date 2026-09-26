"""
Step 1: Triangle quality assessment and isotropic remeshing.

Computes the standard FEM mesh-quality indicators (equiangle skewness, aspect
ratio R/2r, shape-regularity/slenderness, edge and area extrema) plus topology
on the source mesh, performs an isotropic Botsch-Kobbelt remesh, and recomputes
the same indicators. Emits a JSON report, before/after histograms, and the
remeshed surface.

Deliverable 1: the before/after quality table proves the remesh improves the
triangulation for downstream finite-element analysis.

Mesh preparation is part of the FEM workflow, which is why this lives here rather
than beside the rendering stages; it needs no DOLFINx, only Open3D and
gpytoolbox. Run it from the repository root in the `dolfinx-rebel-shape` conda
environment::

    python fem/end_to_end/mesh_quality/analyze_mesh_quality.py
"""

import sys
from pathlib import Path

import tyro

# Repository root: makes `rebel_shape` and this script's own `fem` package
# importable when it is run by path rather than as `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fem.end_to_end.mesh_quality.mesh_quality_report import MeshQualityReport
from fem.end_to_end.mesh_quality.mesh_quality_runner import MeshQualityRunner
from rebel_shape.scripts.end_to_end.end_to_end_config import EndToEndConfig


def run_mesh_quality(config: EndToEndConfig) -> MeshQualityReport:
    """Assess quality, remesh, reassess; write report + histograms + remeshed PLY."""
    return MeshQualityRunner(config).run()


if __name__ == "__main__":
    run_mesh_quality(tyro.cli(EndToEndConfig))
