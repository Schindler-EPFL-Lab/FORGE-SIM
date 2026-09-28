"""
Bake the ground-truth FEM analysis inputs.

Mesh preparation is part of the FEM workflow, so this stage lives beside the
drivers it feeds and runs in the same `dolfinx-forge-sim` conda environment.
It needs no DOLFINx itself, only Mitsuba (to read the source PLY's UVs),
Open3D and gpytoolbox, and hands its results to the drivers as plain `.npz`
files rather than in memory, because the two sides are separate processes:

* `heat_gt.npz`  -> vertices, faces, temperature (per vertex), material
  (per vertex, 1=steel / 0=wood), scale. Fine mesh to resolve the sharp
  checkerboard initial condition and the material interface.
* `modal_gt.npz` -> vertices, faces, scale. Coarser P2-friendly mesh; single
  steel material applied uniformly by the FEM driver.

A `ground_truth_bake_report.json` describing both lands at the output root, so
that what a finished bake produced can be read off without re-running it. It is
the counterpart of step 1's `mesh_quality_report.json`.

Both meshes are isotropically remeshed for quality and kept at the source mesh's
native scale: it and the reconstructed spline already agree to ~0.5%, and with GT
geometry and poses the physical size is the geometry's own.

Run it from the repository root in the `dolfinx-forge-sim` conda
environment::

    python fem/end_to_end/ground_truth_bake/ground_truth_bake.py
"""

import sys
from pathlib import Path

import mitsuba as mi
import tyro

# Repository root: makes `forge_sim` and this script's own `fem` package
# importable when it is run by path rather than as `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fem.end_to_end.ground_truth_bake.ground_truth_bake_report import (
    GroundTruthBakeReport,
)
from fem.end_to_end.ground_truth_bake.ground_truth_bake_runner import (
    GroundTruthBakeRunner,
)
from forge_sim.scripts.end_to_end.end_to_end_config import EndToEndConfig


def run_ground_truth_bake(config: EndToEndConfig) -> GroundTruthBakeReport:
    """Produce heat_gt.npz and modal_gt.npz from the source mesh + config."""
    # The Mitsuba variant is a process-global; the launching script sets it once,
    # before the bake reads the source PLY's UVs, and no module sets it on import.
    mi.set_variant("llvm_ad_rgb")
    return GroundTruthBakeRunner(config).bake()


if __name__ == "__main__":
    run_ground_truth_bake(tyro.cli(EndToEndConfig))
