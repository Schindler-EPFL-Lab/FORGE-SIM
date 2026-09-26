"""
DOLFINx FEM transient-heat driver on a ground-truth triangle mesh.

This is the ground-truth-mesh counterpart to the spline-based FEM heat scripts
(`heat_simulation_spline.py` / `heat_simulation_spline_two_materials.py`), and the
heat-side sibling of `fem_shell_modal_gt_mesh.py`. Instead of tessellating an IGA
spline and projecting fields out of the spline basis, it reads an already-baked
triangle mesh with per-vertex fields from a NumPy `.npz`, baked by its sibling
`ground_truth_bake.py`. Both run in this conda environment, as separate stages in
separate processes (`run_pipeline.py` keeps every stage apart because the baking side
sets a global Mitsuba variant), so the `.npz` is their handoff and the recorded
artifact.

Everything downstream of "where do the per-vertex arrays come from" is the shared
`FEMHeatRun.run_with_xdmf`: the same Crank-Nicolson solve, the same normalized-time-step
eigensolve, the same energy diagnostics and the same custom XDMF format as the spline
runs. The end-to-end experiment compares this run against an IGA run of the same
scenario, and that comparison is only worth making if the two FEM paths cannot
silently drift apart.

Input `.npz` contract (keys):
    vertices     (n, 3) float64:  node positions, already at the physical size the
                                    run is to use (this driver never rescales).
    faces        (m, 3) int:      triangle connectivity.
    temperature  (n,)   float64:  initial condition, normalized units [0, 1].
    material     (n,)   float64:  per-vertex material mask in [0, 1]; the
                                    values above `FEMHeatGTRunner`'s active-material
                                    threshold are the active material.
    scale        scalar float64:  the rescale factor the baking stage already applied.
                                    Recorded for provenance and not read here, since
                                    `vertices` is what the run integrates.

The two materials are named `steel` (active) and `wood` (passive) throughout the
CLI. Those names are slots in the pipeline's wire protocol, not claims about the
substances: `main_scripts/run_pipeline.py` forwards whatever pair of
materials its configuration holds, and for the recorded Suzanne runs the active slot
carries aluminium and the passive slot steel. Which slot carries the heat source is
`heat_source_region`, forwarded from the same shared configuration the IGA run reads;
its default puts the source on the active material, mirroring the "windows heat up"
case of the two-material spline script.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import tyro

# Repository root: makes `rebel_shape` and this script's own `fem` package
# importable when it is run by path rather than as `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from fem.end_to_end.heat_gt.fem_heat_gt_runner import FEMHeatGTRunner
from rebel_shape.scripts.configs.thermal_material_config import ThermalMaterialConfig
from rebel_shape.scripts.end_to_end.heat.thermal_sim_config import ThermalSimConfig


@dataclass(kw_only=True)
class FEMHeatGTMeshParameters(ReverseCli):
    """
    Configuration for a ground-truth-mesh FEM heat run.

    The geometry and initial temperature come from `npz_path`; the per-vertex material
    properties follow from its material mask. Time stepping mirrors the spline-FEM
    scripts so that an IGA comparison is apples-to-apples. `run_fem_heat` is the entry
    point that runs the simulation from a built instance.
    """

    npz_path: Path = Path("outputs/suzanne_e2e/05_fem_heat/heat_gt.npz")
    """Path to the baked `.npz` with vertices/faces/temperature/material/scale."""
    output_folder: Path = Path("outputs/suzanne_e2e/05_fem_heat")
    """Directory for simulation outputs (created if it does not exist)."""

    num_time_steps: int = 500
    """Number of Crank-Nicolson time steps."""
    dt_over_t_relax: float = 1e-5
    """
    Time step normalized by the natural relaxation time T_relax of the discretized
    system (physical dt = `dt_over_t_relax` * T_relax). T_relax is derived from the
    smallest non-zero generalized eigenvalue of the FEM-assembled (K, M), the same
    characterization the IGA solver uses, so the same value yields a near-identical
    physical dt on both.
    """
    save_interval: int = 50
    """Steps between XDMF snapshots and energy-drift reports."""
    fem_degree: int = 1
    """Lagrange polynomial degree for the FEM basis (1 = P1)."""

    heat_source: float = 5.0
    """
    Heat generation on the source material in W/m^2; 0 disables it.

    Ignored while `target_delta_temp` is set, which auto-scales instead.
    """

    heat_source_region: ThermalSimConfig.HeatSourceRegion = (
        ThermalSimConfig.HeatSourceRegion.STEEL
    )
    """
    Which of the two materials carries the volumetric source.

    Forwarded by `main_scripts/run_pipeline.py` out of the same shared
    `ThermalSimConfig` the IGA run reads, so both solvers integrate one scenario. A
    standalone run defaults to that configuration's own default, the active material.
    """

    target_delta_temp: float | None = None
    """
    Target temperature rise [K] of the source material over the run. When set, the
    source field is assembled at unit magnitude and rescaled to Q = dT * rho * c * h /
    t_sim (t_sim = `num_time_steps` * dt), so an insulated `source_material` element
    rises by exactly this much, mirroring `iga_heat_simulation`, which reuses this
    run's dt, step count and thermal mass for an identical Q. Ignored under
    `HeatSourceRegion.NONE`, which names no material to derive the rise from.
    """

    steel_config: ThermalMaterialConfig = field(
        default_factory=lambda: ThermalMaterialConfig(
            conductivity=50.0, density=7850.0, specific_heat=500.0, thickness=0.01
        )
    )
    """
    Properties of the active material slot: mild steel, 1 cm thick.

    Only a default for a standalone invocation. Every pipeline run overrides all four
    values from the shared
    `rebel_shape.scripts.end_to_end.heat.thermal_sim_config.ThermalSimConfig`, so that
    the FEM ground-truth run and the IGA run it is compared against use one set of
    materials.
    """
    wood_config: ThermalMaterialConfig = field(
        default_factory=lambda: ThermalMaterialConfig(
            conductivity=0.15, density=600.0, specific_heat=2400.0, thickness=0.01
        )
    )
    """
    Properties of the passive material slot: softwood, 1 cm thick.

    Overridden per run exactly as `steel_config` is.
    """

    write_xdmf: bool = True
    """Whether to write the solution to XDMF for ParaView / comparison."""

    def __post_init__(self) -> None:
        if not self.npz_path.exists():
            raise ValueError(f"NPZ path {self.npz_path} does not exist.")
        if self.num_time_steps < 1:
            raise ValueError(
                f"`num_time_steps` must be a positive integer, got "
                f"{self.num_time_steps}."
            )
        if self.dt_over_t_relax <= 0.0:
            raise ValueError(
                f"`dt_over_t_relax` must be a positive real number, got "
                f"{self.dt_over_t_relax}."
            )
        if self.save_interval < 1:
            raise ValueError(
                f"`save_interval` must be a positive integer, got {self.save_interval}."
            )
        if self.fem_degree < 1:
            raise ValueError(f"`fem_degree` must be at least 1, got {self.fem_degree}.")
        if self.heat_source < 0.0:
            raise ValueError(
                f"`heat_source` must be non-negative, got {self.heat_source}."
            )
        if self.target_delta_temp is not None and self.target_delta_temp <= 0.0:
            raise ValueError(
                f"`target_delta_temp` must be positive if provided, got "
                f"{self.target_delta_temp}."
            )
        if not isinstance(self.heat_source_region, ThermalSimConfig.HeatSourceRegion):
            raise ValueError(
                f"`heat_source_region` must be a `HeatSourceRegion` member, got "
                f"{self.heat_source_region!r}."
            )

    @property
    def source_material(self) -> ThermalMaterialConfig | None:
        """
        The material `heat_source_region` places the source on.

        :returns: `steel_config` or `wood_config`, or None under
            `HeatSourceRegion.NONE`, which names no material, and therefore no
            thermal mass for `target_delta_temp` to scale a source against.
        """
        if self.heat_source_region is ThermalSimConfig.HeatSourceRegion.STEEL:
            return self.steel_config
        if self.heat_source_region is ThermalSimConfig.HeatSourceRegion.WOOD:
            return self.wood_config
        return None


def run_fem_heat(params: FEMHeatGTMeshParameters) -> None:
    """Runs the shared FEM heat simulation on a baked ground-truth mesh."""
    FEMHeatGTRunner(
        npz_path=params.npz_path,
        output_folder=params.output_folder,
        steel_config=params.steel_config,
        wood_config=params.wood_config,
        heat_source=params.heat_source,
        heat_source_region=params.heat_source_region,
        target_delta_temp=params.target_delta_temp,
        num_time_steps=params.num_time_steps,
        save_interval=params.save_interval,
        dt_over_t_relax=params.dt_over_t_relax,
        fem_degree=params.fem_degree,
        write_xdmf=params.write_xdmf,
    ).run()


if __name__ == "__main__":
    run_fem_heat(params=tyro.cli(FEMHeatGTMeshParameters))
