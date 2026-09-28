"""
Step 6: IGA transient heat on the reconstructed spline (recovered geometry).

Runs the two-material Crank-Nicolson heat equation on the reconstructed spline
using the inpainted temperature field as the initial condition and the inpainted
material field (thresholded) to assign steel/wood properties + the source. This is
the "recovered geometry" half of the heat comparison; the FEM-on-GT-mesh run
(step 5) is the independent ground-truth half.

The physics (material selection, source assembly, Crank-Nicolson time
stepping) is delegated to `simulate_heat_cn_two_materials`, the same
production script an operator would run by hand; this stage only translates
`EndToEndConfig` into its parameters and assembles the report from the
diagnostics it exposes. See `forge_sim.scripts.heat_simulation.
heat_simulation_two_materials` for the two-material physics itself.

Outputs (to `06_iga_heat/`): an XDMF temperature series (IGA-exporter format,
readable by `compare_iga_fem`) and `iga_heat_log.json` with the total thermal
energy and heat-generation rate per saved step, the same diagnostics the FEM
driver logs, so the two are directly comparable.

Run (after steps 2b/3/4):

    python -m forge_sim.scripts.end_to_end.heat.iga_heat_simulation
"""

import json

import drjit as dr
import mitsuba as mi
import tyro

from forge_sim import logger
from forge_sim.isogeometric_analysis.heat_simulation import HeatSimulation
from forge_sim.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)
from forge_sim.scripts.end_to_end.end_to_end_config import EndToEndConfig
from forge_sim.scripts.end_to_end.heat.iga_heat_report import IGAHeatReport
from forge_sim.scripts.end_to_end.heat.thermal_sim_config import ThermalSimConfig
from forge_sim.scripts.heat_simulation.heat_simulation_two_materials import (
    HeatSimulationParametersTwoMaterials,
    simulate_heat_cn_two_materials,
)


def _scalar(x: "mi.TensorXf") -> float:
    """Read a one-element Dr.Jit tensor back as a Python float."""
    return float(x.numpy().reshape(-1)[0])


def _gt_time_step(config: EndToEndConfig) -> float | None:
    """
    The absolute physical dt the FEM ground-truth run derived from its own
    assembled K, M, read back from `05_fem_heat/fem_heat_gt_log.json`. Returns
    `None` when the FEM run has not been executed yet (e.g. running step 6 in
    isolation), in which case the caller falls back to a self-derived dt.
    """
    fem_log = config.fem_heat_dir / "fem_heat_gt_log.json"
    if not fem_log.exists():
        return None
    dt = json.loads(fem_log.read_text()).get("dt")
    return float(dt) if dt is not None else None


def run_iga_heat(config: EndToEndConfig) -> IGAHeatReport:
    """
    Run the IGA heat simulation and write its XDMF series and energy log.

    :raises ValueError: If the temperature and material splines carry different
        numbers of control points, which the config permits by letting the two be
        supplied as independently pre-built artifacts, or if
        `config.thermal.heat_source_region` is not `STEEL`, the only region
        `simulate_heat_cn_two_materials` can express (it always sources the
        thresholded-active material).
    """
    tc = config.thermal
    if tc.heat_source_region is not ThermalSimConfig.HeatSourceRegion.STEEL:
        raise ValueError(
            f"`heat_source_region` must be `STEEL`; got "
            f"{tc.heat_source_region.value!r}. `simulate_heat_cn_two_materials` "
            f"always puts the source on the thresholded-active material, which "
            f"this config calls `STEEL`."
        )

    mi.set_variant("cuda_ad_rgb" if config.use_gpu else "llvm_ad_rgb")

    temp_spline = TessellatedIGASplineSurface.from_file(
        composite_spline_surface_path=config.temperature_spline_path
    )
    material_spline = TessellatedIGASplineSurface.from_file(
        composite_spline_surface_path=config.material_spline_path
    )
    # No rescaling: the spline is at its native scale, which matches the GT mesh
    # (and FEM run) to ~0.5%. The pipeline has GT geometry + poses, so the physical
    # size is the geometry's own.

    n_cp = temp_spline.global_field_values.shape[0]
    n_cp_material = material_spline.global_field_values.shape[0]
    if n_cp_material != n_cp:
        raise ValueError(
            f"The temperature and material splines disagree on the control-point "
            f"count ({n_cp} against {n_cp_material}), so the per-control-point "
            f"material select would not line up with the temperature field. They "
            f"must be inpainted onto the same recovered surface at the same basis "
            f"refinement."
        )

    steel_mask = (
        material_spline.global_field_values > ThermalSimConfig.ACTIVE_MATERIAL_THRESHOLD
    )
    steel_frac = float(steel_mask.numpy().mean())
    if tc.target_delta_temp is not None:
        source_description = f"scaled to a {tc.target_delta_temp} K rise"
    else:
        source_description = f"{tc.heat_source} W/m^2"
    logger.info(
        f"[step 6] steel CP fraction {steel_frac:.3f}, source on "
        f"{tc.heat_source_region.value}: {source_description}"
    )

    # Time step is anchored to the ground-truth (FEM) model so both solvers march
    # on one shared physical clock. Each solver's own relaxation time differs with
    # its geometry (the recovered spline can be a few % "smaller" than the GT mesh),
    # so deriving dt independently would land them at different physical instants
    # after N steps and contaminate the energy comparison with that clock skew.
    # The FEM run (step 5) logs the dt it derived from its own assembled K, M; we
    # reuse that absolute dt here.
    gt_dt = _gt_time_step(config)
    if gt_dt is not None:
        logger.info(
            f"[step 6] Using GT-derived dt = {gt_dt:.6g}s (from the FEM run) so both "
            f"solvers share one physical clock."
        )
    else:
        logger.warning(
            "[step 6] No FEM heat log found; falling back to a self-derived dt "
            "(dt_over_t_relax). The heat comparison will be on mismatched clocks "
            "unless the FEM run (step 5) is executed first."
        )

    out_dir = config.iga_heat_dir
    xdmf_path = out_dir / "iga_heat.xdmf"

    energy_log: list[IGAHeatReport.EnergyEntry] = []

    def record(step_index: int, time: float, heat_simulation: HeatSimulation) -> None:
        """Append this step's live energy diagnostics."""
        energy_log.append(
            IGAHeatReport.EnergyEntry(
                step=step_index,
                time=time,
                total_heat=_scalar(heat_simulation.total_heat),
                heat_generation_rate=_scalar(heat_simulation.heat_generation_rate),
            )
        )

    params = HeatSimulationParametersTwoMaterials(
        spline_path=config.temperature_spline_path,
        facade_mask_path=config.material_spline_path,
        output_folder=out_dir,
        xdmf_path=xdmf_path,
        xdmf_subdivisions=tc.xdmf_subdivisions,
        num_time_steps=tc.num_time_steps,
        dt_over_t_relax=tc.dt_over_t_relax,
        time_step_size=gt_dt,
        save_interval=tc.save_interval,
        facade_mask_threshold=ThermalSimConfig.ACTIVE_MATERIAL_THRESHOLD,
        concrete_config=tc.wood,
        glass_config=tc.steel,
        facade_heat_source=tc.heat_source,
        target_delta_temp=tc.target_delta_temp,
        target_length_scale=None,
        use_gpu=config.use_gpu,
    )

    logger.info(
        f"[step 6] Running {tc.num_time_steps} steps via "
        f"`simulate_heat_cn_two_materials`."
    )
    runner = simulate_heat_cn_two_materials(params=params, on_snapshot=record)

    assert runner.cn_scheme is not None
    dt = runner.cn_scheme.time_step_size
    relaxation_time = runner.cn_scheme.relaxation_time

    source_Q = None
    if tc.target_delta_temp is not None:
        assert runner.heat_simulation is not None
        source_Q = _scalar(dr.max(runner.heat_simulation.source_term))

    summary = IGAHeatReport(
        xdmf=str(xdmf_path),
        dt=dt,
        dt_source=(
            IGAHeatReport.TimeStepSource.GT_FEM
            if gt_dt is not None
            else IGAHeatReport.TimeStepSource.SELF_NORMALIZED
        ),
        relaxation_time=relaxation_time,
        num_time_steps=tc.num_time_steps,
        steel_cp_fraction=steel_frac,
        target_delta_temp=tc.target_delta_temp,
        source_Q=source_Q,
        heat_source=tc.heat_source,
        heat_source_region=tc.heat_source_region,
        total_heat_initial=energy_log[0].total_heat,
        total_heat_final=energy_log[-1].total_heat,
        heat_generation_rate=energy_log[-1].heat_generation_rate,
        energy_log=energy_log,
    )
    (out_dir / "iga_heat_log.json").write_text(
        json.dumps(summary.to_json_dict(), indent=2)
    )
    logger.info(
        f"[step 6] total heat {summary.total_heat_initial:.6g} -> "
        f"{summary.total_heat_final:.6g} J, gen rate "
        f"{summary.heat_generation_rate:.6g} W -> {xdmf_path}"
    )
    return summary


if __name__ == "__main__":
    run_iga_heat(tyro.cli(EndToEndConfig))
