"""
Step 8: IGA Reissner-Mindlin shell modal on the reconstructed spline.

Runs the production IGA shell modal solver on the reconstructed spline (the
recovered geometry, unmodified), with the same single steel material and physical
scale as the FEM-on-GT-mesh run (step 7). Thin wrapper over the production
`reissner_mindlin_modal_analysis` driver so the IGA result is the genuine
pipeline output, written as `modal_analysis_results.npz` + mode-shape XDMF for
the independent eigenvalue comparison (step 8 compare).

Run (after step 2b):  python -m rebel_shape.scripts.end_to_end.modal.iga_modal_analysis
"""

import tyro

from rebel_shape import logger
from rebel_shape.scripts.end_to_end.end_to_end_config import EndToEndConfig
from rebel_shape.scripts.reissner_mindlin_modal_analysis.reissner_mindlin_modal_analysis import (  # noqa: E501
    ReissnerMindlinModalParameters,
    run_modal_analysis,
)


def run_iga_modal(config: EndToEndConfig) -> None:
    """Solve the shell modal problem on the recovered spline at its native scale."""
    steel = config.modal.steel
    params = ReissnerMindlinModalParameters(
        spline_path=config.spline_path,
        output_folder=config.iga_modal_dir,
        num_modes=config.modal.num_modes,
        material_config=steel,
        target_length_scale=None,  # no rescaling: use the spline's native scale
        target_spans=config.modal.iga_modal_spans,  # pin 128 spans (membrane locking)
        use_gpu=config.use_gpu,
    )
    logger.info(
        f"[step 8] IGA modal: {config.modal.num_modes} modes, steel "
        f"E={steel.youngs_modulus:.3g} h={steel.thickness} m, native scale, "
        f"target {config.modal.iga_modal_spans} spans/dir"
    )
    run_modal_analysis(params)
    logger.info(f"[step 8] IGA modal results -> {config.iga_modal_dir}")


if __name__ == "__main__":
    run_iga_modal(tyro.cli(EndToEndConfig))
