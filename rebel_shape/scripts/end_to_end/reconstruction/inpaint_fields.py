"""
Steps 3 & 4: Inpaint the temperature and material fields onto the spline.

Both fields are recovered by the production image-space inpainting model: the
surface is rendered as an area emitter over a black background and per-control-
point field values are optimised so the renders match the measured single-channel
images (`FieldTextureOptimizer`). Temperature uses the thermal images; the binary
material uses the windows-mask images (steel rendered as 1, wood as 0; the 0.5
initial value is the "gray background" of undefined regions).

This mirrors `inpaint_data_spline` but consumes the synthetic dataset through the
deterministic cameras that rendered it (no loader re-framing), keeping the fields in
the same frame as the reconstructed spline.

Deliverables 3 & 4: each run writes `Final_comparison.png` (rendered field vs.
measured field across held-out views) and a field-on-surface verification render.

Run (after step 2b):

    python -m rebel_shape.scripts.end_to_end.reconstruction.inpaint_fields
"""

import os
import subprocess
import sys

import mitsuba as mi
import tyro

from rebel_shape import logger
from rebel_shape.scripts.end_to_end.end_to_end_config import EndToEndConfig
from rebel_shape.scripts.end_to_end.reconstruction.inpaint_fields_runner import (
    FieldInpaintingRunner,
)
from rebel_shape.scripts.end_to_end.reconstruction.inpainted_field import InpaintedField

_ONLY_ENV: str = "E2E_INPAINT_ONLY"
"""
Environment variable naming the single field a worker subprocess is to inpaint.

`run_inpainting` sets it to an :class:`InpaintedField` value when it re-invokes this
module; it is unset in the parent, which is what tells the two roles apart.
"""


def run_inpainting(config: EndToEndConfig) -> None:
    """
    Inpaint the temperature and material fields.

    On the GPU each field is run in its own subprocess: Dr.Jit's OptiX backend
    aborts (PTX parse error) when a second distinct scene's megakernel is compiled
    in the same process, so the two fields cannot share a process. This reproduces
    with zero other GPU activity, so it is not a concurrency issue. On CPU/LLVM
    there is no such limit, so both run in one process.

    :raises ValueError: If `E2E_INPAINT_ONLY` is set to something that is not an
        :class:`InpaintedField` value.
    :raises subprocess.CalledProcessError: If a GPU worker subprocess fails.
    """
    runner = FieldInpaintingRunner(config)
    only = os.environ.get(_ONLY_ENV)
    if only is not None:  # single-field worker (a subprocess spawned below)
        mi.set_variant("cuda_ad_mono" if config.use_gpu else "llvm_ad_mono")
        runner.inpaint(InpaintedField(only))
        return

    if not config.use_gpu:
        mi.set_variant("llvm_ad_mono")
        runner.inpaint(InpaintedField.TEMPERATURE)
        runner.inpaint(InpaintedField.MATERIAL)
        return

    # GPU: one fresh subprocess per field (sequential -> no two-scene crash, no
    # GPU concurrency). Re-invoke this module with the same CLI args + the marker.
    for kind in InpaintedField:
        logger.info(f"[step 3/4] launching GPU inpaint subprocess for {kind.value}")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "rebel_shape.scripts.end_to_end.reconstruction.inpaint_fields",
                *sys.argv[1:],
            ],
            env={**os.environ, _ONLY_ENV: kind.value},
            check=True,
        )


if __name__ == "__main__":
    run_inpainting(tyro.cli(EndToEndConfig))
