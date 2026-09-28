"""
Step 2a: Synthetic multi-view dataset generation (no background extraction).

Renders the textured ground-truth mesh from a Fibonacci camera sphere into a
loader-compatible ThermoScenes dataset:

* `images/`        RGBA PNG (RGB = checkerboard under the env map; alpha = exact
                     object silhouette; no SAM needed, the mask is synthetic).
* `thermal/`       single-channel PNG of the checkerboard temperature field
                     (emissive render of the per-vertex temperature).
* `windows_masks/` single-channel PNG of the binary material field (steel=1).
* `transforms.json`with w, h, fl_x and per-frame file_path / transform_matrix /
                     thermal_file_path / windows_mask_file_path.

The thermal and material renders are emissive (black BSDF + area emitter reading a
per-vertex attribute) so they encode the field directly, with no shading. The same
`transforms.json` later drives both reconstruction (RGB) and inpainting
(thermal/material), guaranteeing identical camera frames.

Run:  python -m forge_sim.scripts.end_to_end.dataset.dataset
"""

import mitsuba as mi
import tyro

from forge_sim.scripts.end_to_end.dataset.dataset_runner import (
    DatasetGenerationRunner,
)
from forge_sim.scripts.end_to_end.dataset.dataset_summary import DatasetSummary
from forge_sim.scripts.end_to_end.end_to_end_config import EndToEndConfig


def run_dataset_generation(config: EndToEndConfig) -> DatasetSummary:
    """
    Render and write the synthetic dataset; return where it went and how it split.

    :raises RuntimeError: If the source mesh has no UVs while the RGB renders are
        configured to use the checkerboard texture (`solid_albedo` is None).
    """
    mi.set_variant("cuda_ad_rgb" if config.use_gpu else "llvm_ad_rgb")
    return DatasetGenerationRunner(config).run()


if __name__ == "__main__":
    run_dataset_generation(tyro.cli(EndToEndConfig))
