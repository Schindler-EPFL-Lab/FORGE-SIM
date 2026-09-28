from pathlib import Path

import mitsuba as mi
import tyro

from forge_sim.data_processing.get_rgb_and_transformation import (
    SaveFormats,
    render_and_save_reference_set,
)


def main(
    input_ply: Path,
    output_folder: Path | None = None,
    sensor_count: int = 19,
    spp_continuous: int = 64,
    spp_projective: int = 16,
    spp: int = 4096,
    image_resolution: int = 512,
    prefix: str = "ref",
) -> None:
    """
    Creates a nerfstudio type dataset for `input_ply` in `output_folder`.

    `spp_continuous` and `spp_projective` are paraeters of the `PRBIntegrator`. `spp`
    stands for  Samples Per Pixel and is the number of samples per pixel for the
    primal rendering step. `spp_continuous` and `spp_projective` are parameters of the
    `PRBIntegrator`. The resolution of each image along x and y is `image_resolution`.
    `prefix` is a str that will be written in frot of each image filename.

    Renders `sensor_count` views and writes:
    - `data/export/<mesh_name>/images/ref_*.{save_format}`
    - `data/export/<mesh_name>/transforms.json`
    """

    if output_folder is None:
        output_folder = Path(__file__).resolve().parents[2] / "data"
    export_root = output_folder / "export"

    mi.set_variant("llvm_ad_rgb")

    mesh_name = input_ply.stem
    out_dir = export_root / mesh_name

    render_and_save_reference_set(
        sensor_count=sensor_count,
        spp_continuous=spp_continuous,
        spp_projective=spp_projective,
        spp=spp,
        image_resolution=image_resolution,
        save_dir=out_dir,
        mesh_path=input_ply,
        image_prefix=prefix,
        save_format=SaveFormats.EXR,
    )


if __name__ == "__main__":
    tyro.cli(main)
