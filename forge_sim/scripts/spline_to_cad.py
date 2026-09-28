from pathlib import Path

import mitsuba as mi
import tyro
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

from forge_sim.meshing.tessellated_spline_surface import TessellatedSplineSurface


def main(spline_file: Path) -> None:
    """
    Convert the json spline file at `spline_file` to a `.step` file of the same name in
    the same directory
    """
    mi.set_variant("llvm_ad_rgb")
    tessellated_spline_surface = TessellatedSplineSurface.from_file(spline_file)
    brep_spline = tessellated_spline_surface.composite_spline.to_brep()

    step_writer = STEPControl_Writer()
    step_writer.Transfer(brep_spline, STEPControl_AsIs)
    output_file = spline_file.parent / (spline_file.stem + ".step")
    step_writer.Write(str(output_file))


if __name__ == "__main__":
    tyro.cli(main)
