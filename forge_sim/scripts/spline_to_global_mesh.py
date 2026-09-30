"""
This script takes an input spline surface and converts it to a global mesh
saved as a PLY file. The script tessellates the spline surface and exports
the resulting triangulated mesh.
"""

from dataclasses import dataclass
from pathlib import Path

import jsonpickle
import mitsuba as mi
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

from forge_sim import logger
from forge_sim.meshing.tessellated_spline_surface import TessellatedSplineSurface


@dataclass(kw_only=True)
class SplineToMeshParameters(ReverseCli):
    """
    Parameters for converting a spline surface to a global mesh PLY file.
    """

    spline_path: Path
    """Path to the input spline surface JSON file."""
    output_path: Path
    """Path where the output PLY file will be saved."""
    tessellation_subdivisions: int = 256
    """
    Number of subdivisions per patch for tessellation. Higher values produce
    more detailed meshes but increase file size and processing time.
    """

    def __post_init__(self) -> None:
        if self.tessellation_subdivisions < 1:
            raise ValueError("`tessellation_subdivisions` must be a positive integer.")
        if not self.spline_path.exists():
            raise ValueError(f"Spline path {self.spline_path} does not exist.")
        if not self.spline_path.is_file():
            raise ValueError(f"Spline path {self.spline_path} is not a file.")


def convert_spline_to_mesh(params: SplineToMeshParameters) -> None:
    """
    Converts a spline surface to a triangulated mesh and saves it as a PLY file.

    This function loads a tessellated spline surface from a JSON file, tessellates
    it with the specified subdivision level, and exports the global mesh as a PLY file.
    The resulting mesh contains vertices, faces, and vertex normals.

    :raises ValueError: If the spline file cannot be loaded or if output directory
        cannot be created.
    """
    mi.set_variant("llvm_ad_mono")
    logger.info("Using the CPU for processing.")

    # Load the spline surface from file
    logger.info(f"Loading spline surface from {params.spline_path}")
    try:
        with open(params.spline_path, "r") as spline_file:
            tessellated_spline_surface: TessellatedSplineSurface = jsonpickle.decode(
                spline_file.read()
            )
    except Exception as e:
        raise ValueError(f"Failed to load spline from {params.spline_path}: {e}")

    logger.info(f"Loaded spline surface '{tessellated_spline_surface.spline_name}'")
    logger.info(f"Number of patches: {len(tessellated_spline_surface)}")

    # Create tessellated mesh with specified subdivisions
    logger.info(
        f"Tessellating surface with {params.tessellation_subdivisions} "
        "subdivisions per patch"
    )
    tessellated_mesh = (
        tessellated_spline_surface.composite_spline.composite_mesh_tesselation(
            subdivisions_per_patch=params.tessellation_subdivisions
        )
    )

    # Ensure output directory exists
    params.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the global mesh as PLY
    logger.info(f"Saving global mesh to {params.output_path}")
    tessellated_mesh._global_mesh.save_to_ply(
        output_dir=params.output_path, save_name=tessellated_mesh.name
    )

    # Log mesh statistics
    num_vertices = tessellated_mesh.global_vertices.shape[0]
    num_faces = tessellated_mesh.global_faces.shape[0]
    logger.info(f"Exported mesh with {num_vertices} vertices and {num_faces} faces")
    logger.info(f"Mesh saved successfully to {params.output_path}")


if __name__ == "__main__":
    parameters = tyro.cli(SplineToMeshParameters)
    convert_spline_to_mesh(params=parameters)

