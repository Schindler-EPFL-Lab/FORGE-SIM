from dataclasses import dataclass

import drjit as dr
import jsonpickle
import mitsuba as mi
import numpy as np
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape import logger
from rebel_shape.meshing.composite_iga_spline_surface import CompositeIGASplineSurface
from rebel_shape.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)
from rebel_shape.regularization.composite_regularization import CompositeRegularization
from rebel_shape.regularization.spline_curvature_loss import SplineCurvatureLoss
from rebel_shape.regularization.spline_g1_loss import SplineG1Loss
from rebel_shape.scene_optimizers.lbfgs import LBFGS
from rebel_shape.splines.cube_initialization import (
    _get_normalizing_tf_matrix,
    unit_cube,
)


@dataclass(kw_only=True)
class SplineSphereGeneratorParameters(ReverseCli):
    """
    Parameters for generating a sphere from a spline surface.
    """

    radius: float = 1.0
    """Radius of the generated sphere. Default: 1.0."""
    degree: int = 3
    """Degree of the spline surface. Default: 3 (cubic)."""
    subdivisions: int = 16
    """Number of subdivisions for the sphere mesh. Default: 16."""
    name: str = "spline_sphere"
    """Name of the generated spline surface. Default: 'spline_sphere'."""
    tessellation_subdivisions: int = 256
    """Number of subdivisions for tessellating the spline surface. Default: 128."""

    output_path: str = "spline_sphere_16.json"
    """
    Path where the generated spline surface will be saved as a JSON file.
    Default: 'spline_sphere.json' in the current directory.
    """
    use_gpu: bool = True
    """
    Whether to use GPU acceleration for optimization. Default: True.
    Note: GPU acceleration can significantly speed up the optimization process,
    especially for higher subdivision levels.
    """

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("`radius` must be a positive real number.")
        if self.subdivisions < 1:
            raise ValueError("`subdivisions` must be a positive integer.")


def generate_spline_sphere(params: SplineSphereGeneratorParameters) -> None:
    """
    Generates a sphere mesh from the given parameters.

    This function creates a sphere mesh with the specified radius and subdivisions.
    The resulting mesh can be used for testing or as a simple geometric primitive.

    :raises ValueError: If the parameters are invalid.
    """
    if params.use_gpu:
        mi.set_variant("cuda_ad_rgb")
        logger.info("Using the GPU for optimization.")
    else:
        mi.set_variant("llvm_ad_rgb")
        logger.info("Using the CPU for optimization.")

    cube = unit_cube(
        name=params.name,
        degree=params.degree,
        spline_subdivisions=params.subdivisions // 4,
        tessellation_subdivisions=256,
    )

    control_points = cube.global_control_points.numpy()
    vertex_magnitudes = np.linalg.norm(control_points, axis=1)
    # Normalize to unit sphere
    cube.global_control_points = mi.TensorXf(
        control_points / vertex_magnitudes[:, np.newaxis]
    )

    logger.info("Optimizing G1 continuity of the cubed sphere.")
    normal_continuity = SplineG1Loss(composite_spline=cube, weight=1.0)
    curvature_energy = SplineCurvatureLoss(tessellated_spline=cube, weight=0.1)
    composite_regularization = CompositeRegularization(
        regularization_terms=[normal_continuity, curvature_energy]
    )

    # Learning rate is chosen heuristically to be small enough to avoid
    # oscillations, but large enough to converge in a reasonable time.
    # quadratic scaling was determined to work well empirically.
    # adam = mi.ad.Adam(lr=2e-2 / params.subdivisions ** 2, uniform=True)
    # adam["cps"] = cube.global_control_points

    def line_search_fn(xk: dict[str, mi.TensorXf]) -> float:
        # Evaluate the composite regularization at the current control points
        cube.global_control_points = xk["cps"]
        return composite_regularization.evaluate()

    iters = [240, 120, 40]

    for stage in range(3):
        if stage > 0:
            cube.composite_spline.h_refinement(factor=2)

        lbfgs = LBFGS(history_size=15)
        lbfgs["cps"] = cube.global_control_points
        lbfgs.line_search_fn = line_search_fn

        for iteration in range(iters[stage]):
            cube.global_control_points = lbfgs["cps"]

            loss = composite_regularization.evaluate()
            dr.backward(loss)
            lbfgs.step(objective=loss)

            if iteration % 10 == 0:
                logger.debug(f"Iteration {iteration:3d}: G1 loss = {loss.numpy():.4g}")

    transform_to_normalized_bbox = _get_normalizing_tf_matrix(
        cube._tessellated_mesh._global_mesh.vertices
    )
    cube.transform_global_control_points(transform_to_normalized_bbox)

    # Scale the sphere to the desired radius
    cube.global_control_points = cube.global_control_points * params.radius

    iga_spline_surface = CompositeIGASplineSurface.from_spline_surface(
        spline_surface=cube.composite_spline,
        per_patch_field_values=[
            dr.zeros(
                dtype=mi.TensorXf,
                shape=(patch.control_points.shape[0], 6),
            )
            for patch in cube.composite_spline
        ],
    )
    tessellated_iga_spline_surface = TessellatedIGASplineSurface(
        iga_spline_surface=iga_spline_surface,
        tessellation_subdivisions_per_patch=params.tessellation_subdivisions,
    )

    # Save the generated spline surface to a JSON file
    with open(params.output_path, "w") as output_file:
        output_file.write(jsonpickle.encode(tessellated_iga_spline_surface))

    logger.info(f"Spline sphere saved successfully to {params.output_path}")


if __name__ == "__main__":
    parameters = tyro.cli(SplineSphereGeneratorParameters)
    generate_spline_sphere(params=parameters)
