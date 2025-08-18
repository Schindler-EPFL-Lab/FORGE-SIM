from dataclasses import dataclass

from rebel_shape.scripts.configs.shape_optimization_config import (
    ShapeOptimizationConfig,
)


@dataclass(kw_only=True)
class SplineShapeOptimizationConfig(ShapeOptimizationConfig):
    """
    Configuration for spline-based shape optimization.

    Many parameters are the same for mesh-based and spline-based shape optimization.
    The spline is defined by its degree, number of knot spans (subdivisions),
    and the number of tessellation subdivisions used for rendering.

    Spline based shape optimization also follows a coarse-to-fine strategy, where the
    number of control points is increased during the optimization process.
    """

    h_refinement_iterations: list[int]
    """
    List of iterations at which the h-refinement is applied, increasing the number
    of knot spans and thus the number of control points in the spline. 
    The values must be strictly increasing positive integers.
    An empty list means no h-refinement is applied.
    """

    degree: int
    """Degree of the spline used for shape optimization"""
    knot_spans: int
    """Number of knot spans in each patch of the spline"""
    tessellation_subdivisions: int
    """Number of tessellation subdivisions used for each patch of the spline"""

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.degree < 1:
            raise ValueError("`spline_degree` must be a positive integer.")
        if self.knot_spans < 1:
            raise ValueError("`spline_subdivisions` must be a positive integer.")
        if self.tessellation_subdivisions < 1:
            raise ValueError("`mesh_subdivisions` must be a positive integer.")
        if self.lbfgs_history_size < 1:
            raise ValueError("`lbfgs_history_size` must be a positive integer.")

        # Check that `h_refinement_iterations` is a list of strictly increasing
        # positive integers
        if len(self.h_refinement_iterations) != 0:
            prev_iteration = 0
            for iteration in self.h_refinement_iterations:
                if iteration < 1:
                    raise ValueError(
                        "`h_refinement_iterations` must be a list of positive integers."
                    )
                if iteration <= prev_iteration:
                    raise ValueError(
                        "`h_refinement_iterations` must be a list of strictly "
                        + "increasing positive integers."
                    )
                prev_iteration = iteration
