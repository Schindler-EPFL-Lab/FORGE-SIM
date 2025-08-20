from dataclasses import dataclass, field
from typing import ClassVar

from rebel_shape.scripts.configs.shape_optimization import ShapeOptimization


@dataclass(kw_only=True)
class SplineShapeOptimization(ShapeOptimization):
    __uninit_h_refinement_iterations: ClassVar[list[int]] = list()
    h_refinement_iterations: list[int] = field(
        default_factory=lambda: SplineShapeOptimization.__uninit_h_refinement_iterations
    )
    """
    List of iterations at which the h-refinement is applied, increasing the number
    of knot spans and thus the number of control points in the spline. The default 
    values are applied if `None` is provided: [5, 10, 15, 25].
    The values must be strictly increasing positive integers.
    """
    spline_degree: int = 3
    """Degree of the spline used for shape optimization"""
    spline_subdivisions: int = 2
    """Number of knot spans in each patch of the spline"""
    mesh_subdivisions: int = 256
    """Number of tessellation subdivisions used for each patch of the spline"""

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.spline_degree < 1:
            raise ValueError("`spline_degree` must be a positive integer.")
        if self.spline_subdivisions < 1:
            raise ValueError("`spline_subdivisions` must be a positive integer.")
        if self.mesh_subdivisions < 1:
            raise ValueError("`mesh_subdivisions` must be a positive integer.")

        if self.h_refinement_iterations is self.__uninit_h_refinement_iterations:
            self.h_refinement_iterations = [5, 10, 15, 25]
        # Check that `h_refinement_iterations` is a list of strictly increasing
        # positive integers
        prev_iteration = 0
        for iteration in self.h_refinement_iterations:
            if iteration < 1:
                raise ValueError(
                    "`h_refinement_iterations` must be a list of positive integers."
                )
            if iteration <= prev_iteration:
                raise ValueError(
                    "`h_refinement_iterations` must be a list of strictly increasing "
                    "positive integers."
                )
            prev_iteration = iteration
