from dataclasses import dataclass, field


@dataclass(kw_only=True)
class ReconstructionConfig:
    """Spline shape+texture reconstruction (mirrors the standard protocol)."""

    degree: int = 3
    """Polynomial degree of the B-spline patches."""
    knot_spans: int = 2
    """Initial knot spans per patch direction on the cubed-sphere init."""
    h_refinement_iterations: list[int] = field(
        default_factory=lambda: [25, 50, 80, 140]
    )
    """Iterations at which the basis is refined by factor 2 (2 -> 32 spans)."""
    iterations: int = 250
    """Total shape+texture optimisation iterations."""
    tessellation_subdivisions: int = 256
    """Per-patch tessellation of the rendered mesh (2*S^2 triangles per patch)."""
    continuous_samples: int = 64
    """Samples per pixel for the continuous (shading) term of the PRB integrator."""
    visibility_samples: int = 16
    """Samples per pixel for the projective and indirect (visibility) terms."""
    texture_resolution: int = 512
    """Square resolution of each patch's optimised albedo texture."""

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError(f"`degree` must be a positive integer, got {self.degree}.")
        if self.knot_spans < 1:
            raise ValueError(
                f"`knot_spans` must be a positive integer, got {self.knot_spans}."
            )
        if any(iteration < 0 for iteration in self.h_refinement_iterations):
            raise ValueError(
                f"`h_refinement_iterations` must all be non-negative, got "
                f"{self.h_refinement_iterations}."
            )
        if self.iterations < 1:
            raise ValueError(
                f"`iterations` must be a positive integer, got {self.iterations}."
            )
        if self.tessellation_subdivisions < 1:
            raise ValueError(
                f"`tessellation_subdivisions` must be a positive integer, got "
                f"{self.tessellation_subdivisions}."
            )
        if self.continuous_samples < 1:
            raise ValueError(
                f"`continuous_samples` must be a positive integer, got "
                f"{self.continuous_samples}."
            )
        if self.visibility_samples < 1:
            raise ValueError(
                f"`visibility_samples` must be a positive integer, got "
                f"{self.visibility_samples}."
            )
        if self.texture_resolution < 1:
            raise ValueError(
                f"`texture_resolution` must be a positive integer, got "
                f"{self.texture_resolution}."
            )
