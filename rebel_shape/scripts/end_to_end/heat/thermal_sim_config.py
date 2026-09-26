from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import ClassVar

from rebel_shape.scripts.configs.thermal_material_config import ThermalMaterialConfig


@dataclass(kw_only=True)
class ThermalSimConfig:
    """
    Heat-equation scenario shared by the FEM (GT) and IGA (recovered) runs.

    Pure two-material diffusion of a checkerboard initial temperature, plus an
    optional source on one material (mirrors the dataset "windows heat up" case).
    The two slots carry a diffusivity contrast; thickness is kept uniform so the
    comparison isolates it.

    The slot names `steel` and `wood` are the pipeline's wire protocol (the
    flags the FEM driver expects), not claims about the substances. The values are
    aluminium and steel respectively, and that is how they should be reported.

    Both solvers march on one clock and one source. The FEM ground-truth run derives
    the absolute time step from its own assembled `K` and `M`; the IGA run reads
    it back from the FEM log along with the step count and the active material's
    thermal mass, so the auto-scaled source magnitude comes out identical on the two
    sides whatever the window length, material or geometry. Running the IGA side before
    the FEM one leaves it to resolve a step of its own, which it warns about. What does
    not close is the
    t=0 energy gap: an exact sharp checkerboard on the FEM mesh against the smooth
    recovered field on the spline is a difference of field provenance, and no FEM
    refinement touches it.
    """

    class TemperatureFieldSource(Enum):
        """
        The coordinates the ground-truth temperature checkerboard is evaluated in.

        The field is the experiment's controlled input: the dataset stage renders it
        into the images the reconstruction sees, and the ground-truth bake evaluates
        the same rule on the FEM analysis mesh. Both sides must agree on it, and two
        runs being compared must agree on it too, so it is stated here rather than
        inferred from the asset.

        It used to be inferred, from whether the source PLY happened to yield
        per-vertex UVs, which made the field a property of the mesh loader rather
        than of the experiment. Changing the loader then moved every heat number for
        an unwrapped asset while leaving the modal ones untouched, a shift that
        reads exactly like a solver result and is recoverable only by re-deriving the
        field from the artifacts. There is deliberately no member that restores that
        inference: a caller states which field it wants, or the call fails.
        """

        WORLD_CHECKER = "world_checker"
        """
        Evaluate the world-space checkerboard at the vertex positions, sized by
        `world_checker_cells`.

        The default. It needs no UVs, so it applies to every asset the experiment
        uses, and its only inputs are the mesh and one integer, where the texture
        below is a separate file whose content nothing pins.
        """
        UV_TEXTURE = "uv_texture"
        """
        Sample `temperature_texture` at the mesh UVs, for an asset carrying an
        unwrap. Raises where the mesh carries none, rather than substituting.
        """

    class HeatSourceRegion(Enum):
        """Which of the two materials carries the volumetric heat source."""

        STEEL = "steel"
        """The `material > 0.5` (active) region: the pipeline's default."""
        WOOD = "wood"
        """The `material < 0.5` (passive) region."""
        NONE = "none"
        """No source anywhere: pure diffusion of the initial field."""

    steel: ThermalMaterialConfig = field(
        default_factory=lambda: ThermalMaterialConfig(
            conductivity=237.0, density=2700.0, specific_heat=900.0, thickness=0.01
        )
    )
    """
    The active (source-carrying) material: the `material > 0.5` region, which
    `material_labels` places on the `x < centre` half and which both solvers load
    with the heat source.
    """
    wood: ThermalMaterialConfig = field(
        default_factory=lambda: ThermalMaterialConfig(
            conductivity=50.0, density=7850.0, specific_heat=500.0, thickness=0.01
        )
    )
    """The passive material: the `material < 0.5` region, which carries no source."""
    temp_cold: float = 0.0
    """Temperature assigned to dark checkerboard squares (normalised units)."""
    temp_hot: float = 1.0
    """Temperature assigned to bright checkerboard squares (normalised units)."""
    temperature_texture: Path = Path("data/textures/checkerboard_texture.jpg")
    """
    Texture sampled at the mesh UVs for the initial temperature field, read as
    luminance and mapped onto `[temp_cold, temp_hot]`; `binary_temperature`
    thresholds it into a hard step instead.

    This file rather than a blurred EXR beside it: the blurred one is not tracked, so
    nothing pins its content, and it was in fact regenerated with a heavier blur at
    some point, which moved the ground-truth field of every target reading it while
    leaving no record. A field the experiment measures against belongs to a file the
    history can vouch for.
    """
    binary_temperature: bool = False
    """
    If True, threshold the temperature texture at 0.5 into a hard 0/1 field, which
    Gibbs-rings on the spline. If False, map the smoothed luminance linearly onto
    `[temp_cold, temp_hot]`, a band-limited field with the same area mean, which
    the spline recovers without ringing.
    """
    world_checker_cells: int = 10
    """
    Checker cells across the longest model axis for the world-space checkerboard.
    Read only when the resolved `temperature_field_source` is `WORLD_CHECKER`.
    """
    temperature_field_source: "ThermalSimConfig.TemperatureFieldSource" = (
        TemperatureFieldSource.WORLD_CHECKER
    )
    """
    Which coordinates the ground-truth temperature field is evaluated in. Every run
    compared against another must agree on it, so it is stated rather than inferred.
    """
    heat_source_region: HeatSourceRegion = HeatSourceRegion.STEEL
    """Which material carries the volumetric source."""
    target_delta_temp: float | None = 0.15
    """
    Target temperature rise [K] of the active material over the run, or None to use
    the absolute `heat_source` instead. Both solvers then scale the source as
    `Q = dT * rho * c * h / t_sim`, `t_sim = num_time_steps * dt`. 0.15 K against
    the [0, 1] checkerboard reads as a legible source without swamping the diffusion.
    """
    heat_source: float = 5.0e-2
    """
    Absolute source magnitude on `heat_source_region` [W/m^2]; 0 disables it.

    Ignored while `target_delta_temp` is set, which auto-scales instead. Use this for
    an explicit, window-dependent magnitude.
    """
    num_time_steps: int = 1000
    """Crank-Nicolson steps marched by both solvers."""
    dt_over_t_relax: float = 1e-5
    """
    Time step normalised by the diffusive relaxation time, chosen to resolve the
    interesting part of the evolution rather than the long equilibration tail. The FEM
    run resolves it against its own model and the IGA run reuses the resulting absolute
    step; resolving it twice would land the two a few % apart in physical time.
    """
    save_interval: int = 100
    """
    XDMF/snapshot cadence (in time steps). With num_time_steps=1000 this writes
    ~10 heat snapshots (steps 0,100,...,1000), keeping the XDMF/HDF5 files small.
    """
    xdmf_subdivisions: int = 256
    """
    Per-patch tessellation for the IGA XDMF export (2*S^2 triangles/patch).

    Six patches at 256 give the 786,432 triangles every recorded run carries, which
    is roughly three times the ground-truth mesh it is compared against. The
    comparison maps the two vertex-wise, so the spline must be sampled at least as
    finely as the mesh; halving this would sample it more coarsely instead.
    """
    fem_remesh_edge: float = 0.012
    """
    Target edge length for the GT FEM heat mesh, fine enough for the sharp checkerboard
    initial condition and the material interface (where P1 over- and undershoots). The
    t=0 energy integral converges at 0.02, a 16x refinement moving it <0.15 %, so this
    buys transient fidelity at the jumps, not a better number; ~260k triangles, cheap.
    """
    inpaint_h_refinement: int = 8
    """
    Basis refinement applied before inpainting the spline fields; 8x is the production
    standard, costing ~4x the control points of 4x. A coarse basis blurs the sharp 0/1
    checkerboard while a finer one saturates the bright plateaus, but the residual
    ~8-10 % field-mean deficit is view-limited by the 19 views, unmoved from 2x to 4x.
    """
    inpaint_iterations: int = 100
    """
    Inpainting epochs. The synthetic fields are perfectly view-consistent (no
    cross-view variation to destabilise the fit), but reaching T=1 through the sigmoid
    parametrisation needs a large logit, so the bright plateaus only saturate with
    enough epochs; on the GPU this is cheap.
    """
    inpaint_learning_rate: float = 0.05
    """
    Adam LR for field inpainting.

    The small 0.01 default is what real-thermal view inconsistency needs; the synthetic
    fields are consistent, so a ~5x larger rate is stable and converges fast.
    """
    inpaint_continuous_samples: int = 16
    """Samples per pixel of the PRB continuous term while a field is being fitted."""
    inpaint_initial_color: float = 0.5
    """
    Starting field value everywhere, and the emitter texture's base colour.

    Mid-grey is the "undefined" value of both fields: it is the midpoint of the [0, 1]
    material mask, so a patch no view observes is pulled towards neither material.
    """
    inpaint_field_texture_resolution: int = 512
    """
    Square resolution of the per-patch field texture the inpainting optimiser
    drives.
    """

    ACTIVE_MATERIAL_THRESHOLD: ClassVar[float] = 0.5
    """
    Value above which a control point or vertex counts as the active
    (source-carrying, `steel`) material rather than the passive one.

    Both `iga_heat_simulation`'s per-control-point select and `FEMHeatGTRunner`'s
    per-vertex select cut the continuous material mask at this one shared value
    (the midpoint of its `[0, 1]` range), so a mismatched cut cannot move the
    material interface between the two sides of the heat comparison.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.heat_source_region, ThermalSimConfig.HeatSourceRegion):
            raise ValueError(
                f"`heat_source_region` must be a `HeatSourceRegion` member, got "
                f"{self.heat_source_region!r}."
            )
        if self.world_checker_cells < 1:
            raise ValueError(
                f"`world_checker_cells` must be a positive integer, got "
                f"{self.world_checker_cells}."
            )
        if self.target_delta_temp is not None and self.target_delta_temp <= 0.0:
            raise ValueError(
                f"`target_delta_temp` must be a positive temperature rise, got "
                f"{self.target_delta_temp}."
            )
        if self.heat_source < 0.0:
            raise ValueError(
                f"`heat_source` must be non-negative, got {self.heat_source}."
            )
        if self.num_time_steps < 1:
            raise ValueError(
                f"`num_time_steps` must be a positive integer, got "
                f"{self.num_time_steps}."
            )
        if self.dt_over_t_relax <= 0.0:
            raise ValueError(
                f"`dt_over_t_relax` must be a positive real number, got "
                f"{self.dt_over_t_relax}."
            )
        if self.save_interval < 1:
            raise ValueError(
                f"`save_interval` must be a positive integer, got {self.save_interval}."
            )
        if self.xdmf_subdivisions < 1:
            raise ValueError(
                f"`xdmf_subdivisions` must be a positive integer, got "
                f"{self.xdmf_subdivisions}."
            )
        if self.fem_remesh_edge <= 0.0:
            raise ValueError(
                f"`fem_remesh_edge` must be a positive edge length, got "
                f"{self.fem_remesh_edge}."
            )
        if self.inpaint_h_refinement < 1:
            raise ValueError(
                f"`inpaint_h_refinement` must be a positive integer, got "
                f"{self.inpaint_h_refinement}."
            )
        if self.inpaint_iterations < 1:
            raise ValueError(
                f"`inpaint_iterations` must be a positive integer, got "
                f"{self.inpaint_iterations}."
            )
        if self.inpaint_learning_rate <= 0.0:
            raise ValueError(
                f"`inpaint_learning_rate` must be a positive real number, got "
                f"{self.inpaint_learning_rate}."
            )
        if self.inpaint_continuous_samples < 1:
            raise ValueError(
                f"`inpaint_continuous_samples` must be a positive integer, got "
                f"{self.inpaint_continuous_samples}."
            )
        if not 0.0 <= self.inpaint_initial_color <= 1.0:
            raise ValueError(
                f"`inpaint_initial_color` must be in [0, 1], got "
                f"{self.inpaint_initial_color}."
            )
        if self.inpaint_field_texture_resolution < 1:
            raise ValueError(
                f"`inpaint_field_texture_resolution` must be a positive integer, got "
                f"{self.inpaint_field_texture_resolution}."
            )
