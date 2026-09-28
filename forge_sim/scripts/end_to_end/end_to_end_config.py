from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from forge_sim.scripts.end_to_end.dataset.dataset_config import DatasetConfig
from forge_sim.scripts.end_to_end.heat.thermal_sim_config import ThermalSimConfig
from forge_sim.scripts.end_to_end.modal.modal_sim_config import ModalSimConfig
from forge_sim.scripts.end_to_end.reconstruction.reconstruction_config import (
    ReconstructionConfig,
)


@dataclass(kw_only=True)
class EndToEndConfig(ReverseCli):
    """
    Top-level configuration for the synthetic end-to-end validation experiment.

    One instance (with its nested sub-configs) drives every stage, so the eight
    steps stay in lock-step on geometry scale, materials, and resolutions. The
    defaults encode the Suzanne proof-of-concept; the structure is dataset-agnostic.

    Every default path is relative and resolves against the working directory, which
    the package already requires to be the repository root (importing `forge_sim`
    opens `outputs/forge_sim.log` there).

    Material convention (binary, geometry-driven): a control/vertex point belongs to
    `steel` when its coordinate along `material_axis` is below the bounding-box
    midpoint of that axis offset by `material_threshold`, otherwise `wood`. The
    defaults cut Suzanne into symmetric halves through the sagittal plane. The same
    rule defines the ground-truth field on the mesh and, via rendered images +
    inpainting, the recovered field on the spline.

    The `gt_*_mesh_path` pair overrides that synthesis for real-image targets, where
    the ground truth is measured rather than generated. Both must be supplied in
    `mesh_path`'s frame, and the temperature one is required whenever the target
    carries no known ground-truth texture (a BuildNet3D facade, say).
    """

    name: str = "suzanne"
    """Short identifier; names output artifacts and the spline surface."""
    mesh_path: Path = Path("data/meshes/suzanne.ply")
    """Ground-truth source mesh (with UVs for the checkerboard mapping)."""
    prebuilt_spline_path: Path | None = Path("data/splines/spline_suzanne.json")
    """
    If set (and it exists), this pre-optimised TessellatedSplineSurface is used as
    the recovered geometry and step 2b (reconstruction) is skipped. Must be in the
    same frame/scale as the GT mesh so the deterministic dataset cameras align with
    it. Set to None to run reconstruction from the dataset instead.
    """
    output_root: Path = Path("outputs/suzanne_e2e")
    """Root directory for all stage outputs."""

    material_axis: int = 0
    """
    Axis (0=x, 1=y, 2=z) the material split runs along. 0 (x) cuts the head into
    symmetric left/right halves through the sagittal plane.
    """
    material_threshold: float = 0.0
    """
    Offset from the bounding-box midpoint along `material_axis`; 0 = symmetric
    halves (steel below the plane, wood above).
    """

    modal_geometry_order: int = 1
    """
    Geometry order of the FEM modal analysis mesh: 1 (flat facets) or 2 (curved).
    Order 2 projects each mid-edge node onto the source mesh, recovering the chord
    area a flat remesh loses; shell mass is `rho * h * area`, so that loss biases
    frequencies high. Features the remesh dropped entirely stay gone.
    """

    repair_source_mesh: bool = True
    """
    Weld and repair `mesh_path` (`fem.meshprep.repaired_mesh.RepairedMesh.from_raw`)
    before remeshing. Suzanne's detached eyeballs would otherwise contribute their own
    rigid-body modes, 18 instead of 6; on an already watertight, manifold,
    single-component mesh every step is at best a no-op and the weld can break it.
    """

    gt_temperature_mesh_path: Path | None = None
    """
    A measured-temperature mesh from `inpaint_data_mesh` (`InpaintType.THERMAL`).
    It replaces the synthetic checkerboard as the FEM ground truth, leaving
    `thermal.temperature_texture`, `world_checker_cells` and
    `binary_temperature` unused.
    """
    gt_material_mesh_path: Path | None = None
    """
    A measured material mask from `inpaint_data_mesh` (`InpaintType.WINDOWS`),
    thresholded at 0.5 like every other consumer. It replaces the geometric
    `material_axis`/`material_threshold` split, which has no meaning for a real
    facade.
    """

    remesh_target_edge: float = 0.025
    """
    Target edge length for the step-1 quality-improvement remesh (deliverable;
    the per-simulation FEM meshes are remeshed independently at their own edge).
    """

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    """Synthetic multi-view dataset generation (step 2a)."""
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    """Spline shape+texture reconstruction (step 2b)."""
    thermal: ThermalSimConfig = field(default_factory=ThermalSimConfig)
    """Heat-equation scenario shared by the FEM and IGA runs (steps 3-6)."""
    modal: ModalSimConfig = field(default_factory=ModalSimConfig)
    """Reissner-Mindlin shell modal scenario shared by both solvers (steps 7-8)."""

    use_gpu: bool = True
    """Use the CUDA Mitsuba variant for rendering-heavy stages (else LLVM/CPU)."""

    FEM_HEAT_DIR_NAME: ClassVar[str] = "05_fem_heat"
    """
    Name of the step-5 FEM ground-truth heat output directory, shared by
    `fem_heat_dir` and `main_scripts/run_pipeline.py` (which needs the path
    without triggering `dir`'s create-on-read side effect).
    """
    FEM_MODAL_DIR_NAME: ClassVar[str] = "07_fem_modal"
    """Name of the step-7 FEM ground-truth modal output directory."""

    def __post_init__(self) -> None:
        if self.material_axis not in (0, 1, 2):
            raise ValueError(
                f"`material_axis` must be 0, 1, or 2, got {self.material_axis}."
            )
        if self.modal_geometry_order not in (1, 2):
            raise ValueError(
                f"`modal_geometry_order` must be 1 (faceted) or 2 (curved), got "
                f"{self.modal_geometry_order}."
            )
        if self.remesh_target_edge <= 0.0:
            raise ValueError(
                f"`remesh_target_edge` must be a positive edge length, got "
                f"{self.remesh_target_edge}."
            )

    def dir_path(self, name: str) -> Path:
        """Return an output sub-directory by name, without creating it."""
        return self.output_root / name

    def dir(self, name: str) -> Path:
        """Return an output sub-directory by name, creating it if needed."""
        path = self.dir_path(name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def mesh_quality_dir(self) -> Path:
        """Step 1: the quality-improved analysis mesh and its quality report."""
        return self.dir("01_mesh_quality")

    @property
    def dataset_dir(self) -> Path:
        """Step 2a: the rendered multi-view dataset (images, thermal, masks)."""
        return self.dir("02_dataset")

    @property
    def reconstruction_dir(self) -> Path:
        """Step 2b: the reconstruction run's splines, textures and comparisons."""
        return self.dir("02_reconstruction")

    @property
    def temperature_dir(self) -> Path:
        """Step 3: the inpainted temperature spline and its validation renders."""
        return self.dir("03_temperature")

    @property
    def material_dir(self) -> Path:
        """Step 4: the inpainted material spline and its validation renders."""
        return self.dir("04_material")

    @property
    def fem_heat_dir(self) -> Path:
        """Step 5: the FEM ground-truth heat run (XDMF series + energy log)."""
        return self.dir(self.FEM_HEAT_DIR_NAME)

    @property
    def iga_heat_dir(self) -> Path:
        """Step 6: the IGA-on-recovered-spline heat run (XDMF series + energy log)."""
        return self.dir("06_iga_heat")

    @property
    def heat_compare_dir(self) -> Path:
        """Step 6 (compare): the FEM-vs-IGA heat comparison table and plot."""
        return self.dir("06_heat_compare")

    @property
    def fem_modal_dir(self) -> Path:
        """Step 7: the FEM ground-truth modal run (eigenvalues + mode shapes)."""
        return self.dir(self.FEM_MODAL_DIR_NAME)

    @property
    def iga_modal_dir(self) -> Path:
        """Step 8: the IGA-on-recovered-spline modal run."""
        return self.dir("08_iga_modal")

    @property
    def modal_compare_dir(self) -> Path:
        """Step 8 (compare): the FEM-vs-IGA modal comparison table and plot."""
        return self.dir("08_modal_compare")

    @property
    def remeshed_mesh_path(self) -> Path:
        """Step-1 quality-improved mesh, in the source frame (analysis input)."""
        return self.mesh_quality_dir / f"{self.name}_remeshed.ply"

    @property
    def reconstructed_spline_path(self) -> Path:
        """Geometry-only TessellatedSplineSurface JSON written by reconstruction."""
        return self.reconstruction_dir / f"{self.name}_spline.json"

    @property
    def spline_path(self) -> Path:
        """
        The recovered geometry the downstream stages consume: the pre-built spline
        if one is supplied (and exists), otherwise the reconstruction output.
        """
        if self.prebuilt_spline_path is not None and self.prebuilt_spline_path.exists():
            return self.prebuilt_spline_path
        return self.reconstructed_spline_path

    @property
    def inpainted_temperature_spline_path(self) -> Path:
        """
        TessellatedIGASplineSurface JSON carrying the temperature written by the
        `inpaint` stage.
        """
        return self.temperature_dir / f"{self.name}_temperature.json"

    @property
    def inpainted_material_spline_path(self) -> Path:
        """
        TessellatedIGASplineSurface JSON carrying the material mask written by the
        `inpaint` stage.
        """
        return self.material_dir / f"{self.name}_material.json"

    @property
    def temperature_spline_path(self) -> Path:
        """
        TessellatedIGASplineSurface JSON carrying the inpainted temperature.

        The `inpaint` stage's own output, and the only source there is: a spline
        field is recovered from the images and never handed in. The ground-truth
        side is manufactured on the mesh instead, which is why no counterpart
        exists here.
        """
        return self.inpainted_temperature_spline_path

    @property
    def material_spline_path(self) -> Path:
        """
        TessellatedIGASplineSurface JSON carrying the inpainted material mask.

        The `inpaint` stage's own output, for the reason given on
        :attr:`temperature_spline_path`.
        """
        return self.inpainted_material_spline_path
