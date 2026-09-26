import sys
from pathlib import Path
from typing import ClassVar

from main_scripts.pipeline.stage import Stage
from rebel_shape.scripts.end_to_end.data_paths import REPO_ROOT
from rebel_shape.scripts.end_to_end.end_to_end_config import EndToEndConfig


class StageCommands:
    """
    Every stage's subprocess argv, built once from one run's configuration.

    `_venv` and `_conda` are the two command shapes a stage can take: a module of
    the active .venv, or a script under the conda interpreter. `commands` is the
    resulting dict, keyed by `Stage`, in canonical run order.
    """

    FEM_STAGE_DIR: ClassVar[Path] = REPO_ROOT / "fem" / "end_to_end"
    """
    Directory holding every stage that runs under the conda interpreter: the two
    mesh-preparation stages and the two DOLFINx ground-truth drivers. They are
    invoked by path rather than as `-m` modules because the conda environment has
    no `rebel_shape` installed.
    """

    def __init__(
        self,
        *,
        cfg: EndToEndConfig,
        conda_python: str,
        use_gpu: bool,
        overrides: list[str],
    ) -> None:
        """Stores the configuration every stage's command is built from."""
        self._cfg = cfg
        self._conda_python = conda_python
        self._use_gpu = use_gpu
        self._overrides = overrides

    def _venv(self, module: str) -> list[str]:
        """
        Command for a stage that runs as a module of the active .venv.

        `module` is the stage's `<subpackage>.<module>` path relative to
        `rebel_shape.scripts.end_to_end`, which is where every stage lives.
        """
        cmd = [sys.executable, "-m", f"rebel_shape.scripts.end_to_end.{module}"]
        if not self._use_gpu:
            cmd.append("--no-use-gpu")
        return cmd + self._overrides

    def _conda(self, script: str) -> list[str]:
        """
        Command for an `EndToEndConfig` stage that runs under the conda interpreter.

        The counterpart of `_venv` for the mesh-preparation stages: they take the
        same configuration as the venv stages and so carry the same flags, but they
        live in `fem/end_to_end/` and are invoked by path, like the two DOLFINx
        drivers `commands` builds directly.
        """
        cmd = [self._conda_python, str(self.FEM_STAGE_DIR / script)]
        if not self._use_gpu:
            cmd.append("--no-use-gpu")
        return cmd + self._overrides

    @property
    def commands(self) -> dict[Stage, list[str]]:
        """Every stage's argv, with the configuration forwarded to both sides."""
        fem_heat_dir = self._cfg.dir_path(EndToEndConfig.FEM_HEAT_DIR_NAME)
        fem_modal_dir = self._cfg.dir_path(EndToEndConfig.FEM_MODAL_DIR_NAME)
        # Forward materials to the FEM heat driver so both heat sides are identical.
        heat_materials: list[str] = []
        for prefix, mat in (
            ("steel", self._cfg.thermal.steel),
            ("wood", self._cfg.thermal.wood),
        ):
            heat_materials += [
                f"--{prefix}-config.conductivity", str(mat.conductivity),
                f"--{prefix}-config.density", str(mat.density),
                f"--{prefix}-config.specific-heat", str(mat.specific_heat),
                f"--{prefix}-config.thickness", str(mat.thickness),
            ]
        # Source: prefer the auto-scaled target ΔT (consistent with the IGA side);
        # fall back to the absolute magnitude when target_delta_temp is None.
        if self._cfg.thermal.target_delta_temp is not None:
            heat_materials += [
                "--target-delta-temp", str(self._cfg.thermal.target_delta_temp)
            ]
        # Forward the structural material to the FEM modal driver so it matches the
        # IGA side exactly; the driver's own defaults are not trusted, since a
        # mismatched thickness scales every FEM frequency by h_iga/h_fem.
        steel = self._cfg.modal.steel
        modal_material = [
            "--material-config.youngs-modulus", str(steel.youngs_modulus),
            "--material-config.poisson-ratio", str(steel.poisson_ratio),
            "--material-config.density", str(steel.density),
            "--material-config.thickness", str(steel.thickness),
            "--material-config.shear-correction-factor",
            str(steel.shear_correction_factor),
        ]
        return {
            Stage.MESH: self._conda("mesh_quality/analyze_mesh_quality.py"),
            Stage.DATASET: self._venv("dataset.dataset"),
            Stage.RECONSTRUCT: self._venv("reconstruction.reconstruct_surface"),
            Stage.INPAINT: self._venv("reconstruction.inpaint_fields"),
            Stage.GT: self._conda("ground_truth_bake/ground_truth_bake.py"),
            Stage.FEM_HEAT: [
                self._conda_python,
                str(self.FEM_STAGE_DIR / "heat_gt" / "fem_heat_gt_mesh.py"),
                "--npz-path",
                str(fem_heat_dir / "heat_gt.npz"),
                "--output-folder",
                str(fem_heat_dir),
                "--heat-source",
                str(self._cfg.thermal.heat_source),
                # The member NAME, which is the only spelling tyro accepts for an
                # enum; `.value` is the JSON-boundary form and would be rejected.
                "--heat-source-region",
                self._cfg.thermal.heat_source_region.name,
                "--num-time-steps",
                str(self._cfg.thermal.num_time_steps),
                "--dt-over-t-relax",
                str(self._cfg.thermal.dt_over_t_relax),
                "--save-interval",
                str(self._cfg.thermal.save_interval),
                *heat_materials,
            ],
            Stage.IGA_HEAT: self._venv("heat.iga_heat_simulation"),
            Stage.CMP_HEAT: self._venv("heat.compare_heat"),
            Stage.FEM_MODAL: [
                self._conda_python,
                str(self.FEM_STAGE_DIR / "modal_gt" / "fem_shell_modal_gt_mesh.py"),
                "--mesh-npz",
                str(fem_modal_dir / "modal_gt.npz"),
                "--output-folder",
                str(fem_modal_dir),
                "--num-modes",
                str(self._cfg.modal.num_modes),
                "--field-degree",
                str(self._cfg.modal.fem_field_degree),
                "--use-bbar",  # block-wise B-bar is fast (minimal consistency error)
                *modal_material,
            ],
            Stage.IGA_MODAL: self._venv("modal.iga_modal_analysis"),
            Stage.CMP_MODAL: self._venv("modal.compare_modal"),
            Stage.SUMMARIZE: self._venv("metrics.summarize"),
        }
