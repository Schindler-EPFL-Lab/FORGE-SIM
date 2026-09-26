from pathlib import Path

import numpy as np
import numpy.typing as npt

from fem.end_to_end.modal_comparison.modal_comparison_scene_config import (
    ModalComparisonSceneConfig,
)
from fem.end_to_end.modal_comparison.spline_modal_comparison_report import (
    SplineModalComparisonReport,
)
from fem.shells.shell_modal_analysis_spline import (
    ShellModalAnalysisParameters,
    run_modal_analysis,
)
from rebel_shape import logger


class SceneModalComparison:
    """
    One scene's FEM-versus-IGA modal comparison: solve, diff, write the report.

    Construction resolves the scene's directories from `scene`; it touches no
    disk, so a scene whose spline the caller could not resolve leaves no empty
    `fem_comparison/` folder behind in the dataset tree to be read later as an
    empty result. `run` makes the directory.
    `fem.end_to_end.heat_comparison.scene_heat_comparison.SceneHeatComparison`
    is built the same way for the same reason.

    Takes the values it needs directly, as `ModalComparisonSceneConfig`,
    rather than a scene identity resolved through a registry: the registry of
    published scenes is `main_scripts`' own subject (`CLAUDE.md` SS3), and a
    comparison for one scene is ordinary library functionality that has no
    need to know the registry exists.
    """

    MODAL_SUBDIR: Path = Path("modal_analysis")
    """Per-scene subdirectory holding the IGA reference modal run."""
    OUTPUT_SUBDIR: Path = Path("fem_comparison")
    """Per-scene subdirectory this comparison writes its artifacts into."""
    ARCHIVE_NAME: str = "modal_analysis_results.npz"
    """
    File name both modal solvers write their spectrum to, read here out of the IGA
    run's own directory and out of this comparison's output directory.
    """
    JSON_REPORT_NAME: str = "modal_comparison_fem_iga.json"
    """Machine-readable comparison; its keys appear in published artifacts."""
    MARKDOWN_REPORT_NAME: str = "modal_comparison_fem_iga.md"
    """The same comparison rendered as a table, for reading."""

    def __init__(
        self,
        *,
        scene: ModalComparisonSceneConfig,
        dataset_root: Path,
        num_modes: int,
        geometry_order: int,
        field_degree: int,
        subdivisions_per_patch: int,
        rb_freq_hz: float,
    ) -> None:
        """
        Pins `scene`'s directories and FEM discretization; touches no disk.

        `dataset_root` locates the scene on disk; `num_modes`,
        `geometry_order`, `field_degree` and `subdivisions_per_patch` are the
        FEM solve knobs `run` builds its solve from. `rb_freq_hz` is the
        rigid-body cutoff the comparison pairs elastic modes above.
        """
        self._scene = scene
        self._num_modes = num_modes
        self._geometry_order = geometry_order
        self._field_degree = field_degree
        self._subdivisions_per_patch = subdivisions_per_patch
        self._rb_freq_hz = rb_freq_hz

        modal_dir = dataset_root / scene.directory / self.MODAL_SUBDIR
        self._iga_archive = modal_dir / self.ARCHIVE_NAME
        self._output_dir = modal_dir / self.OUTPUT_SUBDIR
        self._fem_archive = self._output_dir / self.ARCHIVE_NAME

    def run(self) -> SplineModalComparisonReport:
        """
        Solves the scene with the FEM shell solver and writes the comparison beside it.

        :returns: The comparison, also written to `JSON_REPORT_NAME` and
            `MARKDOWN_REPORT_NAME` in the scene's `fem_comparison` directory.
        :raises ValueError: If `spline_path` does not exist, or if the two
            spectra share no elastic mode.
        :raises FileNotFoundError: If either modal result archive is absent,
            including the FEM one, should the solver exit without writing it.
        :raises KeyError: If either archive holds neither `frequencies` nor
            `eigenvalues`.
        """
        logger.info(
            f"===== {self._scene.directory} "
            f"(tls={self._scene.target_length_scale}) ====="
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._solve_with_fem()

        report = SplineModalComparisonReport.from_spectra(
            scene=str(self._scene.directory),
            fem_frequencies=self.sorted_frequencies(self._fem_archive),
            iga_frequencies=self.sorted_frequencies(self._iga_archive),
            rigid_body_cutoff_hz=self._rb_freq_hz,
            field_degree=self._field_degree,
            subdivisions_per_patch=self._subdivisions_per_patch,
        )

        (self._output_dir / self.JSON_REPORT_NAME).write_text(report.to_json_text())
        (self._output_dir / self.MARKDOWN_REPORT_NAME).write_text(
            report.to_markdown_text(geometry_order=self._geometry_order)
        )
        logger.info(
            f"  {report.scene}: {report.num_compared_elastic_modes} elastic, mean "
            f"{report.mean_rel_error * 100:.3f}% max {report.max_rel_error * 100:.3f}% "
            f"(RB {report.num_rb_fem}/{report.num_rb_iga})"
        )
        return report

    @staticmethod
    def sorted_frequencies(archive_path: Path) -> npt.NDArray[np.float64]:
        """
        Ascending natural frequencies [Hz] out of either modal archive schema.

        The FEM driver stores `frequencies` outright; the IGA driver stores only
        `eigenvalues` = omega^2, whose rigid-body entries come back slightly negative,
        so the magnitude is taken before the root. Sorting is what separates this
        reader from `end_to_end.modal.modal_frequencies`, which must preserve the
        stored order to stay aligned with the mode shapes it indexes; here two
        independent spectra are paired by rank, so both are sorted.

        :raises KeyError: If the archive holds neither `frequencies` nor
            `eigenvalues`.
        """
        archive = np.load(archive_path, allow_pickle=True)
        if "frequencies" in archive:
            frequencies = np.asarray(archive["frequencies"], dtype=float)
        else:
            eigenvalues = np.asarray(archive["eigenvalues"], dtype=float)
            frequencies = np.sqrt(np.abs(eigenvalues)) / (2.0 * np.pi)
        return np.sort(frequencies)

    def _solve_with_fem(self) -> None:
        """
        Solves this scene's spline with the FEM shell solver, unless it already has.

        Calls `run_modal_analysis` directly rather than spawning it as a
        subprocess: this comparison already runs under the DOLFINx conda
        environment the solver needs, the same way
        `fem.end_to_end.heat_comparison.scene_heat_comparison.SceneHeatComparison`
        calls `FEMHeatRun` directly. `use_bbar=True` is pinned explicitly here
        rather than left to `ShellModalAnalysisParameters`'s own default: B-bar
        on in FEM (while off in the IGA core) is a recorded decision every
        published comparison was produced with, not something a future default
        change may silently move.

        :raises ValueError: If `self._scene.spline_path` does not exist.
        """
        if self._fem_archive.exists():
            logger.info(f"  FEM result exists -> skipping ({self.ARCHIVE_NAME})")
            return

        params = ShellModalAnalysisParameters(
            spline_path=self._scene.spline_path,
            output_folder=self._output_dir,
            num_modes=self._num_modes,
            material_config=self._scene.structural_material,
            target_length_scale=self._scene.target_length_scale,
            subdivisions_per_patch=self._subdivisions_per_patch,
            geometry_order=self._geometry_order,
            field_degree=self._field_degree,
            use_bbar=True,
        )
        run_modal_analysis(params)
