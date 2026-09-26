import math
import shutil
from pathlib import Path

from rebel_shape import logger
from rebel_shape.scripts.end_to_end.end_to_end_config import EndToEndConfig


class ReconstructSurfaceRunner:
    """Finalises a reconstruction run by collecting its output artifacts."""

    def __init__(self, config: EndToEndConfig, started_at: float) -> None:
        """
        Stores the `config` a run was driven by and the `started_at` timestamp
        (`time.time()`) the optimisation began at, which `collect_outputs` uses
        to tell this run's checkpoints from an earlier run's.
        """
        self._config = config
        self._started_at = started_at

    def collect_outputs(self) -> None:
        """
        Copy this run's final-iteration spline JSON + 6 patch EXRs to canonical paths.

        Nothing clears the reconstruction directory, so a rerun finds every earlier
        run's checkpoints beside its own, and the timestamp in a checkpoint's name is
        the instant of that save rather than of the run. Candidates are therefore
        restricted by modification time to those written since `started_at`, the
        instant the optimisation began, and only among those is the highest iteration
        taken: an earlier run that reached further would otherwise win.

        The threshold is floored to the second first. A filesystem whose timestamps
        are coarser than `time.time()` truncates, and always downwards, so a
        checkpoint of this run can carry an mtime below `started_at` by up to one
        tick; flooring absorbs exactly that, and no run is preceded by another within
        the same second.

        :raises RuntimeError: If the optimisation wrote no spline checkpoint at all.
        """
        out = self._config.reconstruction_dir
        written_after = math.floor(self._started_at)
        jsons = [
            path
            for path in out.glob(f"{self._config.name}_*_iteration-*.json")
            if path.stat().st_mtime >= written_after
        ]
        if not jsons:
            raise RuntimeError(
                f"This run produced no reconstruction JSON in {out} (checkpoints "
                f"left there by earlier runs are deliberately not considered)."
            )

        latest = max(jsons, key=self._iteration_of)
        shutil.copy(latest, self._config.reconstructed_spline_path)
        logger.info(
            f"[step 2b] Final spline: {latest.name} -> "
            f"{self._config.reconstructed_spline_path.name}"
        )

        all_exrs = [
            path
            for path in out.glob("*-bsdf-reflectance-data_*_iteration-*.exr")
            if path.stat().st_mtime >= written_after
        ]
        if all_exrs:
            last_tex_iter = max(self._iteration_of(exr) for exr in all_exrs)
            exrs = [exr for exr in all_exrs if self._iteration_of(exr) == last_tex_iter]
        else:
            exrs = []
        tex_dir = out / "final_patch_textures"
        tex_dir.mkdir(exist_ok=True)
        for exr in exrs:
            shutil.copy(exr, tex_dir / exr.name)
        logger.info(f"[step 2b] Collected {len(exrs)} patch texture EXRs -> {tex_dir}")

    @staticmethod
    def _iteration_of(path: Path) -> int:
        """The optimisation iteration an artifact's filename records."""
        return int(path.stem.split("iteration-")[-1])
