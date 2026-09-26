import json
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, kw_only=True)
class SplineModalComparisonReport:
    """
    The FEM and IGA elastic spectra on one scene's spline, compared by rank.

    Both sides drop their rigid-body modes and the remaining elastic modes are paired
    by frequency rank, so mode k of one solver always meets mode k of the other.
    Serialised as `modal_comparison_fem_iga.json`; those key names appear in
    published artifacts and must not move.

    It deliberately does not reuse `end_to_end.modal.ModalComparisonReport`, which is
    the same shape for a different pairing: there `fem_gt_hz` names the FEM run on the
    ground-truth mesh, and generalising that field to serve both would rename a second
    published schema to fix the first.
    """

    @dataclass(frozen=True, kw_only=True)
    class Mode:
        """One elastic mode as the two solvers see it."""

        elastic_mode: int
        """One-based elastic mode index, rigid-body modes excluded."""
        fem_spline_hz: float
        """Frequency the FEM shell solver gives on the spline [Hz]."""
        iga_spline_hz: float
        """Frequency the IGA shell solver gives on that same spline [Hz]."""
        rel_error: float
        """`|fem - iga| / iga`, the IGA run being the reference."""

    scene: str
    """Scene directory the two spectra came from, relative to the dataset root."""
    num_rb_fem: int
    """Rigid-body modes the FEM side dropped; 6 for a free-floating closed shell."""
    num_rb_iga: int
    """Rigid-body modes the IGA side dropped; should match `num_rb_fem`."""
    num_compared_elastic_modes: int
    """Elastic modes compared, the shorter of the two spectra."""
    field_degree: int
    """Lagrange degree the FEM fields were solved at."""
    subdivisions_per_patch: int
    """Tessellation the FEM mesh was built at, per patch per direction."""
    mean_rel_error: float
    """Mean relative frequency error over the compared modes."""
    median_rel_error: float
    """Median relative frequency error, robust to a near-degenerate mode swap."""
    max_rel_error: float
    """Largest relative frequency error; inflated by exactly such a swap."""
    modes: list[Mode]
    """Every compared mode, in frequency rank."""

    @classmethod
    def from_spectra(
        cls,
        *,
        scene: str,
        fem_frequencies: npt.NDArray[np.float64],
        iga_frequencies: npt.NDArray[np.float64],
        rigid_body_cutoff_hz: float,
        field_degree: int,
        subdivisions_per_patch: int,
    ) -> "SplineModalComparisonReport":
        """
        Pairs two ascending spectra by elastic rank and summarises their disagreement.

        Everything at or below `rigid_body_cutoff_hz` is taken to be a rigid-body mode
        and is dropped from both sides first; the shorter of the two remaining spectra
        then sets how many modes are compared.

        :returns: The comparison, its `modes` in frequency rank.
        :raises ValueError: If either spectrum holds no mode above the cutoff, which
            leaves nothing to pair and every reported statistic undefined.
        """
        fem_elastic = fem_frequencies[fem_frequencies > rigid_body_cutoff_hz]
        iga_elastic = iga_frequencies[iga_frequencies > rigid_body_cutoff_hz]
        num_compared = min(fem_elastic.size, iga_elastic.size)
        if num_compared == 0:
            raise ValueError(
                f"Scene '{scene}' has no elastic mode to compare: the FEM spectrum "
                f"holds {fem_elastic.size} and the IGA spectrum {iga_elastic.size} "
                f"above the {rigid_body_cutoff_hz} Hz rigid-body cutoff. Solve for "
                f"more modes, or lower the cutoff if the shell is genuinely that soft."
            )

        fem_paired = fem_elastic[:num_compared]
        iga_paired = iga_elastic[:num_compared]
        relative_errors = np.abs(fem_paired - iga_paired) / iga_paired
        modes = [
            cls.Mode(
                elastic_mode=rank + 1,
                fem_spline_hz=float(fem_paired[rank]),
                iga_spline_hz=float(iga_paired[rank]),
                rel_error=float(relative_errors[rank]),
            )
            for rank in range(num_compared)
        ]

        return cls(
            scene=scene,
            num_rb_fem=int((fem_frequencies <= rigid_body_cutoff_hz).sum()),
            num_rb_iga=int((iga_frequencies <= rigid_body_cutoff_hz).sum()),
            num_compared_elastic_modes=num_compared,
            field_degree=field_degree,
            subdivisions_per_patch=subdivisions_per_patch,
            mean_rel_error=float(relative_errors.mean()),
            median_rel_error=float(np.median(relative_errors)),
            max_rel_error=float(relative_errors.max()),
            modes=modes,
        )

    def to_json_text(self) -> str:
        """
        The report as `modal_comparison_fem_iga.json` records it.

        `asdict` walks the nested modes as well and every field is a JSON scalar, so
        no encoder hook is needed. The trailing newline the recorded artifacts lack is
        deliberately not added, so that a regenerated file diffs clean against them.
        """
        return json.dumps(asdict(self), indent=2)

    def to_markdown_text(self, geometry_order: int) -> str:
        """
        The report as the table `modal_comparison_fem_iga.md` records it.

        `geometry_order` is the one FEM knob the JSON does not carry, so the banner
        takes it as an argument rather than reading a field: holding it would add a
        key to a published schema in order to print a line.
        """
        lines = [
            f"# FEM-on-spline vs IGA-on-spline modal: {self.scene}",
            "",
            "Both solvers on the SAME recovered spline (same material + scale). "
            f"FEM: geom_order={geometry_order}, field_deg={self.field_degree}, "
            f"S={self.subdivisions_per_patch}.",
            f"RB modes {self.num_rb_fem}/{self.num_rb_iga} (FEM/IGA). "
            f"Elastic: mean {self.mean_rel_error * 100:.3f}% "
            f"/ median {self.median_rel_error * 100:.3f}% "
            f"/ max {self.max_rel_error * 100:.3f}%.",
            "",
            "| elastic mode | FEM-spline [Hz] | IGA-spline [Hz] | rel err |",
            "|---:|---:|---:|---:|",
        ]
        lines += [
            f"| {mode.elastic_mode} | {mode.fem_spline_hz:.3f} | "
            f"{mode.iga_spline_hz:.3f} | {mode.rel_error * 100:.3f}% |"
            for mode in self.modes
        ]
        return "\n".join(lines) + "\n"
