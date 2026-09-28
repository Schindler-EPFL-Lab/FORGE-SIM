import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class HeatComparisonReport:
    """
    The geometry-robust heat scalars the FEM-GT and IGA-spline runs are compared on.

    A direct field diff is meaningless across two different geometries, so the
    comparison is taken over integrated quantities both solvers log: the total
    thermal energy at the first and last saved step, and the heat-generation rate.
    Serialised as `heat_comparison.json`.
    """

    @dataclass(frozen=True, kw_only=True)
    class Quantity:
        """One logged scalar on both geometries, with their relative difference."""

        fem_gt: float
        """Value on the FEM ground-truth mesh."""
        iga_spline: float
        """Value on the reconstructed IGA spline."""
        rel_error: float
        """`|fem_gt - iga_spline| / |fem_gt|`; NaN when the reference vanishes."""

    @dataclass(frozen=True, kw_only=True)
    class RelaxationTimes:
        """
        Each solver's own diffusive relaxation time [s].

        Reported side by side but not differenced: it is a property of each
        geometry rather than a quantity the two are meant to agree on, and either
        side is None when its run did not log one.
        """

        fem_gt: float | None
        """Relaxation time of the FEM ground-truth model, or None if unlogged."""
        iga_spline: float | None
        """Relaxation time of the IGA spline model, or None if unlogged."""

    total_heat_initial: Quantity
    """Total thermal energy at the first saved step [J]."""
    total_heat_final: Quantity
    """Total thermal energy at the last saved step [J]."""
    heat_generation_rate: Quantity
    """Source power `integral(Q) dA` at the last saved step [W]."""
    relaxation_time: RelaxationTimes
    """Each model's own relaxation time [s]."""

    @classmethod
    def from_json(cls, path: Path) -> "HeatComparisonReport":
        """
        Read back the `heat_comparison.json` `run_heat_comparison` wrote.

        The summary stage reports this table without recomputing it, and reading it
        as the record it was written from is what keeps the two in step: a renamed
        field cannot pass the constructor.
        """
        raw = json.loads(path.read_text())
        return cls(
            total_heat_initial=cls.Quantity(**raw["total_heat_initial"]),
            total_heat_final=cls.Quantity(**raw["total_heat_final"]),
            heat_generation_rate=cls.Quantity(**raw["heat_generation_rate"]),
            relaxation_time=cls.RelaxationTimes(**raw["relaxation_time"]),
        )
