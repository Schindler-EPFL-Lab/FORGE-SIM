from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from fem.heat.fem_simulation_diagnostics import FEMSimulationDiagnostics


@dataclass(frozen=True, kw_only=True)
class FEMHeatSolution:
    """
    The snapshot series a marched FEM heat run produced, plus what it reports.

    Snapshots are in PETSc dof order, which is not the order any consumer wants, so
    the record carries the dof coordinates that map them onto an arbitrary set of
    query points (`gather_at`). Holding them here rather than reaching back into the
    run is what lets the run own no post-march state at all.

    `snapshots` is empty when `FEMHeatRun.march` was asked not to retain the series,
    and `gather_at` then returns an empty list rather than failing, since a caller
    that did not ask for the series is not asking for it here either. Retaining the
    series is single-rank only: `dof_coordinates` comes from
    `tabulate_dof_coordinates`, which includes ghost dofs, while each snapshot comes
    from `getArray`, owned dofs only; the two coincide only when `comm.size == 1`.
    There is no MPI gather in `fem/heat` to reconcile them, so `FEMHeatRun.march`
    raises rather than pair mismatched arrays.
    """

    diagnostics: FEMSimulationDiagnostics
    """The run's own report: the resolved step, T_relax, source power and energies."""
    times: tuple[float, ...]
    """Snapshot times [s], in order, starting at 0.0."""
    snapshots: tuple[npt.NDArray[np.float64], ...]
    """
    One PETSc-dof-ordered temperature array per entry of `times`; empty if not
    retained.
    """
    dof_coordinates: npt.NDArray[np.float64]
    """
    Coordinates of the function space's degrees of freedom, shape (n_dofs, 3).

    This is what `gather_at` matches against, so the record does not have to keep a
    DOLFINx handle alive to be usable.
    """

    def gather_at(
        self, query_vertices: npt.NDArray[np.float64]
    ) -> list[npt.NDArray[np.float64]]:
        """
        Every snapshot resampled into the order of `query_vertices` by nearest dof.

        For P1 the dofs are the vertices, so this is a permutation and every value is
        copied rather than combined; at higher degrees each query point takes its
        nearest dof's value. `query_vertices` is an (n, 3) array of points, and the
        returned arrays are (n,), in its own order.

        :returns: One array per snapshot, in time order; empty when the series was
            not retained.
        """
        index = cKDTree(self.dof_coordinates).query(query_vertices, k=1)[1]
        return [snapshot[index] for snapshot in self.snapshots]
