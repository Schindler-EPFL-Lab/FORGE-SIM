from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from rebel_shape import logger


@dataclass(frozen=True, kw_only=True)
class VertexMapping:
    """
    Nearest-vertex correspondence from the IGA mesh onto the FEM mesh.

    The two solvers discretise the same surface independently, so their vertex
    arrays share neither an ordering nor exact coordinates. `iga_to_fem` pairs
    them by nearest neighbour and records how far the matches had to reach;
    `gather` then brings a FEM field into IGA vertex ordering.

    That direction is deliberate. Scattering FEM values into IGA slots through a
    FEM->IGA nearest map leaves some IGA vertices unwritten whenever that map is
    not a perfect bijection, which it never quite is, because the two meshes
    carry tiny (float32) tessellation position differences. Those unwritten
    slots stay zero and inject spurious errors of the size of the temperature
    itself, swamping the real difference. Gathering guarantees every IGA vertex
    receives its FEM counterpart, with no gaps.
    """

    MATCH_TOLERANCE: ClassVar[float] = 1e-4
    """
    Distance above which a match is logged as suspect.

    Heuristic and diagnostic only: every IGA vertex is matched regardless and
    nothing computed depends on the magnitude, which is why exceeding it warns
    rather than raises. It sits far below the vertex spacing of either mesh, so
    a comparison between two genuinely different resolutions trips the warning
    as a matter of course and a quiet run is the exception, not the rule.
    """

    indices: npt.NDArray[np.int64]
    """`(n_iga_vertices,)` index of each IGA vertex's nearest FEM vertex."""
    max_distance: float
    """Largest distance any IGA vertex had to reach for its match."""
    mean_distance: float
    """Mean of those distances, over every IGA vertex."""

    @classmethod
    def iga_to_fem(
        cls,
        iga_vertices: npt.NDArray[np.float64],
        fem_vertices: npt.NDArray[np.float64],
    ) -> "VertexMapping":
        """
        Matches every vertex of `iga_vertices` to its nearest in `fem_vertices`.

        `iga_vertices` is `(n_iga, 3)` and `fem_vertices` `(n_fem, 3)`; the match
        runs against a KD-tree of the latter. Every IGA vertex is matched
        whatever the distance, and a match reaching further than
        `MATCH_TOLERANCE` only logs a warning.

        :returns: The mapping, carrying the distance diagnostics of the match.
        """
        logger.info("Building IGA->FEM vertex mapping (nearest neighbour)...")
        fem_tree = cKDTree(fem_vertices)
        distances, indices = fem_tree.query(iga_vertices, k=1)

        max_distance = float(np.max(distances))
        mean_distance = float(np.mean(distances))
        logger.debug(
            f"Vertex mapping: max distance = {max_distance:.2e}, "
            f"mean = {mean_distance:.2e}"
        )
        if max_distance > cls.MATCH_TOLERANCE:
            logger.warning(
                f"Large vertex distance detected: {max_distance:.2e} > "
                f"{cls.MATCH_TOLERANCE:.2e}. Meshes may not be compatible."
            )
        logger.info("Vertex mapping complete.")

        return cls(
            indices=indices.astype(np.int64),
            max_distance=max_distance,
            mean_distance=mean_distance,
        )

    def gather(
        self, fem_temperatures: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Brings `fem_temperatures`, shaped (n_steps, n_fem_vertices), onto the IGA
        vertices.

        A gather rather than a scatter, for the reason the class docstring
        gives: every IGA slot is written exactly once, by construction.

        :returns: An (n_steps, n_iga_vertices) array holding, for each IGA
            vertex, the value of the FEM vertex matched to it.
        """
        return fem_temperatures[:, self.indices]
