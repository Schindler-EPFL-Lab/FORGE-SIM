import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

from forge_sim import logger


@dataclass(frozen=True, kw_only=True)
class XDMFSolution:
    """
    One temperature time series, as written by either solver to an XDMF/HDF5 pair.

    Both sides write the same custom layout (`Mesh/Vertices`, `Mesh/Topology`,
    `Function/Step_N`): the IGA side through its exporter and the FEM side
    through `xdmf_time_series.XDMFTimeSeries.write`; DOLFINx's native
    `XDMFFile.write_function` is deliberately never used (it corrupts repeated
    snapshot writes), so that layout is the only one read here. The mesh and the
    fields live in the `.h5`, while the snapshot times live in the `.xdmf`
    sidecar as one `<Time Value="...">` element per grid. `from_file` is the way
    in.
    """

    vertices: npt.NDArray[np.float64]
    """`(n_vertices, 3)` node coordinates."""
    faces: npt.NDArray[np.int64]
    """
    `(n_faces, 3)` triangle connectivity, in whichever integer dtype the file
    carries: the IGA exporter writes int32, `XDMFTimeSeries` int64.
    """
    times: tuple[float, ...]
    """Snapshot times, in file order."""
    temperatures: npt.NDArray[np.float64]
    """
    `(n_steps, n_vertices)` temperatures, one row per entry of `times`, and
    empty when the file carries no `Function` group.
    """

    @classmethod
    def from_file(cls, xdmf_path: Path) -> "XDMFSolution":
        """
        Reads the solution written to `xdmf_path` and its companion `.h5`.

        The companion is found by replacing the suffix of `xdmf_path`, so a
        written pair stays readable after being copied elsewhere.

        :returns: The mesh, the snapshot times and the temperature series the
            pair carries.
        :raises FileNotFoundError: If the companion HDF5 file is missing.
        :raises ValueError: If that file does not carry the expected layout.
        """
        h5_path = xdmf_path.with_suffix(".h5")

        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        times = cls._parse_times(xdmf_path)
        logger.debug(f"Found {len(times)} time steps in XDMF")

        with h5py.File(h5_path, "r") as h5_file:
            if "Mesh" not in h5_file or "Vertices" not in h5_file["Mesh"]:
                raise ValueError(
                    f"Could not find mesh data in {h5_path}: expected a "
                    "`Mesh/Vertices` dataset, as written by the IGA XDMF exporter "
                    "and by `xdmf_time_series.XDMFTimeSeries.write`."
                )

            vertices = np.array(h5_file["Mesh/Vertices"][:])
            faces = np.array(h5_file["Mesh/Topology"][:])

            temperatures = []
            if "Function" in h5_file:
                function_group = h5_file["Function"]
                step_keys = sorted(
                    [key for key in function_group.keys() if key.startswith("Step_")],
                    key=lambda key: int(key.split("_")[1]),
                )
                for step_key in step_keys:
                    temperatures.append(
                        np.array(function_group[step_key][:]).flatten()
                    )

            temperatures = np.array(temperatures) if temperatures else np.array([])

        logger.debug(
            f"Loaded: {vertices.shape[0]} vertices, {faces.shape[0]} faces, "
            f"{len(temperatures)} temperature snapshots"
        )

        return cls(
            vertices=vertices, faces=faces, times=times, temperatures=temperatures
        )

    @staticmethod
    def _parse_times(xdmf_path: Path) -> tuple[float, ...]:
        """
        Reads the snapshot times out of the `.xdmf` descriptor at `xdmf_path`.

        The times are the one part of a solution that does not live in the HDF5
        payload, so they are parsed from the XML rather than read beside the
        fields.

        :returns: One time per `<Time Value="...">` element, in document order.
        """
        root = ElementTree.parse(xdmf_path).getroot()

        times = []
        for element in root.iter("Time"):
            if "Value" in element.attrib:
                times.append(float(element.attrib["Value"]))

        return tuple(times)
