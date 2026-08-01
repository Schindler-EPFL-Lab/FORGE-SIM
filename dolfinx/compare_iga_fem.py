"""
Compare IGA and FEM heat simulation results from XDMF files.

This script loads temperature time series from two XDMF files (IGA and FEM
solutions), computes the difference at each time step, and reports L2 and
L∞ error norms. Optionally, it can export a difference time series to XDMF
for visualization in ParaView.

The comparison requires that both solutions use the same mesh (or compatible
meshes with matching vertex ordering). For IGA-FEM comparison, both should
use the same tessellation resolution.

Example usage:
    python compare_iga_fem.py \\
        --iga-xdmf ./iga_outputs/solution.xdmf \\
        --fem-xdmf ./fem_outputs/solution.xdmf \\
        --output-xdmf ./comparison/difference.xdmf

The script will report errors for each time step and provide summary
statistics at the end.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import tyro

# Add rebel-shape to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses_reverse_cli.reverse_cli import ReverseCli

from rebel_shape import logger

from xdmf_time_series import XdmfLayout, write_time_series_xdmf


@dataclass(kw_only=True)
class CompareIGAFEMParameters(ReverseCli):
    """
    Parameters for comparing IGA and FEM heat simulation results.
    """

    iga_xdmf: Path
    """Path to the IGA solution XDMF file."""
    fem_xdmf: Path
    """Path to the FEM solution XDMF file."""
    output_xdmf: Path | None = None
    """Optional path for output difference XDMF file."""
    relative_errors: bool = True
    """Whether to compute relative errors (vs absolute)."""

    def __post_init__(self) -> None:
        if not self.iga_xdmf.exists():
            raise ValueError(f"IGA XDMF file not found: {self.iga_xdmf}")
        if not self.fem_xdmf.exists():
            raise ValueError(f"FEM XDMF file not found: {self.fem_xdmf}")


def load_xdmf_data(
    xdmf_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[float], np.ndarray]:
    """
    Loads mesh and temperature data from an XDMF/HDF5 file pair.

    Both solvers write the same custom layout -- `Mesh/Vertices`,
    `Mesh/Topology`, `Function/Step_N` -- via the IGA exporter and
    `xdmf_time_series.write_time_series_xdmf` respectively; DOLFINx's native
    `XDMFFile.write_function` is deliberately never used (it corrupts repeated
    snapshot writes), so that layout is the only one read here.

    :param xdmf_path: Path to the XDMF file
    :returns: Tuple of (vertices, faces, times, temperatures)
        - vertices: (n_vertices, 3) array of vertex coordinates
        - faces: (n_faces, 3) array of face connectivity
        - times: List of time values for each snapshot
        - temperatures: (n_steps, n_vertices) array of temperature values

    :raises FileNotFoundError: If the companion HDF5 file is missing.
    :raises ValueError: If the HDF5 file does not carry the expected layout.
    """
    h5_path = xdmf_path.with_suffix(".h5")

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    # Time values live in the XDMF descriptor, not the HDF5 payload.
    times = parse_times_from_xdmf(xdmf_path)
    logger.debug(f"Found {len(times)} time steps in XDMF")

    with h5py.File(h5_path, "r") as f:
        if "Mesh" not in f or "Vertices" not in f["Mesh"]:
            raise ValueError(
                f"Could not find mesh data in {h5_path}: expected a "
                "`Mesh/Vertices` dataset, as written by the IGA XDMF exporter "
                "and by `xdmf_time_series.write_time_series_xdmf`."
            )

        vertices = np.array(f["Mesh/Vertices"][:])
        faces = np.array(f["Mesh/Topology"][:])

        temperatures = []
        if "Function" in f:
            func_group = f["Function"]
            step_keys = sorted(
                [k for k in func_group.keys() if k.startswith("Step_")],
                key=lambda x: int(x.split("_")[1]),
            )
            for step_key in step_keys:
                temperatures.append(np.array(func_group[step_key][:]).flatten())

        temperatures = np.array(temperatures) if temperatures else np.array([])

    logger.debug(
        f"Loaded: {vertices.shape[0]} vertices, {faces.shape[0]} faces, "
        f"{len(temperatures)} temperature snapshots"
    )

    return vertices, faces, times, temperatures


def parse_times_from_xdmf(xdmf_path: Path) -> list[float]:
    """
    Parses time values from an XDMF file.

    :param xdmf_path: Path to the XDMF file
    :returns: List of time values
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xdmf_path)
    root = tree.getroot()

    times = []
    for time_elem in root.iter("Time"):
        if "Value" in time_elem.attrib:
            times.append(float(time_elem.attrib["Value"]))

    return times


def build_vertex_mapping(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """
    Builds a mapping from source to target vertices using KD-tree.

    For each source vertex, finds the nearest target vertex. This handles
    cases where meshes have the same vertices but in different orders.

    :param source_vertices: Source mesh vertices (n_source, 3)
    :param target_vertices: Target mesh vertices (n_target, 3)
    :param tolerance: Distance above which the mapping is reported as suspect.
        Every source vertex is mapped regardless; exceeding this only logs a
        warning, since the two discretisations legitimately disagree by more than
        any fixed length once `target_length_scale` has rescaled the geometry.
    :returns: Array of shape (n_source,) mapping source indices to target indices
    """
    from scipy.spatial import cKDTree

    # Build KD-tree of target vertices
    target_tree = cKDTree(target_vertices)

    # Query nearest target vertex for each source vertex
    distances, mapping = target_tree.query(source_vertices, k=1)

    # Validate mapping quality
    max_dist = np.max(distances)
    mean_dist = np.mean(distances)

    logger.debug(
        f"Vertex mapping: max distance = {max_dist:.2e}, mean = {mean_dist:.2e}"
    )

    if max_dist > tolerance:
        logger.warning(
            f"Large vertex distance detected: {max_dist:.2e} > {tolerance:.2e}. "
            f"Meshes may not be compatible."
        )

    return mapping.astype(np.int64)


def compute_errors(
    iga_temps: np.ndarray,
    fem_temps: np.ndarray,
    relative: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes error metrics between IGA and FEM solutions.

    :param iga_temps: IGA temperature array (n_steps, n_vertices)
    :param fem_temps: FEM temperature array (n_steps, n_vertices)
    :param relative: Whether to compute relative errors
    :returns: Tuple of (difference, l2_errors, linf_errors)
    """
    difference = iga_temps - fem_temps

    l2_errors = np.zeros(len(iga_temps))
    linf_errors = np.zeros(len(iga_temps))

    for i in range(len(iga_temps)):
        diff = difference[i]

        if relative:
            # Relative error using IGA as reference
            ref_norm = np.sqrt(np.sum(iga_temps[i] ** 2))
            if ref_norm > 1e-10:
                l2_errors[i] = np.sqrt(np.sum(diff**2)) / ref_norm
                linf_errors[i] = np.max(np.abs(diff)) / np.max(np.abs(iga_temps[i]))
            else:
                l2_errors[i] = np.sqrt(np.sum(diff**2))
                linf_errors[i] = np.max(np.abs(diff))
        else:
            l2_errors[i] = np.sqrt(np.sum(diff**2))
            linf_errors[i] = np.max(np.abs(diff))

    return difference, l2_errors, linf_errors


def compare_solutions(params: CompareIGAFEMParameters) -> None:
    """
    Main comparison function.

    Loads both solutions, builds vertex mapping to handle different orderings,
    computes errors, and optionally writes difference.
    """
    logger.info(f"Loading IGA solution from {params.iga_xdmf}")
    iga_vertices, iga_faces, iga_times, iga_temps = load_xdmf_data(params.iga_xdmf)

    logger.info(f"Loading FEM solution from {params.fem_xdmf}")
    fem_vertices, fem_faces, fem_times, fem_temps = load_xdmf_data(params.fem_xdmf)

    # Validate compatibility
    logger.info(f"IGA: {iga_vertices.shape[0]} vertices, {len(iga_times)} time steps")
    logger.info(f"FEM: {fem_vertices.shape[0]} vertices, {len(fem_times)} time steps")

    # Map each IGA vertex to its nearest FEM vertex and GATHER the FEM field onto
    # the IGA ordering. The opposite (scattering FEM values into IGA slots via a
    # FEM->IGA nearest map) leaves some IGA vertices unwritten whenever the
    # nearest-neighbour map is not a perfect bijection -- which it never quite is,
    # because the two meshes carry tiny (float32) tessellation position
    # differences. Those unwritten slots stay zero and inject spurious
    # O(temperature) errors that swamp the real difference. Gathering guarantees
    # every IGA vertex receives its FEM counterpart, with no gaps.
    logger.info("Building IGA->FEM vertex mapping (nearest neighbour)...")
    iga_to_fem_mapping = build_vertex_mapping(
        source_vertices=iga_vertices,
        target_vertices=fem_vertices,
        tolerance=1e-4,
    )
    logger.info("Vertex mapping complete.")

    # Match time steps (use the last n_steps from each; they share the same
    # physical times when the runs use matched final_time + save_interval).
    n_steps = min(len(iga_times), len(fem_times))
    if len(iga_times) != len(fem_times):
        logger.warning(
            f"Time step count mismatch: IGA {len(iga_times)} vs FEM {len(fem_times)}"
        )
        logger.warning(f"Using last {n_steps} steps for comparison")

    iga_temps_aligned = iga_temps[-n_steps:]
    fem_temps_aligned = fem_temps[-n_steps:]
    times = iga_times[-n_steps:]

    # Gather FEM temperatures onto the IGA vertex ordering (no gaps).
    fem_temps_remapped = fem_temps_aligned[:, iga_to_fem_mapping]

    # Compute errors
    logger.info("Computing errors...")
    error_type = "relative" if params.relative_errors else "absolute"

    difference, l2_errors, linf_errors = compute_errors(
        iga_temps_aligned, fem_temps_remapped, relative=params.relative_errors
    )

    # Report errors for each time step
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Error Analysis ({error_type})")
    logger.info(f"{'=' * 60}")
    logger.info(f"{'Step':>6} {'Time':>10} {'L2 Error':>14} {'L∞ Error':>14}")
    logger.info(f"{'-' * 60}")

    for i, (t, l2, linf) in enumerate(zip(times, l2_errors, linf_errors)):
        logger.info(f"{i:>6} {t:>10.2f} {l2:>14.6e} {linf:>14.6e}")

    # Summary statistics
    logger.info(f"{'-' * 60}")
    logger.info(f"{'Summary Statistics':^60}")
    logger.info(f"{'-' * 60}")
    logger.info(f"Mean L2 error:  {np.mean(l2_errors):.6e}")
    logger.info(f"Max L2 error:   {np.max(l2_errors):.6e}")
    logger.info(f"Mean L∞ error:  {np.mean(linf_errors):.6e}")
    logger.info(f"Max L∞ error:   {np.max(linf_errors):.6e}")
    logger.info(f"{'=' * 60}")

    # Write difference XDMF if requested. The layout / precision arguments are
    # this writer's historical formatting, kept verbatim so the files it produces
    # stay byte-identical to the ones it produced before the writers were merged;
    # they are cosmetic and say nothing about the data (see `xdmf_time_series`).
    if params.output_xdmf is not None:
        write_time_series_xdmf(
            params.output_xdmf,
            # The whole comparison lives in IGA vertex ordering -- `difference`
            # has one column per IGA vertex (the FEM field was gathered onto that
            # ordering above) and `iga_faces` indexes the full IGA vertex array.
            # So the geometry written here must be the complete IGA mesh. An
            # earlier truncation to `min(n_iga, n_fem)` was a no-op whenever the
            # FEM mesh was at least as fine, and produced geometry inconsistent
            # with both the topology and the attribute when it was not.
            iga_vertices,
            iga_faces,
            times,
            difference,
            attribute_name="Difference",
            layout=XdmfLayout.INDENTED,
            declare_float_precision=False,
            create_parent_dirs=True,
        )
        logger.info(f"Wrote difference XDMF to {params.output_xdmf}")


if __name__ == "__main__":
    parameters = tyro.cli(CompareIGAFEMParameters)
    compare_solutions(params=parameters)
