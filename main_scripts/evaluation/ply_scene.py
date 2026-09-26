from enum import Enum


class PlyScene(Enum):
    """
    One of the six PLY reference scenes the Chamfer sweeps evaluate, by identity
    alone.

    `value` is the scene's stem under `EvalAllPlyParameters.DATASET_ROOT`
    (`data/meshes/<value>.ply`), and the matching reconstruction folder name
    under `validation_inputs_root`/splines/ or /large-steps/, and does not move.

    tyro renders a `--scenes-names` argument by member NAME, not by value, so a
    command line now spells `--scenes-names ARMADILLO` where it used to spell
    `--scenes-names armadillo`.
    """

    ARMADILLO = "armadillo"
    """The armadillo mesh."""
    BUNNY = "bunny"
    """The Stanford bunny mesh."""
    CAR_1 = "car_1"
    """Car mesh, variant 1."""
    CAR_24 = "car_24"
    """Car mesh, variant 24."""
    SPOT = "spot"
    """The spot (cow) mesh."""
    SUZANNE = "suzanne"
    """The Suzanne mesh."""
