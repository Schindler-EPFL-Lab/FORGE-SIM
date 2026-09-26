from enum import Enum


class RealScene(Enum):
    """
    One of the four real (Thermoscenes) reference scenes the NVS sweeps evaluate,
    by identity alone.

    `value` is the directory name these sweeps have always taken as a
    `--scenes-names` entry, under both `validation_inputs_root`/data/ and
    /splines/ or /large-steps/, and does not move.

    tyro renders a `--scenes-names` argument by member NAME, not by value, so a
    command line now spells `--scenes-names CAR` where it used to spell
    `--scenes-names extracted-car`.
    """

    BUILDINGA_SPRING = "extracted-buildingA-spring"
    """The BuildingA-spring facade."""
    CAR = "extracted-car"
    """The car scene."""
    WOODSHED_SUN = "extracted-woodshed-sun"
    """The woodshed scene, sun-lit variant."""
    HEATER_WATER_KETTLE = "extracted_heater_water_kettle"
    """The heater / water-kettle scene."""
