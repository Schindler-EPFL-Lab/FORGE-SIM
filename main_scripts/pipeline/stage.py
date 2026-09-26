from enum import Enum


class Stage(Enum):
    """One step of the end-to-end pipeline, in canonical run order."""

    MESH = "mesh"
    DATASET = "dataset"
    RECONSTRUCT = "reconstruct"
    INPAINT = "inpaint"
    GT = "gt"
    FEM_HEAT = "fem_heat"
    IGA_HEAT = "iga_heat"
    CMP_HEAT = "cmp_heat"
    FEM_MODAL = "fem_modal"
    IGA_MODAL = "iga_modal"
    CMP_MODAL = "cmp_modal"
    SUMMARIZE = "summarize"
