from enum import Enum

from main_scripts.vggt_pipeline.pipeline_steps import PipelineSteps


class PipelineStage(Enum):
    """
    A subset of the pipeline worth running on its own, named for what it produces.

    Every subset that inpaints also repairs the image paths first, since inpainting
    reads the entries that step fixes and the fix is idempotent.
    """

    MASK_FIX_ONLY = "mask_fix_only"
    THERMAL_INPAINT_ONLY = "thermal_inpaint_only"
    WINDOWS_INPAINT_ONLY = "windows_inpaint_only"
    INPAINTING_ONLY = "inpainting_only"
    HEAT_SIM_ONLY = "heat_sim_only"
    INPAINTING_AND_HEAT = "inpainting_and_heat"
    MODAL_ONLY = "modal_only"
    ALL = "all"

    @property
    def steps(self) -> PipelineSteps:
        """
        The five per-step switches this selection turns on.

        :raises ValueError: If a member has been added without a row here.
        """
        if self is PipelineStage.MASK_FIX_ONLY:
            return PipelineSteps(
                mask_fix=True,
                thermal_inpaint=False,
                windows_inpaint=False,
                heat_sim=False,
                modal=False,
            )
        if self is PipelineStage.THERMAL_INPAINT_ONLY:
            return PipelineSteps(
                mask_fix=True,
                thermal_inpaint=True,
                windows_inpaint=False,
                heat_sim=False,
                modal=False,
            )
        if self is PipelineStage.WINDOWS_INPAINT_ONLY:
            return PipelineSteps(
                mask_fix=True,
                thermal_inpaint=False,
                windows_inpaint=True,
                heat_sim=False,
                modal=False,
            )
        if self is PipelineStage.INPAINTING_ONLY:
            return PipelineSteps(
                mask_fix=True,
                thermal_inpaint=True,
                windows_inpaint=True,
                heat_sim=False,
                modal=False,
            )
        if self is PipelineStage.HEAT_SIM_ONLY:
            return PipelineSteps(
                mask_fix=False,
                thermal_inpaint=False,
                windows_inpaint=False,
                heat_sim=True,
                modal=False,
            )
        if self is PipelineStage.INPAINTING_AND_HEAT:
            return PipelineSteps(
                mask_fix=True,
                thermal_inpaint=True,
                windows_inpaint=True,
                heat_sim=True,
                modal=False,
            )
        if self is PipelineStage.MODAL_ONLY:
            return PipelineSteps(
                mask_fix=False,
                thermal_inpaint=False,
                windows_inpaint=False,
                heat_sim=False,
                modal=True,
            )
        if self is PipelineStage.ALL:
            return PipelineSteps(
                mask_fix=True,
                thermal_inpaint=True,
                windows_inpaint=True,
                heat_sim=True,
                modal=True,
            )
        raise ValueError(f"Unhandled stage {self!r}")
