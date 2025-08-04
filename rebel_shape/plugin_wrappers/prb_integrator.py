from enum import Enum

from rebel_shape.plugin_wrappers.mitsuba_plugin import MitsubaPlugin


class GuidingOptions(Enum):
    """
    Ray guiding strucures supproted by Mitsuba 3. Guiding can improve convergence
    in complex scenes with many reflections, but can hurt it otherwise.
    """

    NONE = "none"
    """Do not use a guiding structure. Can improve performance for simple scenes"""
    GRID = "grid"
    """Simple guiding structure suitable when there is not too much empty space"""
    OCTREE = "octree"
    """
    Mitsuba 3 default. The octree data structure is more flexible than a grid.
    Recommended for more complex scenes with many ray samples.
    """


class PRBIntegrator(MitsubaPlugin):
    """
    Pyhon wrapper around Projective sampling Path Replay Backpropagation (PRB)
    https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html

    """

    def __init__(
        self,
        max_bounces: int = 20,
        bounces_without_russian_roulette: int = 5,
        spp_continuous: int = 16,
        spp_projective: int = 16,
        spp_indirect: int = 16,
        guiding: GuidingOptions = GuidingOptions.OCTREE,
        guiding_rounds: int = 1,
    ) -> None:
        """
        `max_bounces` and `bounces_without_russian_roulette` determine the accuracy
        of global illumination and increase compute time; open scenes can likely do
        with a lower number of bounces. Setting `max_bounces = 2` is equivalent to
        using 'direct projective' Mitsuba integrator (not implemented here). It is
        recommended that `max_bounces > 4` and `bounces_without_russian_roulette > 3`
        to avoid excessive variance (unless there is a special reason).

        spp is a shorthand for Samples Per Pixel

        Since Mitsuba uses Monte Carlo simulation to estimate gradients instead of
        pure AD, one can tune how many samples are used to estimate the 3 components
        of the integral that defines the gradient. `spp_continuous` determines how
        mant samples are dedicated continuous properties such as textures, while
        `spp_projective` samples dedicated to primary visibility changes (e.g. vertex
        positions change and something is no longer visible) and `spp_indirect`
        indirect visibility changes (e.g. changing the vertex position leads to the
        shape occluding itself). Note that explicitly passing `spp` in `mi.render`
        will override these settings.

        Finally, for rendering difficult scenes or SDS paths it is useful to add
        path guiding ('steers' rays where they make a bigger contribution). The
        larger the number of `guiding_rounds` the better the structure, but make
        sure that this does not take up too much compute time!
        """
        if max_bounces < 1:
            raise ValueError("There has to be at least one bounce to obtain gradients")

        if bounces_without_russian_roulette < 1:
            raise ValueError("There has to be at least one bounce to obtain gradients")

        if spp_continuous < 1:
            raise ValueError("'spp_continuous' must be a positive integer.")
        if spp_projective < 1:
            raise ValueError("'spp_projective' must be a positive integer.")
        if spp_indirect < 1:
            raise ValueError("'spp_indirect' must be a positive integer.")

        if guiding_rounds < 1 and guiding != GuidingOptions.NONE:
            raise ValueError(
                "There has to be at least one guiding round if guiding is enabled."
            )

        self._max_bounces = max_bounces
        self._bounces_without_russian_roulette = bounces_without_russian_roulette
        self._spp_continuous = spp_continuous
        self._spp_projective = spp_projective
        self._spp_indirect = spp_indirect
        self._guiding = guiding
        self._guiding_rounds = guiding_rounds

    @property
    def mitsuba_dict(self) -> dict[str, str | int]:
        """
        Get the Mitsuba compatible dictionary representation of this integrator

        :returns: A dictionary that is passed to the Mitsuba scene description dict.
        """
        return {
            "type": "prb_projective",
            "max_depth": self._max_bounces,
            "rr_depth": self._bounces_without_russian_roulette,
            "sppc": self._spp_continuous,
            "sppp": self._spp_projective,
            "sppi": self._spp_indirect,
            "guiding": self._guiding.value,
            "guiding_rounds": self._guiding_rounds,
        }
