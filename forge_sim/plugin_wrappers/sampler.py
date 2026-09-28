from enum import Enum

from forge_sim.plugin_wrappers.mitsuba_plugin import MitsubaPlugin


class Sampler(MitsubaPlugin):
    """
    Pyhon wrapper around
    https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html
    """

    class Type(Enum):
        """Represents sampler types supported by Mitsuba 3."""

        INDEPENDENT = "independent"
        """Mitsuba 3 default, takes purely random samples."""
        STRATIFIED = "stratified"
        """
        Number of samples should be a square number, subdividies the pixel 
        domain into strata and takes a random sample in each of them. In
        principle this reduced clumping and reduces noise.
        """
        MULTIJITTER = "multijitter"
        """
        Uses advanced strategies to produce spaced out random samples. Good 
        for high quality results and improved convergence. Might tweak
        the number of samples to approach a square number.
        """
        ORTHOGONAL = "orthogonal"
        """
        The number of samples has to be a square of a prime number. 
        Further refines concepts introduced in `MULTIJITTER`.
        """
        LDSAMPLER = "ldsampler"
        """
        Legacy Mitsuba 1 sampler. Requires a power of 2 number of samples.
        Generally has more clumping than `MULTIJITTER` or `ORTHOGONAL`, but 
        higher quality than `STRATIFIED`.
        """

    def __init__(
        self,
        type: "Sampler.Type" = Type.INDEPENDENT,
        sample_count: int = 16,
        random_seed: int = 0,
    ) -> None:
        """
        Note that `sample_count` value can be overriden by passing an explicit
        `spp` value at a later time (but is not recommended, see `PerspectiveCamera`
        for more details).

        :raises ValueError: if `sample_count` is non-positive or `random_seed`
        is a negative integer.
        """
        if sample_count <= 0:
            raise ValueError(
                f"'sample_count' must be a positive integer, got {sample_count}"
            )

        if random_seed < 0:
            raise ValueError(
                f"'random_seed' must be a non-negative integer, got {random_seed}"
            )

        self._type = type
        self._sample_count = sample_count
        self._random_seed = random_seed

    @property
    def sample_count(self):
        """
        Returns how many samples the sampler will draw if rendering is invoked.
        """
        return self._sample_count

    @property
    def mitsuba_dict(self) -> dict[str, str | int]:
        """
        Get the Mitsuba compatible dictionary representation of this sampler

        :returns: A dictionary that is passed to the Mitsuba scene description dict.
        """
        return {
            "type": self._type.value,
            "sample_count": self._sample_count,
            "seed": self._random_seed,
        }
