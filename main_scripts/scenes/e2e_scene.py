from enum import Enum


class E2EScene(Enum):
    """
    One of the four synthetic targets the end-to-end validation covers.

    Keyed by identity alone; `E2ESceneProfile` holds what each one needs. The set is
    fixed by the recorded runs the published tables report, so a fifth target is a new
    experiment rather than a new member.

    It lives in `main_scripts/`, not in either library: enumerating the datasets an
    experiment ran on is the driver's subject, and `forge_sim` and `fem` provide the
    machinery for one target at a time without naming any (`CLAUDE.md` SS3).

    tyro renders a `--scenes` argument by member NAME, so a command line spells
    `SUZANNE` where a directory is named `suzanne`.
    """

    SUZANNE = "suzanne"
    """Blender's Suzanne. Carries an unwrap, which is deliberately not used."""
    SPOT = "spot"
    """Keenan Crane's Spot: the one target whose temperature comes from its UVs."""
    BUNNY = "bunny"
    """The Stanford bunny, hole-filled before use."""
    ARMADILLO = "armadillo"
    """The Stanford armadillo."""
