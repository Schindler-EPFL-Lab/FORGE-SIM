from enum import Enum


class VGGTScene(Enum):
    """
    One of the eight published VGGT / BuildNet3D scenes, by identity alone.

    Two registries key themselves by it: `VGGTSceneThermalProfile`'s and
    `VGGTSceneProfile`'s, both in `main_scripts.scenes`. Keyed by raw strings
    instead, a scene name exists nowhere as a checkable type, and a typo in a
    `--scenes` argument parses cleanly and fails only once the dataset has already
    loaded.

    Carries no material, path or geometry data: those stay in the per-purpose
    registries that key off this enum, so a scene is added or renamed in exactly one
    of them rather than in three places at once. `value` is the on-disk / JSON
    identity string this project used before this type existed, and does not move.

    It lives in `main_scripts/`, not `forge_sim/` or `fem/`: the set of
    published scenes is a dataset registry, which is `main_scripts`' own subject
    (`CLAUDE.md` SS3). Neither solver library needs to know a scene's identity to
    run a comparison for it; only the driver that resolves which scenes exist and
    calls the library once per scene does.

    tyro renders a `--scenes` argument by member NAME, not by value, so a command
    line now spells `--scenes BUILDNET_1` where it used to spell
    `--scenes buildnet-1`.
    """

    CAR = "car"
    """The car scene: steel body, glazed windshield."""
    BUILDINGA_SPRING = "buildinga_spring"
    """The BuildingA-spring facade."""
    LION = "lion"
    """The lion sculpture: single-material, no facade mask."""
    WOODSHED = "woodshed"
    """The woodshed: timber walls, glazed openings."""
    BUILDNET_1 = "buildnet-1"
    """BuildNet3D synthetic facade 1."""
    BUILDNET_2 = "buildnet-2"
    """BuildNet3D synthetic facade 2."""
    BUILDNET_3 = "buildnet-3"
    """BuildNet3D synthetic facade 3."""
    BUILDNET_4 = "buildnet-4"
    """BuildNet3D synthetic facade 4."""

    @classmethod
    def for_value(cls, value: str) -> "VGGTScene | None":
        """
        The scene whose `value` is `value`, or `None` when no scene carries it.

        The constructor raises on an unknown value, which makes a caller testing
        whether a directory name is a known scene write a `try`/`except` around
        what is a membership question. This answers it directly.
        """
        return next((scene for scene in cls if scene.value == value), None)
