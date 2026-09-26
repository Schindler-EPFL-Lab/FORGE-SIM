from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class SceneFailure:
    """One scene the run could not finish, kept for the summary at the end."""

    scene_name: str
    """Name of the scene, as it was selected or auto-detected."""
    message: str
    """Text of the exception that stopped it."""
