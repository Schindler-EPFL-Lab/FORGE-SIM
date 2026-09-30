from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from forge_sim.scripts.end_to_end.heat.thermal_sim_config import ThermalSimConfig
from main_scripts.scenes.e2e_scene import E2EScene


@dataclass(frozen=True, kw_only=True)
class E2ESceneProfile:
    """
    What one end-to-end target needs that is not the same for all of them.

    Only the ground-truth temperature rule lives here so far, because it is the one
    input that differs per target and that nothing else records: it is the experiment's
    controlled field, shared by the dataset render and the ground-truth bake, and two
    runs compared against each other must agree on it.

    It is stated per scene rather than left to the config default so that reading this
    table answers which field a target was run with. It was previously inferred from
    whether the target's mesh happened to carry UVs, which made the answer a property
    of the loader and moved every heat number for one target when the loader changed.

    The pre-built spline is per scene for a stronger reason: a recovered spline is
    only valid for the scene it was recovered from, so a target with none registered
    must reconstruct rather than inherit another target's spline from the config
    default.
    """

    SCENE_PROFILES: ClassVar[dict[E2EScene, "E2ESceneProfile"]] = {}
    """Every registered target, in the order the recorded runs report them."""

    mesh_path: Path
    """Ground-truth source mesh, relative to the repository root."""
    prebuilt_spline_path: Path | None
    """
    Recovered spline for this target, relative to the repository root, or `None` when
    none is shipped and the pipeline must reconstruct it from the dataset.
    """
    temperature_field_source: ThermalSimConfig.TemperatureFieldSource
    """Which coordinates this target's ground-truth temperature is evaluated in."""

    @classmethod
    def for_scene(cls, scene: E2EScene) -> "E2ESceneProfile":
        """
        The registered profile for `scene`.

        :raises ValueError: If `scene` has no entry. Every member is registered, so
            this signals the table has fallen out of step with the enum.
        """
        profile = cls.SCENE_PROFILES.get(scene)
        if profile is None:
            raise ValueError(
                f"No E2ESceneProfile registered for {scene.name}. Registered: "
                f"{sorted(member.name for member in cls.SCENE_PROFILES)}."
            )
        return profile


_WORLD = ThermalSimConfig.TemperatureFieldSource.WORLD_CHECKER
_UV = ThermalSimConfig.TemperatureFieldSource.UV_TEXTURE

E2ESceneProfile.SCENE_PROFILES = {
    E2EScene.SUZANNE: E2ESceneProfile(
        mesh_path=Path("data/meshes/suzanne.ply"),
        prebuilt_spline_path=Path("data/splines/spline_suzanne.json"),
        temperature_field_source=_WORLD,
    ),
    E2EScene.SPOT: E2ESceneProfile(
        mesh_path=Path("data/meshes/spot.ply"),
        prebuilt_spline_path=None,
        temperature_field_source=_UV,
    ),
    E2EScene.BUNNY: E2ESceneProfile(
        mesh_path=Path("data/meshes/bunny.ply"),
        prebuilt_spline_path=None,
        temperature_field_source=_WORLD,
    ),
    E2EScene.ARMADILLO: E2ESceneProfile(
        mesh_path=Path("data/meshes/armadillo.ply"),
        prebuilt_spline_path=None,
        temperature_field_source=_WORLD,
    ),
}
