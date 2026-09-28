from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from forge_sim.scripts.configs.structural_material_config import (
    StructuralMaterialConfig,
)
from main_scripts.scenes.vggt_scene import VGGTScene
from main_scripts.scenes.vggt_scene_thermal_profile import VGGTSceneThermalProfile


@dataclass(frozen=True, kw_only=True)
class VGGTSceneProfile:
    """
    Per-scene knobs that differ between buildings, vehicles and sculptures.

    One registry for the whole scene, not one per consumer: it used to be two
    classes of the same name, `main_scripts/vggt_pipeline/scene_profile.py` and
    `fem/end_to_end/heat_comparison/scene_profile.py`, each carrying only the
    fields its own side needed and unable to import the other. This carries the
    union, and lives in `main_scripts/`: the registry of published scenes is
    this package's own subject (`CLAUDE.md` SS3); `forge_sim/` and `fem/` are
    libraries that take a scene's values as ordinary arguments and never need to
    know the registry exists.

    The materials and the geometry scale are not among the per-scene fields:
    those are the thermal problem statement, held in `thermal` so the FEM
    cross-validation reads the same values the IGA pipeline runs with rather
    than a second copy of them. `for_scene` is the way in.
    """

    _STEEL_STRUCT: ClassVar[StructuralMaterialConfig] = StructuralMaterialConfig(
        youngs_modulus=210e9,
        poisson_ratio=0.3,
        density=7850.0,
        thickness=0.002,
        shear_correction_factor=5.0 / 6.0,
    )
    """
    Structural steel, 2 mm: the shell material of the vehicle scenes. The
    thickness matches the thermal side's steel preset, so a scene's thermal and
    modal runs describe the same skin.
    """
    _CONCRETE_STRUCT: ClassVar[StructuralMaterialConfig] = StructuralMaterialConfig(
        youngs_modulus=40e9,
        poisson_ratio=0.2,
        density=2400.0,
        thickness=0.2,
        shear_correction_factor=5.0 / 6.0,
    )
    """Concrete, 0.2 m: the shell material of the building scenes."""
    _WOOD_STRUCT: ClassVar[StructuralMaterialConfig] = StructuralMaterialConfig(
        youngs_modulus=11e9,
        poisson_ratio=0.35,
        density=600.0,
        thickness=0.02,
        shear_correction_factor=5.0 / 6.0,
    )
    """Softwood, 20 mm: the shell material of the woodshed."""
    _WOOL_STRUCT: ClassVar[StructuralMaterialConfig] = StructuralMaterialConfig(
        youngs_modulus=3.0e6,
        poisson_ratio=0.3,
        density=100.0,
        thickness=0.02,
        shear_correction_factor=5.0 / 6.0,
    )
    """
    Wool, 20 mm: the shell material of the lion scene, paired with its thermal
    counterpart's fluff insulation.
    """

    SCENE_PROFILES: ClassVar[dict[VGGTScene, "VGGTSceneProfile"]]
    """
    The profile each published scene was run with.

    Total over `VGGTScene`: every member has an entry. A test in `tests/` pins
    this registry's key set equal to `VGGTSceneThermalProfile`'s, so a scene
    carrying a shell material but no thermal profile, or the reverse, fails
    at import rather than only once its stage is reached.
    """
    DEFAULT_PROFILE: ClassVar["VGGTSceneProfile"]
    """
    Profile for a scene not listed above: concrete over the whole surface,
    rescaled to 4 m, at the single-material IGA stem. Names neither a window
    nor a shell material, so an unregistered scene's masked heat run and any
    modal run are rejected rather than run on a guess. Not keyed by a
    `VGGTScene`; a caller whose raw scene name matches none falls back to
    this directly rather than through `for_scene`.
    """

    directory: Path | None
    """
    Scene's directory under the dataset root, relative to it. Equal to the
    scene's own identity string for every scene except `VGGTScene.BUILDNET_4`,
    which lives one directory deeper on disk (`buildnet-4/buildnet-4`).
    `None` only for `DEFAULT_PROFILE`, which has no registered entry and so
    no directory to name; its caller falls back to the scene's own literal
    name instead.
    """
    iga_base_name: str = "heat_simulation"
    """
    Stem of the IGA reference heat run's `.h5` / `.xdmf` pair inside the scene
    directory. Held rather than derived from `thermal.is_two_material`: the
    producer routes on whether the dataset carries a facade mask, so a scene
    with a window material and no mask would silently get the wrong stem from
    a derived value. Defaults to the single-material stem, which is also
    correct for `DEFAULT_PROFILE`, since it names no window material.
    """
    thermal: VGGTSceneThermalProfile
    """
    Materials and geometry scale this scene's heat runs are defined by. The FEM
    heat comparison reads the plain values off this profile and passes them to
    its own one-scene entry point; it never reads this class.
    """
    h_refinement_factor: int = 4
    """Knot-span refinement applied to the reconstruction before inpainting."""
    inpaint_initial_color: float = 0.1
    """Value the inpainted field is initialised to before optimisation."""
    structural_material: StructuralMaterialConfig | None = None
    """
    Shell properties for the modal analysis. A scene has no default shell to
    fall back on, so leaving this unset is rejected rather than defaulted for
    any scene the modal step is asked to run.
    """
    modal_target_spans: int = 128
    """
    Target IGA modal knot spans per direction, pinned across splines regardless
    of the base the reconstruction came out at. 128 escapes the membrane
    locking a thin-shell discretisation suffers when under-refined; the VGGT
    splines are 64-base, so this is a factor of two.
    """

    @classmethod
    def for_scene(cls, scene: VGGTScene) -> "VGGTSceneProfile":
        """
        The registered profile for `scene`.

        :raises ValueError: If `scene` has no entry in `SCENE_PROFILES`. Every
            `VGGTScene` member is registered, so this signals the registry has
            fallen out of sync with the enum, not a bad `scene`. A caller
            holding a raw scene name rather than a `VGGTScene` resolves it with
            `VGGTScene.for_value` first, falling back to `DEFAULT_PROFILE`
            directly when that returns `None` rather than reaching this method.
        """
        profile = cls.SCENE_PROFILES.get(scene)
        if profile is None:
            raise ValueError(
                f"No VGGTSceneProfile registered for {scene.name}. Registered "
                f"scenes: {sorted(member.name for member in cls.SCENE_PROFILES)}."
            )
        return profile


VGGTSceneProfile.SCENE_PROFILES = {
    VGGTScene.CAR: VGGTSceneProfile(
        directory=Path("car"),
        iga_base_name="heat_simulation_two_materials",
        thermal=VGGTSceneThermalProfile.for_scene(VGGTScene.CAR),
        structural_material=VGGTSceneProfile._STEEL_STRUCT,
    ),
    VGGTScene.BUILDINGA_SPRING: VGGTSceneProfile(
        directory=Path("buildinga_spring"),
        iga_base_name="heat_simulation_two_materials",
        thermal=VGGTSceneThermalProfile.for_scene(VGGTScene.BUILDINGA_SPRING),
        structural_material=VGGTSceneProfile._CONCRETE_STRUCT,
    ),
    VGGTScene.LION: VGGTSceneProfile(
        directory=Path("lion"),
        iga_base_name="heat_simulation",
        thermal=VGGTSceneThermalProfile.for_scene(VGGTScene.LION),
        structural_material=VGGTSceneProfile._WOOL_STRUCT,
    ),
    VGGTScene.WOODSHED: VGGTSceneProfile(
        directory=Path("woodshed"),
        iga_base_name="heat_simulation_two_materials",
        thermal=VGGTSceneThermalProfile.for_scene(VGGTScene.WOODSHED),
        structural_material=VGGTSceneProfile._WOOD_STRUCT,
    ),
    VGGTScene.BUILDNET_1: VGGTSceneProfile(
        directory=Path("buildnet-1"),
        iga_base_name="heat_simulation_two_materials",
        thermal=VGGTSceneThermalProfile.for_scene(VGGTScene.BUILDNET_1),
        structural_material=VGGTSceneProfile._CONCRETE_STRUCT,
    ),
    VGGTScene.BUILDNET_2: VGGTSceneProfile(
        directory=Path("buildnet-2"),
        iga_base_name="heat_simulation_two_materials",
        thermal=VGGTSceneThermalProfile.for_scene(VGGTScene.BUILDNET_2),
        structural_material=VGGTSceneProfile._CONCRETE_STRUCT,
    ),
    VGGTScene.BUILDNET_3: VGGTSceneProfile(
        directory=Path("buildnet-3"),
        iga_base_name="heat_simulation_two_materials",
        thermal=VGGTSceneThermalProfile.for_scene(VGGTScene.BUILDNET_3),
        structural_material=VGGTSceneProfile._CONCRETE_STRUCT,
    ),
    # buildnet-4 lives one directory deeper than the rest, so its `directory` is the
    # nested path while its thermal profile is registered under the scene's own name.
    VGGTScene.BUILDNET_4: VGGTSceneProfile(
        directory=Path("buildnet-4/buildnet-4"),
        iga_base_name="heat_simulation_two_materials",
        thermal=VGGTSceneThermalProfile.for_scene(VGGTScene.BUILDNET_4),
        structural_material=VGGTSceneProfile._CONCRETE_STRUCT,
    ),
}

VGGTSceneProfile.DEFAULT_PROFILE = VGGTSceneProfile(
    directory=None,
    thermal=VGGTSceneThermalProfile.DEFAULT_VGGT_SCENE_THERMAL_PROFILE,
)
