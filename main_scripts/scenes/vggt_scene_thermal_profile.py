from dataclasses import dataclass
from typing import ClassVar

from forge_sim.scripts.configs.thermal_material_config import ThermalMaterialConfig
from main_scripts.scenes.vggt_scene import VGGTScene


@dataclass(frozen=True, kw_only=True)
class VGGTSceneThermalProfile:
    """
    The thermal problem statement of each published VGGT scene: materials and
    geometry scale, not solver code.

    Two solvers handed different conductivities agree or disagree by accident, so
    the IGA runner (`main_scripts/run_vggt_pipeline.py`) and the driver of the
    FEM cross-validation (`main_scripts/compare_all_scenes.py`)
    both read a scene's materials from here rather than each carrying its own
    copy, then pass the plain values on to the FEM solver's own one-scene entry
    point; the registry itself never crosses into `fem/`. Sharing inputs
    strengthens the cross-validation; only sharing the discretisation would
    weaken it, and neither solver is described here.

    Whether a scene is single- or two-material is carried by `window_material` alone
    rather than by a separate label, so the two cannot disagree. `target_length_scale`
    has no default because it is a per-scene physical fact: a scene that is registered
    without stating it would silently inherit another scene's size. `for_scene` is the
    way in.
    """

    _STEEL: ClassVar[ThermalMaterialConfig] = ThermalMaterialConfig(
        conductivity=50.0, density=7850.0, specific_heat=500.0, thickness=0.002
    )
    """Structural steel as a 2 mm skin: the thermal material of the vehicle scenes."""
    _CONCRETE: ClassVar[ThermalMaterialConfig] = ThermalMaterialConfig(
        conductivity=2.0, density=2250.0, specific_heat=900.0, thickness=0.2
    )
    """Concrete as a 0.2 m wall: the thermal material of the building scenes."""
    _WOOD: ClassVar[ThermalMaterialConfig] = ThermalMaterialConfig(
        conductivity=0.15, density=600.0, specific_heat=2400.0, thickness=0.02
    )
    """Softwood as a 20 mm board: the thermal material of the woodshed."""
    _FLUFF: ClassVar[ThermalMaterialConfig] = ThermalMaterialConfig(
        conductivity=0.04, density=100.0, specific_heat=1400.0, thickness=0.02
    )
    """
    Low-density insulation as a 20 mm layer: the thermal material of the lion scene.
    """
    _BUILDING_GLASS: ClassVar[ThermalMaterialConfig] = ThermalMaterialConfig(
        conductivity=0.9, density=2500.0, specific_heat=840.0, thickness=0.01
    )
    """
    Soda-lime glazing as a 10 mm pane: the window material of every scene housed in a
    building: the five facades and the woodshed.

    These are the paper's "All Buildings" row, whose caption groups the woodshed with
    the facades; only the car glazes differently. The values match
    `heat_simulation_two_materials`' own default, and naming them here is what stops an
    edit to that script from silently moving six published scenes.
    """
    _CAR_GLASS: ClassVar[ThermalMaterialConfig] = ThermalMaterialConfig(
        conductivity=0.9, density=2500.0, specific_heat=840.0, thickness=0.001
    )
    """Soda-lime glazing as a 1 mm pane: the window material of the car scene."""

    VGGT_SCENE_THERMAL_PROFILES: ClassVar[dict[VGGTScene, "VGGTSceneThermalProfile"]]
    """
    The thermal profile each published scene was run with, keyed by scene identity.

    Total over `VGGTScene`: every member has an entry. The four BuildNet scenes share
    the building materials with `buildinga_spring` but are kept at their natural scale,
    since they are synthetic and already metric.
    """
    DEFAULT_VGGT_SCENE_THERMAL_PROFILE: ClassVar["VGGTSceneThermalProfile"]
    """
    Thermal profile for a scene not registered above: concrete over the whole surface,
    rescaled to 4 m.

    It names no window material, so an unregistered scene whose dataset carries a
    facade mask is rejected rather than run on a guess; registering the scene is what
    records the parameters its run was produced with.
    """

    base_material: ThermalMaterialConfig
    """
    Thermal properties of the material covering the whole surface. It feeds the
    `--concrete-config` flag whatever the scene is actually made of, that flag being
    named for the building case the two-material entry point was written for.
    """
    window_material: ThermalMaterialConfig | None = None
    """
    Thermal properties of the source-carrying window material, or None for a scene
    that is genuinely single-material and is then solved with no heat source at all.
    A scene whose dataset carries a facade mask is solved with two materials, so
    leaving this unset for one is rejected by the runner rather than defaulted.
    """
    target_length_scale: float | None
    """
    Longest control-point bbox axis [m] to rescale the scene to, or None to keep it
    at the scale the reconstruction came out at.
    """
    window_target_delta_temp: float | None = 0.5
    """
    Temperature rise [K] the window material is scaled to reach over a two-material
    run, for a scene whose `window_material` is set; unused otherwise. Must match
    what the compared IGA run actually used: both two-material entry points
    independently default `target_delta_temp` to this same 0.5, and
    `run_vggt_pipeline` forwards no override, so this default is what every
    registered scene's reference run was produced with.
    """

    def __post_init__(self) -> None:
        if self.target_length_scale is not None and self.target_length_scale <= 0.0:
            raise ValueError(
                f"`target_length_scale` must be positive if provided, got "
                f"{self.target_length_scale}."
            )
        if (
            self.window_target_delta_temp is not None
            and self.window_target_delta_temp <= 0.0
        ):
            raise ValueError(
                f"`window_target_delta_temp` must be positive if provided, got "
                f"{self.window_target_delta_temp}."
            )
        if self.window_material is not None and self.window_target_delta_temp is None:
            raise ValueError(
                f"A profile carrying a `window_material` must state the "
                f"`window_target_delta_temp` its run scales that material to, and "
                f"this one leaves it None. Unlike `target_length_scale`, where None "
                f"means the geometry is left alone, None here does not disable the "
                f"scaling: the source keeps its raw unit magnitude, so the two "
                f"solvers integrate different sources and their disagreement is "
                f"reported as if it were solver error. Window material: "
                f"{self.window_material}."
            )

    @classmethod
    def for_scene(cls, scene: VGGTScene) -> "VGGTSceneThermalProfile":
        """
        The registered thermal profile for `scene`.

        :raises ValueError: If `scene` has no entry in `VGGT_SCENE_THERMAL_PROFILES`.
            Every `VGGTScene` member is registered, so this signals the registry has
            fallen out of sync with the enum, not a bad `scene`.
        """
        profile = cls.VGGT_SCENE_THERMAL_PROFILES.get(scene)
        if profile is None:
            raise ValueError(
                f"No thermal profile registered for {scene.name}. Registered scenes: "
                f"{sorted(member.name for member in cls.VGGT_SCENE_THERMAL_PROFILES)}."
            )
        return profile

    @property
    def is_two_material(self) -> bool:
        """Whether a facade mask splits the surface into base and window materials."""
        return self.window_material is not None

    @property
    def target_length_scale_argument(self) -> str:
        """
        `target_length_scale` as the downstream CLIs receive it.

        tyro renders a `float | None` parameter as a `{None}|FLOAT` value rather than
        a `--no-...` flag, so declining to rescale is spelled by passing the literal
        string `None`. That is what `str(None)` yields anyway, but going through one
        named conversion is what stops the heat and modal steps from agreeing by
        accident.
        """
        return str(self.target_length_scale)


VGGTSceneThermalProfile.VGGT_SCENE_THERMAL_PROFILES = {
    VGGTScene.CAR: VGGTSceneThermalProfile(
        base_material=VGGTSceneThermalProfile._STEEL,
        window_material=VGGTSceneThermalProfile._CAR_GLASS,
        target_length_scale=4.0,
    ),
    VGGTScene.BUILDINGA_SPRING: VGGTSceneThermalProfile(
        base_material=VGGTSceneThermalProfile._CONCRETE,
        window_material=VGGTSceneThermalProfile._BUILDING_GLASS,
        target_length_scale=15.0,
    ),
    VGGTScene.LION: VGGTSceneThermalProfile(
        base_material=VGGTSceneThermalProfile._FLUFF,
        target_length_scale=0.3,
    ),
    VGGTScene.WOODSHED: VGGTSceneThermalProfile(
        base_material=VGGTSceneThermalProfile._WOOD,
        window_material=VGGTSceneThermalProfile._BUILDING_GLASS,
        target_length_scale=2.5,
    ),
    VGGTScene.BUILDNET_1: VGGTSceneThermalProfile(
        base_material=VGGTSceneThermalProfile._CONCRETE,
        window_material=VGGTSceneThermalProfile._BUILDING_GLASS,
        target_length_scale=None,
    ),
    VGGTScene.BUILDNET_2: VGGTSceneThermalProfile(
        base_material=VGGTSceneThermalProfile._CONCRETE,
        window_material=VGGTSceneThermalProfile._BUILDING_GLASS,
        target_length_scale=None,
    ),
    VGGTScene.BUILDNET_3: VGGTSceneThermalProfile(
        base_material=VGGTSceneThermalProfile._CONCRETE,
        window_material=VGGTSceneThermalProfile._BUILDING_GLASS,
        target_length_scale=None,
    ),
    VGGTScene.BUILDNET_4: VGGTSceneThermalProfile(
        base_material=VGGTSceneThermalProfile._CONCRETE,
        window_material=VGGTSceneThermalProfile._BUILDING_GLASS,
        target_length_scale=None,
    ),
}

VGGTSceneThermalProfile.DEFAULT_VGGT_SCENE_THERMAL_PROFILE = VGGTSceneThermalProfile(
    base_material=VGGTSceneThermalProfile._CONCRETE, target_length_scale=4.0
)
