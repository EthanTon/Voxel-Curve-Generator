from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..Core.block.states import BlockState
from .classify import DEFAULT_LOOSE, DEFAULT_TOPS, BlockNames
from .slopes import PASSES, POCKET, PORTAL, REPAIR_PASSES, RIDGE_WIDTH, ROCK_BAND, WALL_BAND
from .subsurface import REACH, ROCK_BODY, STONE_SHARE, TILE, RockRule
from .terrain import SUBSURFACE
from ..Core.coordinates import ColumnPos
from ..Format.schematics import parse_state_key
from .layers import KERNELS, Blend
from .materialize import Strata
from .surface_mask import SurfaceMaskSpec, read_surface_mask


class Intensity(Enum):
    NONE = "none"
    CLEAR = "clear"
    SHAPE = "shape"


class Look(Enum):
    NATURAL = "natural"
    ENGINEERED = "engineered"
    MINIMAL = "minimal"


class Action(Enum):
    SLOPE_DOWN = "slope_down"
    SLOPE_UP = "slope_up"
    LEVEL = "level"
    GAUSSIAN = "gaussian"
    CIRCULAR = "circular"
    MOUND = "mound"
    OFFSET = "offset"
    KEEP = "keep"


@dataclass(frozen=True, slots=True)
class Keepout:
    least: ColumnPos
    most: ColumnPos
    reason: str = ""

    def holds(self, x: int, z: int) -> bool:
        return self.least.x <= x <= self.most.x and self.least.z <= z <= self.most.z


# A step is measured from the nearest track column's lowest block ("footing")
# or highest ("crest").
BASES = ("footing", "crest")


@dataclass(frozen=True, slots=True)
class Step:
    action: Action
    blend: Blend = Blend.RAISE
    ratio: float = 0.0
    height: float = 0.0
    width: float = 0.0
    reach: float = 0.0
    feather: float | None = None
    softness: float | None = None
    base: str = "footing"
    # Level ground this wide beside the track before a slope starts; the
    # plan's shoulder if unset.
    shoulder: float | None = None


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    steps: tuple[Step, ...]


@dataclass(frozen=True, slots=True)
class Span:
    start: str | ColumnPos | None = None
    stop: str | ColumnPos | None = None
    intensity: Intensity = Intensity.SHAPE
    profile: str = "at_grade"
    transition: int = 8
    order: int = 0
    # Clearance above the build; None uses the plan's, 0 clears only
    # enclosed space (tunnels).
    headroom: int | None = None
    cross_section: str | None = None
    section: str | None = None
    surface_mask: SurfaceMaskSpec | None = None

    def __post_init__(self) -> None:
        if self.section is not None:
            if not isinstance(self.section, str) or not self.section:
                raise ValueError("a terraform span's section must be a nonempty name")
            if self.start is not None or self.stop is not None:
                raise ValueError("a terraform span uses either section or from/to")
        elif self.start is None or self.stop is None:
            raise ValueError("a terraform span needs section or both from and to")


@dataclass(frozen=True, slots=True)
class Follows:
    fitting: str
    profile: str = "footing"
    intensity: Intensity = Intensity.SHAPE
    reach: float = 0.0
    order: int = 20
    reach_mode: str = "fixed"
    reach_min: float = 6.0
    reach_max: float = 22.0
    height_response: float = 0.08
    terrain_response: float = 0.2
    surface_mask: SurfaceMaskSpec | None = None


@dataclass(frozen=True, slots=True)
class SmoothingTransition:
    source: str
    target: str
    behavior: str = "auto"
    reach: float | None = None
    side: str = "both"
    bidirectional: bool = True


@dataclass(frozen=True, slots=True)
class Smoothing:
    mode: str = "fixed"
    size: float | None = None
    min_size: float = 0.75
    max_size: float = 3.5
    strength: float = 1.0
    passes: int = 1
    terrain_response: float = 1.0
    kernel: str = "gaussian"
    # Direction of material movement, separate from fixed/adaptive brush size.
    behavior: str = "auto"
    stable_change: float = 1.0
    decision_threshold: float = 1.0
    transition_reach: float = 12.0
    adaptive_transitions: bool = True
    preserve_cliffs: bool = True
    # Broad native-contour blending, then bounded block-face repair.
    contour_strength: float = 0.65
    contour_scale: float = 3.0
    contact_blend: float = 1.0
    face_iterations: int = 2
    face_min_exposed: int = 3
    face_depth: int = 4
    profiles: Mapping[str, str] = field(default_factory=dict)
    transitions: tuple[SmoothingTransition, ...] = ()


@dataclass(frozen=True, slots=True)
class Brush:
    # A smoothing brush over the finished ground, as a player would run one:
    # `passes` Gaussian passes of `radius` over every column the works touch
    # and `margin` beyond, fading out over `falloff`. The track, its shoulder
    # and protected ground hold their height.
    radius: float = 1.5
    passes: int = 3
    margin: float = 4.0
    falloff: float = 6.0


@dataclass(frozen=True, slots=True)
class Embankment:
    adaptive: bool = False
    height_response: float = 0.06
    terrain_response: float = 0.2
    max_ratio: float = 4.5
    # Blocks across per block down for a fill's side slope; None takes it
    # from the surrounding terrain.
    ratio: float | None = None
    # A bank's end spills down under the first span of a bridge.
    spill: bool = True
    # The blocks that only ever top the ground.
    tops: BlockNames = DEFAULT_TOPS
    # A fill copies the commonest block at each depth under the columns
    # within `layering_reach` of it, tallied `layering_tile` columns square
    # at a time.
    layering_reach: int = REACH
    layering_tile: int = TILE
    # Eased, exact linear, or varied natural contours for banks and cuts.
    shape: str = "eased"
    water_spill: bool = False
    # Extra room for a bank's toe, without widening cuts into a hillside.
    max_reach: int = 0


@dataclass(frozen=True, slots=True)
class Cutting:
    # Blocks across per block up for a cut's side slope; None takes it from
    # the surrounding terrain.
    ratio: float | None = None
    # Stone is cut straight down beside the track, the soil above it set
    # back at the ratio.
    rock: bool = True
    # The ground that is soil rather than rock; rock starts where the
    # `rock_body` blocks down from it are at least `stone_share` rock, and a
    # face is rock where `stone_share` of it is.
    loose: BlockNames = DEFAULT_LOOSE
    rock_body: int = ROCK_BODY
    stone_share: float = STONE_SHARE
    # How far out from the formation a cut through rock stands as a face,
    # its soil stripped to the rock's top, and how far behind a trench's
    # walls the ground is held at their top.
    rock_band: float = ROCK_BAND
    wall_band: float = WALL_BAND
    # New rock faces near a portal can be steeper than loose soil. This
    # bounds their ledges, rather than pinning a tall vertical rock remnant.
    rock_drop: int = 2

    @property
    def rock_rule(self) -> RockRule:
        return RockRule(self.loose, self.rock_body, self.stone_share)


@dataclass(frozen=True, slots=True)
class Sculpt:
    # Shape every column the works change from the structure's contacts:
    # blocks of rise a column cut through rock and through the soil over
    # it, and of fall a column down a bank, each wandering across its range,
    # with rock faces broken into ledges this many blocks apart ((0, 0) for
    # none).
    enabled: bool = False
    rock_rise: tuple[float, float] = (2.5, 4.0)
    soil_rise: tuple[float, float] = (0.25, 0.5)
    bank_rise: tuple[float, float] = (0.25, 0.45)
    ledges: tuple[int, int] = (0, 0)
    # Read how steep the land's own hillsides run round each cut and keep
    # its faces within that, inside the ranges above.
    adapt: bool = True
    # A separate finishing pass over existing bank fill after sculpting.
    bank_radius: float = 3.0
    bank_passes: int = 4


@dataclass(frozen=True, slots=True)
class Slopes:
    # How far out from a tunnel's lining its portal may stand as a face, how
    # many passes each phase of the repair may take, how far round a drop it
    # could not close held ground may keep a face against it, how many
    # passes the slopes may take to spread out from the track, and how many
    # columns across a ridge or a groove may be and still be levelled.
    portal_reach: float = PORTAL
    repair_passes: int = REPAIR_PASSES
    pocket: int = POCKET
    cone_passes: int = PASSES
    ridge_width: int = RIDGE_WIDTH
    # Maximum rise between adjacent columns in newly shaped terrain.
    max_drop: int = 1
    # How much new ground wanders from a regular slope, 0 to 1: a portal's
    # graded hill runs up to 1 + variation times gentler in places, and the
    # saved ground's bumps show through the works, so steps vary in width
    # and contours stop echoing the track.
    variation: float = 0.0
    seed: int = 0


@dataclass(frozen=True, slots=True)
class TerrainMask:
    preserve_water: bool = False
    water_buffer: int = 0
    preserve_structures: bool = False
    slope_to_water: bool = False


@dataclass(frozen=True, slots=True)
class CrossSection:
    name: str
    left: str = "auto"
    right: str = "auto"
    sample_distance: int = 8
    sample_window: int = 2
    cut_threshold: float = 2.0
    fill_threshold: float = 2.0
    bridge_threshold: float = 9.0
    water_mode: str = "bridge"
    cut_base: str = "footing"
    min_run: int = 20


@dataclass(frozen=True, slots=True)
class Foliage:
    clear: bool = True
    revegetate: bool = False


@dataclass(frozen=True, slots=True)
class Bridge:
    # Smooth the saved land independently of the deck's elevation.
    natural_only: bool = False
    radius: float = 2.0
    strength: float = 0.5
    max_change: float = 2.0
    reach: float = 4.0
    clear_underneath: bool = True


DEFAULT_STRATA = Strata(
    surface=BlockState.create("minecraft:grass_block"),
    subsurface=BlockState.create("minecraft:dirt"),
    filler=BlockState.create("minecraft:stone"),
)


@dataclass(frozen=True, slots=True)
class Stretch:
    start: int
    stop: int
    intensity: Intensity
    profile: str
    transition: int
    order: int
    headroom: int | None = None
    cross_section: str | None = None
    surface_mask: SurfaceMaskSpec | None = None


@dataclass(frozen=True, slots=True)
class Tunnel:
    # Separate from the rail-height slope profiles: only exposed lining and
    # already shaped mouths seed changes in the exterior native terrain.
    sides: bool = False
    reach: float = 12.0
    max_change: int = 4
    min_cover: int = 2


@dataclass(frozen=True, slots=True)
class Plan:
    look: Look = Look.NATURAL
    domain: str = "terrain"
    default: Intensity = Intensity.NONE
    margin: int = 24
    reach: float = 32.0
    # Side slopes hold their ratio until this far inside the reach, then fade
    # out; None fades them across the whole reach.
    feather: float | None = None
    # Level ground this wide beside the track before a side slope starts.
    shoulder: float = 0.0
    headroom: int = 6
    world: Path | None = None
    dimension: str | None = None
    strata: Strata = DEFAULT_STRATA
    foliage: Foliage = Foliage()
    bridge: Bridge = Bridge()
    tunnel: Tunnel = Tunnel()
    smoothing: Smoothing = Smoothing()
    brush: Brush = Brush()
    embankment: Embankment = Embankment()
    cutting: Cutting = Cutting()
    sculpt: Sculpt = Sculpt()
    slopes: Slopes = Slopes()
    terrain_mask: TerrainMask = TerrainMask()
    surface_mask: SurfaceMaskSpec | None = None
    spans: tuple[Span, ...] = ()
    follows: tuple[Follows, ...] = ()
    keepouts: tuple[Keepout, ...] = ()
    profiles: Mapping[str, Profile] = field(default_factory=dict)
    cross_sections: Mapping[str, CrossSection] = field(default_factory=dict)
    # Include rail columns beyond the sections' footprint in the ground's
    # alignment, with their support surface one block below the rail.
    include_rails: bool = False
    # Let neighbouring earthworks continue through other section profiles.
    bleed: bool = False
    # Preserve natural ground around named fittings, such as bridge piers.
    preserve_fittings: tuple[str, ...] = ()
    fitting_buffer: int = 0
    # Waterlog capable build blocks placed below the saved water surface.
    waterlog: bool = False
    # The usable track space clears native blocks even in protected columns,
    # and stays dry instead of inheriting the saved water surface.
    clear_track: bool = False
    # Optional original terrain, separate from a world already pasted into.
    source_world: Path | None = None
    # Blocks of a custom terrain that are ground, besides those known by name.
    ground: BlockNames = BlockNames(())

    @property
    def terrain_world(self) -> Path | None:
        return self.source_world or self.world

    def stretches(
        self, locate, length: int,
        section_spans: Mapping[str, Sequence[tuple[int, int]]] | None = None,
    ) -> tuple[Stretch, ...]:
        if not self.spans:
            return (Stretch(0, length - 1, self.default, "at_grade", 0, 0,
                            surface_mask=self.surface_mask),)
        stretches = []
        for span in self.spans:
            if span.section is None:
                ranges = (tuple(sorted((locate(span.start), locate(span.stop)))),)
            else:
                if section_spans is None or span.section not in section_spans:
                    raise ValueError(
                        f"terraform span names section {span.section!r}, "
                        "but that section has no assigned path intervals"
                    )
                ranges = section_spans[span.section]
            for start, stop in ranges:
                stretches.append(Stretch(
                    start, stop, span.intensity, span.profile, span.transition,
                    span.order, span.headroom, span.cross_section,
                    span.surface_mask,
                ))
        return tuple(stretches)

    def profile(self, name: str) -> Profile:
        found = self.profiles.get(name, BUILT_IN.get(name))
        if found is None:
            known = ", ".join(sorted({*self.profiles, *BUILT_IN}))
            raise KeyError(f"no profile named {name!r}. Known profiles: {known}")
        return found

    def cross_section(self, name: str) -> CrossSection:
        try:
            return self.cross_sections[name]
        except KeyError:
            raise KeyError(f"no cross section named {name!r}") from None


FLAT_SLOPE_FOOTING = (
    Step(Action.LEVEL, Blend.SET, reach=3.0, feather=1.0),
    Step(Action.SLOPE_DOWN, Blend.RAISE, ratio=2.0, feather=2.0),
)
GAUSSIAN_FOOTING = (Step(Action.GAUSSIAN, Blend.SET, width=3.0, feather=2.0),)
CIRCULAR_FOOTING = (Step(Action.CIRCULAR, Blend.SET, width=3.0, feather=0.0),)


BUILT_IN = {
    "at_grade": Profile(
        "at_grade",
        (
            Step(Action.SLOPE_DOWN, Blend.RAISE),
            Step(Action.SLOPE_UP, Blend.LOWER),
        ),
    ),
    # Filled under and beside, and cut back where the ground beside stands
    # higher, rather than left as a face at the edge of the clearance.
    "embankment": Profile(
        "embankment",
        (
            Step(Action.SLOPE_DOWN, Blend.RAISE),
            Step(Action.SLOPE_UP, Blend.LOWER),
        ),
    ),
    "cutting": Profile("cutting", (Step(Action.SLOPE_UP, Blend.LOWER),)),
    "trench": Profile("trench", (Step(Action.SLOPE_UP, Blend.LOWER),)),
    "bridge": Profile("bridge", (Step(Action.KEEP, Blend.RESTORE),)),
    "tunnel": Profile("tunnel", (Step(Action.KEEP, Blend.RESTORE),)),
    # Cut back, or filled up, to the top of retaining walls, so the walls
    # hold the ground and never stand up out of it.
    "retained_cutting": Profile(
        "retained_cutting",
        (
            Step(Action.SLOPE_UP, Blend.LOWER, base="crest"),
            Step(Action.SLOPE_DOWN, Blend.RAISE, base="crest"),
        ),
    ),
    "pad": Profile("pad", (Step(Action.LEVEL, Blend.SET),)),
    "footing_flat_slope": Profile("footing_flat_slope", FLAT_SLOPE_FOOTING),
    "footing_gaussian": Profile("footing_gaussian", GAUSSIAN_FOOTING),
    "footing_circular": Profile("footing_circular", CIRCULAR_FOOTING),
    # Alias of footing_circular.
    "footing": Profile("footing", CIRCULAR_FOOTING),
    "abutment": Profile(
        "abutment",
        (
            Step(Action.LEVEL, Blend.SET, reach=5.0, feather=0.0),
            Step(Action.SLOPE_DOWN, Blend.RAISE, ratio=1.5, feather=2.0),
        ),
    ),
    "berm": Profile("berm", (Step(Action.MOUND, Blend.RAISE, height=4.0, width=10.0),)),
}


def load(path: Path | str) -> Plan:
    path = Path(path)
    return read(tomllib.loads(path.read_text()), path.parent)


def read(data: Mapping, base: Path = Path(".")) -> Plan:
    held = data.get("terraform", data)
    world = held.get("world")
    source_world = held.get("source_world")
    include_rails = held.get("include_rails", False)
    if not isinstance(include_rails, bool):
        raise ValueError("include_rails must be true or false")
    bleed = held.get("bleed", False)
    if type(bleed) is not bool:
        raise ValueError("bleed must be true or false")
    waterlog = held.get("waterlog", False)
    if type(waterlog) is not bool:
        raise ValueError("waterlog must be true or false")
    clear_track = held.get("clear_track", False)
    if type(clear_track) is not bool:
        raise ValueError("clear_track must be true or false")
    preserve_fittings = held.get("preserve_fittings", ())
    if not isinstance(preserve_fittings, (list, tuple)) or any(
        not isinstance(name, str) or not name for name in preserve_fittings
    ):
        raise ValueError("preserve_fittings must list fitting names")
    plan = Plan(
        look=_enum(Look, held.get("look"), Look.NATURAL),
        domain=held.get("domain", "terrain"),
        default=_enum(Intensity, held.get("default"), Intensity.NONE),
        margin=held.get("margin", 24),
        reach=held.get("reach", 32.0),
        feather=None if held.get("feather") is None else _positive(held["feather"], "feather"),
        shoulder=_nonnegative(held.get("shoulder", 0.0), "shoulder"),
        headroom=held.get("headroom", 6),
        world=None if world is None else base / world,
        source_world=None if source_world is None else base / source_world,
        dimension=held.get("dimension"),
        include_rails=include_rails,
        bleed=bleed,
        preserve_fittings=tuple(preserve_fittings),
        fitting_buffer=_whole(held.get("fitting_buffer", 0), "fitting_buffer", 0),
        waterlog=waterlog,
        clear_track=clear_track,
        ground=_block_names(held.get("ground"), "ground", BlockNames(())),
        strata=_strata(held.get("strata", {})),
        foliage=_foliage(held.get("foliage", {})),
        bridge=_bridge(held.get("bridge", {})),
        tunnel=_tunnel(held.get("tunnel", {})),
        smoothing=_smoothing(held.get("smoothing", {})),
        brush=_brush(held.get("brush", {})),
        embankment=_embankment(held.get("embankment", {})),
        cutting=_cutting(held.get("cutting", {})),
        sculpt=_sculpt(held.get("sculpt", {})),
        slopes=_slopes(held.get("slopes", {})),
        terrain_mask=_terrain_mask(held.get("terrain_mask", {})),
        surface_mask=read_surface_mask(held.get("surface_mask")),
        spans=tuple(_span(each) for each in data.get("span", ())),
        follows=tuple(_follows(each) for each in data.get("follows", ())),
        keepouts=tuple(_keepout(each) for each in data.get("keep_out", ())),
        profiles={
            name: _profile(name, each)
            for name, each in data.get("profile", {}).items()
        },
        cross_sections={
            name: _cross_section(name, each)
            for name, each in data.get("cross_section", {}).items()
        },
    )
    known_profiles = set(BUILT_IN) | set(plan.profiles)
    requested = set(plan.smoothing.profiles)
    for rule in plan.smoothing.transitions:
        requested.update((rule.source, rule.target))
    unknown = requested - known_profiles
    if unknown:
        raise ValueError("smoothing names unknown profiles: " + ", ".join(sorted(unknown)))
    return plan


def _span(data: Mapping) -> Span:
    return Span(
        start=_marker(data, "from") if "from" in data else None,
        stop=_marker(data, "to") if "to" in data else None,
        intensity=_enum(Intensity, data.get("mode"), Intensity.SHAPE),
        profile=data.get("profile", "at_grade"),
        transition=data.get("transition", 8),
        order=data.get("order", 0),
        headroom=data.get("headroom"),
        cross_section=data.get("cross_section"),
        section=data.get("section"),
        surface_mask=read_surface_mask(data.get("surface_mask")),
    )


def _follows(data: Mapping) -> Follows:
    if "fitting" not in data:
        raise ValueError("a terraform follows needs a 'fitting'")
    mode = _choice(data.get("reach_mode", "fixed"), ("fixed", "adaptive"), "reach_mode")
    minimum = _positive(data.get("reach_min", 6.0), "reach_min")
    maximum = _positive(data.get("reach_max", 22.0), "reach_max")
    if maximum < minimum:
        raise ValueError("reach_max must be at least reach_min")
    return Follows(
        fitting=data["fitting"],
        profile=data.get("profile", "footing"),
        intensity=_enum(Intensity, data.get("mode"), Intensity.SHAPE),
        reach=data.get("reach", 0.0),
        order=data.get("order", 20),
        reach_mode=mode,
        reach_min=minimum,
        reach_max=maximum,
        height_response=_nonnegative(data.get("height_response", 0.08), "height_response"),
        terrain_response=_nonnegative(data.get("terrain_response", 0.2), "terrain_response"),
        surface_mask=read_surface_mask(data.get("surface_mask")),
    )


def _smoothing(data: Mapping) -> Smoothing:
    mode = _choice(data.get("mode", "fixed"), ("fixed", "adaptive"), "smoothing.mode")
    size = data.get("size")
    minimum = _positive(data.get("min_size", 0.75), "smoothing.min_size")
    maximum = _positive(data.get("max_size", 3.5), "smoothing.max_size")
    if maximum < minimum:
        raise ValueError("smoothing.max_size must be at least min_size")
    strength = float(data.get("strength", 1.0))
    passes = data.get("passes", 1)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("smoothing.strength must be between 0 and 1")
    if type(passes) is not int or passes < 1:
        raise ValueError("smoothing.passes must be a positive integer")
    profiles = data.get("profiles", {})
    transitions = data.get("transitions", ())
    if not isinstance(profiles, Mapping) or any(not isinstance(name, str) or not name for name in profiles):
        raise ValueError("smoothing.profiles must map profile names to behaviors")
    if not isinstance(transitions, (list, tuple)) or any(not isinstance(each, Mapping) for each in transitions):
        raise ValueError("smoothing.transitions must be a list of transition rules")
    adaptive = data.get("adaptive_transitions", True)
    if type(adaptive) is not bool:
        raise ValueError("smoothing.adaptive_transitions must be true or false")
    preserve_cliffs = data.get("preserve_cliffs", True)
    if type(preserve_cliffs) is not bool:
        raise ValueError("smoothing.preserve_cliffs must be true or false")
    exposed = _whole(data.get("face_min_exposed", 3), "smoothing.face_min_exposed", 2)
    if exposed > 6:
        raise ValueError("smoothing.face_min_exposed must be between 2 and 6")
    contour_scale = _smoothing_amount(data.get("contour_scale", 3.0), "contour_scale")
    if contour_scale <= 0:
        raise ValueError("smoothing.contour_scale must be positive")
    return Smoothing(
        mode=mode,
        size=None if size is None else _positive(size, "smoothing.size"),
        min_size=minimum,
        max_size=maximum,
        strength=strength,
        passes=passes,
        terrain_response=_nonnegative(data.get("terrain_response", 1.0), "smoothing.terrain_response"),
        kernel=_choice(data.get("kernel", "gaussian"), KERNELS, "smoothing.kernel"),
        behavior=_smoothing_behavior(data.get("behavior", "auto")),
        stable_change=_smoothing_amount(data.get("stable_change", 1.0), "stable_change"),
        decision_threshold=_smoothing_amount(data.get("decision_threshold", 1.0), "decision_threshold"),
        transition_reach=_smoothing_amount(data.get("transition_reach", 12.0), "transition_reach"),
        adaptive_transitions=adaptive,
        preserve_cliffs=preserve_cliffs,
        contour_strength=_fraction(data.get("contour_strength", .65), "smoothing.contour_strength"),
        contour_scale=contour_scale,
        contact_blend=_fraction(data.get("contact_blend", 1.0), "smoothing.contact_blend"),
        face_iterations=_whole(data.get("face_iterations", 2), "smoothing.face_iterations", 0),
        face_min_exposed=exposed,
        face_depth=_whole(data.get("face_depth", 4), "smoothing.face_depth", 0),
        profiles={name: _smoothing_behavior(value) for name, value in profiles.items()},
        transitions=tuple(_smoothing_transition(each) for each in transitions),
    )


def _smoothing_behavior(value: str) -> str:
    return _choice(value, ("auto", "grow", "stable", "melt"), "smoothing.behavior")


def _smoothing_amount(value: float, name: str) -> float:
    from math import isfinite
    if isinstance(value, bool):
        raise ValueError(f"smoothing.{name} must be a number of blocks")
    value = float(value)
    if not isfinite(value) or value < 0:
        raise ValueError(f"smoothing.{name} must be finite and nonnegative")
    return value


def _smoothing_transition(data: Mapping) -> SmoothingTransition:
    source, target = data.get("from"), data.get("to")
    if not isinstance(source, str) or not source or not isinstance(target, str) or not target:
        raise ValueError("smoothing.transitions need nonempty from and to profile names")
    bidirectional = data.get("bidirectional", True)
    if type(bidirectional) is not bool:
        raise ValueError("smoothing.transitions.bidirectional must be true or false")
    return SmoothingTransition(
        source, target, _smoothing_behavior(data.get("behavior", "auto")),
        None if data.get("reach") is None else _smoothing_amount(data["reach"], "transitions.reach"),
        _choice(data.get("side", "both"), ("left", "right", "both"), "smoothing.transitions.side"),
        bidirectional,
    )


def _brush(data: Mapping) -> Brush:
    passes = data.get("passes", 3)
    if type(passes) is not int or passes < 0:
        raise ValueError("brush.passes must be a whole number, 0 for none")
    return Brush(
        radius=_positive(data.get("radius", 1.5), "brush.radius"),
        passes=passes,
        margin=_nonnegative(data.get("margin", 4.0), "brush.margin"),
        falloff=_positive(data.get("falloff", 6.0), "brush.falloff"),
    )


def _embankment(data: Mapping) -> Embankment:
    water_spill = data.get("water_spill", False)
    if type(water_spill) is not bool:
        raise ValueError("embankment.water_spill must be true or false")
    return Embankment(
        adaptive=bool(data.get("adaptive", False)),
        height_response=_nonnegative(data.get("height_response", 0.06), "embankment.height_response"),
        terrain_response=_nonnegative(data.get("terrain_response", 0.2), "embankment.terrain_response"),
        max_ratio=_positive(data.get("max_ratio", 4.5), "embankment.max_ratio"),
        ratio=None if data.get("ratio") is None else _positive(data["ratio"], "embankment.ratio"),
        spill=bool(data.get("spill", True)),
        tops=_block_names(data.get("tops"), "embankment.tops", DEFAULT_TOPS),
        layering_reach=_whole(data.get("layering_reach", REACH), "embankment.layering_reach", 0),
        layering_tile=_whole(data.get("layering_tile", TILE), "embankment.layering_tile", 1),
        shape=_choice(data.get("shape", "eased"), ("eased", "linear", "natural"), "embankment.shape"),
        water_spill=water_spill,
        max_reach=_whole(data.get("max_reach", 0), "embankment.max_reach", 0),
    )


def _whole(value, name: str, least: int) -> int:
    if type(value) is not int or value < least:
        raise ValueError(f"{name} must be a whole number of blocks, at least {least}")
    return value


def _cutting(data: Mapping) -> Cutting:
    ratio = data.get("ratio")
    return Cutting(
        ratio=None if ratio is None else _positive(ratio, "cutting.ratio"),
        rock=bool(data.get("rock", True)),
        loose=_block_names(data.get("loose"), "cutting.loose", DEFAULT_LOOSE),
        rock_body=_rock_body(data.get("rock_body", ROCK_BODY)),
        stone_share=_stone_share(data.get("stone_share", STONE_SHARE)),
        rock_band=_blocks(data.get("rock_band", ROCK_BAND), "cutting.rock_band"),
        wall_band=_blocks(data.get("wall_band", WALL_BAND), "cutting.wall_band"),
        rock_drop=_whole(data.get("rock_drop", 2), "cutting.rock_drop", 1),
    )


def _sculpt(data: Mapping) -> Sculpt:
    if not isinstance(data, Mapping):
        raise ValueError("sculpt must be a table")
    unknown = set(data) - {"enabled", "rock_rise", "soil_rise", "bank_rise", "ledges", "adapt",
                           "bank_radius", "bank_passes"}
    if unknown:
        raise ValueError(f"unknown sculpt option(s): {', '.join(sorted(unknown))}")
    default = Sculpt()
    return Sculpt(
        enabled=_flag(data.get("enabled", default.enabled), "sculpt.enabled"),
        rock_rise=_span_of(data.get("rock_rise", default.rock_rise), "sculpt.rock_rise", float),
        soil_rise=_span_of(data.get("soil_rise", default.soil_rise), "sculpt.soil_rise", float),
        bank_rise=_span_of(data.get("bank_rise", default.bank_rise), "sculpt.bank_rise", float),
        ledges=_span_of(data.get("ledges", default.ledges), "sculpt.ledges", int, 0),
        adapt=_flag(data.get("adapt", default.adapt), "sculpt.adapt"),
        bank_radius=_positive(_smoothing_amount(data.get("bank_radius", default.bank_radius),
                                                "sculpt.bank_radius"), "sculpt.bank_radius"),
        bank_passes=_whole(data.get("bank_passes", default.bank_passes), "sculpt.bank_passes", 0),
    )


def _flag(value, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be true or false")
    return value


def _span_of(value, name: str, kind, least: float = 1e-6) -> tuple:
    # A [low, high] range of positive numbers, or whole numbers from `least`.
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(each, bool) or not isinstance(each, (int, float)) for each in value)
            or (kind is int and any(each != int(each) for each in value))
            or value[0] < least or value[0] > value[1]):
        whole = "whole numbers" if kind is int else "numbers"
        raise ValueError(f"{name} must be [low, high] {whole}, low at least {least:g} and at most high")
    return kind(value[0]), kind(value[1])


def _blocks(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a number of blocks, at least 0")
    return float(value)


def _rock_body(value) -> int:
    if type(value) is not int or not 1 <= value <= SUBSURFACE:
        raise ValueError(f"cutting.rock_body must be a whole number of blocks from 1 to {SUBSURFACE}")
    return value


def _stone_share(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < value <= 1.0:
        raise ValueError("cutting.stone_share must be a share above 0 and at most 1")
    return float(value)


def _slopes(data: Mapping) -> Slopes:
    return Slopes(
        portal_reach=_blocks(data.get("portal_reach", PORTAL), "slopes.portal_reach"),
        repair_passes=_whole(data.get("repair_passes", REPAIR_PASSES), "slopes.repair_passes", 0),
        pocket=_whole(data.get("pocket", POCKET), "slopes.pocket", 0),
        cone_passes=_whole(data.get("cone_passes", PASSES), "slopes.cone_passes", 0),
        ridge_width=_whole(data.get("ridge_width", RIDGE_WIDTH), "slopes.ridge_width", 0),
        max_drop=_whole(data.get("max_drop", 1), "slopes.max_drop", 1),
        variation=_fraction(data.get("variation", 0.0), "slopes.variation"),
        seed=_whole(data.get("seed", 0), "slopes.seed", 0),
    )


def _fraction(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a number from 0 to 1")
    return float(value)


def _block_names(data, name: str, default: BlockNames) -> BlockNames:
    if data is None:
        return default
    if not isinstance(data, (list, tuple)) or not all(
        isinstance(each, str) and ":" in each for each in data
    ):
        raise ValueError(
            f"{name} must list block names or patterns such as "
            '"minecraft:dirt" or "minecraft:*sand"'
        )
    return BlockNames(tuple(data))


def _terrain_mask(data: Mapping) -> TerrainMask:
    buffer = data.get("water_buffer", 0)
    if type(buffer) is not int or buffer < 0:
        raise ValueError("terrain_mask.water_buffer must be a nonnegative integer")
    slope_to_water = data.get("slope_to_water", False)
    if type(slope_to_water) is not bool:
        raise ValueError("terrain_mask.slope_to_water must be true or false")
    if slope_to_water and not data.get("preserve_water", False):
        raise ValueError("terrain_mask.slope_to_water requires preserve_water")
    return TerrainMask(
        preserve_water=bool(data.get("preserve_water", False)),
        water_buffer=buffer,
        preserve_structures=bool(data.get("preserve_structures", False)),
        slope_to_water=slope_to_water,
    )


def _cross_section(name: str, data: Mapping) -> CrossSection:
    sides = ("auto", "at_grade", "cutting", "embankment", "bridge")
    sample = data.get("sample_distance", 8)
    window = data.get("sample_window", 2)
    min_run = data.get("min_run", 20)
    if type(sample) is not int or sample < 1:
        raise ValueError(f"cross_section.{name}.sample_distance must be positive")
    if type(window) is not int or window < 0:
        raise ValueError(f"cross_section.{name}.sample_window must be nonnegative")
    if type(min_run) is not int or min_run < 1:
        raise ValueError(f"cross_section.{name}.min_run must be positive")
    return CrossSection(
        name=name,
        left=_choice(data.get("left", "auto"), sides, f"cross_section.{name}.left"),
        right=_choice(data.get("right", "auto"), sides, f"cross_section.{name}.right"),
        sample_distance=sample,
        sample_window=window,
        cut_threshold=_nonnegative(data.get("cut_threshold", 2.0), f"cross_section.{name}.cut_threshold"),
        fill_threshold=_nonnegative(data.get("fill_threshold", 2.0), f"cross_section.{name}.fill_threshold"),
        bridge_threshold=_positive(data.get("bridge_threshold", 9.0), f"cross_section.{name}.bridge_threshold"),
        water_mode=_choice(data.get("water_mode", "bridge"), ("bridge", "at_grade"), f"cross_section.{name}.water_mode"),
        cut_base=_base(data.get("cut_base", "footing")),
        min_run=min_run,
    )


def _choice(value: str, allowed: tuple[str, ...], name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}")
    return value


def _positive(value: float, name: str) -> float:
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative(value: float, name: str) -> float:
    value = float(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _profile(name: str, data: Mapping) -> Profile:
    steps = data.get("steps", [data])
    return Profile(name, tuple(_step(each) for each in steps))


def _step(data: Mapping) -> Step:
    action = _enum(Action, data.get("action"), Action.SLOPE_DOWN)
    return Step(
        action=action,
        blend=_enum(
            Blend, data.get("blend"),
            Blend.SET if action in (Action.GAUSSIAN, Action.CIRCULAR) else Blend.RAISE,
        ),
        ratio=data.get("ratio", 0.0),
        height=data.get("height", 0.0),
        width=data.get("width", 0.0),
        reach=data.get("reach", 0.0),
        feather=data.get("feather"),
        softness=data.get("softness"),
        base=_base(data.get("from", "footing")),
        shoulder=None if data.get("shoulder") is None else _nonnegative(data["shoulder"], "shoulder"),
    )


def _base(value: str) -> str:
    if value not in BASES:
        raise ValueError(f"a step is measured from {' or '.join(BASES)}, not {value!r}")
    return value


def _keepout(data: Mapping) -> Keepout:
    box = data["box"]
    return Keepout(ColumnPos.coerce(box[0]), ColumnPos.coerce(box[1]), data.get("reason", ""))


def _tunnel(data: Mapping) -> Tunnel:
    extra = set(data) - {"sides", "reach", "max_change", "min_cover"}
    if extra:
        raise ValueError(f"unknown tunnel options: {sorted(extra)}")
    if "sides" in data and type(data["sides"]) is not bool:
        raise ValueError("tunnel.sides must be true or false")
    return Tunnel(
        sides=data.get("sides", False),
        reach=_positive(_blocks(data.get("reach", 12.0), "tunnel.reach"), "tunnel.reach"),
        max_change=_whole(data.get("max_change", 4), "tunnel.max_change", 0),
        min_cover=_whole(data.get("min_cover", 2), "tunnel.min_cover", 1),
    )


def _bridge(data: Mapping) -> Bridge:
    extra = set(data) - {"natural_only", "radius", "strength", "max_change", "reach", "clear_underneath"}
    if extra:
        raise ValueError(f"unknown bridge options: {sorted(extra)}")
    for key in ("natural_only", "clear_underneath"):
        if key in data and type(data[key]) is not bool:
            raise ValueError(f"bridge.{key} must be true or false")
    strength = _nonnegative(data.get("strength", 0.5), "bridge.strength")
    if strength > 1:
        raise ValueError("bridge.strength must be 0..1")
    return Bridge(
        natural_only=data.get("natural_only", False),
        radius=_positive(data.get("radius", 2.0), "bridge.radius"),
        strength=strength,
        max_change=_nonnegative(data.get("max_change", 2.0), "bridge.max_change"),
        reach=_positive(data.get("reach", 4.0), "bridge.reach"),
        clear_underneath=data.get("clear_underneath", True),
    )


def _foliage(data: Mapping) -> Foliage:
    return Foliage(data.get("clear", True), data.get("revegetate", False))


def _strata(data: Mapping) -> Strata:
    if not data:
        return DEFAULT_STRATA
    return Strata(
        surface=parse_state_key(data.get("surface", "minecraft:grass_block")),
        subsurface=parse_state_key(data.get("subsurface", "minecraft:dirt")),
        filler=parse_state_key(data.get("filler", "minecraft:stone")),
        depth=data.get("depth", 3),
    )


def _marker(data: Mapping, key: str) -> str | ColumnPos:
    if key not in data:
        raise ValueError(f"a terraform span needs a {key!r}")
    value = data[key]
    return value if isinstance(value, str) else ColumnPos.coerce(value)


def _enum(kind, value, default):
    if value is None:
        return default
    try:
        return kind(value) if not isinstance(value, str) else kind(value.lower())
    except ValueError:
        known = ", ".join(each.value for each in kind)
        raise ValueError(f"{value!r} is not one of {known}") from None
