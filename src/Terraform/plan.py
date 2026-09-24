from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..Core.block.states import BlockState
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
class Smoothing:
    mode: str = "fixed"
    size: float | None = None
    min_size: float = 0.75
    max_size: float = 3.5
    strength: float = 1.0
    passes: int = 1
    terrain_response: float = 1.0
    kernel: str = "gaussian"


@dataclass(frozen=True, slots=True)
class Embankment:
    adaptive: bool = False
    height_response: float = 0.06
    terrain_response: float = 0.2
    max_ratio: float = 4.5


@dataclass(frozen=True, slots=True)
class Cutting:
    # Blocks across per block up for a cut's side slope; None takes it from
    # the surrounding terrain.
    ratio: float | None = None


@dataclass(frozen=True, slots=True)
class TerrainMask:
    preserve_water: bool = False
    water_buffer: int = 0
    preserve_structures: bool = False


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
class Plan:
    look: Look = Look.NATURAL
    domain: str = "terrain"
    default: Intensity = Intensity.NONE
    margin: int = 24
    reach: float = 32.0
    # Side slopes hold their ratio until this far inside the reach, then fade
    # out; None fades them across the whole reach.
    feather: float | None = None
    headroom: int = 6
    world: Path | None = None
    strata: Strata = DEFAULT_STRATA
    foliage: Foliage = Foliage()
    smoothing: Smoothing = Smoothing()
    embankment: Embankment = Embankment()
    cutting: Cutting = Cutting()
    terrain_mask: TerrainMask = TerrainMask()
    surface_mask: SurfaceMaskSpec | None = None
    spans: tuple[Span, ...] = ()
    follows: tuple[Follows, ...] = ()
    keepouts: tuple[Keepout, ...] = ()
    profiles: Mapping[str, Profile] = field(default_factory=dict)
    cross_sections: Mapping[str, CrossSection] = field(default_factory=dict)

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
    "embankment": Profile("embankment", (Step(Action.SLOPE_DOWN, Blend.RAISE),)),
    "cutting": Profile("cutting", (Step(Action.SLOPE_UP, Blend.LOWER),)),
    "trench": Profile("trench", (Step(Action.SLOPE_UP, Blend.LOWER),)),
    "bridge": Profile("bridge", (Step(Action.KEEP, Blend.RESTORE),)),
    "tunnel": Profile("tunnel", (Step(Action.KEEP, Blend.RESTORE),)),
    # Sloped back up from the top of retaining walls.
    "retained_cutting": Profile(
        "retained_cutting", (Step(Action.SLOPE_UP, Blend.LOWER, base="crest"),)
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
    return Plan(
        look=_enum(Look, held.get("look"), Look.NATURAL),
        domain=held.get("domain", "terrain"),
        default=_enum(Intensity, held.get("default"), Intensity.NONE),
        margin=held.get("margin", 24),
        reach=held.get("reach", 32.0),
        feather=None if held.get("feather") is None else _positive(held["feather"], "feather"),
        headroom=held.get("headroom", 6),
        world=None if world is None else base / world,
        strata=_strata(held.get("strata", {})),
        foliage=_foliage(held.get("foliage", {})),
        smoothing=_smoothing(held.get("smoothing", {})),
        embankment=_embankment(held.get("embankment", {})),
        cutting=_cutting(held.get("cutting", {})),
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
    return Smoothing(
        mode=mode,
        size=None if size is None else _positive(size, "smoothing.size"),
        min_size=minimum,
        max_size=maximum,
        strength=strength,
        passes=passes,
        terrain_response=_nonnegative(data.get("terrain_response", 1.0), "smoothing.terrain_response"),
        kernel=_choice(data.get("kernel", "gaussian"), KERNELS, "smoothing.kernel"),
    )


def _embankment(data: Mapping) -> Embankment:
    return Embankment(
        adaptive=bool(data.get("adaptive", False)),
        height_response=_nonnegative(data.get("height_response", 0.06), "embankment.height_response"),
        terrain_response=_nonnegative(data.get("terrain_response", 0.2), "embankment.terrain_response"),
        max_ratio=_positive(data.get("max_ratio", 4.5), "embankment.max_ratio"),
    )


def _cutting(data: Mapping) -> Cutting:
    ratio = data.get("ratio")
    return Cutting(ratio=None if ratio is None else _positive(ratio, "cutting.ratio"))


def _terrain_mask(data: Mapping) -> TerrainMask:
    buffer = data.get("water_buffer", 0)
    if type(buffer) is not int or buffer < 0:
        raise ValueError("terrain_mask.water_buffer must be a nonnegative integer")
    return TerrainMask(
        preserve_water=bool(data.get("preserve_water", False)),
        water_buffer=buffer,
        preserve_structures=bool(data.get("preserve_structures", False)),
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
    )


def _base(value: str) -> str:
    if value not in BASES:
        raise ValueError(f"a step is measured from {' or '.join(BASES)}, not {value!r}")
    return value


def _keepout(data: Mapping) -> Keepout:
    box = data["box"]
    return Keepout(ColumnPos.coerce(box[0]), ColumnPos.coerce(box[1]), data.get("reason", ""))


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
