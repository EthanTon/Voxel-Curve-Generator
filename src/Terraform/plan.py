from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..Core.block.states import BlockState
from ..Core.coordinates import ColumnPos
from ..Format.schematics import parse_state_key
from .layers import Blend
from .materialize import Strata


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


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    steps: tuple[Step, ...]


@dataclass(frozen=True, slots=True)
class Span:
    start: str | ColumnPos
    stop: str | ColumnPos
    intensity: Intensity = Intensity.SHAPE
    profile: str = "at_grade"
    transition: int = 8
    order: int = 0


@dataclass(frozen=True, slots=True)
class Follows:
    fitting: str
    profile: str = "footing"
    intensity: Intensity = Intensity.SHAPE
    reach: float = 0.0
    order: int = 20


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


@dataclass(frozen=True, slots=True)
class Plan:
    look: Look = Look.NATURAL
    domain: str = "terrain"
    default: Intensity = Intensity.NONE
    margin: int = 24
    reach: float = 32.0
    headroom: int = 6
    world: Path | None = None
    strata: Strata = DEFAULT_STRATA
    foliage: Foliage = Foliage()
    spans: tuple[Span, ...] = ()
    follows: tuple[Follows, ...] = ()
    keepouts: tuple[Keepout, ...] = ()
    profiles: Mapping[str, Profile] = field(default_factory=dict)

    def stretches(self, locate, length: int) -> tuple[Stretch, ...]:
        if not self.spans:
            return (Stretch(0, length - 1, self.default, "at_grade", 0, 0),)
        return tuple(
            Stretch(
                *sorted((locate(span.start), locate(span.stop))),
                span.intensity,
                span.profile,
                span.transition,
                span.order,
            )
            for span in self.spans
        )

    def profile(self, name: str) -> Profile:
        found = self.profiles.get(name, BUILT_IN.get(name))
        if found is None:
            known = ", ".join(sorted({*self.profiles, *BUILT_IN}))
            raise KeyError(f"no profile named {name!r}. Known profiles: {known}")
        return found


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
    "pad": Profile("pad", (Step(Action.LEVEL, Blend.SET),)),
    "footing": Profile(
        "footing",
        (
            Step(Action.LEVEL, Blend.SET, reach=3.0, feather=1.0),
            Step(Action.SLOPE_DOWN, Blend.RAISE, ratio=2.0, feather=2.0),
        ),
    ),
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
        headroom=held.get("headroom", 6),
        world=None if world is None else base / world,
        strata=_strata(held.get("strata", {})),
        foliage=_foliage(held.get("foliage", {})),
        spans=tuple(_span(each) for each in data.get("span", ())),
        follows=tuple(_follows(each) for each in data.get("follows", ())),
        keepouts=tuple(_keepout(each) for each in data.get("keep_out", ())),
        profiles={
            name: _profile(name, each)
            for name, each in data.get("profile", {}).items()
        },
    )


def _span(data: Mapping) -> Span:
    return Span(
        start=_marker(data, "from"),
        stop=_marker(data, "to"),
        intensity=_enum(Intensity, data.get("mode"), Intensity.SHAPE),
        profile=data.get("profile", "at_grade"),
        transition=data.get("transition", 8),
        order=data.get("order", 0),
    )


def _follows(data: Mapping) -> Follows:
    if "fitting" not in data:
        raise ValueError("a terraform follows needs a 'fitting'")
    return Follows(
        fitting=data["fitting"],
        profile=data.get("profile", "footing"),
        intensity=_enum(Intensity, data.get("mode"), Intensity.SHAPE),
        reach=data.get("reach", 0.0),
        order=data.get("order", 20),
    )


def _profile(name: str, data: Mapping) -> Profile:
    steps = data.get("steps", [data])
    return Profile(name, tuple(_step(each) for each in steps))


def _step(data: Mapping) -> Step:
    return Step(
        action=_enum(Action, data.get("action"), Action.SLOPE_DOWN),
        blend=_enum(Blend, data.get("blend"), Blend.RAISE),
        ratio=data.get("ratio", 0.0),
        height=data.get("height", 0.0),
        width=data.get("width", 0.0),
        reach=data.get("reach", 0.0),
        feather=data.get("feather"),
        softness=data.get("softness"),
    )


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
