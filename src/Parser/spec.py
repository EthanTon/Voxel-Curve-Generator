from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..Core.coordinates import BlockPos, ColumnPos, Vec3
from ..Core.Domain import ConnectivityRules, PlacementRules, TerrainRules
from ..Core.block.connectivity_options import ConnectionOverride
from ..Core.modes import ElevationMode, PlacementMode, StepMode
from ..Stamping import DEFAULT_RESOLUTION, Heading


@dataclass(frozen=True, slots=True)
class PathSpec:
    # Positions and vectors as given. The only path work is the lead: points
    # extended straight off either end so the ends are framed like any other.
    points: tuple[BlockPos, ...]
    gradients: tuple[Vec3, ...]
    lead: int = 0


class Measure(Enum):
    # What a spacing or a dash is counted in: blocks along the path, or the
    # path's own points.
    DISTANCE = "distance"
    INDEX = "index"


class SectionElevation(Enum):
    FOLLOW = "follow"
    MIN = "min"
    MAX = "max"


@dataclass(frozen=True, slots=True)
class SpanSpec:
    # The columns a section's stretch of path starts and stops at. The
    # marker names are kept for the messages, not for the geometry.
    start: ColumnPos
    stop: ColumnPos
    start_marker: str | None = None
    stop_marker: str | None = None

    def __str__(self) -> str:
        return f"{_where(self.start, self.start_marker)}"\
               f"..{_where(self.stop, self.stop_marker)}"


@dataclass(frozen=True, slots=True)
class SectionSpec:
    name: str
    domain: str
    schematic: Path
    placement_mode: PlacementMode | None = None
    width: int = 0
    depth: int = 0
    elevation: SectionElevation = SectionElevation.FOLLOW


@dataclass(frozen=True, slots=True)
class AssignmentSpec:
    # Which sections are swept over one stretch of the path. Sections say what
    # they are; assignments say where they are used, so the two can be written
    # apart and an assignment can be worked out rather than written.
    span: SpanSpec
    sections: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DomainSpec:
    # Every rule is optional, and an unset one is inherited from the parents
    # before falling back to the world's own default. `order` is when this
    # domain's blocks go into the world, so a later one covers an earlier.
    name: str
    parents: tuple[str, ...] = ()
    order: int = 0
    properties: Mapping[str, object] = field(default_factory=dict)
    connectivity: ConnectivityRules = ConnectivityRules()
    placement: PlacementRules = PlacementRules()
    terrain: TerrainRules = TerrainRules()


@dataclass(frozen=True, slots=True)
class InteractionSpec:
    # A rule between two domains, inherited by both sides' descendants.
    source: str
    target: str
    connect: bool | None = None
    overwrite: bool | None = None
    overrides: tuple[ConnectionOverride, ...] = ()
    set_properties: Mapping[str, str] | None = None
    properties: Mapping[str, object] = field(default_factory=dict)
    grants: Mapping[str, object] = field(default_factory=dict)
    symmetric: bool = True


@dataclass(frozen=True, slots=True)
class StructureSpec:
    # One schematic per authored heading: a resolution of 90 needs one, 45
    # needs two, 22.5 needs four. It sits every `spacing` along its sections,
    # and wherever the path has turned `angle` since the last one.
    name: str
    domain: str
    sections: tuple[str, ...]
    headings: Mapping[Heading, Path]
    resolution: float = 90.0
    spacing: float = 0.0
    angle: float = 0.0
    measure: Measure = Measure.DISTANCE
    offset: BlockPos = BlockPos(0, 0, 0)
    placement_mode: PlacementMode | None = None
    elevation_mode: ElevationMode = ElevationMode.RIGID


@dataclass(frozen=True, slots=True)
class WireSpec:
    # A line from pole to pole, where `spacing` and `angle` put the poles.
    # One placement per run, so connectivity sees the whole wire.
    #
    # `between` names two other wires instead of sections, and the line goes
    # across from one to the other: a dropper hanging a contact wire off the
    # messenger above it.
    name: str
    domain: str
    schematic: Path
    sections: tuple[str, ...] = ()
    between: tuple[str, ...] = ()
    offset: BlockPos = BlockPos(0, 0, 0)
    spacing: float = 0.0
    angle: float = 0.0
    measure: Measure = Measure.DISTANCE
    step: StepMode = StepMode.MANHATTAN
    placement_mode: PlacementMode | None = None


@dataclass(frozen=True, slots=True)
class LaneSpec:
    # A dash of `on` blocks then a gap of `off`, over and over.
    name: str
    domain: str
    sections: tuple[str, ...]
    on: int
    off: int
    schematic: Path | None = None
    block: str | None = None
    measure: Measure = Measure.INDEX
    offset: BlockPos = BlockPos(0, 0, 0)
    placement_mode: PlacementMode | None = None


@dataclass(frozen=True, slots=True)
class OutputSpec:
    path: Path
    name: str = "build"
    data_version: int = 3700


@dataclass(frozen=True, slots=True)
class BuildSpec:
    output: OutputSpec
    path: PathSpec
    markers: Mapping[str, ColumnPos] = field(default_factory=dict)
    sections: tuple[SectionSpec, ...] = ()
    assignments: tuple[AssignmentSpec, ...] = ()
    domains: tuple[DomainSpec, ...] = ()
    interactions: tuple[InteractionSpec, ...] = ()
    structures: tuple[StructureSpec, ...] = ()
    wires: tuple[WireSpec, ...] = ()
    lanes: tuple[LaneSpec, ...] = ()

    def marker(self, name: str) -> ColumnPos:
        if name not in self.markers:
            known = ", ".join(sorted(self.markers)) or "none"
            raise KeyError(f"No marker named {name!r}. Known markers: {known}")
        return self.markers[name]

    def section(self, name: str) -> SectionSpec:
        return _named(self.sections, name, "section")

    def domain(self, name: str) -> DomainSpec:
        return _named(self.domains, name, "domain")

    def structure(self, name: str) -> StructureSpec:
        return _named(self.structures, name, "structure")

    def wire(self, name: str) -> WireSpec:
        return _named(self.wires, name, "wire")

    def lane(self, name: str) -> LaneSpec:
        return _named(self.lanes, name, "lane")


def _where(column: ColumnPos, marker: str | None) -> str:
    return marker or f"{tuple(column)}"


def _named(items, name: str, kind: str):
    for item in items:
        if item.name == name:
            return item
    known = ", ".join(sorted(item.name for item in items)) or "none"
    raise KeyError(f"No {kind} named {name!r}. Known {kind}s: {known}")
