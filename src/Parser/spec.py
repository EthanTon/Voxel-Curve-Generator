from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..Core.coordinates import BlockPos, ColumnPos, Vec3
from ..Core.CrossSection.turns import JOIN_DEGREES
from ..Core.Domain import ConnectivityRules, PlacementRules, ReplacementRules, TerrainRules
from ..Core.block.connectivity_options import ConnectionOverride
from ..Core.modes import ElevationMode, PlacementMode, StepMode
from ..Core.Feature.dependent_path import PathProjection
from ..Stamping import Heading
from ..Terraform.plan import Plan as TerraformPlan
from ..Terraform.surface_mask import SurfaceMaskSpec


@dataclass(frozen=True, slots=True)
class PathSpec:
    # `lead` points are added straight off each end. `diagonal_degrees` is
    # how near a diagonal a heading must be for the join to be laid there.
    points: tuple[BlockPos, ...]
    gradients: tuple[Vec3, ...]
    lead: int = 0
    diagonal_degrees: float = JOIN_DEGREES


class Measure(Enum):
    # Spacing counted in blocks along the path or in path points.
    DISTANCE = "distance"
    INDEX = "index"


@dataclass(frozen=True, slots=True)
class FeaturePathSpec:
    # A reusable selection from an existing feature's world path.
    name: str
    source: str
    spacing: float = 0.0
    angle: float = 0.0
    measure: Measure = Measure.DISTANCE
    include_end: bool = True
    projection: PathProjection | None = None
    min_distance: float = 0.0
    sections: tuple[str, ...] = ()


class WireGeometry(Enum):
    # FOLLOW lies on the section column at every point, at each point's own
    # level; LINE runs straight and rigid between anchors.
    FOLLOW = "follow"
    LINE = "line"


class SectionElevation(Enum):
    FOLLOW = "follow"
    MIN = "min"
    MAX = "max"


@dataclass(frozen=True, slots=True)
class SpanSpec:
    # Marker names are kept for messages only.
    start: ColumnPos
    stop: ColumnPos
    start_marker: str | None = None
    stop_marker: str | None = None

    def __str__(self) -> str:
        return f"{_where(self.start, self.start_marker)}"\
               f"..{_where(self.stop, self.stop_marker)}"


@dataclass(frozen=True, slots=True)
class StationSpanSpec:
    # Inclusive indices in the authored path, before padding is added.
    start: int
    stop: int

    def __post_init__(self) -> None:
        if (any(not isinstance(i, int) or isinstance(i, bool) for i in (self.start, self.stop))
                or not 0 <= self.start <= self.stop):
            raise ValueError("assignment stations need nonnegative inclusive start <= stop indices")

    def __str__(self) -> str:
        return f"stations {self.start}..{self.stop}"


@dataclass(frozen=True, slots=True)
class SectionSpec:
    # `step` applies to every column unless `steps` overrides it by "x,z",
    # or at one height of a column by "x,y,z".
    # MANHATTAN keeps a column face-connected along the path (rails, fences).
    name: str
    domain: str
    schematic: Path
    placement_mode: PlacementMode | None = None
    width: int = 0
    depth: int = 0
    elevation: SectionElevation = SectionElevation.FOLLOW
    step: StepMode = StepMode.CHEBYSHEV
    steps: Mapping[tuple[int, ...], StepMode] = field(default_factory=dict)
    # Optional domains for raised blocks on the left and right of the deck.
    edge_domains: tuple[str, str] | None = None
    # Columns this far from the centre also belong to the edge at deck level.
    edge_from: int | None = None
    surface_mask: SurfaceMaskSpec | None = None
    # The local height the ground is shaped to meet; the lowest block if unset.
    formation: int | None = None
    # Blocks of the section the earthworks may cover: left out of the track
    # the ground is shaped round, and replaced wherever it fills.
    terrain_overrides: frozenset[str] = frozenset()
    # Optional local lateral columns; everything else, including the bed,
    # remains structural even when it uses one of the same block names.
    terrain_overrides_columns: frozenset[int] | None = None
    # Rotate authored assets in memory; composite halves share one layout.
    schematic_turns: int = 0
    left_schematic: Path | None = None
    right_schematic: Path | None = None
    half_schematic_turns: int = 0
    surface_mask_min_y: int | None = None
    surface_mask_columns: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class AssignmentSpec:
    # The sections swept over one stretch of the path.
    span: SpanSpec | StationSpanSpec
    sections: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DomainSpec:
    # Unset rules are inherited. Higher `order` domains are laid later and
    # cover lower ones.
    name: str
    parents: tuple[str, ...] = ()
    order: int = 0
    properties: Mapping[str, object] = field(default_factory=dict)
    connectivity: ConnectivityRules = ConnectivityRules()
    placement: PlacementRules = PlacementRules()
    terrain: TerrainRules = TerrainRules()
    replacement: ReplacementRules | None = None


@dataclass(frozen=True, slots=True)
class InteractionSpec:
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
    # One schematic per authored heading (four at 22.5 degrees), placed every
    # `spacing` or wherever the path has turned `angle`.
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
    # "heading" picks a variant by travel direction; "layout" turns one
    # schematic with the section's layout, for structures spanning its
    # columns.
    turn: str = "heading"
    surface_mask: SurfaceMaskSpec | None = None
    # Use this wire's feature path, including each run's boundary anchors.
    wire: str | None = None
    min_distance: float = 0.0
    # Structures naming the same feature path are spaced as one, across all
    # their sections, each laying the anchors on its own. A declared
    # [[feature_path]] instead supplies shared world placement points.
    feature_path: str | None = None
    schematic_turns: int = 0


@dataclass(frozen=True, slots=True)
class WireSpec:
    # A strand along the sections (see WireGeometry; a line's anchors are
    # placed by `spacing` and `angle`), or with `between`, droppers across
    # two other wires. `step` joins its blocks by faces or by corners in 3D.
    name: str
    domain: str
    schematic: Path
    sections: tuple[str, ...] = ()
    between: tuple[str, ...] = ()
    offset: BlockPos = BlockPos(0, 0, 0)
    geometry: WireGeometry = WireGeometry.FOLLOW
    spacing: float = 0.0
    angle: float = 0.0
    measure: Measure = Measure.DISTANCE
    step: StepMode = StepMode.MANHATTAN
    placement_mode: PlacementMode | None = None
    surface_mask: SurfaceMaskSpec | None = None
    min_distance: float = 0.0
    # A declared dependent path supplies the wire's world placement points.
    feature_path: str | None = None


@dataclass(frozen=True, slots=True)
class LaneSpec:
    # Dashes of `on` path points separated by gaps of `off`.
    name: str
    domain: str
    sections: tuple[str, ...]
    on: int
    off: int
    schematic: Path | None = None
    block: str | None = None
    measure: Measure = Measure.INDEX
    offset: BlockPos = BlockPos(0, 0, 0)
    step: StepMode = StepMode.MANHATTAN
    placement_mode: PlacementMode | None = None
    surface_mask: SurfaceMaskSpec | None = None


@dataclass(frozen=True, slots=True)
class SplitSpec:
    # Export pieces of at most `max_span` per axis and `max_blocks` cells.
    # With `terrain`, pieces also carry the surrounding ground (see export.py).
    directory: Path
    max_blocks: int = 4_000_000
    max_span: int = 128
    terrain: bool = False
    terrain_depth: int = 16
    # Skip the duplicate, full bounding-box schematic when only pieces
    # are needed; write() then returns their manifest.
    combined: bool = True


@dataclass(frozen=True, slots=True)
class OutputSpec:
    path: Path
    name: str = "build"
    data_version: int = 3700
    # "absolute", "relative", or None for each format's default.
    mode: str | None = None
    split: SplitSpec | None = None


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
    terraform: TerraformPlan | None = None
    surface_world: Path | None = None
    surface_dimension: str | None = None
    feature_paths: tuple[FeaturePathSpec, ...] = ()

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

    def feature_path(self, name: str) -> FeaturePathSpec:
        return _named(self.feature_paths, name, "feature path")


def _where(column: ColumnPos, marker: str | None) -> str:
    return marker or f"{tuple(column)}"


def _named(items, name: str, kind: str):
    for item in items:
        if item.name == name:
            return item
    known = ", ".join(sorted(item.name for item in items)) or "none"
    raise KeyError(f"No {kind} named {name!r}. Known {kind}s: {known}")
