from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...Core.coordinates import BlockPos
from ...Parser.spec import BuildSpec, FeaturePathSpec, FormationSpec, OutputSpec

Polygon = tuple[tuple[float, float], ...]
Triangle = tuple[tuple[float, float, float], ...]


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    name: str
    build: BuildSpec
    source: Path
    terrain_priority: int = 0
    priority: int = 0
    suppressed: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RegionSpec:
    name: str
    domain: str
    boundary: Polygon
    block: str
    top: float = 0
    grade: tuple[float, float] = (0, 0)
    thickness: int = 1
    holes: tuple[Polygon, ...] = ()
    triangles: tuple[Triangle, ...] = ()
    terrain_component: str | None = None


@dataclass(frozen=True, slots=True)
class OwnershipSpec:
    owner: str
    boundary: Polygon
    heights: tuple[int, int] | None = None
    components: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    priority: int = 0


@dataclass(frozen=True, slots=True)
class ConnectionSpec:
    source: str
    target: str
    source_station: int
    target_station: int
    tolerance: float = 1


@dataclass(frozen=True, slots=True)
class BindingSpec:
    branch: str
    target: str
    source: str


@dataclass(frozen=True, slots=True)
class FormationRule:
    formation: FormationSpec
    components: tuple[str, ...]
    boundary: Polygon = ()
    heights: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class CompositionSpec:
    output: OutputSpec
    components: tuple[ComponentSpec, ...]
    regions: tuple[RegionSpec, ...] = ()
    ownership: tuple[OwnershipSpec, ...] = ()
    connections: tuple[ConnectionSpec, ...] = ()
    origin: BlockPos | None = None
    feature_paths: tuple[FeaturePathSpec, ...] = ()
    bindings: tuple[BindingSpec, ...] = ()
    sources: tuple[Path, ...] = ()
    formations: tuple[FormationRule, ...] = ()
