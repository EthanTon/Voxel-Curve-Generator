from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from ..block.connectivity import ConnectivityDomainId
from .rules import ConnectivityRules, PlacementRules, TerrainRules


DomainId = ConnectivityDomainId
DomainRef = DomainId | str


def _frozen_mapping(values: Mapping[str, object] | None) -> Mapping[str, object]:
    return MappingProxyType(dict(values or {}))


@dataclass(frozen=True, slots=True)
class Domain:
    """A named group of placements that share connectivity and placement rules.

    Domains inherit from ``parents``: any rule left as None, and any property
    not set here, is taken from the nearest parent that defines it.
    """

    domain_id: DomainId
    name: str
    parents: tuple[DomainId, ...] = ()
    properties: Mapping[str, object] = field(default_factory=dict)
    connectivity: ConnectivityRules = ConnectivityRules()
    placement: PlacementRules = PlacementRules()
    terrain: TerrainRules = TerrainRules()

    def __post_init__(self) -> None:
        object.__setattr__(self, "parents", tuple(self.parents))
        object.__setattr__(self, "properties", _frozen_mapping(self.properties))
        if self.domain_id in self.parents:
            raise ValueError(f"Domain {self.name!r} cannot be its own parent")
        if len(set(self.parents)) != len(self.parents):
            raise ValueError(f"Domain {self.name!r} lists a parent twice")
