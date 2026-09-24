from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from ..block.connectivity import ConnectivityDomainId
from .replacement import ReplacementRules
from .rules import ConnectivityRules, PlacementRules, TerrainRules


DomainId = ConnectivityDomainId
DomainRef = DomainId | str


def _frozen_mapping(values: Mapping[str, object] | None) -> Mapping[str, object]:
    return MappingProxyType(dict(values or {}))


class DomainLookup:
    # Resolves a domain by id or name; subclasses fill `_domains` and `_names`.
    _domains: dict[DomainId, Domain]
    _names: dict[str, DomainId]

    def __contains__(self, domain: object) -> bool:
        if isinstance(domain, str):
            return domain in self._names
        return domain in self._domains

    def id_of(self, domain: DomainRef) -> DomainId:
        if isinstance(domain, str):
            try:
                return self._names[domain]
            except KeyError:
                raise KeyError(f"Unknown domain {domain!r}") from None
        if domain not in self._domains:
            raise KeyError(f"Unknown domain {domain!r}")
        return domain

    def get(self, domain: DomainRef) -> Domain:
        return self._domains[self.id_of(domain)]


@dataclass(frozen=True, slots=True)
class Domain:
    """A named group of placements sharing connectivity and placement rules.

    Rules left as None, and unset properties, come from the nearest parent.
    """

    domain_id: DomainId
    name: str
    parents: tuple[DomainId, ...] = ()
    properties: Mapping[str, object] = field(default_factory=dict)
    connectivity: ConnectivityRules = ConnectivityRules()
    placement: PlacementRules = PlacementRules()
    terrain: TerrainRules = TerrainRules()
    replacement: ReplacementRules | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parents", tuple(self.parents))
        object.__setattr__(self, "properties", _frozen_mapping(self.properties))
        if self.domain_id in self.parents:
            raise ValueError(f"Domain {self.name!r} cannot be its own parent")
        if len(set(self.parents)) != len(self.parents):
            raise ValueError(f"Domain {self.name!r} lists a parent twice")
