from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

from ..block.registry import BlockRegistry
from .domain import Domain, DomainId, DomainRef
from .interaction import DomainInteraction
from .policy import DomainPolicy, DomainProfile
from .rules import ConnectivityRules, PlacementRules, TerrainRules


if TYPE_CHECKING:
    from ..World.block_world import BlockWorld, PlacementId
    from ..block.connectivity import NeighborResolver
    from ..block.connectivity_options import ConnectionOverride
    from ..Field.block_field import BlockField


class DomainRegistry:
    """Mutable collection of domains and the interactions between them.

    Query and apply the rules through ``compile``, which returns a cached
    ``DomainPolicy`` until the registry changes. Worlds and resolvers keep the
    policy they were given, so call ``bind`` and ``resolver`` again after
    editing domains.

    Automatic ids start at 1: ``BlockWorld`` uses domain 0 for placements and
    base blocks that were never assigned one. Register a domain with
    ``domain_id=0`` to give that default domain rules of its own.
    """

    def __init__(
        self,
        domains: Iterable[Domain] = (),
        interactions: Iterable[DomainInteraction] = (),
    ) -> None:
        self._domains: dict[DomainId, Domain] = {}
        self._names: dict[str, DomainId] = {}
        self._interactions: list[DomainInteraction] = []
        self._next_id = 1
        self._revision = 0
        self._compiled: DomainPolicy | None = None
        for domain in domains:
            self.register(domain)
        for interaction in interactions:
            self.add_interaction(interaction)

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def domains(self) -> tuple[Domain, ...]:
        return tuple(self._domains.values())

    @property
    def interactions(self) -> tuple[DomainInteraction, ...]:
        return tuple(self._interactions)

    def __contains__(self, domain: object) -> bool:
        if isinstance(domain, str):
            return domain in self._names
        return domain in self._domains

    def __iter__(self) -> Iterator[Domain]:
        return iter(tuple(self._domains.values()))

    def __len__(self) -> int:
        return len(self._domains)

    def id_of(self, domain: DomainRef | Domain) -> DomainId:
        if isinstance(domain, Domain):
            domain = domain.domain_id
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

    def create(
        self,
        name: str,
        *,
        parents: Iterable[DomainRef | Domain] = (),
        properties: Mapping[str, object] | None = None,
        connectivity: ConnectivityRules | None = None,
        placement: PlacementRules | None = None,
        terrain: TerrainRules | None = None,
        domain_id: DomainId | None = None,
    ) -> Domain:
        if domain_id is None:
            while self._next_id in self._domains:
                self._next_id += 1
            domain_id = self._next_id
        return self.register(
            Domain(
                domain_id=domain_id,
                name=name,
                parents=tuple(self.id_of(parent) for parent in parents),
                properties=properties or {},
                connectivity=connectivity or ConnectivityRules(),
                placement=placement or PlacementRules(),
                terrain=terrain or TerrainRules(),
            )
        )

    def register(self, domain: Domain) -> Domain:
        if not domain.name:
            raise ValueError("Domain name must not be empty")
        if domain.domain_id in self._domains:
            raise ValueError(f"Domain id {domain.domain_id} is already registered")
        if domain.name in self._names:
            raise ValueError(f"Domain name {domain.name!r} is already registered")
        for parent in domain.parents:
            if parent not in self._domains:
                raise ValueError(
                    f"Domain {domain.name!r} has unknown parent {parent}; "
                    "register parents first"
                )
        self._domains[domain.domain_id] = domain
        self._names[domain.name] = domain.domain_id
        self._changed()
        return domain

    def update(self, domain: DomainRef | Domain, **changes: Any) -> Domain:
        """Replace fields of a registered domain, keeping its id."""
        domain_id = self.id_of(domain)
        if "domain_id" in changes:
            raise ValueError("A domain's id cannot change")
        if "parents" in changes:
            changes["parents"] = tuple(
                self.id_of(parent) for parent in changes["parents"]
            )
        current = self._domains[domain_id]
        updated = dataclasses.replace(current, **changes)
        if updated.name != current.name and updated.name in self._names:
            raise ValueError(f"Domain name {updated.name!r} is already registered")

        domains = dict(self._domains)
        domains[domain_id] = updated
        # Compiling validates parents and rejects inheritance cycles.
        policy = DomainPolicy(domains.values(), self._interactions)

        del self._names[current.name]
        self._names[updated.name] = domain_id
        self._domains = domains
        self._changed()
        self._compiled = policy
        return updated

    def interact(
        self,
        source: DomainRef | Domain,
        target: DomainRef | Domain,
        **options: Any,
    ) -> DomainInteraction:
        """Declare a rule between two domains; see ``DomainInteraction``."""
        return self.add_interaction(
            DomainInteraction(self.id_of(source), self.id_of(target), **options)
        )

    def add_interaction(self, interaction: DomainInteraction) -> DomainInteraction:
        self.id_of(interaction.source)
        self.id_of(interaction.target)
        self._interactions.append(interaction)
        self._changed()
        return interaction

    def remove_interaction(self, interaction: DomainInteraction) -> None:
        for index, existing in enumerate(self._interactions):
            if existing is interaction:
                del self._interactions[index]
                self._changed()
                return
        raise ValueError("Interaction is not registered")

    def compile(self) -> DomainPolicy:
        if self._compiled is None:
            self._compiled = DomainPolicy(self._domains.values(), self._interactions)
        return self._compiled

    def profile(self, domain: DomainRef | Domain) -> DomainProfile:
        return self.compile().profile(self.id_of(domain))

    def resolver(
        self,
        registry: BlockRegistry,
        *,
        overrides: Iterable[ConnectionOverride] = (),
        **options: Any,
    ) -> NeighborResolver:
        return self.compile().resolver(registry, overrides=overrides, **options)

    def bind(
        self,
        world: BlockWorld,
        *,
        terrain: DomainRef | Domain | None = None,
    ) -> None:
        self.compile().bind(
            world,
            terrain=None if terrain is None else self.id_of(terrain),
        )

    def place(
        self,
        world: BlockWorld,
        field: BlockField,
        domain: DomainRef | Domain,
        *args: Any,
        **kwargs: Any,
    ) -> PlacementId:
        return self.compile().place(world, field, self.id_of(domain), *args, **kwargs)

    def _changed(self) -> None:
        self._revision += 1
        self._compiled = None
