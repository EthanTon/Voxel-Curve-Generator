from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping
import dataclasses
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeVar

from ..block.connectivity import BlockConnectionPredicate, NeighborResolver
from ..block.connectivity_options import (
    ConnectionAction,
    ConnectionOverride,
    ConnectivityOptions,
)
from ..block.registry import BlockRegistry
from ..constants import DEFAULT_BLOCK, DEFAULT_PLACEMENT_MODE, DEFAULT_TERRAIN_MASK_MODE
from ..coordinates import BlockPos
from ..modes import PlacementMode, TerrainMaskMode
from .domain import Domain, DomainId, DomainLookup, DomainRef
from .interaction import DomainInteraction
from .replacement import ReplacementRules


if TYPE_CHECKING:
    from ..Field.block_field import BlockField
    from ..World.block_world import BlockTransform, BlockWorld, PlacementId


_T = TypeVar("_T")
# (combined lineage distance, farthest side's distance, -declaration order)
_Rank = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class DomainProfile:
    """A domain's effective rules after inheritance and interaction grants."""

    domain: Domain
    lineage: tuple[DomainId, ...]
    connects_to_self: bool
    connects_to_any: bool
    placement_mode: PlacementMode
    replace: str
    replacement: ReplacementRules | None
    protected: bool
    is_terrain: bool
    terrain_mask_mode: TerrainMaskMode
    properties: Mapping[str, object]
    stairs_join_diagonals: bool = False

    @property
    def domain_id(self) -> DomainId:
        return self.domain.domain_id

    @property
    def name(self) -> str:
        return self.domain.name


class _PolicyRule:
    __slots__ = ("policy",)

    def __init__(self, policy: DomainPolicy) -> None:
        self.policy = policy


class DomainConnectionRule(_PolicyRule):
    """``NeighborResolver.domains_can_connect`` backed by a compiled policy."""

    __slots__ = ()

    def __call__(self, source: DomainId, neighbor: DomainId) -> bool:
        return self.policy.can_connect(source, neighbor)

    @property
    def policy_token(self) -> tuple[object, ...]:
        return self.policy.connection_token


class DomainStairRule(_PolicyRule):
    """``NeighborResolver.stairs_join_diagonals`` backed by a compiled policy."""

    __slots__ = ()

    def __call__(self, domain: DomainId) -> bool:
        return self.policy.stairs_join_diagonals(domain)

    @property
    def policy_token(self) -> tuple[object, ...]:
        return self.policy.connection_token


class DomainPlacementRule(_PolicyRule):
    """``BlockWorld.placement_policy`` backed by a compiled policy."""

    __slots__ = ()

    def __call__(self, upper: DomainId, lower: DomainId) -> bool:
        return self.policy.can_overwrite(upper, lower)

    @property
    def policy_token(self) -> tuple[object, ...]:
        return self.policy.placement_token


class DomainPolicy(DomainLookup):
    """An immutable compiled view of domains and interactions; recompile and
    rebind after editing the registry.
    """

    def __init__(
        self,
        domains: Iterable[Domain],
        interactions: Iterable[DomainInteraction] = (),
    ) -> None:
        self._domains: dict[DomainId, Domain] = {}
        self._names: dict[str, DomainId] = {}
        for domain in domains:
            if domain.domain_id in self._domains:
                raise ValueError(f"Duplicate domain id {domain.domain_id}")
            if domain.name in self._names:
                raise ValueError(f"Duplicate domain name {domain.name!r}")
            self._domains[domain.domain_id] = domain
            self._names[domain.name] = domain.domain_id
        self._interactions = tuple(interactions)
        self._validate()

        self._lineages = {
            domain_id: self._linearize(domain_id) for domain_id in self._domains
        }
        self._members = {
            domain_id: frozenset(
                other
                for other, lineage in self._lineages.items()
                if domain_id in lineage
            )
            for domain_id in self._domains
        }
        self._profiles = {
            domain_id: self._build_profile(domain_id) for domain_id in self._domains
        }
        self._matches: dict[
            tuple[DomainId, DomainId],
            tuple[tuple[bool, DomainInteraction], ...],
        ] = {}
        self._connect_cache: dict[tuple[DomainId, DomainId], bool] = {}
        self._overwrite_cache: dict[tuple[DomainId, DomainId], bool] = {}

        overrides = self._compile_overrides()
        self.options = ConnectivityOptions(overrides) if overrides else None
        self.connection_rule = DomainConnectionRule(self)
        self.placement_rule = DomainPlacementRule(self)

        # Each token covers only the inputs its rule reads, so unrelated edits
        # keep resolved world state.
        self.connection_token: tuple[object, ...] = (
            "domain-connectivity",
            tuple(
                (
                    domain_id,
                    self._lineages[domain_id],
                    profile.connects_to_self,
                    profile.connects_to_any,
                )
                for domain_id, profile in sorted(self._profiles.items())
            ),
            tuple(
                (item.source, item.target, item.connect, item.symmetric)
                for item in self._interactions
                if item.connect is not None
            ),
        )
        self.placement_token: tuple[object, ...] = (
            "domain-placement",
            tuple(
                (domain_id, self._lineages[domain_id], profile.protected)
                for domain_id, profile in sorted(self._profiles.items())
            ),
            tuple(
                (item.source, item.target, item.overwrite)
                for item in self._interactions
                if item.overwrite is not None
            ),
        )

    @property
    def domains(self) -> tuple[Domain, ...]:
        return tuple(self._domains.values())

    @property
    def interactions(self) -> tuple[DomainInteraction, ...]:
        return self._interactions

    def profile(self, domain: DomainRef) -> DomainProfile:
        return self._profiles[self.id_of(domain)]

    def lineage(self, domain: DomainRef) -> tuple[DomainId, ...]:
        """The domain followed by its ancestors, nearest first."""
        return self._lineages[self.id_of(domain)]

    def members(self, domain: DomainRef) -> frozenset[DomainId]:
        """The domain and every domain that inherits from it."""
        return self._members[self.id_of(domain)]

    def is_a(self, domain: DomainRef, ancestor: DomainRef) -> bool:
        return self.id_of(ancestor) in self.lineage(domain)

    def terrain_domain(self) -> DomainId | None:
        """The single top-level terrain domain, if one is declared."""
        roots = [
            domain_id
            for domain_id, profile in self._profiles.items()
            if profile.is_terrain
            and not any(
                self._profiles[parent].is_terrain
                for parent in self._domains[domain_id].parents
            )
        ]
        if len(roots) > 1:
            names = ", ".join(sorted(self._domains[item].name for item in roots))
            raise ValueError(f"Several terrain domains are declared: {names}")
        return roots[0] if roots else None

    def can_connect(self, source: DomainId, neighbor: DomainId) -> bool:
        # Unregistered ids are allowed for a world's implicit domain 0.
        key = (source, neighbor)
        cached = self._connect_cache.get(key)
        if cached is not None:
            return cached
        result = self._first(
            source, neighbor, lambda forward, item: item.connect
        )
        if result is None:
            if source == neighbor:
                profile = self._profiles.get(source)
                result = True if profile is None else profile.connects_to_self
            else:
                result = self._connects_to_any(source) or self._connects_to_any(
                    neighbor
                )
        self._connect_cache[key] = result
        return result

    def can_overwrite(self, upper: DomainId, lower: DomainId) -> bool:
        key = (upper, lower)
        cached = self._overwrite_cache.get(key)
        if cached is not None:
            return cached
        result = self._first(
            upper,
            lower,
            lambda forward, item: item.overwrite if forward else None,
        )
        if result is None:
            profile = self._profiles.get(lower)
            result = upper == lower or profile is None or not profile.protected
        self._overwrite_cache[key] = result
        return result

    def interactions_between(
        self,
        source: DomainRef,
        target: DomainRef,
    ) -> tuple[DomainInteraction, ...]:
        """Interactions that apply from source to target, most specific first."""
        seen: set[int] = set()
        found: list[DomainInteraction] = []
        for _, interaction in self._matches_for(self.id_of(source), self.id_of(target)):
            if id(interaction) not in seen:
                seen.add(id(interaction))
                found.append(interaction)
        return tuple(found)

    def interaction_properties(
        self,
        source: DomainRef,
        target: DomainRef,
    ) -> Mapping[str, object]:
        """Relationship properties for a pair; more specific interactions win."""
        merged: dict[str, object] = {}
        for interaction in reversed(self.interactions_between(source, target)):
            merged.update(interaction.properties)
        return MappingProxyType(merged)

    def resolver(
        self,
        registry: BlockRegistry,
        *,
        blocks_can_connect: BlockConnectionPredicate | None = None,
        overrides: Iterable[ConnectionOverride] = (),
    ) -> NeighborResolver:
        """A resolver applying these rules; extra ``overrides`` win ties."""
        combined = (
            *(() if self.options is None else self.options.overrides),
            *overrides,
        )
        return NeighborResolver(
            registry,
            domains_can_connect=self.connection_rule,
            blocks_can_connect=blocks_can_connect,
            stairs_join_diagonals=DomainStairRule(self),
            options=ConnectivityOptions(combined) if combined else None,
        )

    def bind(self, world: BlockWorld, *, terrain: DomainRef | None = None) -> None:
        """Apply this policy's placement rules and terrain domain to a world."""
        terrain_id = self.terrain_domain() if terrain is None else self.id_of(terrain)
        if terrain_id is not None:
            world.set_base_domain(terrain_id)
        world.set_placement_policy(self.placement_rule)

    def place(
        self,
        world: BlockWorld,
        field: BlockField,
        domain: DomainRef,
        offset: BlockPos = BlockPos(0, 0, 0),
        *,
        placement_mode: PlacementMode | None = None,
        terrain_mask_mode: TerrainMaskMode | None = None,
        replace: str | None = None,
        quarter_turns: int = 0,
        transform: BlockTransform | None = None,
        sequence_index: int | None = None,
    ) -> PlacementId:
        """Add a field to a world using the domain's placement defaults."""
        profile = self.profile(domain)
        return world.add_field(
            field,
            offset,
            profile.placement_mode if placement_mode is None else placement_mode,
            (
                profile.terrain_mask_mode
                if terrain_mask_mode is None
                else terrain_mask_mode
            ),
            profile.replace if replace is None else replace,
            block_replacement=profile.replacement,
            quarter_turns=quarter_turns,
            transform=transform,
            connectivity_domain=profile.domain_id,
            sequence_index=sequence_index,
        )

    def _validate(self) -> None:
        for domain in self._domains.values():
            for parent in domain.parents:
                if parent not in self._domains:
                    raise ValueError(
                        f"Domain {domain.name!r} has unknown parent {parent}"
                    )
        for interaction in self._interactions:
            for end in (interaction.source, interaction.target):
                if end not in self._domains:
                    raise ValueError(f"Interaction refers to unknown domain {end}")

        visiting: set[DomainId] = set()
        finished: set[DomainId] = set()

        def visit(domain_id: DomainId) -> None:
            if domain_id in finished:
                return
            if domain_id in visiting:
                raise ValueError(
                    f"Domain inheritance cycle through "
                    f"{self._domains[domain_id].name!r}"
                )
            visiting.add(domain_id)
            for parent in self._domains[domain_id].parents:
                visit(parent)
            visiting.discard(domain_id)
            finished.add(domain_id)

        for domain_id in self._domains:
            visit(domain_id)

    def _linearize(self, domain_id: DomainId) -> tuple[DomainId, ...]:
        # Breadth-first, so every parent comes before any grandparent.
        lineage: list[DomainId] = []
        seen: set[DomainId] = set()
        queue = deque([domain_id])
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            lineage.append(current)
            queue.extend(self._domains[current].parents)
        return tuple(lineage)

    def _inherit(
        self,
        lineage: tuple[DomainId, ...],
        read: Callable[[Domain], _T | None],
    ) -> _T | None:
        for ancestor in lineage:
            value = read(self._domains[ancestor])
            if value is not None:
                return value
        return None

    def _build_profile(self, domain_id: DomainId) -> DomainProfile:
        lineage = self._lineages[domain_id]
        properties: dict[str, object] = {}
        for ancestor in reversed(lineage):
            for interaction in self._interactions:
                if interaction.source == ancestor or (
                    interaction.symmetric and interaction.target == ancestor
                ):
                    properties.update(interaction.grants)
            properties.update(self._domains[ancestor].properties)

        def inherit(read: Callable[[Domain], _T | None], default: _T) -> _T:
            value = self._inherit(lineage, read)
            return default if value is None else value

        return DomainProfile(
            domain=self._domains[domain_id],
            lineage=lineage,
            connects_to_self=inherit(lambda d: d.connectivity.connects_to_self, True),
            connects_to_any=inherit(lambda d: d.connectivity.connects_to_any, False),
            stairs_join_diagonals=inherit(
                lambda d: d.connectivity.stairs_join_diagonals, False
            ),
            placement_mode=inherit(lambda d: d.placement.mode, DEFAULT_PLACEMENT_MODE),
            replace=inherit(lambda d: d.placement.replace, DEFAULT_BLOCK),
            replacement=self._inherit(lineage, lambda d: d.replacement),
            protected=inherit(lambda d: d.placement.protected, False),
            is_terrain=inherit(lambda d: d.terrain.is_terrain, False),
            terrain_mask_mode=inherit(
                lambda d: d.terrain.mask_mode, DEFAULT_TERRAIN_MASK_MODE
            ),
            properties=MappingProxyType(properties),
        )

    def _matches_for(
        self,
        source: DomainId,
        target: DomainId,
    ) -> tuple[tuple[bool, DomainInteraction], ...]:
        key = (source, target)
        cached = self._matches.get(key)
        if cached is not None:
            return cached
        source_index = {
            domain_id: index
            for index, domain_id in enumerate(self._lineages.get(source, (source,)))
        }
        target_index = {
            domain_id: index
            for index, domain_id in enumerate(self._lineages.get(target, (target,)))
        }
        ranked: list[tuple[_Rank, bool, DomainInteraction]] = []
        for order, interaction in enumerate(self._interactions):
            ends = [(True, interaction.source, interaction.target)]
            if interaction.symmetric:
                ends.append((False, interaction.target, interaction.source))
            for forward, near, far in ends:
                i = source_index.get(near)
                j = target_index.get(far)
                if i is not None and j is not None:
                    ranked.append(((i + j, max(i, j), -order), forward, interaction))
        ranked.sort(key=lambda item: item[0])
        matches = tuple((forward, interaction) for _, forward, interaction in ranked)
        self._matches[key] = matches
        return matches

    def _first(
        self,
        source: DomainId,
        target: DomainId,
        read: Callable[[bool, DomainInteraction], bool | None],
    ) -> bool | None:
        for forward, interaction in self._matches_for(source, target):
            value = read(forward, interaction)
            if value is not None:
                return value
        return None

    def stairs_join_diagonals(self, domain_id: DomainId) -> bool:
        profile = self._profiles.get(domain_id)
        return profile is not None and profile.stairs_join_diagonals

    def _connects_to_any(self, domain_id: DomainId) -> bool:
        profile = self._profiles.get(domain_id)
        return profile is not None and profile.connects_to_any

    def _compile_overrides(self) -> tuple[ConnectionOverride, ...]:
        # The last match wins, so order general to specific: boundary patches,
        # domain overrides, pair overrides, each shallow lineage first.
        patches: list[ConnectionOverride] = []
        domain_overrides: list[ConnectionOverride] = []
        pair_overrides: list[ConnectionOverride] = []

        for domain in sorted(
            self._domains.values(),
            key=lambda item: len(self._lineages[item.domain_id]),
        ):
            members = self._members[domain.domain_id]
            for override in domain.connectivity.overrides:
                domain_overrides.append(
                    dataclasses.replace(override, source_domains=members)
                )

        for interaction in sorted(
            self._interactions,
            key=lambda item: (
                len(self._lineages[item.source]) + len(self._lineages[item.target])
            ),
        ):
            sources = self._members[interaction.source]
            targets = self._members[interaction.target]
            if interaction.set_properties:
                sides = [(sources, targets)]
                if interaction.symmetric:
                    sides.append((targets, sources))
                for near, far in sides:
                    patches.append(
                        ConnectionOverride(
                            action=ConnectionAction.DEFAULT,
                            source_domains=near,
                            neighbor_domains=far,
                            set_properties=interaction.set_properties,
                        )
                    )
            for override in interaction.overrides:
                pair_overrides.append(
                    dataclasses.replace(
                        override,
                        source_domains=sources,
                        neighbor_domains=targets,
                    )
                )

        return (*patches, *domain_overrides, *pair_overrides)
