from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import floor

from ...Core.World.block_world import BlockWorld
from ...Core.coordinates import BlockPos
from .regions import contains
from .spec import OwnershipSpec


class Ownership:
    def __init__(self, rules: tuple[OwnershipSpec, ...]):
        self.buckets: dict[tuple[int, int], list[OwnershipSpec]] = {}
        for rule in rules:
            xs, zs = zip(*rule.boundary)
            for x in range(floor(min(xs) / 16), floor(max(xs) / 16) + 1):
                for z in range(floor(min(zs) / 16), floor(max(zs) / 16) + 1):
                    self.buckets.setdefault((x, z), []).append(rule)

    def owner_at(self, component: str, domain: str, position: BlockPos) -> str | None:
        matching = [rule for rule in self.buckets.get((position.x // 16, position.z // 16), ())
                    if (not rule.components or component in rule.components)
                    and (not rule.domains or domain in rule.domains)
                    and (rule.heights is None or rule.heights[0] <= position.y <= rule.heights[1])
                    and contains(rule.boundary, position.x + 0.5, position.z + 0.5)]
        if not matching:
            return None
        priority = max(rule.priority for rule in matching)
        owners = {rule.owner for rule in matching if rule.priority == priority}
        if len(owners) != 1:
            raise ValueError(f"conflicting ownership at {position.as_tuple()} in domain {domain!r}; set different priorities")
        return owners.pop()

    def allows(self, component: str, domain: str, position: BlockPos) -> bool:
        owner = self.owner_at(component, domain, position)
        return owner is None or owner == component


@dataclass(frozen=True, slots=True)
class OwnedMask:
    ownership: Ownership
    component: str
    domain: str
    surface: Callable[[BlockPos], bool] | None = None

    @property
    def spec(self):
        return getattr(self.surface, "spec", None)

    def __call__(self, position: BlockPos) -> bool:
        return self.ownership.allows(self.component, self.domain, position) and (
            self.surface is None or self.surface(position)
        )


class OwnedWorld(BlockWorld):
    def __init__(self, registry, rules: tuple[OwnershipSpec, ...]):
        super().__init__(registry)
        self.ownership = Ownership(rules)
        self.component: str | None = None
        self.domains = None
        self.owners: dict[int, str | None] = {}

    def add_field(self, field, *args, **options):
        if self.component is not None:
            domain = self.domains.get(options.get("connectivity_domain", 0)).name
            options["surface_mask"] = OwnedMask(self.ownership, self.component, domain, options.get("surface_mask"))
        placement = super().add_field(field, *args, **options)
        self.owners[placement] = self.component
        return placement

    def owner_at(self, position: BlockPos) -> str | None:
        source = self.source_at(position)
        if source is None:
            return None
        owner = self.owners.get(source.placement_id)
        if owner is None:
            # Automatically added joint supports inherit the component immediately above them.
            above = self.source_at(position + (0, 1, 0))
            owner = None if above is None else self.owners.get(above.placement_id)
            if owner is not None:
                owner = self.ownership.owner_at(owner, self.domains.get(source.domain).name, position) or owner
        return owner
