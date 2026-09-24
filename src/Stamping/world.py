from __future__ import annotations

from ..Core.Domain import DomainRegistry
from ..Core.World.block_world import BlockWorld, ResolutionDelta
from ..Core.block.registry import BlockRegistry


def resolve(
    world: BlockWorld,
    registry: BlockRegistry,
    domains: DomainRegistry,
    *,
    everything: bool = False,
) -> ResolutionDelta:
    # Run once every feature is placed.
    resolver = domains.resolver(registry)
    return world.resolve_all(resolver) if everything else world.resolve_dirty(resolver)
