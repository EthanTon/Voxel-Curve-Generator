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
    # Connectivity, once every feature is in. Per stamp would resolve each
    # feature against a world the rest had not reached yet.
    resolver = domains.resolver(registry)
    return world.resolve_all(resolver) if everything else world.resolve_dirty(resolver)
