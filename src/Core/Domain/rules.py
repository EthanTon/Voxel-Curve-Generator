from __future__ import annotations

from dataclasses import dataclass

from ..block.connectivity_options import ConnectionOverride
from ..modes import PlacementMode, TerrainMaskMode


# Every rule field uses None for "inherit from the parent domains". Values that
# are still None after inheritance fall back to the world defaults.


@dataclass(frozen=True, slots=True)
class ConnectivityRules:
    """How blocks in a domain connect to their neighbors.

    ``connects_to_self`` controls connections between blocks of the same
    domain. ``connects_to_any`` lets the domain connect to every domain that
    has no explicit interaction with it, which suits terrain that fences and
    walls should attach to.

    ``overrides`` are block-level exceptions scoped to this domain and its
    descendants. Their source domains are filled in when the policy compiles,
    so they must not set ``source_domains`` themselves.
    """

    connects_to_self: bool | None = None
    connects_to_any: bool | None = None
    overrides: tuple[ConnectionOverride, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "overrides", tuple(self.overrides))
        for override in self.overrides:
            if override.source_domains is not None:
                raise ValueError(
                    "Domain overrides are scoped automatically; "
                    "leave source_domains unset"
                )


@dataclass(frozen=True, slots=True)
class PlacementRules:
    """Defaults for placing fields into a world under a domain.

    ``mode`` and ``replace`` are the defaults ``DomainRegistry.place`` passes
    to ``BlockWorld.add_field``. ``protected`` stops other domains from
    overwriting this domain's blocks unless an interaction allows it.
    """

    mode: PlacementMode | None = None
    replace: str | None = None
    protected: bool | None = None


@dataclass(frozen=True, slots=True)
class TerrainRules:
    """How a domain relates to terrain.

    ``is_terrain`` marks the domain that owns a world's base blocks.
    ``mask_mode`` is the default terrain mask for placements in this domain;
    it only takes effect with ``PlacementMode.TERRAIN_MASK``.
    """

    is_terrain: bool | None = None
    mask_mode: TerrainMaskMode | None = None
