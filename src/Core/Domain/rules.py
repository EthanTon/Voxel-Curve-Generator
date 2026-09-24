from __future__ import annotations

from dataclasses import dataclass

from ..block.connectivity_options import ConnectionOverride
from ..modes import PlacementMode, TerrainMaskMode


# None means inherit from the parents, then fall back to the world default.


@dataclass(frozen=True, slots=True)
class ConnectivityRules:
    """How blocks in a domain connect to their neighbors.

    ``connects_to_any`` connects to every domain without an explicit
    interaction. ``stairs_join_diagonals`` lets stairs take corner shapes
    from diagonal neighbours. ``overrides`` are scoped to this domain and its
    descendants, so they must leave ``source_domains`` unset.
    """

    connects_to_self: bool | None = None
    connects_to_any: bool | None = None
    overrides: tuple[ConnectionOverride, ...] = ()
    stairs_join_diagonals: bool | None = None

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
    """Placement defaults. ``protected`` blocks overwrites by other domains
    unless an interaction allows them.
    """

    mode: PlacementMode | None = None
    replace: str | None = None
    protected: bool | None = None


@dataclass(frozen=True, slots=True)
class TerrainRules:
    """``is_terrain`` marks the domain owning a world's base blocks;
    ``mask_mode`` applies with ``PlacementMode.TERRAIN_MASK``.
    """

    is_terrain: bool | None = None
    mask_mode: TerrainMaskMode | None = None
