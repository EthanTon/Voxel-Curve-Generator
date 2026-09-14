from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ..block.connectivity_options import ConnectionOverride
from .domain import DomainId, _frozen_mapping


@dataclass(frozen=True, slots=True)
class DomainInteraction:
    """A rule between two domains, inherited by both sides' descendants.

    When several interactions match a pair of domains, the one declared
    closest to both domains wins, and the latest one wins a tie. A source or
    target of a parent domain covers that parent and all of its children, so
    ``DomainInteraction(transport, transport, connect=True)`` joins every
    transport subdomain together.

    ``connect``
        Whether source blocks connect to target blocks. None defers to the
        domains' own connectivity rules.
    ``overwrite``
        Whether source placements may replace target blocks. False forbids it;
        True exempts the pair from the target's ``protected`` rule. The
        placement mode still decides the rest. This is always directional.
    ``overrides``
        Block-level connection overrides scoped to this domain pair. Use the
        override's own ``symmetric`` flag to mirror it.
    ``set_properties``
        Block state patch applied to source blocks that border target blocks.
    ``properties``
        Metadata describing the relationship, read with
        ``DomainPolicy.interaction_properties``.
    ``grants``
        Properties added to the source domain's profile, and to the target's
        when the interaction is symmetric. A domain's own properties still
        take precedence over granted ones.
    ``symmetric``
        Also apply ``connect``, ``set_properties``, ``properties``, and
        ``grants`` from the target's side.
    """

    source: DomainId
    target: DomainId
    connect: bool | None = None
    overwrite: bool | None = None
    overrides: tuple[ConnectionOverride, ...] = ()
    set_properties: Mapping[str, str] | None = None
    properties: Mapping[str, object] = field(default_factory=dict)
    grants: Mapping[str, object] = field(default_factory=dict)
    symmetric: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "overrides", tuple(self.overrides))
        object.__setattr__(self, "properties", _frozen_mapping(self.properties))
        object.__setattr__(self, "grants", _frozen_mapping(self.grants))
        if self.set_properties is not None:
            object.__setattr__(self, "set_properties", dict(self.set_properties))
        for override in self.overrides:
            if (
                override.source_domains is not None
                or override.neighbor_domains is not None
            ):
                raise ValueError(
                    "Interaction overrides are scoped automatically; "
                    "leave source_domains and neighbor_domains unset"
                )
