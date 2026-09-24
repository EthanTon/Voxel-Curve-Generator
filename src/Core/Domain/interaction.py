from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ..block.connectivity_options import ConnectionOverride
from .domain import DomainId, _frozen_mapping


@dataclass(frozen=True, slots=True)
class DomainInteraction:
    """A rule between two domains and their descendants.

    The interaction declared closest to both domains wins; the latest breaks
    ties.

    ``connect``: whether source blocks connect to target blocks (None defers).
    ``overwrite``: whether source placements may replace target blocks; always
    directional. ``overrides``: connection overrides scoped to the pair.
    ``set_properties``: state patch for source blocks bordering the target.
    ``properties``: relationship metadata. ``grants``: properties added to the
    source's profile (and the target's when symmetric), below its own.
    ``symmetric``: also apply connect, set_properties, properties and grants
    from the target's side.
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
