from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

from .states import BlockState


ConnectivityDomainId = int
Direction = str


class ConnectionAction(Enum):
    """What an override does after it matches a block pair."""

    DEFAULT = "default"
    CONNECT = "connect"
    DISCONNECT = "disconnect"


@dataclass(frozen=True, slots=True)
class ConnectionDecision:
    action: ConnectionAction = ConnectionAction.DEFAULT
    value: str | None = None
    set_properties: Mapping[str, str] | None = None


DEFAULT_CONNECTION_DECISION = ConnectionDecision()


@dataclass(frozen=True, slots=True)
class ConnectionOverride:
    """A scoped exception to normal Minecraft connectivity.

    ``value`` selects a non-standard visual connection when the source rule has
    more than two states. Examples are ``"tall"`` for a wall and ``"up"`` for
    redstone. Omitting it uses that rule's normal connected value.

    Block, behavior, tag, domain, property, and direction selectors can be
    combined. ``set_properties`` applies an arbitrary state patch after the
    built-in resolver, including for blocks whose neighbor rule is ``NONE``.

    Direction is always viewed from the source block. With ``symmetric=True``,
    a rule for east also matches the reverse block looking west.
    """

    action: ConnectionAction = ConnectionAction.CONNECT
    source_domains: frozenset[ConnectivityDomainId] | None = None
    neighbor_domains: frozenset[ConnectivityDomainId] | None = None
    source_blocks: frozenset[str] | None = None
    neighbor_blocks: frozenset[str] | None = None
    source_behaviors: frozenset[str] | None = None
    neighbor_behaviors: frozenset[str] | None = None
    directions: frozenset[Direction] | None = None
    value: str | None = None
    symmetric: bool = False
    source_tags: frozenset[str] | None = None
    neighbor_tags: frozenset[str] | None = None
    source_properties: Mapping[str, frozenset[str]] | None = None
    neighbor_properties: Mapping[str, frozenset[str]] | None = None
    set_properties: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        for name in (
            "source_domains",
            "neighbor_domains",
            "source_blocks",
            "neighbor_blocks",
            "source_behaviors",
            "neighbor_behaviors",
            "source_tags",
            "neighbor_tags",
            "directions",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, frozenset):
                object.__setattr__(self, name, frozenset(value))
        for name in ("source_properties", "neighbor_properties"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    {
                        key: values
                        if isinstance(values, frozenset)
                        else frozenset(
                            (values,) if isinstance(values, str) else values
                        )
                        for key, values in value.items()
                    },
                )
        if self.set_properties is not None:
            object.__setattr__(self, "set_properties", dict(self.set_properties))


class ConnectivityOptions:
    """Ordered user overrides; the last matching rule wins."""

    def __init__(
        self,
        overrides: Iterable[ConnectionOverride] = (),
    ) -> None:
        self.overrides = tuple(overrides)

    def decide(
        self,
        source: BlockState,
        source_domain: ConnectivityDomainId,
        neighbor: BlockState,
        neighbor_domain: ConnectivityDomainId,
        direction: Direction,
        source_behavior: str | None = None,
        neighbor_behavior: str | None = None,
        source_tags: frozenset[str] = frozenset(),
        neighbor_tags: frozenset[str] = frozenset(),
    ) -> ConnectionDecision:
        for override in reversed(self.overrides):
            if _matches(
                override,
                source,
                source_domain,
                neighbor,
                neighbor_domain,
                direction,
                source_behavior,
                neighbor_behavior,
                source_tags,
                neighbor_tags,
            ):
                return ConnectionDecision(
                    override.action, override.value, override.set_properties
                )
            if override.symmetric and _matches_reverse(
                override,
                source,
                source_domain,
                neighbor,
                neighbor_domain,
                direction,
                source_behavior,
                neighbor_behavior,
                source_tags,
                neighbor_tags,
            ):
                return ConnectionDecision(
                    override.action, override.value, override.set_properties
                )
        return DEFAULT_CONNECTION_DECISION


def _matches(
    override: ConnectionOverride,
    source: BlockState,
    source_domain: ConnectivityDomainId,
    neighbor: BlockState,
    neighbor_domain: ConnectivityDomainId,
    direction: Direction,
    source_behavior: str | None,
    neighbor_behavior: str | None,
    source_tags: frozenset[str],
    neighbor_tags: frozenset[str],
) -> bool:
    return (
        _contains(override.source_domains, source_domain)
        and _contains(override.neighbor_domains, neighbor_domain)
        and _contains(override.source_blocks, source.name)
        and _contains(override.neighbor_blocks, neighbor.name)
        and _contains(override.source_behaviors, source_behavior)
        and _contains(override.neighbor_behaviors, neighbor_behavior)
        and _contains_all(source_tags, override.source_tags)
        and _contains_all(neighbor_tags, override.neighbor_tags)
        and _properties_match(source, override.source_properties)
        and _properties_match(neighbor, override.neighbor_properties)
        and _contains(override.directions, direction)
    )


def _matches_reverse(
    override: ConnectionOverride,
    source: BlockState,
    source_domain: ConnectivityDomainId,
    neighbor: BlockState,
    neighbor_domain: ConnectivityDomainId,
    direction: Direction,
    source_behavior: str | None,
    neighbor_behavior: str | None,
    source_tags: frozenset[str],
    neighbor_tags: frozenset[str],
) -> bool:
    return (
        _contains(override.source_domains, neighbor_domain)
        and _contains(override.neighbor_domains, source_domain)
        and _contains(override.source_blocks, neighbor.name)
        and _contains(override.neighbor_blocks, source.name)
        and _contains(override.source_behaviors, neighbor_behavior)
        and _contains(override.neighbor_behaviors, source_behavior)
        and _contains_all(neighbor_tags, override.source_tags)
        and _contains_all(source_tags, override.neighbor_tags)
        and _properties_match(neighbor, override.source_properties)
        and _properties_match(source, override.neighbor_properties)
        and _contains(override.directions, _opposite(direction))
    )


def _contains(values: frozenset[object] | None, value: object) -> bool:
    return values is None or value in values


def _contains_all(
    actual: frozenset[str], required: frozenset[str] | None
) -> bool:
    return required is None or required <= actual


def _properties_match(
    state: BlockState,
    required: Mapping[str, frozenset[str]] | None,
) -> bool:
    return required is None or all(
        state.get(key) in values for key, values in required.items()
    )


def _opposite(direction: Direction) -> Direction:
    return {
        "down": "up",
        "up": "down",
        "north": "south",
        "east": "west",
        "south": "north",
        "west": "east",
    }[direction]
