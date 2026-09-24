from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import cache
from typing import Callable, Iterable, Iterator, Mapping, MutableMapping, Protocol

from .connectivity_options import (
    ConnectionAction,
    ConnectionDecision,
    ConnectivityOptions,
    DEFAULT_CONNECTION_DECISION,
)
from .info import BlockInfo, NeighborRule, SolidFace
from .palette import BlockId
from .registry import BlockRegistry
from .states import BlockState
from ..coordinates import BlockPos


ConnectivityDomainId = int
Direction = str
_UNSEEN = object()

HORIZONTAL_DIRECTIONS: tuple[Direction, ...] = (
    "north",
    "east",
    "south",
    "west",
)
ALL_DIRECTIONS: tuple[Direction, ...] = (
    "down",
    "up",
    *HORIZONTAL_DIRECTIONS,
)
DIRECTION_OFFSETS: dict[Direction, BlockPos] = {
    "down": BlockPos(0, -1, 0),
    "up": BlockPos(0, 1, 0),
    "north": BlockPos(0, 0, -1),
    "east": BlockPos(1, 0, 0),
    "south": BlockPos(0, 0, 1),
    "west": BlockPos(-1, 0, 0),
}
_NEIGHBOR_DELTAS = tuple(DIRECTION_OFFSETS[direction].as_tuple() for direction in ALL_DIRECTIONS)
EDGE_OFFSETS: tuple[BlockPos, ...] = tuple(
    BlockPos(x, y, z)
    for x in (-1, 0, 1)
    for y in (-1, 0, 1)
    for z in (-1, 0, 1)
    if x * x + y * y + z * z == 2
)

def _toward(offset: BlockPos, direction: Direction) -> bool:
    axis = DIRECTION_OFFSETS[direction]
    return offset.x * axis.x + offset.y * axis.y + offset.z * axis.z == 1


# Each edge offset with the face directions it lies toward.
EDGE_DIRECTIONS: tuple[tuple[BlockPos, tuple[Direction, ...]], ...] = tuple(
    (offset, tuple(direction for direction in ALL_DIRECTIONS if _toward(offset, direction)))
    for offset in EDGE_OFFSETS
)
EDGES_TOWARD: dict[Direction, tuple[BlockPos, ...]] = {
    direction: tuple(offset for offset in EDGE_OFFSETS if _toward(offset, direction))
    for direction in ALL_DIRECTIONS
}
OPPOSITE_DIRECTION: dict[Direction, Direction] = {
    "down": "up",
    "up": "down",
    "north": "south",
    "east": "west",
    "south": "north",
    "west": "east",
}
FACE_BY_DIRECTION: dict[Direction, SolidFace] = {
    "down": SolidFace.DOWN,
    "up": SolidFace.UP,
    "north": SolidFace.NORTH,
    "east": SolidFace.EAST,
    "south": SolidFace.SOUTH,
    "west": SolidFace.WEST,
}
WALL_POST_COVER = 1
WALL_COVER_BY_DIRECTION: dict[Direction, int] = {
    "north": 2,
    "east": 4,
    "south": 8,
    "west": 16,
}

DomainPredicate = Callable[[ConnectivityDomainId, ConnectivityDomainId], bool]
BlockConnectionPredicate = Callable[
    [BlockState, BlockInfo, BlockState, BlockInfo, Direction],
    bool,
]


@dataclass(frozen=True, slots=True)
class DependencySpec:
    # How far beyond the six face neighbours a change must re-resolve.
    radius: BlockPos = BlockPos(0, 0, 0)
    horizontal_range: int = 0
    stream_halo: int | None = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius", BlockPos.coerce(self.radius))

    def affected_positions(self, position: BlockPos) -> set[BlockPos]:
        position = BlockPos.coerce(position)
        x, y, z = position.x, position.y, position.z
        return {BlockPos(x + dx, y + dy, z + dz) for dx, dy, dz in _affected_offsets(self)}


@cache
def _affected_offsets(spec: DependencySpec) -> tuple[tuple[int, int, int], ...]:
    offsets = []
    if any(spec.radius):
        radius = spec.radius
        offsets.extend(
            (dx, dy, dz)
            for dx in range(-radius.x, radius.x + 1)
            for dy in range(-radius.y, radius.y + 1)
            for dz in range(-radius.z, radius.z + 1)
        )
    for direction in HORIZONTAL_DIRECTIONS:
        offset = DIRECTION_OFFSETS[direction]
        for distance in range(1, spec.horizontal_range + 1):
            offsets.append((offset.x * distance, 0, offset.z * distance))
    return tuple(offsets)


DEFAULT_DEPENDENCY_SPEC = DependencySpec()
BUILTIN_DEPENDENCY_SPECS: dict[NeighborRule, DependencySpec] = {
    NeighborRule.REDSTONE: DependencySpec(radius=BlockPos(1, 1, 1)),
    # Stairs can take their shape from a diagonal neighbour.
    NeighborRule.STAIR: DependencySpec(radius=BlockPos(1, 0, 1)),
    NeighborRule.MOSS_CARPET: DependencySpec(radius=BlockPos(1, 1, 1)),
    NeighborRule.RAIL: DependencySpec(radius=BlockPos(2, 2, 2), stream_halo=2),
    NeighborRule.TRIPWIRE: DependencySpec(horizontal_range=42, stream_halo=42),
    NeighborRule.TRIPWIRE_HOOK: DependencySpec(horizontal_range=42, stream_halo=42),
    NeighborRule.LEAVES: DependencySpec(stream_halo=7),
    NeighborRule.SHELF: DependencySpec(stream_halo=None),
    NeighborRule.POINTED_DRIPSTONE: DependencySpec(stream_halo=None),
    NeighborRule.CHEST: DependencySpec(stream_halo=None),
    NeighborRule.SCAFFOLDING: DependencySpec(stream_halo=None),
}


@dataclass(frozen=True, slots=True)
class ConnectivityCell:
    block_id: BlockId
    domain: ConnectivityDomainId = 0
    source_token: object | None = None
    placement_id: int | None = None
    sequence_index: int | None = None


class ConnectivityView(Protocol):
    # Minimal mutable world interface required by neighbor resolution.

    def source_at(self, position: BlockPos) -> ConnectivityCell | None: ...

    def effective_at(self, position: BlockPos) -> ConnectivityCell | None: ...

    def write_resolution(
        self,
        position: BlockPos,
        source_token: object | None,
        block_id: BlockId | None,
    ) -> None: ...

    def clear_resolution(self, position: BlockPos) -> None: ...

    def iter_source_positions(self) -> Iterator[BlockPos]: ...


class _MappingConnectivityView:
    # Adapts plain position -> block id mappings to ConnectivityView.
    def __init__(
        self,
        blocks: Mapping[BlockPos, BlockId],
        domains: Mapping[BlockPos, ConnectivityDomainId] | None = None,
    ) -> None:
        self.blocks = blocks
        self.domains = domains or {}

    def source_at(self, position: BlockPos) -> ConnectivityCell | None:
        block_id = self.blocks.get(position)
        if block_id is None:
            return None
        return ConnectivityCell(
            block_id=block_id,
            domain=self.domains.get(position, 0),
            source_token=(position, block_id),
        )

    def effective_at(self, position: BlockPos) -> ConnectivityCell | None:
        return self.source_at(position)

    def write_resolution(
        self,
        position: BlockPos,
        source_token: object | None,
        block_id: BlockId | None,
    ) -> None:
        if not isinstance(self.blocks, MutableMapping):
            raise TypeError("Resolution requires a mutable block mapping")
        if block_id is None:
            self.blocks.pop(position, None)
        else:
            self.blocks[position] = block_id

    def clear_resolution(self, position: BlockPos) -> None:
        return None

    def iter_source_positions(self) -> Iterator[BlockPos]:
        return (BlockPos.coerce(position) for position in self.blocks)


class _EffectiveMapping(Mapping[BlockPos, int]):
    # One field of each effective cell, for handlers written against mappings.
    def __init__(self, view: ConnectivityView, field_name: str) -> None:
        self.view = view
        self.field_name = field_name

    def __getitem__(self, position: BlockPos) -> int:
        cell = self.view.effective_at(position)
        if cell is None:
            raise KeyError(position)
        return getattr(cell, self.field_name)

    def __iter__(self) -> Iterator[BlockPos]:
        return (
            position
            for position in self.view.iter_source_positions()
            if self.view.effective_at(position) is not None
        )

    def __len__(self) -> int:
        return sum(1 for _ in self)


@dataclass(frozen=True, slots=True)
class NeighborBlock:
    position: BlockPos
    block_id: BlockId
    state: BlockState
    info: BlockInfo
    domain: ConnectivityDomainId
    source_token: object | None = None
    placement_id: int | None = None
    sequence_index: int | None = None


@dataclass(slots=True)
class NeighborContext:
    """Read-only neighborhood supplied to a connectivity rule handler."""

    resolver: "NeighborResolver"
    position: BlockPos
    block_id: BlockId
    state: BlockState
    info: BlockInfo
    source: ConnectivityCell
    view: ConnectivityView
    _blocks: dict[BlockPos, NeighborBlock | None] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    @property
    def domain(self) -> ConnectivityDomainId:
        return self.source.domain

    @property
    def placement_id(self) -> int | None:
        return self.source.placement_id

    @property
    def sequence_index(self) -> int | None:
        return self.source.sequence_index

    @property
    def blocks(self) -> Mapping[BlockPos, BlockId]:
        return _EffectiveMapping(self.view, "block_id")

    @property
    def domains(self) -> Mapping[BlockPos, ConnectivityDomainId]:
        return _EffectiveMapping(self.view, "domain")

    def neighbor(self, direction: Direction) -> NeighborBlock | None:
        return self.neighbor_at(DIRECTION_OFFSETS[direction])

    def neighbor_at(self, offset: BlockPos) -> NeighborBlock | None:
        return self.block_at(_add_position(self.position, offset))

    def block_at(self, position: BlockPos) -> NeighborBlock | None:
        # Memoized: the view does not change while one position resolves.
        found = self._blocks.get(position, _UNSEEN)
        if found is not _UNSEEN:
            return found
        cell = self.view.effective_at(position)
        found = None if cell is None else NeighborBlock(
            position=position,
            block_id=cell.block_id,
            state=self.resolver.registry.get_state(cell.block_id),
            info=self.resolver.registry.get_info(cell.block_id),
            domain=cell.domain,
            source_token=cell.source_token,
            placement_id=cell.placement_id,
            sequence_index=cell.sequence_index,
        )
        self._blocks[position] = found
        return found

    def connects_domain(self, other: NeighborBlock | ConnectivityCell) -> bool:
        return self.resolver.domains_can_connect(self.domain, other.domain)

    def allows(self, direction: Direction, default: Callable[[NeighborBlock], bool]) -> bool:
        # Whether the neighbour toward `direction` counts: an override decides
        # if one matches, otherwise `default` does.
        neighbor = self.neighbor(direction)
        decision = self.connection_decision(direction)
        return bool(
            neighbor is not None
            and decision.action != ConnectionAction.DISCONNECT
            and (decision.action == ConnectionAction.CONNECT or default(neighbor))
        )

    def connects(
        self,
        direction: Direction,
        accepted_rules: frozenset[NeighborRule] = frozenset(),
        *,
        solid_faces: bool = True,
    ) -> bool:
        neighbor = self.neighbor(direction)
        if neighbor is None:
            return self.diagonal_decision(direction).action == ConnectionAction.CONNECT

        decision = self.connection_decision(direction)
        if decision.action == ConnectionAction.CONNECT:
            return True
        if decision.action == ConnectionAction.DISCONNECT:
            return False

        if self.diagonal_decision(direction).action == ConnectionAction.CONNECT:
            return True

        if not self.connects_domain(neighbor):
            return False

        predicate = self.resolver.blocks_can_connect
        if predicate is not None:
            return predicate(
                self.state,
                self.info,
                neighbor.state,
                neighbor.info,
                direction,
            )

        if neighbor.info.neighbor_rule in accepted_rules:
            if neighbor.info.neighbor_rule == NeighborRule.FENCE_GATE:
                return _gate_connects_to_direction(neighbor.state, direction)
            if (
                self.info.neighbor_rule == NeighborRule.FENCE
                and neighbor.info.neighbor_rule == NeighborRule.FENCE
            ):
                return _same_fence_group(
                    self.state,
                    self.info,
                    neighbor.state,
                    neighbor.info,
                )
            return True

        if not solid_faces:
            return False

        touching_face = FACE_BY_DIRECTION[OPPOSITE_DIRECTION[direction]]
        return bool(
            not neighbor.info.connection_exception
            and neighbor.info.solid_faces & touching_face
        )

    def connection_decision(self, direction: Direction) -> ConnectionDecision:
        neighbor = self.neighbor(direction)
        return self.connection_decision_for(neighbor, direction)

    def connection_decision_for(
        self,
        neighbor: NeighborBlock | None,
        direction: Direction,
    ) -> ConnectionDecision:
        if neighbor is None or self.resolver.options is None:
            return DEFAULT_CONNECTION_DECISION
        offset = neighbor.position - self.position
        return self.resolver.options.decide(
            self.state,
            self.domain,
            neighbor.state,
            neighbor.domain,
            direction,
            self.info.behavior,
            neighbor.info.behavior,
            self.info.tags,
            neighbor.info.tags,
            tuple(offset),
        )

    def diagonal_decision(self, direction: Direction) -> ConnectionDecision:
        if self.resolver.options is None or not self.resolver.has_edge_overrides:
            return DEFAULT_CONNECTION_DECISION
        for offset in EDGES_TOWARD[direction]:
            neighbor = self.neighbor_at(offset)
            decision = self.connection_decision_for(neighbor, direction)
            if decision.action != ConnectionAction.DEFAULT:
                return decision
        return DEFAULT_CONNECTION_DECISION


NeighborHandler = Callable[[NeighborContext], BlockState | None]


class NeighborResolver:
    def __init__(
        self,
        registry: BlockRegistry,
        *,
        domains_can_connect: DomainPredicate | None = None,
        blocks_can_connect: BlockConnectionPredicate | None = None,
        options: ConnectivityOptions | None = None,
        stairs_join_diagonals: Callable[[ConnectivityDomainId], bool] | None = None,
    ) -> None:
        self.registry = registry
        self.domains_can_connect = domains_can_connect or _same_domain
        self.stairs_join_diagonals = stairs_join_diagonals
        self.blocks_can_connect = blocks_can_connect
        self.options = options
        self.has_edge_overrides = bool(
            options and any(override.max_distance > 1 for override in options.overrides)
        )
        self._dependency_specs = dict(BUILTIN_DEPENDENCY_SPECS)
        self._handlers: dict[NeighborRule, NeighborHandler] = {
            NeighborRule.NONE: _unchanged,
            NeighborRule.FENCE: _resolve_fence,
            NeighborRule.PANE: _resolve_pane,
            NeighborRule.WALL: _resolve_wall,
            NeighborRule.FENCE_GATE: _resolve_fence_gate,
            NeighborRule.STAIR: _resolve_stair,
            NeighborRule.RAIL: _resolve_rail,
            NeighborRule.REDSTONE: _resolve_redstone,
            NeighborRule.TRIPWIRE: _resolve_tripwire,
            NeighborRule.CHORUS: _resolve_chorus,
            NeighborRule.MULTIFACE: _resolve_multiface,
            NeighborRule.VINE: _resolve_vine,
            NeighborRule.FIRE: _resolve_fire,
            NeighborRule.MOSS_CARPET: _resolve_moss_carpet,
            NeighborRule.SCAFFOLDING: _resolve_scaffolding,
            NeighborRule.SHELF: _resolve_shelf,
            NeighborRule.POINTED_DRIPSTONE: _resolve_pointed_dripstone,
            NeighborRule.LEAVES: _resolve_leaves,
            NeighborRule.CHEST: _resolve_chest,
            NeighborRule.STEM: _resolve_stem,
            NeighborRule.HANGING: _resolve_hanging,
            NeighborRule.SUPPORT: _resolve_support,
            NeighborRule.MULTIBLOCK: _resolve_multiblock,
            NeighborRule.PROPAGATED: _unchanged,
            NeighborRule.HUGE_MUSHROOM: _resolve_huge_mushroom,
            NeighborRule.SNOWY: _resolve_snowy,
            NeighborRule.GROWING_PLANT: _resolve_growing_plant,
            NeighborRule.HANGING_MOSS: _resolve_hanging_moss,
            NeighborRule.CAMPFIRE: _resolve_campfire,
            NeighborRule.TRIPWIRE_HOOK: _resolve_tripwire_hook,
        }

    def register_handler(
        self,
        rule: NeighborRule,
        handler: NeighborHandler,
        dependency_spec: DependencySpec | None = None,
    ) -> None:
        self._handlers[rule] = handler
        if dependency_spec is not None:
            self._dependency_specs[rule] = dependency_spec

    def dependency_spec(self, rule: NeighborRule) -> DependencySpec:
        return self._dependency_specs.get(rule, DEFAULT_DEPENDENCY_SPEC)

    @property
    def policy_token(self) -> tuple[object, ...]:
        # Changes whenever resolution rules change, invalidating cached results.
        return (
            _predicate_token(self.domains_can_connect),
            _predicate_token(self.blocks_can_connect),
            _predicate_token(self.stairs_join_diagonals),
            repr(None if self.options is None else self.options.overrides),
            tuple(
                sorted(
                    (rule.value, id(handler))
                    for rule, handler in self._handlers.items()
                )
            ),
            tuple(
                sorted(
                    (rule.value, spec)
                    for rule, spec in self._dependency_specs.items()
                )
            ),
        )

    def dependency_positions(
        self,
        position: BlockPos,
        rules: Iterable[NeighborRule],
    ) -> set[BlockPos]:
        affected: set[BlockPos] = set()
        for rule in rules:
            affected.update(self.dependency_spec(rule).affected_positions(position))
        return affected

    def resolve_state(
        self,
        position: BlockPos,
        blocks: Mapping[BlockPos, BlockId] | ConnectivityView,
        domains: Mapping[BlockPos, ConnectivityDomainId] | None = None,
    ) -> BlockState | None:
        view = _as_connectivity_view(blocks, domains)
        source = view.source_at(position)
        if source is None:
            return None

        info = self.registry.get_info(source.block_id)
        context = NeighborContext(
            resolver=self,
            position=position,
            block_id=source.block_id,
            state=self.registry.get_state(source.block_id),
            info=info,
            source=source,
            view=view,
        )
        state = self._handlers[info.neighbor_rule](context)
        if state is None:
            return None
        if self.options is None:
            return state
        updates: dict[str, str] = {}
        for direction in ALL_DIRECTIONS:
            decision = context.connection_decision(direction)
            if decision.set_properties:
                updates.update(decision.set_properties)
        if self.has_edge_overrides:
            for offset, directions in EDGE_DIRECTIONS:
                neighbor = context.neighbor_at(offset)
                if neighbor is None:
                    continue
                for direction in directions:
                    decision = context.connection_decision_for(neighbor, direction)
                    if decision.set_properties:
                        updates.update(decision.set_properties)
        return _with_properties(state, updates) if updates else state

    def resolve_id(
        self,
        position: BlockPos,
        blocks: Mapping[BlockPos, BlockId] | ConnectivityView,
        domains: Mapping[BlockPos, ConnectivityDomainId] | None = None,
    ) -> BlockId | None:
        view = _as_connectivity_view(blocks, domains)
        source = view.source_at(position)
        if source is None:
            return None

        state = self.resolve_state(position, view)
        if state is None:
            return None
        if state == self.registry.get_state(source.block_id):
            return source.block_id
        return self.registry.get_or_add_variant(source.block_id, state)

    def resolve_dirty(
        self,
        blocks: MutableMapping[BlockPos, BlockId] | ConnectivityView,
        dirty: Iterable[BlockPos | object],
        domains: Mapping[BlockPos, ConnectivityDomainId] | None = None,
    ) -> set[BlockPos]:
        view = _as_connectivity_view(blocks, domains)
        initial: set[BlockPos] = set()
        for item in dirty:
            position, changed_rules = _dirty_position_and_rules(item, self.registry)
            neighbors = neighboring_positions(position)
            initial.add(position)
            initial.update(neighbors)
            if self.has_edge_overrides:
                initial.update(_add_position(position, offset) for offset in EDGE_OFFSETS)
            nearby_rules = {
                self.registry.get_info(cell.block_id).neighbor_rule
                for nearby in (position, *neighbors)
                if (cell := view.source_at(nearby)) is not None
            }
            nearby_rules.update(changed_rules)
            initial.update(self.dependency_positions(position, nearby_rules))

        queue = deque(
            position
            for position in sorted(initial, key=_position_order)
            if view.source_at(position) is not None
        )
        queued = set(queue)
        changed: set[BlockPos] = set()

        while queue:
            position = queue.popleft()
            queued.discard(position)
            source = view.source_at(position)
            if source is None:
                continue
            old_effective = view.effective_at(position)
            old_rule = self.registry.get_info(source.block_id).neighbor_rule

            new_id = self.resolve_id(position, view)
            view.write_resolution(position, source.source_token, new_id)
            new_effective = view.effective_at(position)
            old_id = None if old_effective is None else old_effective.block_id
            effective_id = None if new_effective is None else new_effective.block_id
            if effective_id == old_id:
                continue
            changed.add(position)

            affected = set(neighboring_positions(position))
            if self.has_edge_overrides:
                affected.update(_add_position(position, offset) for offset in EDGE_OFFSETS)
            new_rule = (
                old_rule
                if new_effective is None
                else self.registry.get_info(new_effective.block_id).neighbor_rule
            )
            affected.update(self.dependency_positions(position, {old_rule, new_rule}))
            for neighbor_position in sorted(affected, key=_position_order):
                if (
                    view.source_at(neighbor_position) is not None
                    and neighbor_position not in queued
                ):
                    queue.append(neighbor_position)
                    queued.add(neighbor_position)

        return changed

    def resolve_all(
        self,
        blocks: MutableMapping[BlockPos, BlockId] | ConnectivityView,
        domains: Mapping[BlockPos, ConnectivityDomainId] | None = None,
    ) -> set[BlockPos]:
        view = _as_connectivity_view(blocks, domains)
        return self.resolve_dirty(view, view.iter_source_positions())


def _position_order(position: BlockPos) -> tuple[int, int, int]:
    return position.x, position.y, position.z


def _as_connectivity_view(
    blocks: Mapping[BlockPos, BlockId] | ConnectivityView,
    domains: Mapping[BlockPos, ConnectivityDomainId] | None = None,
) -> ConnectivityView:
    if all(
        hasattr(blocks, name)
        for name in (
            "source_at",
            "effective_at",
            "write_resolution",
            "clear_resolution",
            "iter_source_positions",
        )
    ):
        return blocks  # type: ignore[return-value]
    return _MappingConnectivityView(blocks, domains)


def _dirty_position_and_rules(
    item: BlockPos | object,
    registry: BlockRegistry,
) -> tuple[BlockPos, set[NeighborRule]]:
    if isinstance(item, (BlockPos, tuple)) and len(item) == 3:
        return BlockPos.coerce(item), set()
    position = getattr(item, "position")
    rules: set[NeighborRule] = set()
    for name in ("before", "after"):
        cell = getattr(item, name, None)
        if cell is not None:
            rules.add(registry.get_info(cell.block_id).neighbor_rule)
    return position, rules


def neighboring_positions(position: BlockPos) -> tuple[BlockPos, ...]:
    position = BlockPos.coerce(position)
    x, y, z = position.x, position.y, position.z
    return tuple(BlockPos(x + dx, y + dy, z + dz) for dx, dy, dz in _NEIGHBOR_DELTAS)


def _same_domain(left: ConnectivityDomainId, right: ConnectivityDomainId) -> bool:
    return left == right


def _predicate_token(predicate: object) -> object:
    # Predicates may describe their rules by value; identity is the fallback.
    token = getattr(predicate, "policy_token", None)
    return id(predicate) if token is None else token


def _add_position(left: BlockPos, right: BlockPos) -> BlockPos:
    left, right = BlockPos.coerce(left), BlockPos.coerce(right)
    return BlockPos(left.x + right.x, left.y + right.y, left.z + right.z)


def _with_properties(state: BlockState, updates: Mapping[str, str]) -> BlockState:
    properties = dict(state.properties)
    properties.update(updates)
    return BlockState.create(state.name, properties)


def _boolean_connections(
    context: NeighborContext,
    directions: Iterable[Direction],
    accepted_rules: frozenset[NeighborRule],
    *,
    solid_faces: bool = True,
) -> BlockState:
    return _with_properties(
        context.state,
        {
            direction: str(
                context.connects(
                    direction,
                    accepted_rules,
                    solid_faces=solid_faces,
                )
            ).lower()
            for direction in directions
        },
    )


def _unchanged(context: NeighborContext) -> BlockState:
    return context.state


def _resolve_fence(context: NeighborContext) -> BlockState:
    return _boolean_connections(
        context,
        HORIZONTAL_DIRECTIONS,
        frozenset((NeighborRule.FENCE, NeighborRule.FENCE_GATE)),
    )


def _resolve_pane(context: NeighborContext) -> BlockState:
    return _boolean_connections(
        context,
        HORIZONTAL_DIRECTIONS,
        frozenset((NeighborRule.PANE, NeighborRule.WALL)),
    )


def _resolve_wall(context: NeighborContext) -> BlockState:
    connected = {
        direction: _wall_connection(context, direction)
        for direction in HORIZONTAL_DIRECTIONS
    }
    above = context.neighbor("up")
    above_wall_cover = (
        above.info.wall_cover
        if above is not None and above.info.wall_cover
        else (
            31
            if above is not None and above.info.solid_faces & SolidFace.DOWN
            else 0
        )
    )
    covered_from_above = bool(
        above is not None
        and context.connects_domain(above)
        and above_wall_cover & WALL_POST_COVER
    )
    force_post = bool(
        above is not None
        and context.connects_domain(above)
        and (
            "wall_post_override" in above.info.tags
            or (
                above.info.neighbor_rule == NeighborRule.WALL
                and above.state.get("up") == "true"
            )
        )
    )
    if above is not None and context.connects_domain(above):
        connected = {
            direction: (
                "tall"
                if value == "low"
                and above_wall_cover & WALL_COVER_BY_DIRECTION[direction]
                else value
            )
            for direction, value in connected.items()
        }
    straight = (
        connected["north"] == "low"
        and connected["south"] == "low"
        and connected["east"] == "none"
        and connected["west"] == "none"
    ) or (
        connected["east"] == "low"
        and connected["west"] == "low"
        and connected["north"] == "none"
        and connected["south"] == "none"
    )
    updates = dict(connected)
    updates["up"] = str(force_post or covered_from_above or not straight).lower()
    return _with_properties(context.state, updates)


def _wall_connection(context: NeighborContext, direction: Direction) -> str:
    neighbor = context.neighbor(direction)
    if neighbor is None:
        diagonal = context.diagonal_decision(direction)
        return _wall_height(diagonal) if diagonal.action == ConnectionAction.CONNECT else "none"

    decision = context.connection_decision(direction)
    if decision.action == ConnectionAction.DISCONNECT:
        return "none"
    if decision.action == ConnectionAction.CONNECT:
        return _wall_height(decision)

    diagonal = context.diagonal_decision(direction)
    if diagonal.action == ConnectionAction.CONNECT:
        return _wall_height(diagonal)

    if not context.connects_domain(neighbor):
        return "none"

    predicate = context.resolver.blocks_can_connect
    if predicate is not None:
        return (
            "low"
            if predicate(
                context.state,
                context.info,
                neighbor.state,
                neighbor.info,
                direction,
            )
            else "none"
        )

    if neighbor.info.neighbor_rule in (
        NeighborRule.WALL,
        NeighborRule.FENCE_GATE,
    ):
        if (
            neighbor.info.neighbor_rule == NeighborRule.FENCE_GATE
            and not _gate_connects_to_direction(neighbor.state, direction)
        ):
            return "none"
        return "low"

    touching_face = FACE_BY_DIRECTION[OPPOSITE_DIRECTION[direction]]
    if (
        not neighbor.info.connection_exception
        and neighbor.info.solid_faces & touching_face
    ):
        return "low"
    return "none"


def _wall_height(decision: ConnectionDecision) -> str:
    return decision.value if decision.value in ("low", "tall") else "low"


def _resolve_fence_gate(context: NeighborContext) -> BlockState:
    facing = context.state.get("facing")
    if facing in ("north", "south"):
        directions = ("east", "west")
    elif facing in ("east", "west"):
        directions = ("north", "south")
    else:
        directions = ()

    in_wall = any(
        context.connects(
            direction,
            frozenset((NeighborRule.WALL,)),
            solid_faces=False,
        )
        for direction in directions
    )
    return _with_properties(context.state, {"in_wall": str(in_wall).lower()})


def _resolve_stair(context: NeighborContext) -> BlockState:
    facing = context.state.get("facing")
    half = context.state.get("half")
    if facing not in HORIZONTAL_DIRECTIONS:
        return context.state

    left = _counter_clockwise(facing)
    front = context.neighbor(facing)
    if _matching_stair(context, front, half, facing):
        neighbor_facing = front.state.get("facing")
        if (
            neighbor_facing in (left, OPPOSITE_DIRECTION[left])
            and _can_take_stair_shape(
                context,
                OPPOSITE_DIRECTION[neighbor_facing],
                facing,
                half,
            )
        ):
            shape = "outer_left" if neighbor_facing == left else "outer_right"
            return _with_properties(context.state, {"shape": shape})

    back = context.neighbor(OPPOSITE_DIRECTION[facing])
    if _matching_stair(context, back, half, OPPOSITE_DIRECTION[facing]):
        neighbor_facing = back.state.get("facing")
        if (
            neighbor_facing in (left, OPPOSITE_DIRECTION[left])
            and _can_take_stair_shape(context, neighbor_facing, facing, half)
        ):
            shape = "inner_left" if neighbor_facing == left else "inner_right"
            return _with_properties(context.state, {"shape": shape})

    diagonal = _diagonal_stair_shape(context, facing, left, half)
    if diagonal is not None:
        return _with_properties(context.state, {"shape": diagonal})
    joined = _stray_in_diagonal_run(context, facing, half)
    if joined is not None:
        return _with_properties(context.state, joined)
    return _with_properties(context.state, {"shape": "straight"})


_DIAGONALS = (
    (BlockPos(1, 0, 1), BlockPos(-1, 0, -1)),
    (BlockPos(1, 0, -1), BlockPos(-1, 0, 1)),
)
_AROUND = tuple(
    BlockPos(dx, 0, dz)
    for dx in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if dx or dz
)


def _stray_in_diagonal_run(
    context: NeighborContext, facing: Direction, half: str | None
) -> dict[str, str] | None:
    # A lone stair inside a diagonal run of another facing (left by two
    # layouts meeting at a join) adopts the run's facing and shape.
    rule = context.resolver.stairs_join_diagonals
    if rule is None or not rule(context.domain):
        return None
    if any(_same_run_at(context, offset, facing, half) for offset in _AROUND):
        return None
    for first, second in _DIAGONALS:
        for other in HORIZONTAL_DIRECTIONS:
            if other == facing:
                continue
            if _same_run_at(context, first, other, half) and _same_run_at(
                context, second, other, half
            ):
                left = _counter_clockwise(other)
                shape = _diagonal_stair_shape(context, other, left, half)
                return {"facing": other, "shape": shape or "straight"}
    return None


def _diagonal_stair_shape(
    context: NeighborContext, facing: Direction, left: Direction, half: str | None
) -> str | None:
    # Where a run of stairs steps sideways, the stairs either side of the
    # step touch only at a corner. The one before turns outer toward the
    # next, the one after turns inner from the last, unless a stair beside
    # it on that side continues the run straight.
    rule = context.resolver.stairs_join_diagonals
    if rule is None or not rule(context.domain):
        return None
    right = OPPOSITE_DIRECTION[left]
    ahead, behind = DIRECTION_OFFSETS[facing], DIRECTION_OFFSETS[OPPOSITE_DIRECTION[facing]]
    for turned, prefix in ((ahead, "outer"), (behind, "inner")):
        for side, name in ((left, "left"), (right, "right")):
            step = DIRECTION_OFFSETS[side]
            if _same_run_at(context, step, facing, half):
                continue
            if _same_run_at(context, turned + step, facing, half):
                return f"{prefix}_{name}"
    return None


def _same_run_at(
    context: NeighborContext, offset: BlockPos, facing: Direction, half: str | None
) -> bool:
    neighbor = context.neighbor_at(offset)
    return (
        neighbor is not None
        and neighbor.info.neighbor_rule == NeighborRule.STAIR
        and neighbor.state.get("facing") == facing
        and neighbor.state.get("half") == half
        and context.connects_domain(neighbor)
    )


def _matching_stair(
    context: NeighborContext,
    neighbor: NeighborBlock | None,
    half: str | None,
    direction: Direction,
) -> bool:
    decision = context.connection_decision_for(neighbor, direction)
    if decision.action != ConnectionAction.DEFAULT:
        return decision.action == ConnectionAction.CONNECT and neighbor is not None
    return bool(
        neighbor is not None
        and neighbor.info.neighbor_rule == NeighborRule.STAIR
        and neighbor.state.get("half") == half
        and context.connects_domain(neighbor)
    )


def _can_take_stair_shape(
    context: NeighborContext,
    direction: Direction,
    facing: Direction,
    half: str | None,
) -> bool:
    return not _same_run_at(context, DIRECTION_OFFSETS[direction], facing, half)


def _resolve_rail(context: NeighborContext) -> BlockState:
    candidates = _selected_rail_candidates(
        context,
        context.position,
        context.state,
        context.info,
        context.domain,
    )
    mutual: dict[Direction, tuple[NeighborBlock, bool, bool]] = {}
    for direction, candidate in candidates.items():
        neighbor, _, forced = candidate
        if forced:
            mutual[direction] = candidate
            continue
        reverse_candidates = _selected_rail_candidates(
            context,
            neighbor.position,
            neighbor.state,
            neighbor.info,
            neighbor.domain,
        )
        reverse = reverse_candidates.get(OPPOSITE_DIRECTION[direction])
        if reverse is not None and reverse[0].position == context.position:
            mutual[direction] = candidate

    connected = {
        direction: direction in mutual for direction in HORIZONTAL_DIRECTIONS
    }
    ascending = {
        direction: direction in mutual and mutual[direction][1]
        for direction in HORIZONTAL_DIRECTIONS
    }

    shape = _rail_shape(
        connected,
        context.state.get("shape"),
        allow_curves=context.info.behavior not in ("powered_rail", "detector_rail"),
    )
    if shape == "north_south":
        if ascending["north"]:
            shape = "ascending_north"
        elif ascending["south"]:
            shape = "ascending_south"
    elif shape == "east_west":
        if ascending["east"]:
            shape = "ascending_east"
        elif ascending["west"]:
            shape = "ascending_west"

    return _with_properties(context.state, {"shape": shape})


def _selected_rail_candidates(
    context: NeighborContext,
    position: BlockPos,
    state: BlockState,
    info: BlockInfo,
    domain: ConnectivityDomainId,
) -> dict[Direction, tuple[NeighborBlock, bool, bool]]:
    candidates: dict[Direction, tuple[NeighborBlock, bool, bool]] = {}
    for direction in HORIZONTAL_DIRECTIONS:
        offset_position = _add_position(position, DIRECTION_OFFSETS[direction])
        locations = (
            (offset_position, False),
            (_add_position(offset_position, DIRECTION_OFFSETS["up"]), True),
            (_add_position(offset_position, DIRECTION_OFFSETS["down"]), False),
        )
        for neighbor_position, ascending in locations:
            neighbor = context.block_at(neighbor_position)
            allowed, forced = _rail_pair_allowed(
                context, state, info, domain, neighbor, direction
            )
            if allowed and neighbor is not None:
                candidates[direction] = (neighbor, ascending, forced)
                break

    if len(candidates) <= 2:
        return candidates
    current_shape = state.get("shape") or ""
    preferred_axis = (
        ("north", "south")
        if "north" in current_shape or "south" in current_shape
        else ("east", "west")
    )
    priority = (
        *preferred_axis,
        *(
            direction
            for direction in HORIZONTAL_DIRECTIONS
            if direction not in preferred_axis
        ),
    )
    selected = [direction for direction in priority if direction in candidates][:2]
    return {direction: candidates[direction] for direction in selected}


def _rail_pair_allowed(
    context: NeighborContext,
    source_state: BlockState,
    source_info: BlockInfo,
    source_domain: ConnectivityDomainId,
    neighbor: NeighborBlock | None,
    direction: Direction,
) -> tuple[bool, bool]:
    if neighbor is None:
        return False, False
    decision = DEFAULT_CONNECTION_DECISION
    if context.resolver.options is not None:
        decision = context.resolver.options.decide(
            source_state,
            source_domain,
            neighbor.state,
            neighbor.domain,
            direction,
            source_info.behavior,
            neighbor.info.behavior,
            source_info.tags,
            neighbor.info.tags,
        )
    if decision.action == ConnectionAction.CONNECT:
        return True, True
    if decision.action == ConnectionAction.DISCONNECT:
        return False, False
    return (
        neighbor.info.neighbor_rule == NeighborRule.RAIL
        and context.resolver.domains_can_connect(source_domain, neighbor.domain),
        False,
    )


def _rail_shape(
    connected: Mapping[Direction, bool],
    current_shape: str | None,
    *,
    allow_curves: bool = True,
) -> str:
    north = connected["north"]
    east = connected["east"]
    south = connected["south"]
    west = connected["west"]

    if not allow_curves:
        north_south = north or south
        east_west = east or west
        current_is_north_south = current_shape in (
            "north_south",
            "ascending_north",
            "ascending_south",
        )
        if north_south and (not east_west or current_is_north_south):
            return "north_south"
        if east_west:
            return "east_west"
        if north_south:
            return "north_south"
    if north and east and not south and not west:
        return "north_east"
    if east and south and not west and not north:
        return "south_east"
    if south and west and not north and not east:
        return "south_west"
    if west and north and not east and not south:
        return "north_west"
    if (north or south) and not (east or west):
        return "north_south"
    if east or west:
        return "east_west"
    if current_shape in (
        "north_south",
        "ascending_north",
        "ascending_south",
    ):
        return "north_south"
    return "east_west"


def _resolve_redstone(context: NeighborContext) -> BlockState:
    updates = {
        direction: _redstone_connection(context, direction)
        for direction in HORIZONTAL_DIRECTIONS
    }
    connected = [direction for direction, value in updates.items() if value != "none"]
    was_dot = all(
        context.state.get(direction) == "none"
        for direction in HORIZONTAL_DIRECTIONS
    )
    if connected or not was_dot:
        has_north_south = any(
            updates[direction] != "none" for direction in ("north", "south")
        )
        has_east_west = any(
            updates[direction] != "none" for direction in ("east", "west")
        )
        if not has_north_south:
            for direction in ("east", "west"):
                if (
                    context.connection_decision(direction).action
                    != ConnectionAction.DISCONNECT
                ):
                    updates[direction] = "side"
        if not has_east_west:
            for direction in ("north", "south"):
                if (
                    context.connection_decision(direction).action
                    != ConnectionAction.DISCONNECT
                ):
                    updates[direction] = "side"
    return _with_properties(context.state, updates)


def _redstone_connection(context: NeighborContext, direction: Direction) -> str:
    decision = context.connection_decision(direction)
    if decision.action == ConnectionAction.DISCONNECT:
        return "none"
    if decision.action == ConnectionAction.CONNECT:
        return decision.value if decision.value in ("side", "up") else "side"

    neighbor = context.neighbor(direction)
    if neighbor is None:
        return "none"
    if not context.connects_domain(neighbor):
        return "none"
    if _is_redstone_connector(neighbor, direction):
        return "side"

    above_current = context.neighbor("up")
    current_top_open = above_current is None or not (
        above_current.info.solid_faces & SolidFace.DOWN
    )
    if current_top_open and neighbor.info.solid_faces:
        above_neighbor = context.block_at(_add_position(neighbor.position, DIRECTION_OFFSETS["up"]))
        if _is_compatible_redstone(context, above_neighbor, direction):
            return "up"

    if not neighbor.info.solid_faces:
        below_neighbor = context.block_at(_add_position(neighbor.position, DIRECTION_OFFSETS["down"]))
        if _is_compatible_redstone(context, below_neighbor, direction):
            return "side"
    return "none"


def _is_redstone_connector(neighbor: NeighborBlock, direction: Direction) -> bool:
    if neighbor.info.neighbor_rule == NeighborRule.REDSTONE:
        return True
    if neighbor.info.behavior in ("repeater", "comparator"):
        facing = neighbor.state.get("facing")
        return _same_axis(facing, direction)
    if neighbor.info.behavior == "observer":
        return neighbor.state.get("facing") == OPPOSITE_DIRECTION[direction]
    return neighbor.info.signal_source


def _is_compatible_redstone(
    context: NeighborContext,
    neighbor: NeighborBlock | None,
    direction: Direction,
) -> bool:
    return bool(
        neighbor is not None
        and context.connects_domain(neighbor)
        and _is_redstone_connector(neighbor, direction)
    )


def _resolve_tripwire(context: NeighborContext) -> BlockState:
    updates = {
        direction: str(_tripwire_connects(context, direction)).lower()
        for direction in HORIZONTAL_DIRECTIONS
    }
    if context.state.get("attached") is not None:
        attached, _ = _tripwire_line_through(context)
        updates["attached"] = str(attached).lower()
    return _with_properties(context.state, updates)


def _tripwire_connects(context: NeighborContext, direction: Direction) -> bool:
    decision = context.connection_decision(direction)
    if decision.action != ConnectionAction.DEFAULT:
        return decision.action == ConnectionAction.CONNECT
    neighbor = context.neighbor(direction)
    if neighbor is None or not context.connects_domain(neighbor):
        return False
    if neighbor.info.neighbor_rule == NeighborRule.TRIPWIRE:
        return True
    return (
        neighbor.info.behavior == "trip_wire_hook"
        and neighbor.state.get("facing") == OPPOSITE_DIRECTION[direction]
    )


def _resolve_tripwire_hook(context: NeighborContext) -> BlockState:
    facing = context.state.get("facing")
    if facing not in HORIZONTAL_DIRECTIONS:
        return context.state
    support_direction = OPPOSITE_DIRECTION[facing]
    if not context.allows(
        support_direction,
        lambda support: context.connects_domain(support)
        and _supports_face(support.info, FACE_BY_DIRECTION[facing]),
    ):
        return None
    attached, powered = _scan_tripwire_from_hook(context, facing)
    return _with_properties(
        context.state,
        {"attached": str(attached).lower(), "powered": str(powered).lower()},
    )


def _scan_tripwire_from_hook(
    context: NeighborContext, direction: Direction
) -> tuple[bool, bool]:
    powered = False
    for distance in range(1, 43):
        position = _add_position(
            context.position,
            tuple(value * distance for value in DIRECTION_OFFSETS[direction]),
        )
        neighbor = context.block_at(position)
        if neighbor is None or not context.connects_domain(neighbor):
            return False, False
        decision = context.connection_decision_for(neighbor, direction)
        if decision.action == ConnectionAction.DISCONNECT:
            return False, False
        if neighbor.info.neighbor_rule == NeighborRule.TRIPWIRE:
            powered = powered or (
                neighbor.state.get("powered") == "true"
                and neighbor.state.get("disarmed") != "true"
            )
            continue
        attached = bool(
            distance > 1
            and neighbor.info.neighbor_rule == NeighborRule.TRIPWIRE_HOOK
            and neighbor.state.get("facing") == OPPOSITE_DIRECTION[direction]
        )
        return attached, powered if attached else False
    return False, False


def _tripwire_line_through(context: NeighborContext) -> tuple[bool, bool]:
    for first, second in (("north", "south"), ("east", "west")):
        left = _scan_tripwire_endpoint(context, first)
        right = _scan_tripwire_endpoint(context, second)
        if left[0] and right[0] and left[1] + right[1] <= 41:
            return True, left[2] or right[2] or context.state.get("powered") == "true"
    return False, False


def _scan_tripwire_endpoint(
    context: NeighborContext, direction: Direction
) -> tuple[bool, int, bool]:
    powered = False
    for distance in range(1, 42):
        position = _add_position(
            context.position,
            tuple(value * distance for value in DIRECTION_OFFSETS[direction]),
        )
        neighbor = context.block_at(position)
        if neighbor is None or not context.connects_domain(neighbor):
            return False, distance, powered
        if neighbor.info.neighbor_rule == NeighborRule.TRIPWIRE:
            powered = powered or (
                neighbor.state.get("powered") == "true"
                and neighbor.state.get("disarmed") != "true"
            )
            continue
        is_hook = (
            neighbor.info.neighbor_rule == NeighborRule.TRIPWIRE_HOOK
            and neighbor.state.get("facing") == OPPOSITE_DIRECTION[direction]
        )
        return is_hook, distance, powered
    return False, 42, powered


def _resolve_chorus(context: NeighborContext) -> BlockState:
    return _with_properties(
        context.state,
        {
            direction: str(_chorus_connects(context, direction)).lower()
            for direction in ALL_DIRECTIONS
        },
    )


def _resolve_huge_mushroom(context: NeighborContext) -> BlockState:
    updates: dict[str, str] = {}
    for direction in ALL_DIRECTIONS:
        joined = context.allows(
            direction,
            lambda neighbor: neighbor.state.name == context.state.name
            and context.connects_domain(neighbor),
        )
        updates[direction] = str(not joined).lower()
    return _with_properties(context.state, updates)


def _resolve_snowy(context: NeighborContext) -> BlockState:
    snowy = context.allows(
        "up", lambda above: "snow" in above.info.tags and context.connects_domain(above)
    )
    return _with_properties(context.state, {"snowy": str(snowy).lower()})


_GROWING_PLANTS: dict[str, tuple[str, str, Direction]] = {
    "minecraft:kelp": ("minecraft:kelp", "minecraft:kelp_plant", "up"),
    "minecraft:kelp_plant": ("minecraft:kelp", "minecraft:kelp_plant", "up"),
    "minecraft:twisting_vines": (
        "minecraft:twisting_vines",
        "minecraft:twisting_vines_plant",
        "up",
    ),
    "minecraft:twisting_vines_plant": (
        "minecraft:twisting_vines",
        "minecraft:twisting_vines_plant",
        "up",
    ),
    "minecraft:cave_vines": (
        "minecraft:cave_vines",
        "minecraft:cave_vines_plant",
        "down",
    ),
    "minecraft:cave_vines_plant": (
        "minecraft:cave_vines",
        "minecraft:cave_vines_plant",
        "down",
    ),
    "minecraft:weeping_vines": (
        "minecraft:weeping_vines",
        "minecraft:weeping_vines_plant",
        "down",
    ),
    "minecraft:weeping_vines_plant": (
        "minecraft:weeping_vines",
        "minecraft:weeping_vines_plant",
        "down",
    ),
}


def _resolve_growing_plant(context: NeighborContext) -> BlockState:
    head, body, direction = _GROWING_PLANTS[context.state.name]
    continues = context.allows(
        direction,
        lambda neighbor: neighbor.state.name in (head, body)
        and context.connects_domain(neighbor),
    )
    target = body if continues else head
    if target == context.state.name:
        return context.state
    from .vanilla import vanilla_catalog

    state = vanilla_catalog().default_state(target)
    shared = {
        key: value
        for key, value in context.state.properties
        if state.get(key) is not None
    }
    return _with_properties(state, shared)


def _resolve_hanging_moss(context: NeighborContext) -> BlockState | None:
    supported = context.allows(
        "up",
        lambda above: context.connects_domain(above)
        and (
            above.state.name == context.state.name
            or _supports_face(above.info, SolidFace.DOWN)
        ),
    )
    if not supported:
        return None
    below = context.neighbor("down")
    tip = not (
        below is not None
        and below.state.name == context.state.name
        and context.connects_domain(below)
    )
    return _with_properties(context.state, {"tip": str(tip).lower()})


def _resolve_campfire(context: NeighborContext) -> BlockState:
    signal = context.allows(
        "down",
        lambda below: below.state.name == "minecraft:hay_block"
        and context.connects_domain(below),
    )
    return _with_properties(context.state, {"signal_fire": str(signal).lower()})


def _chorus_connects(context: NeighborContext, direction: Direction) -> bool:
    decision = context.connection_decision(direction)
    if decision.action != ConnectionAction.DEFAULT:
        return decision.action == ConnectionAction.CONNECT
    if context.connects(
        direction,
        frozenset((NeighborRule.CHORUS,)),
        solid_faces=False,
    ):
        return True
    neighbor = context.neighbor(direction)
    if neighbor is None:
        return False
    if neighbor.info.behavior == "chorus_flower":
        return context.connects_domain(neighbor)
    return direction == "down" and (
        "supports_chorus_plant" in neighbor.info.tags
        or neighbor.state.name == "minecraft:end_stone"
    )


def _resolve_multiface(context: NeighborContext) -> BlockState:
    return _preserve_supported_faces(context, ALL_DIRECTIONS)


def _resolve_vine(context: NeighborContext) -> BlockState:
    return _preserve_supported_faces(context, ("up", *HORIZONTAL_DIRECTIONS))


def _preserve_supported_faces(
    context: NeighborContext,
    directions: Iterable[Direction],
) -> BlockState:
    updates: dict[str, str] = {}
    infer_attachments = not any(
        context.state.get(direction) == "true" for direction in directions
    )
    for direction in directions:
        if context.state.get(direction) is None:
            continue
        decision = context.connection_decision(direction)
        if decision.action == ConnectionAction.CONNECT:
            attached = True
        elif decision.action == ConnectionAction.DISCONNECT:
            attached = False
        else:
            attached = (
                (infer_attachments or context.state.get(direction) == "true")
                and context.connects(direction, frozenset())
            )
        updates[direction] = str(attached).lower()
    return _with_properties(context.state, updates)


def _resolve_fire(context: NeighborContext) -> BlockState:
    return _with_properties(
        context.state,
        {
            direction: str(_fire_connects(context, direction)).lower()
            for direction in ("up", *HORIZONTAL_DIRECTIONS)
        },
    )


def _fire_connects(context: NeighborContext, direction: Direction) -> bool:
    decision = context.connection_decision(direction)
    if decision.action != ConnectionAction.DEFAULT:
        return decision.action == ConnectionAction.CONNECT
    neighbor = context.neighbor(direction)
    return bool(
        neighbor is not None
        and neighbor.info.flammable
        and context.connects_domain(neighbor)
    )


def _resolve_moss_carpet(context: NeighborContext) -> BlockState:
    below = context.neighbor("down")
    above = context.neighbor("up")
    bottom = context.state.get("bottom") == "true"
    survives = bool(
        below is not None
        and context.connects_domain(below)
        and (
            bottom
            or (
                below.info.neighbor_rule == NeighborRule.MOSS_CARPET
                and below.state.get("bottom") == "true"
            )
        )
    )
    if not survives:
        return None
    updates: dict[str, str] = {"bottom": str(bottom).lower()}
    for direction in HORIZONTAL_DIRECTIONS:
        decision = context.connection_decision(direction)
        if decision.action == ConnectionAction.DISCONNECT:
            updates[direction] = "none"
            continue
        if decision.action == ConnectionAction.CONNECT:
            updates[direction] = _wall_height(decision)
            continue

        supported = _moss_side_supported(context, context.position, direction)
        side = (
            "low"
            if supported and bottom
            else (context.state.get(direction) or "none")
            if supported
            else "none"
        )
        if side == "low" and above is not None and (
            above.info.neighbor_rule == NeighborRule.MOSS_CARPET
            and above.state.get(direction) not in (None, "none")
            and above.state.get("bottom") == "false"
            and context.connects_domain(above)
        ):
            side = "tall"
        if not bottom and below is not None and (
            below.info.neighbor_rule == NeighborRule.MOSS_CARPET
            and below.state.get(direction) == "none"
            and not _moss_side_supported(context, below.position, direction)
        ):
            side = "none"
        updates[direction] = side
    if not bottom and not any(
        updates[direction] != "none" for direction in HORIZONTAL_DIRECTIONS
    ):
        return None
    return _with_properties(context.state, updates)


def _moss_side_supported(
    context: NeighborContext, origin: BlockPos, direction: Direction
) -> bool:
    neighbor = context.block_at(_add_position(origin, DIRECTION_OFFSETS[direction]))
    return bool(
        neighbor is not None
        and context.connects_domain(neighbor)
        and _supports_face(
            neighbor.info, FACE_BY_DIRECTION[OPPOSITE_DIRECTION[direction]]
        )
    )


def _resolve_scaffolding(context: NeighborContext) -> BlockState | None:
    below = context.neighbor("down")
    below_decision = context.connection_decision("down")
    if below is not None and (
        below_decision.action == ConnectionAction.CONNECT
        or (
            below_decision.action == ConnectionAction.DEFAULT
            and context.connects_domain(below)
        )
    ):
        if below.info.neighbor_rule == NeighborRule.SCAFFOLDING:
            distance = _integer_property(below.state, "distance", 7)
        elif below_decision.action == ConnectionAction.CONNECT:
            distance = 0
        elif below.info.solid_faces & SolidFace.UP:
            distance = 0
        else:
            distance = 7
    else:
        distance = 7

    for direction in HORIZONTAL_DIRECTIONS:
        neighbor = context.neighbor(direction)
        decision = context.connection_decision(direction)
        if (
            neighbor is not None
            and (
                neighbor.info.neighbor_rule == NeighborRule.SCAFFOLDING
                or decision.action == ConnectionAction.CONNECT
            )
            and decision.action != ConnectionAction.DISCONNECT
            and (
                decision.action == ConnectionAction.CONNECT
                or context.connects_domain(neighbor)
            )
        ):
            distance = min(
                distance,
                (
                    _integer_property(neighbor.state, "distance", 7)
                    if neighbor.info.neighbor_rule == NeighborRule.SCAFFOLDING
                    else 0
                )
                + 1,
            )
    distance = min(distance, 7)
    if distance == 7:
        return None
    bottom = distance > 0 and (
        below is None or below.info.neighbor_rule != NeighborRule.SCAFFOLDING
    )
    return _with_properties(
        context.state,
        {"distance": str(distance), "bottom": str(bottom).lower()},
    )


def _resolve_shelf(context: NeighborContext) -> BlockState:
    if context.state.get("powered") != "true":
        return _with_properties(context.state, {"side_chain": "unconnected"})
    facing = context.state.get("facing")
    if facing not in HORIZONTAL_DIRECTIONS:
        return context.state

    left_direction = _counter_clockwise(facing)
    right_direction = _clockwise(facing)
    left_count = _shelf_run_length(context, left_direction)
    right_count = _shelf_run_length(context, right_direction)
    index = left_count
    group_index = index % 3
    group_size = min(3, left_count + 1 + right_count - (index - group_index))
    if group_size == 1:
        part = "unconnected"
    elif group_index == 0:
        part = "left"
    elif group_index == group_size - 1:
        part = "right"
    else:
        part = "center"
    return _with_properties(context.state, {"side_chain": part})


def _shelf_run_length(context: NeighborContext, direction: Direction) -> int:
    length = 0
    position = context.position
    while True:
        position = _add_position(position, DIRECTION_OFFSETS[direction])
        neighbor = context.block_at(position)
        if neighbor is None:
            return length
        decision = context.connection_decision_for(neighbor, direction)
        if decision.action == ConnectionAction.DISCONNECT:
            return length
        if decision.action != ConnectionAction.CONNECT and not (
            neighbor.info.neighbor_rule == NeighborRule.SHELF
            and neighbor.state.get("powered") == "true"
            and neighbor.state.get("facing") == context.state.get("facing")
            and context.connects_domain(neighbor)
        ):
            return length
        length += 1


def _resolve_pointed_dripstone(context: NeighborContext) -> BlockState:
    vertical_direction = context.state.get("vertical_direction")
    if vertical_direction not in ("up", "down"):
        return context.state

    backward = OPPOSITE_DIRECTION[vertical_direction]
    base_distance = _dripstone_run_length(context, backward, vertical_direction)
    tip_distance = _dripstone_run_length(
        context,
        vertical_direction,
        vertical_direction,
    )
    if base_distance == 0 and tip_distance == 0:
        thickness = "tip"
    elif base_distance == 0:
        thickness = "base"
    elif tip_distance == 0:
        merges = context.allows(
            vertical_direction,
            lambda ahead: ahead.info.neighbor_rule == NeighborRule.POINTED_DRIPSTONE
            and ahead.state.name == context.state.name
            and context.connects_domain(ahead),
        )
        facing_back = merges and (
            context.neighbor(vertical_direction).state.get("vertical_direction") == backward
        )
        thickness = "tip_merge" if facing_back else "tip"
    elif tip_distance == 1:
        thickness = "frustum"
    else:
        thickness = "middle"
    return _with_properties(context.state, {"thickness": thickness})


def _dripstone_run_length(
    context: NeighborContext,
    scan_direction: Direction,
    vertical_direction: Direction,
) -> int:
    length = 0
    position = context.position
    while True:
        position = _add_position(position, DIRECTION_OFFSETS[scan_direction])
        neighbor = context.block_at(position)
        decision = context.connection_decision_for(neighbor, scan_direction)
        if (
            neighbor is None
            or decision.action == ConnectionAction.DISCONNECT
            or (
                decision.action != ConnectionAction.CONNECT
                and (
                    neighbor.info.neighbor_rule != NeighborRule.POINTED_DRIPSTONE
                    or neighbor.state.name != context.state.name
                    or neighbor.state.get("vertical_direction") != vertical_direction
                    or not context.connects_domain(neighbor)
                )
            )
        ):
            return length
        length += 1


def _resolve_leaves(context: NeighborContext) -> BlockState:
    distance = 7
    for direction in ALL_DIRECTIONS:
        neighbor = context.neighbor(direction)
        if neighbor is None:
            continue
        decision = context.connection_decision_for(neighbor, direction)
        if decision.action == ConnectionAction.DISCONNECT:
            continue
        if decision.action == ConnectionAction.DEFAULT and not (
            context.connects_domain(neighbor)
            and (
                "logs" in neighbor.info.tags
                or neighbor.info.neighbor_rule == NeighborRule.LEAVES
            )
        ):
            continue
        if neighbor.info.neighbor_rule == NeighborRule.LEAVES:
            candidate = _integer_property(neighbor.state, "distance", 7) + 1
        else:
            candidate = 1
        distance = min(distance, candidate)
    return _with_properties(context.state, {"distance": str(min(7, distance))})


def _resolve_chest(context: NeighborContext) -> BlockState:
    facing = context.state.get("facing")
    if facing not in HORIZONTAL_DIRECTIONS:
        return context.state
    left_direction = _counter_clockwise(facing)
    right_direction = _clockwise(facing)
    left_count = _chest_run_length(context, left_direction)
    if left_count % 2 == 1:
        return _with_properties(context.state, {"type": "right"})
    if _chest_run_length(context, right_direction) > 0:
        return _with_properties(context.state, {"type": "left"})
    return _with_properties(context.state, {"type": "single"})


def _chest_run_length(context: NeighborContext, direction: Direction) -> int:
    length = 0
    position = context.position
    while True:
        position = _add_position(position, DIRECTION_OFFSETS[direction])
        neighbor = context.block_at(position)
        if not _matching_chest(context, neighbor, direction):
            return length
        length += 1


def _matching_chest(
    context: NeighborContext,
    neighbor: NeighborBlock | None,
    direction: Direction,
) -> bool:
    if neighbor is None:
        return False
    decision = context.connection_decision_for(neighbor, direction)
    if decision.action != ConnectionAction.DEFAULT:
        return decision.action == ConnectionAction.CONNECT
    return bool(
        neighbor.info.neighbor_rule == NeighborRule.CHEST
        and neighbor.state.name == context.state.name
        and neighbor.state.get("facing") == context.state.get("facing")
        and context.connects_domain(neighbor)
    )


def _resolve_stem(context: NeighborContext) -> BlockState:
    is_melon = "melon" in context.state.name
    fruit = "minecraft:melon" if is_melon else "minecraft:pumpkin"
    attached_name = (
        "minecraft:attached_melon_stem"
        if is_melon
        else "minecraft:attached_pumpkin_stem"
    )
    for direction in HORIZONTAL_DIRECTIONS:
        if context.allows(
            direction,
            lambda neighbor: neighbor.state.name == fruit and context.connects_domain(neighbor),
        ) and (
            context.info.behavior == "attached_stem" or context.state.get("age") == "7"
        ):
            return BlockState.create(attached_name, {"facing": direction})
    if context.info.behavior == "attached_stem":
        stem_name = "minecraft:melon_stem" if is_melon else "minecraft:pumpkin_stem"
        return BlockState.create(stem_name, {"age": "7"})
    return context.state


def _resolve_hanging(context: NeighborContext) -> BlockState:
    hanging = context.state.get("hanging") == "true"
    direction = "up" if hanging else "down"
    supported = context.allows(
        direction,
        lambda support: context.connects_domain(support)
        and (
            _supports_face(support.info, FACE_BY_DIRECTION[OPPOSITE_DIRECTION[direction]])
            or (hanging and support.info.behavior in ("chain", "weathering_copper_chain"))
        ),
    )
    return context.state if supported else None


def _resolve_support(context: NeighborContext) -> BlockState | None:
    direction = context.state.get("support")
    if direction not in ALL_DIRECTIONS:
        face = context.state.get("face")
        attachment = context.state.get("attachment")
        if attachment == "double_wall" and context.state.get(
            "facing"
        ) in HORIZONTAL_DIRECTIONS:
            facing = context.state.get("facing")
            if any(
                _has_attachment_support(context, side)
                for side in (_clockwise(facing), _counter_clockwise(facing))
            ):
                return context.state
            return None
        if face == "ceiling":
            direction = "up"
        elif face == "wall" and context.state.get("facing") in HORIZONTAL_DIRECTIONS:
            direction = OPPOSITE_DIRECTION[context.state.get("facing")]
        elif attachment == "ceiling":
            direction = "up"
        elif attachment in ("single_wall",) and context.state.get(
            "facing"
        ) in HORIZONTAL_DIRECTIONS:
            direction = OPPOSITE_DIRECTION[context.state.get("facing")]
        elif context.info.behavior in (
            "ceiling_hanging_sign",
            "hanging_roots",
            "spore_blossom",
        ):
            direction = "up"
        elif context.info.behavior in (
            "wall_banner",
            "wall_coral_fan",
            "wall_hanging_sign",
            "wall_sign",
            "wall_torch",
            "ladder",
            "redstone_wall_torch",
            "wall_skull",
            "player_wall_head",
            "piglinwallskull",
            "wither_wall_skull",
        ) and context.state.get("facing") in HORIZONTAL_DIRECTIONS:
            direction = OPPOSITE_DIRECTION[context.state.get("facing")]
        elif context.info.behavior == "amethyst_cluster" and context.state.get(
            "facing"
        ) in ALL_DIRECTIONS:
            direction = OPPOSITE_DIRECTION[context.state.get("facing")]
        else:
            direction = "down"

    return context.state if _has_attachment_support(context, direction) else None


def _has_attachment_support(
    context: NeighborContext, direction: Direction
) -> bool:
    neighbor = context.neighbor(direction)
    decision = context.connection_decision(direction)
    if decision.action == ConnectionAction.CONNECT:
        return True
    if decision.action == ConnectionAction.DISCONNECT or neighbor is None:
        return False
    support_face = FACE_BY_DIRECTION[OPPOSITE_DIRECTION[direction]]
    supported = (
        bool((neighbor.info.rigid_faces or neighbor.info.solid_faces) & support_face)
        if context.info.behavior in ("pressure_plate", "weighted_pressure_plate")
        else _supports_face(neighbor.info, support_face)
    )
    if context.info.behavior == "ceiling_hanging_sign" and (
        neighbor.info.behavior == "ceiling_hanging_sign"
    ):
        supported = True
    return bool(
        context.connects_domain(neighbor)
        and supported
    )


def _resolve_multiblock(context: NeighborContext) -> BlockState | None:
    behavior = context.info.behavior
    if behavior == "bed":
        facing = context.state.get("facing")
        part = context.state.get("part")
        if facing not in HORIZONTAL_DIRECTIONS or part not in ("head", "foot"):
            return context.state
        direction = facing if part == "foot" else OPPOSITE_DIRECTION[facing]
        partner = context.neighbor(direction)
        if not _valid_multiblock_partner(context, partner, direction):
            return None
        if (
            partner.state.name != context.state.name
            or partner.state.get("part") == part
            or partner.state.get("facing") != facing
        ):
            return None
        if part == "head":
            return _with_properties(
                context.state, {"occupied": partner.state.get("occupied") or "false"}
            )
        return context.state

    half = context.state.get("half")
    if half not in ("upper", "lower"):
        return context.state
    if half == "lower" and behavior == "door" and not _has_attachment_support(
        context, "down"
    ):
        return None
    direction = "down" if half == "upper" else "up"
    partner = context.neighbor(direction)
    if not _valid_multiblock_partner(context, partner, direction):
        return None
    if (
        partner.info.behavior != behavior
        or partner.state.name != context.state.name
        or partner.state.get("half") == half
    ):
        return None
    if half == "upper" and behavior in ("door", "small_dripleaf"):
        shared = {
            key: value
            for key, value in partner.state.properties
            if context.state.get(key) is not None and key != "half"
        }
        return _with_properties(context.state, shared)
    return context.state


def _valid_multiblock_partner(
    context: NeighborContext,
    partner: NeighborBlock | None,
    direction: Direction,
) -> bool:
    if partner is None:
        return False
    decision = context.connection_decision_for(partner, direction)
    return decision.action == ConnectionAction.CONNECT or (
        decision.action == ConnectionAction.DEFAULT
        and context.connects_domain(partner)
    )


def _supports_face(info: BlockInfo, face: SolidFace) -> bool:
    return bool((info.center_faces or info.solid_faces) & face)


def _gate_connects_to_direction(state: BlockState, direction: Direction) -> bool:
    facing = state.get("facing")
    return facing in HORIZONTAL_DIRECTIONS and not _same_axis(facing, direction)


def _same_fence_group(
    left: BlockState,
    left_info: BlockInfo,
    right: BlockState,
    right_info: BlockInfo,
) -> bool:
    left_is_wood = "wooden_fences" in left_info.tags or (
        not left_info.tags and left.name != "minecraft:nether_brick_fence"
    )
    right_is_wood = "wooden_fences" in right_info.tags or (
        not right_info.tags and right.name != "minecraft:nether_brick_fence"
    )
    return left_is_wood == right_is_wood


def _same_axis(left: str | None, right: str | None) -> bool:
    return (
        left in ("north", "south") and right in ("north", "south")
    ) or (left in ("east", "west") and right in ("east", "west"))


def _clockwise(direction: Direction) -> Direction:
    return HORIZONTAL_DIRECTIONS[
        (HORIZONTAL_DIRECTIONS.index(direction) + 1) % 4
    ]


def _counter_clockwise(direction: Direction) -> Direction:
    return HORIZONTAL_DIRECTIONS[
        (HORIZONTAL_DIRECTIONS.index(direction) - 1) % 4
    ]


def _integer_property(state: BlockState, name: str, default: int) -> int:
    try:
        return int(state.get(name) or default)
    except ValueError:
        return default
