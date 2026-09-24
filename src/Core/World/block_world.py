from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..Field.block_field import BlockBounds, BlockField
from ..Domain.replacement import ReplacementRules
from ..block.connectivity import ConnectivityCell, ConnectivityDomainId
from ..block.palette import BlockId
from ..block.registry import BlockRegistry
from ..block.rotation import mirror_block_state, rotate_block_state
from ..block.states import BlockState
from ..constants import (
    AIR_NAMES,
    DEFAULT_BLOCK,
    DEFAULT_PLACEMENT_MODE,
    DEFAULT_TERRAIN_MASK_MODE,
)
from ..modes import PlacementMode, TerrainMaskMode
from ..coordinates import BlockPos, ChunkPos, quarter_turn


if TYPE_CHECKING:
    from ..block.connectivity import NeighborResolver


PlacementId = int
_UNSEEN = object()
TerrainMask = Callable[[BlockPos, TerrainMaskMode], bool]
# (placing domain, existing domain) -> whether to replace a non-air block.
PlacementPolicy = Callable[[ConnectivityDomainId, ConnectivityDomainId], bool]


@runtime_checkable
class YOffset(Protocol):
    def __call__(self, x: int, z: int) -> int: ...

    @property
    def span(self) -> tuple[int, int]: ...


@runtime_checkable
class ColumnMask(Protocol):
    # The columns a placement may occupy; cells outside are never added.
    def contains(self, x: float, z: float) -> bool: ...


@dataclass(frozen=True, slots=True)
class BlockTransform:
    origin: BlockPos = BlockPos(0, 0, 0)
    quarter_turns: int = 0
    y_offset: YOffset | None = None
    # Local axis ("x" or "z") flipped before turning; block states flip too.
    mirror: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", BlockPos.coerce(self.origin))
        object.__setattr__(self, "quarter_turns", self.quarter_turns % 4)
        if self.mirror not in (None, "x", "z"):
            raise ValueError(f"A mirror flips x or z, not {self.mirror!r}")

    def _place(self, position: BlockPos) -> BlockPos:
        position = BlockPos.coerce(position)
        x, y, z = position
        if self.mirror == "x":
            x = -x
        elif self.mirror == "z":
            z = -z
        x, z = quarter_turn(x, z, self.quarter_turns)
        return BlockPos(x + self.origin.x, y + self.origin.y, z + self.origin.z)

    def apply(self, position: BlockPos) -> BlockPos:
        placed = self._place(position)
        if self.y_offset is None:
            return placed
        # y_offset is applied last, in world coordinates.
        return BlockPos(
            placed.x,
            placed.y + self.y_offset(placed.x, placed.z),
            placed.z,
        )

    def inverse(self, position: BlockPos) -> BlockPos:
        position = BlockPos.coerce(position)
        x, y, z = position
        if self.y_offset is not None:
            y -= self.y_offset(x, z)
        x -= self.origin.x
        y -= self.origin.y
        z -= self.origin.z
        x, z = quarter_turn(x, z, -self.quarter_turns)
        if self.mirror == "x":
            x = -x
        elif self.mirror == "z":
            z = -z
        return BlockPos(x, y, z)

    def apply_bounds(self, bounds: BlockBounds) -> BlockBounds:
        corners = (
            self._place(BlockPos(x, y, z))
            for x in (bounds.minimum.x, bounds.maximum.x)
            for y in (bounds.minimum.y, bounds.maximum.y)
            for z in (bounds.minimum.z, bounds.maximum.z)
        )
        first = next(corners)
        minimum = list(first)
        maximum = list(first)
        for position in corners:
            for axis, value in enumerate(position):
                minimum[axis] = min(minimum[axis], value)
                maximum[axis] = max(maximum[axis], value)
        if self.y_offset is not None:
            low, high = self.y_offset.span
            minimum[1] += low
            maximum[1] += high
        return BlockBounds(BlockPos(*minimum), BlockPos(*maximum))


@dataclass(slots=True)
class BlockPlacement:
    placement_id: PlacementId
    field: BlockField
    transform: BlockTransform
    placement_mode: PlacementMode
    terrain_mask_mode: TerrainMaskMode
    replace: str
    block_replacement: ReplacementRules | None
    replacement_ids: tuple[BlockId, ...]
    connectivity_domain: int
    sequence_index: int | None
    order: int
    revision: int = 0
    clip: ColumnMask | None = None
    surface_mask: Callable[[BlockPos], bool] | None = None

    @property
    def offset(self) -> BlockPos:
        return self.transform.origin

    def occupies(self, position: BlockPos) -> bool:
        return (self.clip is None or self.clip.contains(position.x, position.z)) and (
            self.surface_mask is None or self.surface_mask(position)
        )

    @property
    def bounds(self) -> BlockBounds | None:
        local_bounds = self.field.bounds
        return (
            None
            if local_bounds is None
            else self.transform.apply_bounds(local_bounds)
        )


@dataclass(frozen=True, slots=True)
class CellChange:
    position: BlockPos
    before: ConnectivityCell | None
    after: ConnectivityCell | None


@dataclass(frozen=True, slots=True)
class ResolutionDelta:
    changed: frozenset[BlockPos]


@dataclass(frozen=True, slots=True)
class _OverlayEntry:
    source_token: object | None
    policy_epoch: int
    block_id: BlockId | None


class BlockWorld:
    # Block fields placed by transform, composed lazily per position, with
    # connectivity results kept as an overlay.

    def __init__(
        self,
        registry: BlockRegistry | None = None,
        *,
        chunk_size: int = 16,
        base_blocks: Mapping[BlockPos, BlockId] | None = None,
        base_domain: int = 0,
        terrain_mask: TerrainMask | None = None,
        placement_policy: PlacementPolicy | None = None,
        surface_reference: object | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.registry = registry if registry is not None else BlockRegistry()
        self.chunk_size = chunk_size
        self.base_blocks = base_blocks if base_blocks is not None else {}
        self.base_domain = base_domain
        self.terrain_mask = terrain_mask
        self.placement_policy = placement_policy
        self.surface_reference = surface_reference
        self.instances: list[BlockPlacement] = []
        self._placements: dict[PlacementId, BlockPlacement] = {}
        self._position_placements: dict[ChunkPos, dict[BlockPos, list[PlacementId]]] = {}
        self._footprints: dict[PlacementId, tuple[BlockPos, ...]] = {}
        self._source_snapshot: dict[BlockPos, ConnectivityCell | None] | None = None
        self._overlay: dict[ChunkPos, dict[BlockPos, _OverlayEntry]] = {}
        self._palette_translation: dict[tuple[int, BlockId, int], BlockId] = {}
        self._air_ids: dict[BlockId, bool] = {}
        self._dirty_changes: dict[BlockPos, CellChange] = {}
        self._dirty_all = False
        self._next_placement_id = 1
        self._next_order = 0
        self._policy_epoch = 0
        self._resolver_policy_token: tuple[object, ...] | None = None

    @property
    def policy_epoch(self) -> int:
        return self._policy_epoch

    @property
    def overlay_size(self) -> int:
        return sum(len(entries) for entries in self._overlay.values())

    @property
    def palette_translation_size(self) -> int:
        return len(self._palette_translation)

    def add_field(
        self,
        field: BlockField,
        offset: BlockPos = BlockPos(0, 0, 0),
        placement_mode: PlacementMode = DEFAULT_PLACEMENT_MODE,
        terrain_mask_mode: TerrainMaskMode = DEFAULT_TERRAIN_MASK_MODE,
        replace: str = DEFAULT_BLOCK,
        *,
        block_replacement: ReplacementRules | None = None,
        quarter_turns: int = 0,
        transform: BlockTransform | None = None,
        connectivity_domain: int = 0,
        sequence_index: int | None = None,
        clip: ColumnMask | None = None,
        surface_mask: Callable[[BlockPos], bool] | None = None,
    ) -> PlacementId:
        if placement_mode == PlacementMode.TERRAIN_MASK and self.terrain_mask is None:
            raise ValueError("TERRAIN_MASK placement requires a terrain_mask callback")
        offset = BlockPos.coerce(offset)
        if transform is not None and (offset != BlockPos(0, 0, 0) or quarter_turns):
            raise ValueError("Use either transform or offset/quarter_turns")
        field.freeze()
        placement_id = self._next_placement_id
        self._next_placement_id += 1
        self._next_order += 1
        placement = BlockPlacement(
            placement_id=placement_id,
            field=field,
            transform=transform or BlockTransform(offset, quarter_turns),
            placement_mode=placement_mode,
            terrain_mask_mode=terrain_mask_mode,
            replace=replace,
            block_replacement=block_replacement,
            replacement_ids=(
                tuple(
                    self._replacement_id(state)
                    for state, _ in block_replacement.blocks
                )
                if block_replacement is not None else ()
            ),
            connectivity_domain=connectivity_domain,
            sequence_index=sequence_index,
            order=self._next_order,
            clip=clip,
            surface_mask=surface_mask,
        )
        positions = tuple(self._placement_positions(placement))
        before = self._sources(positions)
        self.instances.append(placement)
        self._placements[placement_id] = placement
        self._index_placement(placement, positions)
        self._record_source_changes(before, positions)
        return placement_id

    def _replacement_id(self, state: BlockState) -> BlockId:
        try:
            return self.registry.get_id(state)
        except KeyError:
            return self.registry.register(state)

    def refresh_surface_masks(self) -> None:
        # Re-applies masks that read the designed terrain once it is known.
        for placement in tuple(self.instances):
            if (placement.surface_mask is None or
                    getattr(getattr(placement.surface_mask, "spec", None), "reference", None)
                    != "designed"):
                continue
            self._reindex(placement, lambda: None)

    def remove_placement(self, placement_id: PlacementId) -> tuple[CellChange, ...]:
        placement = self._placements[placement_id]
        positions = self._footprints[placement_id]
        before = self._sources(positions)
        self._unindex_placement(placement)
        self.instances.remove(placement)
        del self._placements[placement_id]
        return self._record_source_changes(before, positions)

    def move_placement(
        self,
        placement_id: PlacementId,
        offset: BlockPos | None = None,
        *,
        quarter_turns: int | None = None,
        transform: BlockTransform | None = None,
    ) -> tuple[CellChange, ...]:
        placement = self._placements[placement_id]
        if transform is not None and (offset is not None or quarter_turns is not None):
            raise ValueError("Use either transform or offset/quarter_turns")
        new_transform = transform or BlockTransform(
            offset if offset is not None else placement.transform.origin,
            placement.transform.quarter_turns if quarter_turns is None else quarter_turns,
        )

        def move() -> None:
            placement.transform = new_transform

        return self._reindex(placement, move)

    def set_placement_domain(
        self,
        placement_id: PlacementId,
        connectivity_domain: int,
    ) -> tuple[CellChange, ...]:
        placement = self._placements[placement_id]
        if placement.connectivity_domain == connectivity_domain:
            return ()
        positions = self._footprints[placement_id]
        before = self._sources(positions)
        placement.connectivity_domain = connectivity_domain
        placement.revision += 1
        self._source_snapshot = None
        return self._record_source_changes(before, positions)

    def set_base_domain(self, domain: ConnectivityDomainId) -> None:
        if domain == self.base_domain:
            return
        self.base_domain = domain
        self.invalidate_connectivity_policy()

    def set_placement_policy(self, policy: PlacementPolicy | None) -> None:
        # A policy with different rules invalidates the resolved overlay.
        previous = self.placement_policy
        self.placement_policy = policy
        if _policy_token(previous) != _policy_token(policy):
            self.invalidate_connectivity_policy()

    def source_at(self, position: BlockPos) -> ConnectivityCell | None:
        position = BlockPos.coerce(position)
        snapshot = self._source_snapshot
        if snapshot is not None:
            cell = snapshot.get(position, _UNSEEN)
            if cell is not _UNSEEN:
                return cell
        cell = self._composed_cell(position)
        if cell is None or self._is_air(cell.block_id):
            cell = None
        if snapshot is not None:
            snapshot[position] = cell
        return cell

    def get_placement(self, placement_id: PlacementId) -> BlockPlacement:
        return self._placements[placement_id]

    def get_block_id(
        self,
        position: BlockPos,
        *,
        resolved: bool = True,
    ) -> BlockId | None:
        cell = self.effective_at(position) if resolved else self.source_at(position)
        return None if cell is None else cell.block_id

    def get_block(
        self,
        position: BlockPos,
        *,
        resolved: bool = True,
    ) -> BlockState | None:
        block_id = self.get_block_id(position, resolved=resolved)
        return None if block_id is None else self.registry.get_state(block_id)

    def has_block(self, position: BlockPos, *, resolved: bool = True) -> bool:
        return self.get_block_id(position, resolved=resolved) is not None

    def __contains__(self, position: BlockPos) -> bool:
        return self.has_block(position)

    def effective_at(self, position: BlockPos) -> ConnectivityCell | None:
        position = BlockPos.coerce(position)
        source = self.source_at(position)
        chunk = self._chunk_for(position)
        entries = self._overlay.get(chunk)
        entry = None if entries is None else entries.get(position)
        if entry is None:
            return source
        if (
            source is None
            or entry.source_token != source.source_token
            or entry.policy_epoch != self._policy_epoch
        ):
            self.clear_resolution(position)
            return source
        if entry.block_id is None:
            return None
        return ConnectivityCell(
            block_id=entry.block_id,
            domain=source.domain,
            source_token=source.source_token,
            placement_id=source.placement_id,
            sequence_index=source.sequence_index,
        )

    def write_resolution(
        self,
        position: BlockPos,
        source_token: object | None,
        block_id: BlockId | None,
    ) -> None:
        position = BlockPos.coerce(position)
        source = self.source_at(position)
        if source is None or source.source_token != source_token:
            self.clear_resolution(position)
            return
        if block_id == source.block_id:
            self.clear_resolution(position)
            return
        chunk = self._chunk_for(position)
        self._overlay.setdefault(chunk, {})[position] = _OverlayEntry(
            source_token=source_token,
            policy_epoch=self._policy_epoch,
            block_id=block_id,
        )

    def clear_resolution(self, position: BlockPos) -> None:
        position = BlockPos.coerce(position)
        chunk = self._chunk_for(position)
        entries = self._overlay.get(chunk)
        if entries is None:
            return
        entries.pop(position, None)
        if not entries:
            del self._overlay[chunk]

    def invalidate_connectivity_policy(self) -> None:
        self._source_snapshot = None
        self._policy_epoch += 1
        self._overlay.clear()
        self._dirty_changes.clear()
        self._dirty_all = True

    def iter_source_positions(self) -> Iterator[BlockPos]:
        for chunk in sorted(self._position_placements):
            for position in sorted(self._position_placements[chunk]):
                if self.source_at(position) is not None:
                    yield position

    def iter_resolved_chunk(self, chunk: ChunkPos) -> Iterator[tuple[BlockPos, BlockId]]:
        for position in sorted(self._position_placements.get(chunk, ())):
            cell = self.effective_at(position)
            if cell is not None:
                yield position, cell.block_id

    def iter_resolved_chunks(
        self,
        resolver: NeighborResolver | None = None,
    ) -> Iterator[tuple[ChunkPos, dict[BlockPos, BlockId]]]:
        if resolver is not None:
            self.resolve_dirty(resolver)
        for chunk in sorted(self._position_placements):
            blocks = dict(self.iter_resolved_chunk(chunk))
            if blocks:
                yield chunk, blocks

    def drain_dirty_changes(self) -> Iterable[CellChange | BlockPos]:
        if self._dirty_all:
            self._dirty_all = False
            self._dirty_changes.clear()
            return self.iter_source_positions()
        changes = tuple(self._dirty_changes.values())
        self._dirty_changes.clear()
        return changes

    def resolve_dirty(self, resolver: NeighborResolver) -> ResolutionDelta:
        self._prepare_resolver(resolver)
        self._source_snapshot = {}
        try:
            changed = resolver.resolve_dirty(self, self.drain_dirty_changes())
        finally:
            self._source_snapshot = None
        return ResolutionDelta(frozenset(changed))

    def resolve_all(self, resolver: NeighborResolver) -> ResolutionDelta:
        self._prepare_resolver(resolver)
        self._dirty_changes.clear()
        self._dirty_all = False
        self._overlay.clear()
        self._source_snapshot = {}
        try:
            changed = resolver.resolve_all(self)
        finally:
            self._source_snapshot = None
        return ResolutionDelta(frozenset(changed))

    def _prepare_resolver(self, resolver: NeighborResolver) -> None:
        token = resolver.policy_token
        if (
            self._resolver_policy_token is not None
            and token != self._resolver_policy_token
        ):
            self.invalidate_connectivity_policy()
        self._resolver_policy_token = token

    def _composed_cell(self, position: BlockPos) -> ConnectivityCell | None:
        base_id = self.base_blocks.get(position)
        current = None if base_id is None else ConnectivityCell(
            block_id=base_id,
            domain=self.base_domain,
            source_token=("base", position, base_id),
            placement_id=None,
            sequence_index=None,
        )
        candidates = self._position_placements.get(self._chunk_for(position), {}).get(position, ())
        if len(candidates) > 1:
            candidates = sorted(candidates, key=lambda placement_id: self._placements[placement_id].order)
        for placement_id in candidates:
            placement = self._placements[placement_id]
            if not placement.occupies(position):
                continue
            local_position = placement.transform.inverse(position)
            local_id = placement.field.get_block_id(local_position)
            if local_id is None:
                continue
            block_id = self._world_block_id(placement, local_id)
            if not self._placement_applies(placement, position, current):
                continue
            replacement = placement.block_replacement
            if replacement is not None:
                source_name = placement.field.palette.get_state(local_id).name
                if (
                    source_name not in AIR_NAMES
                    and (replacement.source_blocks is None
                         or source_name in replacement.source_blocks)
                ):
                    block_id = placement.replacement_ids[replacement.index_at(position)]
            current = ConnectivityCell(
                block_id=block_id,
                domain=placement.connectivity_domain,
                source_token=(
                    placement.placement_id,
                    placement.revision,
                    placement.field.revision,
                    local_position,
                    block_id,
                ),
                placement_id=placement.placement_id,
                sequence_index=placement.sequence_index,
            )
        return current

    def _placement_applies(
        self,
        placement: BlockPlacement,
        position: BlockPos,
        current: ConnectivityCell | None,
    ) -> bool:
        if (
            self.placement_policy is not None
            and current is not None
            and not self._is_air(current.block_id)
            and not self.placement_policy(placement.connectivity_domain, current.domain)
        ):
            return False
        mode = placement.placement_mode
        if mode == PlacementMode.OVERWRITE:
            return True
        if mode in (PlacementMode.KEEP_EXISTING, PlacementMode.REPLACE_AIR_ONLY):
            return current is None or self._is_air(current.block_id)
        if mode == PlacementMode.REPLACE_MATCHING:
            current_name = (
                DEFAULT_BLOCK
                if current is None
                else self.registry.get_state(current.block_id).name
            )
            return current_name == placement.replace
        if mode == PlacementMode.TERRAIN_MASK:
            return bool(
                self.terrain_mask is not None
                and self.terrain_mask(position, placement.terrain_mask_mode)
            )
        raise ValueError(f"Unknown placement mode: {mode}")

    def _world_block_id(
        self,
        placement: BlockPlacement,
        local_id: BlockId,
    ) -> BlockId:
        cache_key = (
            id(placement.field.palette),
            local_id,
            placement.transform.quarter_turns,
            placement.transform.mirror,
        )
        cached = self._palette_translation.get(cache_key)
        if cached is not None:
            return cached
        state = placement.field.palette.get_state(local_id)
        world_id = (
            local_id
            if placement.field.palette is self.registry.palette
            else self.registry.register(state)
        )
        turned = self.registry.get_state(world_id)
        if placement.transform.mirror is not None:
            turned = mirror_block_state(
                turned,
                self.registry.get_info(world_id),
                placement.transform.mirror,
            )
        if placement.transform.quarter_turns:
            turned = rotate_block_state(
                turned,
                self.registry.get_info(world_id),
                placement.transform.quarter_turns,
            )
        if turned != self.registry.get_state(world_id):
            world_id = self.registry.get_or_add_variant(world_id, turned)
        self._palette_translation[cache_key] = world_id
        return world_id

    def _sources(self, positions: Iterable[BlockPos]) -> dict[BlockPos, ConnectivityCell | None]:
        return {position: self.source_at(position) for position in positions}

    def _reindex(
        self, placement: BlockPlacement, change: Callable[[], None]
    ) -> tuple[CellChange, ...]:
        # Re-walks a placement's footprint after `change` alters it.
        old = set(self._footprints[placement.placement_id])
        before = self._sources(old)
        self._unindex_placement(placement)
        change()
        placement.revision += 1
        new = set(self._placement_positions(placement))
        before.update(self._sources(new - old))
        self._index_placement(placement, new)
        return self._record_source_changes(before, old | new)

    def _record_source_changes(
        self,
        before: Mapping[BlockPos, ConnectivityCell | None],
        positions: Iterable[BlockPos],
    ) -> tuple[CellChange, ...]:
        changes: list[CellChange] = []
        for position in positions:
            previous = before.get(position)
            current = self.source_at(position)
            if previous == current:
                continue
            self.clear_resolution(position)
            existing = self._dirty_changes.get(position)
            change = CellChange(
                position,
                previous if existing is None else existing.before,
                current,
            )
            if change.before == change.after:
                self._dirty_changes.pop(position, None)
                continue
            self._dirty_changes[position] = change
            changes.append(change)
        return tuple(changes)

    def _placement_positions(self, placement: BlockPlacement) -> Iterator[BlockPos]:
        for local_position, _ in placement.field.iter_block_ids():
            position = placement.transform.apply(local_position)
            if placement.occupies(position):
                yield position

    def _index_placement(
        self,
        placement: BlockPlacement,
        positions: Iterable[BlockPos] | None = None,
    ) -> None:
        if positions is None:
            positions = self._placement_positions(placement)
        footprint = tuple(positions)
        self._footprints[placement.placement_id] = footprint
        self._source_snapshot = None
        for position in footprint:
            chunk = self._chunk_for(position)
            occupants = self._position_placements.setdefault(chunk, {}).setdefault(position, [])
            occupants.append(placement.placement_id)

    def _unindex_placement(
        self,
        placement: BlockPlacement,
    ) -> None:
        footprint = self._footprints.pop(placement.placement_id)
        self._source_snapshot = None
        for position in footprint:
            chunk = self._chunk_for(position)
            cells = self._position_placements[chunk]
            occupants = cells[position]
            occupants.remove(placement.placement_id)
            if not occupants:
                del cells[position]
            if not cells:
                del self._position_placements[chunk]

    def _chunk_for(self, position: BlockPos) -> ChunkPos:
        position = BlockPos.coerce(position)
        return ChunkPos(
            position.x // self.chunk_size,
            position.y // self.chunk_size,
            position.z // self.chunk_size,
        )

    def _is_air(self, block_id: BlockId) -> bool:
        air = self._air_ids.get(block_id)
        if air is None:
            air = self._air_ids[block_id] = self.registry.get_state(block_id).name in AIR_NAMES
        return air


def _policy_token(policy: object) -> object:
    token = getattr(policy, "policy_token", None)
    return id(policy) if token is None else token
