from __future__ import annotations

from array import array
from bisect import bisect_left
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from ..block.palette import BlockId, BlockPalette
from ..block.states import BlockState

from ..coordinates import BlockPos, ChunkPos


_STORAGE_CHUNK_SIZE = 16
_DENSE_CHUNK_VOLUME = _STORAGE_CHUNK_SIZE**3
_MUTABLE_DENSE_THRESHOLD = 256
_FROZEN_DENSE_THRESHOLD = 2731
_EMPTY_BLOCK_ID = 0xFFFFFFFF
_ChunkKey = tuple[int, int, int]


class _BlockChunk:
    __slots__ = ("_count", "_dense", "_frozen", "_ids", "_indices", "_sparse")

    def __init__(self) -> None:
        self._count = 0
        self._dense: array[int] | None = None
        self._frozen = False
        self._ids: array[int] | None = None
        self._indices: array[int] | None = None
        self._sparse: dict[int, BlockId] | None = {}

    def get(self, index: int) -> BlockId | None:
        if self._dense is not None:
            value = self._dense[index]
            return None if value == _EMPTY_BLOCK_ID else value
        if self._sparse is not None:
            return self._sparse.get(index)
        assert self._indices is not None and self._ids is not None
        offset = bisect_left(self._indices, index)
        if offset == len(self._indices) or self._indices[offset] != index:
            return None
        return self._ids[offset]

    def set(self, index: int, block_id: BlockId) -> BlockId | None:
        if block_id < 0 or block_id >= _EMPTY_BLOCK_ID:
            raise ValueError("block_id must fit in an unsigned 32-bit value")
        if self._dense is not None:
            previous = self._dense[index]
            self._dense[index] = block_id
            if previous == _EMPTY_BLOCK_ID:
                self._count += 1
                return None
            return previous
        assert self._sparse is not None
        previous = self._sparse.get(index)
        self._sparse[index] = block_id
        if previous is None:
            self._count += 1
            if self._count >= _MUTABLE_DENSE_THRESHOLD:
                self._make_dense()
        return previous

    def remove(self, index: int) -> BlockId | None:
        if self._dense is not None:
            previous = self._dense[index]
            if previous == _EMPTY_BLOCK_ID:
                return None
            self._dense[index] = _EMPTY_BLOCK_ID
            self._count -= 1
            return previous
        assert self._sparse is not None
        previous = self._sparse.pop(index, None)
        if previous is not None:
            self._count -= 1
        return previous

    def freeze(self) -> None:
        if self._frozen:
            return
        if self._dense is not None and self._count < _FROZEN_DENSE_THRESHOLD:
            items = (
                (index, block_id)
                for index, block_id in enumerate(self._dense)
                if block_id != _EMPTY_BLOCK_ID
            )
            self._indices, self._ids = _compact_items(items)
            self._dense = None
        elif self._sparse is not None:
            items = sorted(self._sparse.items())
            self._indices, self._ids = _compact_items(items)
            self._sparse = None
        self._frozen = True

    def items(self) -> Iterator[tuple[int, BlockId]]:
        if self._dense is not None:
            for index, block_id in enumerate(self._dense):
                if block_id != _EMPTY_BLOCK_ID:
                    yield index, block_id
            return
        if self._sparse is not None:
            yield from self._sparse.items()
            return
        assert self._indices is not None and self._ids is not None
        yield from zip(self._indices, self._ids)

    def __len__(self) -> int:
        return self._count

    def _make_dense(self) -> None:
        assert self._sparse is not None
        dense = array("I", [_EMPTY_BLOCK_ID]) * _DENSE_CHUNK_VOLUME
        for index, block_id in self._sparse.items():
            dense[index] = block_id
        self._dense = dense
        self._sparse = None


def _compact_items(
    items: Iterable[tuple[int, BlockId]],
) -> tuple[array[int], array[int]]:
    indices = array("H")
    block_ids = array("I")
    for index, block_id in items:
        indices.append(index)
        block_ids.append(block_id)
    return indices, block_ids


@dataclass(frozen=True, slots=True)
class BlockBounds:
    minimum: BlockPos
    maximum: BlockPos

    def __post_init__(self) -> None:
        object.__setattr__(self, "minimum", BlockPos.coerce(self.minimum))
        object.__setattr__(self, "maximum", BlockPos.coerce(self.maximum))

    def contains(self, position: BlockPos) -> bool:
        position = BlockPos.coerce(position)
        return (
            self.minimum.x <= position.x <= self.maximum.x
            and self.minimum.y <= position.y <= self.maximum.y
            and self.minimum.z <= position.z <= self.maximum.z
        )


class BlockField:
    # A reusable sparse block template in local coordinates.

    def __init__(self, palette: BlockPalette | None = None) -> None:
        self._chunks: dict[_ChunkKey, _BlockChunk] = {}
        self._block_count = 0
        self._palette = palette if palette is not None else BlockPalette()
        self._revision = 0
        self._frozen = False
        self._bounds: BlockBounds | None = None
        self._minimum: list[int] | None = None
        self._maximum: list[int] | None = None
        self._bounds_dirty = False

    @property
    def palette(self) -> BlockPalette:
        return self._palette

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def bounds(self) -> BlockBounds | None:
        if self._bounds_dirty:
            self._recalculate_bounds()
            self._bounds_dirty = False
        if (
            self._bounds is None
            and self._minimum is not None
            and self._maximum is not None
        ):
            self._bounds = BlockBounds(
                BlockPos(*self._minimum),
                BlockPos(*self._maximum),
            )
        return self._bounds

    def freeze(self) -> None:
        # Field freezing protects every placement that shares this template.
        for chunk in self._chunks.values():
            chunk.freeze()
        self._frozen = True

    def set_block(self, position: BlockPos, block: BlockState) -> None:
        self._ensure_mutable()
        self.set_block_id(position, self._palette.add(block))

    def set_block_id(self, position: BlockPos, block_id: BlockId) -> None:
        self._ensure_mutable()
        position = BlockPos.coerce(position)
        self._palette.get_state(block_id)
        chunk_position, local_index = _storage_location(position)
        chunk = self._chunks.get(chunk_position)
        if chunk is None:
            chunk = _BlockChunk()
            self._chunks[chunk_position] = chunk
        previous = chunk.set(local_index, block_id)
        if previous == block_id:
            return
        if previous is None:
            self._block_count += 1
            self._include_in_bounds(position)
        self._revision += 1

    def get_block(self, position: BlockPos) -> BlockState | None:
        block_id = self.get_block_id(position)
        return None if block_id is None else self._palette.get_state(block_id)

    def get_block_id(self, position: BlockPos) -> BlockId | None:
        chunk_position, local_index = _storage_location(BlockPos.coerce(position))
        chunk = self._chunks.get(chunk_position)
        return None if chunk is None else chunk.get(local_index)

    def remove_block(self, position: BlockPos) -> None:
        self._ensure_mutable()
        position = BlockPos.coerce(position)
        chunk_position, local_index = _storage_location(position)
        chunk = self._chunks.get(chunk_position)
        if chunk is None or chunk.remove(local_index) is None:
            return
        self._block_count -= 1
        if not chunk:
            del self._chunks[chunk_position]
        self._revision += 1
        self._bounds_dirty = True

    def iter_block_ids(
        self,
        bounds: BlockBounds | None = None,
    ) -> Iterator[tuple[BlockPos, BlockId]]:
        for chunk_position, chunk in self._chunks.items():
            for local_index, block_id in chunk.items():
                position = _block_position(chunk_position, local_index)
                if bounds is None or bounds.contains(position):
                    yield position, block_id

    def occupied_chunks(self, chunk_size: int = 16) -> frozenset[ChunkPos]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_size == _STORAGE_CHUNK_SIZE:
            return frozenset(ChunkPos(*chunk) for chunk in self._chunks)
        return frozenset(
            ChunkPos(
                position.x // chunk_size,
                position.y // chunk_size,
                position.z // chunk_size,
            )
            for position, _ in self.iter_block_ids()
        )

    def has_block(self, position: BlockPos) -> bool:
        return self.get_block_id(position) is not None

    def __contains__(self, position: BlockPos) -> bool:
        return self.has_block(position)

    def __len__(self) -> int:
        return self._block_count

    def _ensure_mutable(self) -> None:
        if self._frozen:
            raise RuntimeError("BlockField is frozen")

    def _include_in_bounds(self, position: BlockPos) -> None:
        if self._bounds_dirty:
            self._recalculate_bounds()
            self._bounds_dirty = False
        if self._minimum is None or self._maximum is None:
            self._minimum = [position.x, position.y, position.z]
            self._maximum = [position.x, position.y, position.z]
            self._bounds = None
            return
        self._minimum[0] = min(self._minimum[0], position.x)
        self._minimum[1] = min(self._minimum[1], position.y)
        self._minimum[2] = min(self._minimum[2], position.z)
        self._maximum[0] = max(self._maximum[0], position.x)
        self._maximum[1] = max(self._maximum[1], position.y)
        self._maximum[2] = max(self._maximum[2], position.z)
        self._bounds = None

    def _recalculate_bounds(self) -> None:
        if not self._chunks:
            self._minimum = None
            self._maximum = None
            self._bounds = None
            return
        positions = (position for position, _ in self.iter_block_ids())
        first = next(positions)
        minimum = [first.x, first.y, first.z]
        maximum = [first.x, first.y, first.z]
        for position in positions:
            for axis, value in enumerate(position):
                minimum[axis] = min(minimum[axis], value)
                maximum[axis] = max(maximum[axis], value)
        self._minimum = minimum
        self._maximum = maximum
        self._bounds = None


def _storage_location(position: BlockPos) -> tuple[_ChunkKey, int]:
    chunk_x, local_x = divmod(position.x, _STORAGE_CHUNK_SIZE)
    chunk_y, local_y = divmod(position.y, _STORAGE_CHUNK_SIZE)
    chunk_z, local_z = divmod(position.z, _STORAGE_CHUNK_SIZE)
    local_index = (local_y << 8) | (local_z << 4) | local_x
    return (chunk_x, chunk_y, chunk_z), local_index


def _block_position(chunk: _ChunkKey, local_index: int) -> BlockPos:
    local_x = local_index & 0xF
    local_z = (local_index >> 4) & 0xF
    local_y = (local_index >> 8) & 0xF
    return BlockPos(
        chunk[0] * _STORAGE_CHUNK_SIZE + local_x,
        chunk[1] * _STORAGE_CHUNK_SIZE + local_y,
        chunk[2] * _STORAGE_CHUNK_SIZE + local_z,
    )
