from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from ..block.palette import BlockId, BlockPalette
from ..block.states import BlockState


BlockPos = tuple[int, int, int]
ChunkPos = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class BlockBounds:
    minimum: BlockPos
    maximum: BlockPos

    def contains(self, position: BlockPos) -> bool:
        return all(
            lower <= value <= upper
            for value, lower, upper in zip(position, self.minimum, self.maximum)
        )


class BlockField:
    # A reusable sparse block template in local coordinates.

    def __init__(self, palette: BlockPalette | None = None) -> None:
        self._field: dict[BlockPos, BlockId] = {}
        self._palette = palette if palette is not None else BlockPalette()
        self._revision = 0
        self._frozen = False
        self._bounds: BlockBounds | None = None
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
            self._bounds = self._calculate_bounds()
            self._bounds_dirty = False
        return self._bounds

    def freeze(self) -> None:
        # Field freezing protects every placement that shares this template.
        self._frozen = True

    def set_block(self, position: BlockPos, block: BlockState) -> None:
        self._ensure_mutable()
        self.set_block_id(position, self._palette.add(block))

    def set_block_id(self, position: BlockPos, block_id: BlockId) -> None:
        self._ensure_mutable()
        self._palette.get_state(block_id)
        previous = self._field.get(position)
        if previous == block_id:
            return
        self._field[position] = block_id
        self._revision += 1
        self._include_in_bounds(position)

    def get_block(self, position: BlockPos) -> BlockState | None:
        block_id = self.get_block_id(position)
        return None if block_id is None else self._palette.get_state(block_id)

    def get_block_id(self, position: BlockPos) -> BlockId | None:
        return self._field.get(position)

    def remove_block(self, position: BlockPos) -> None:
        self._ensure_mutable()
        if self._field.pop(position, None) is None:
            return
        self._revision += 1
        self._bounds_dirty = True

    def iter_block_ids(
        self,
        bounds: BlockBounds | None = None,
    ) -> Iterator[tuple[BlockPos, BlockId]]:
        for position, block_id in self._field.items():
            if bounds is None or bounds.contains(position):
                yield position, block_id

    def occupied_chunks(self, chunk_size: int = 16) -> frozenset[ChunkPos]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        return frozenset(
            tuple(value // chunk_size for value in position)
            for position in self._field
        )

    def has_block(self, position: BlockPos) -> bool:
        return position in self._field

    def __contains__(self, position: BlockPos) -> bool:
        return self.has_block(position)

    def __len__(self) -> int:
        return len(self._field)

    def _ensure_mutable(self) -> None:
        if self._frozen:
            raise RuntimeError("BlockField is frozen")

    def _include_in_bounds(self, position: BlockPos) -> None:
        if self._bounds_dirty:
            self._bounds = self._calculate_bounds()
            self._bounds_dirty = False
        if self._bounds is None:
            self._bounds = BlockBounds(position, position)
            self._bounds_dirty = False
            return
        self._bounds = BlockBounds(
            tuple(
                min(left, right)
                for left, right in zip(self._bounds.minimum, position)
            ),
            tuple(
                max(left, right)
                for left, right in zip(self._bounds.maximum, position)
            ),
        )

    def _calculate_bounds(self) -> BlockBounds | None:
        if not self._field:
            return None
        positions = iter(self._field)
        first = next(positions)
        minimum = list(first)
        maximum = list(first)
        for position in positions:
            for axis, value in enumerate(position):
                minimum[axis] = min(minimum[axis], value)
                maximum[axis] = max(maximum[axis], value)
        return BlockBounds(tuple(minimum), tuple(maximum))
