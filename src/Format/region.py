from __future__ import annotations

from array import array
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import floor
from typing import TYPE_CHECKING

from ..Core.Field.block_field import BlockBounds, BlockField
from ..Core.block.palette import BlockId, BlockPalette
from ..Core.block.states import BlockState
from ..Core.coordinates import BlockPos, ChunkPos

if TYPE_CHECKING:
    from ..Core.World.block_world import BlockWorld
    from ..Core.block.connectivity import NeighborResolver


AIR = BlockState.create("minecraft:air")
AIR_INDEX = 0
AIR_NAMES = frozenset({"minecraft:air", "minecraft:cave_air", "minecraft:void_air"})
SECTION_SIZE = 16


@dataclass(frozen=True, slots=True)
class RegionEntity:
    # An entity riding along with the blocks. `data` holds the rest of the
    # entity NBT untouched: a display entity's transformation, brightness and
    # block_state belong to whoever authored it, not to this exporter.
    id: str
    pos: tuple[float, float, float]
    tags: tuple[str, ...] = ()
    data: object | None = None

    @property
    def block_pos(self) -> BlockPos:
        # Entity positions are doubles. Flooring gives one rule for which
        # cell, and so which piece, an entity belongs to.
        return BlockPos(*(floor(value) for value in self.pos))

    def moved(self, offset: BlockPos) -> RegionEntity:
        return RegionEntity(
            self.id,
            tuple(self.pos[axis] + offset[axis] for axis in range(3)),
            self.tags,
            self.data,
        )


class Axis(Enum):
    X = 0
    Y = 1
    Z = 2


class Overflow(Enum):
    RAISE = "raise"
    SUBDIVIDE = "subdivide"


@dataclass(frozen=True, slots=True)
class SplitBudget:
    # max_blocks counts non-air blocks, or the whole box when count_air is set.
    max_blocks: int
    count_air: bool = False
    # None picks the longest extent.
    axis: Axis | None = None
    # None trims each piece flush to its content. A bounds tiles that
    # footprint instead, holding every piece out to it.
    extent: BlockBounds | None = None
    # Cut planes along the split axis, so a swept path can break where its
    # stations are rather than where the bounding box says.
    cuts: tuple[int, ...] = ()
    # A count budget cannot bound the box: a sparse stretch adds almost no
    # blocks and grows a run without limit. max_span caps every extent.
    max_span: int | None = None
    overflow: Overflow = Overflow.SUBDIVIDE
    name_template: str = "{name}_{index:03d}"

    def __post_init__(self) -> None:
        if self.max_blocks <= 0:
            raise ValueError("max_blocks must be positive")
        if self.max_span is not None and self.max_span <= 0:
            raise ValueError("max_span must be positive")
        object.__setattr__(self, "cuts", tuple(sorted(set(self.cuts))))

    def capped(self, span: int) -> SplitBudget:
        # Tighten max_span without loosening a stricter caller value.
        limit = span if self.max_span is None else min(self.max_span, span)
        return SplitBudget(
            self.max_blocks,
            self.count_air,
            self.axis,
            self.extent,
            self.cuts,
            limit,
            self.overflow,
            self.name_template,
        )


class BlockRegion:
    # A bounded box of cells with a local palette: one litematica region, one
    # .schem volume, or the source .mca sections are sliced from.
    #
    # Cells run x fastest, then z, then y, the order all three formats use.

    __slots__ = ("name", "origin", "size", "palette", "cells", "occupancy", "entities")

    def __init__(
        self,
        name: str,
        origin: Sequence[int],
        size: Sequence[int],
        palette: BlockPalette | None = None,
    ) -> None:
        origin = BlockPos.coerce(origin)
        size = BlockPos.coerce(size)
        if size.x <= 0 or size.y <= 0 or size.z <= 0:
            raise ValueError(f"region size must be positive, got {tuple(size)}")
        if palette is None:
            palette = BlockPalette()
        if len(palette) == 0:
            palette.add(AIR)
        if palette.get_state(AIR_INDEX) != AIR:
            raise ValueError("a region palette holds minecraft:air at index 0")
        volume = size.x * size.y * size.z
        self.name = name
        self.origin = origin
        self.size = size
        self.palette = palette
        self.cells = array("I", bytes(volume * 4))
        self.occupancy = bytearray(volume)
        self.entities: list[RegionEntity] = []

    @classmethod
    def from_field(
        cls,
        field: BlockField,
        *,
        name: str,
        origin: Sequence[int] = BlockPos(0, 0, 0),
    ) -> BlockRegion:
        bounds = field.bounds
        if bounds is None:
            raise ValueError("cannot build a region from an empty field")
        offset = BlockPos.coerce(origin)
        region = cls(name, bounds.minimum + offset, _extent_of(bounds))
        for position, block_id in field.iter_block_ids():
            region.set_state(position + offset, field.palette.get_state(block_id))
        return region

    @classmethod
    def from_world(
        cls,
        world: BlockWorld,
        *,
        name: str,
        bounds: BlockBounds | None = None,
        resolver: NeighborResolver | None = None,
    ) -> BlockRegion:
        # A resolver settles dirty connectivity first; without one the world
        # exports whatever it already settled on.
        collected: dict[BlockPos, BlockId] = {}
        for _, blocks in world.iter_resolved_chunks(resolver):
            for position, block_id in blocks.items():
                if bounds is None or bounds.contains(position):
                    collected[position] = block_id
        if bounds is None:
            if not collected:
                raise ValueError("cannot build a region from an empty world")
            bounds = _bounds_of(collected)
        region = cls(name, bounds.minimum, _extent_of(bounds))
        for position, block_id in collected.items():
            region.set_state(position, world.registry.get_state(block_id))
        return region

    @property
    def volume(self) -> int:
        return self.size.x * self.size.y * self.size.z

    @property
    def bounds(self) -> BlockBounds:
        return BlockBounds(self.origin, self.origin + (self.size - (1, 1, 1)))

    @property
    def filled_count(self) -> int:
        # Cells this region has an opinion about, air included.
        return sum(self.occupancy)

    @property
    def block_count(self) -> int:
        # Non-air cells: litematica's TotalBlocks, and the split budget's count.
        air = self._air_indices()
        return sum(
            1
            for index, flag in enumerate(self.occupancy)
            if flag and self.cells[index] not in air
        )

    @property
    def has_content(self) -> bool:
        return any(self.occupancy)

    def linear_index(self, position: Sequence[int]) -> int:
        position = BlockPos.coerce(position)
        x = position.x - self.origin.x
        y = position.y - self.origin.y
        z = position.z - self.origin.z
        if not (0 <= x < self.size.x and 0 <= y < self.size.y and 0 <= z < self.size.z):
            raise IndexError(f"{tuple(position)} lies outside {self.name}")
        return (y * self.size.z + z) * self.size.x + x

    def position_at(self, index: int) -> BlockPos:
        x, rest = index % self.size.x, index // self.size.x
        z, y = rest % self.size.z, rest // self.size.z
        return BlockPos(self.origin.x + x, self.origin.y + y, self.origin.z + z)

    def set_state(self, position: Sequence[int], state: BlockState) -> None:
        index = self.linear_index(position)
        self.cells[index] = self.palette.add(state)
        self.occupancy[index] = 1

    def state_at(self, position: Sequence[int]) -> BlockState:
        return self.palette.get_state(self.cells[self.linear_index(position)])

    def is_occupied(self, position: Sequence[int]) -> bool:
        return bool(self.occupancy[self.linear_index(position)])

    def iter_cells(self) -> Iterator[int]:
        # Untouched cells read as air, which is what a dense format wants.
        return iter(self.cells)

    def iter_occupied(self) -> Iterator[tuple[BlockPos, BlockState]]:
        for index, flag in enumerate(self.occupancy):
            if flag:
                yield self.position_at(index), self.palette.get_state(self.cells[index])

    def add_entity(self, entity: RegionEntity) -> None:
        self.entities.append(entity)

    def translated(self, offset: Sequence[int]) -> BlockRegion:
        offset = BlockPos.coerce(offset)
        moved = BlockRegion(self.name, self.origin + offset, self.size)
        moved.palette = self.palette
        moved.cells = array("I", self.cells)
        moved.occupancy = bytearray(self.occupancy)
        moved.entities = [entity.moved(offset) for entity in self.entities]
        return moved

    def sliced(self, bounds: BlockBounds, *, name: str | None = None) -> BlockRegion:
        # Bounds may reach past this region; the overhang is untouched air.
        # The palette is built from what lands inside, so pieces stay minimal.
        out = BlockRegion(self.name if name is None else name, bounds.minimum, _extent_of(bounds))
        source = self.bounds
        for y in range(max(bounds.minimum.y, source.minimum.y), min(bounds.maximum.y, source.maximum.y) + 1):
            for z in range(max(bounds.minimum.z, source.minimum.z), min(bounds.maximum.z, source.maximum.z) + 1):
                for x in range(max(bounds.minimum.x, source.minimum.x), min(bounds.maximum.x, source.maximum.x) + 1):
                    position = BlockPos(x, y, z)
                    index = self.linear_index(position)
                    if self.occupancy[index]:
                        out.set_state(position, self.palette.get_state(self.cells[index]))
        out.entities = [e for e in self.entities if bounds.contains(e.block_pos)]
        return out

    def iter_sections(self) -> Iterator[tuple[ChunkPos, BlockRegion]]:
        # World-aligned 16^3 boxes for anvil.py. Sections nothing touched are
        # skipped; a section of explicit air is not.
        bounds = self.bounds
        for y in _section_range(bounds.minimum.y, bounds.maximum.y):
            for z in _section_range(bounds.minimum.z, bounds.maximum.z):
                for x in _section_range(bounds.minimum.x, bounds.maximum.x):
                    base = BlockPos(x * SECTION_SIZE, y * SECTION_SIZE, z * SECTION_SIZE)
                    section = self.sliced(
                        BlockBounds(base, base + (SECTION_SIZE - 1,) * 3),
                        name=f"{self.name}_s{x}_{y}_{z}",
                    )
                    if section.has_content:
                        yield ChunkPos(x, y, z), section

    def _air_indices(self) -> frozenset[int]:
        return frozenset(
            index
            for index in range(len(self.palette))
            if self.palette.get_state(index).name in AIR_NAMES
        )

    def __repr__(self) -> str:
        return (
            f"BlockRegion(name={self.name!r}, origin={tuple(self.origin)}, "
            f"size={tuple(self.size)}, blocks={self.block_count})"
        )


def enclosing_bounds(regions: Sequence[BlockRegion]) -> BlockBounds:
    if not regions:
        raise ValueError("no regions to enclose")
    return BlockBounds(
        BlockPos(*(min(r.bounds.minimum[a] for r in regions) for a in range(3))),
        BlockPos(*(max(r.bounds.maximum[a] for r in regions) for a in range(3))),
    )


def split_by_budget(region: BlockRegion, budget: SplitBudget) -> list[BlockRegion]:
    # Pieces tile the split axis in ascending order, which is also paste
    # order. A boundary only falls between slabs, so coverage is lossless.
    if budget.extent is not None:
        _require_within(region, budget.extent)
    axis = budget.axis if budget.axis is not None else _longest_axis(region.size)
    pieces: list[BlockRegion] = []
    for segment, extent in _cut_segments(region, budget, axis):
        pieces.extend(_split_axis(segment, budget, axis, extent, (axis,)))
    for index, piece in enumerate(pieces):
        piece.name = budget.name_template.format(name=region.name, index=index)
    return pieces


@dataclass(slots=True)
class _Run:
    minimum: list[int]
    maximum: list[int]
    count: int

    def joined(self, other: _Run) -> _Run:
        return _Run(
            [min(self.minimum[a], other.minimum[a]) for a in range(3)],
            [max(self.maximum[a], other.maximum[a]) for a in range(3)],
            self.count + other.count,
        )

    @property
    def extents(self) -> tuple[int, ...]:
        return tuple(self.maximum[a] - self.minimum[a] + 1 for a in range(3))

    @property
    def box(self) -> BlockBounds:
        return BlockBounds(BlockPos(*self.minimum), BlockPos(*self.maximum))


def _exceeds(run: _Run, budget: SplitBudget) -> bool:
    # The box is settled before it is measured: trimming flush shrinks it, so
    # measuring an untrimmed run would under-fill every piece.
    extents = run.extents
    blocks = extents[0] * extents[1] * extents[2] if budget.count_air else run.count
    if blocks > budget.max_blocks:
        return True
    return budget.max_span is not None and max(extents) > budget.max_span


def _split_axis(
    region: BlockRegion,
    budget: SplitBudget,
    axis: Axis,
    extent: BlockBounds | None,
    used: tuple[Axis, ...],
) -> list[BlockRegion]:
    pieces: list[BlockRegion] = []
    run: _Run | None = None
    for slab in _slabs(region, axis, extent):
        if run is None:
            run = slab
        elif _exceeds(run.joined(slab), budget):
            pieces.append(region.sliced(run.box))
            run = slab
        else:
            run = run.joined(slab)
            continue
        if _exceeds(run, budget):
            pieces.extend(_overflow(region, run, budget, axis, extent, used))
            run = None
    if run is not None:
        pieces.append(region.sliced(run.box))
    return pieces


def _overflow(
    region: BlockRegion,
    run: _Run,
    budget: SplitBudget,
    axis: Axis,
    extent: BlockBounds | None,
    used: tuple[Axis, ...],
) -> list[BlockRegion]:
    extents = run.extents
    remaining = [candidate for candidate in Axis if candidate not in used]
    if budget.overflow is Overflow.RAISE or not remaining:
        raise ValueError(f"{region.name} at {extents} does not fit the budget")
    following = max(remaining, key=lambda candidate: extents[candidate.value])
    return _split_axis(
        region.sliced(run.box),
        budget,
        following,
        None if extent is None else _intersect(extent, run.box),
        used + (following,),
    )


def _slabs(region: BlockRegion, axis: Axis, extent: BlockBounds | None) -> list[_Run]:
    # One pass over the occupied cells. Bounds follow occupancy, air included,
    # since placed air is still footprint; the count follows non-air cells.
    index = axis.value
    air = region._air_indices()
    runs: dict[int, _Run] = {}
    for cell, flag in enumerate(region.occupancy):
        if not flag:
            continue
        position = region.position_at(cell)
        components = list(position.as_tuple())
        blocks = 0 if region.cells[cell] in air else 1
        run = runs.get(components[index])
        if run is None:
            runs[components[index]] = _Run(components, list(components), blocks)
            continue
        for component in range(3):
            run.minimum[component] = min(run.minimum[component], components[component])
            run.maximum[component] = max(run.maximum[component], components[component])
        run.count += blocks
    for entity in region.entities:
        components = list(entity.block_pos.as_tuple())
        run = runs.get(components[index])
        if run is None:
            runs[components[index]] = _Run(components, list(components), 0)
        else:
            for component in range(3):
                run.minimum[component] = min(run.minimum[component], components[component])
                run.maximum[component] = max(run.maximum[component], components[component])
    if extent is None:
        # Flush with the build: empty leading and trailing slabs never enter a
        # run, while a gap between filled slabs still costs span.
        return [runs[key] for key in sorted(runs)]
    # Flush with the footprint: every slab is present with the footprint's
    # cross-section, so pieces tile it whether or not the build reaches them.
    slabs = []
    for coordinate in range(extent.minimum[index], extent.maximum[index] + 1):
        minimum = list(extent.minimum.as_tuple())
        maximum = list(extent.maximum.as_tuple())
        minimum[index] = maximum[index] = coordinate
        run = runs.get(coordinate)
        slabs.append(_Run(minimum, maximum, 0 if run is None else run.count))
    return slabs


def _cut_segments(
    region: BlockRegion,
    budget: SplitBudget,
    axis: Axis,
) -> list[tuple[BlockRegion, BlockBounds | None]]:
    extent = budget.extent
    span = region.bounds if extent is None else extent
    if not budget.cuts:
        return [(region, extent)]
    low, high = span.minimum[axis.value], span.maximum[axis.value]
    edges = [low, *(cut for cut in budget.cuts if low < cut <= high), high + 1]
    segments = []
    for start, stop in zip(edges, edges[1:]):
        if start >= stop:
            continue
        box = _axis_slice(span, axis, start, stop - 1)
        segment = region.sliced(box)
        if extent is None and not segment.has_content:
            continue
        segments.append((segment, None if extent is None else _intersect(extent, box)))
    return segments


def _axis_slice(bounds: BlockBounds, axis: Axis, low: int, high: int) -> BlockBounds:
    minimum = list(bounds.minimum.as_tuple())
    maximum = list(bounds.maximum.as_tuple())
    minimum[axis.value], maximum[axis.value] = low, high
    return BlockBounds(BlockPos(*minimum), BlockPos(*maximum))


def _intersect(left: BlockBounds, right: BlockBounds) -> BlockBounds:
    return BlockBounds(
        BlockPos(*(max(left.minimum[a], right.minimum[a]) for a in range(3))),
        BlockPos(*(min(left.maximum[a], right.maximum[a]) for a in range(3))),
    )


def _require_within(region: BlockRegion, extent: BlockBounds) -> None:
    # Dropping blocks outside a stated footprint is found three schematics
    # into pasting, so it is an error.
    for position, _ in region.iter_occupied():
        if not extent.contains(position):
            raise ValueError(f"{region.name} places a block at {tuple(position)}, outside the extent")


def _section_range(low: int, high: int) -> range:
    return range(low // SECTION_SIZE, high // SECTION_SIZE + 1)


def _longest_axis(size: BlockPos) -> Axis:
    return max(Axis, key=lambda axis: size[axis.value])


def _extent_of(bounds: BlockBounds) -> BlockPos:
    return bounds.maximum - bounds.minimum + (1, 1, 1)


def _bounds_of(positions) -> BlockBounds:
    return BlockBounds(
        BlockPos(*(min(p[a] for p in positions) for a in range(3))),
        BlockPos(*(max(p[a] for p in positions) for a in range(3))),
    )
