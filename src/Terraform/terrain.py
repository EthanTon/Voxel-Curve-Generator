from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..Core.Field.block_field import BlockField
from ..Core.block.states import BlockState
from ..Core.coordinates import BlockPos
from .classify import BlockClass, classify, is_tree
from .grid import Field, Grid


CHUNK = 16
ABSENT = -32768
SOLID = (BlockClass.GROUND, BlockClass.STRUCTURE)


def region_directory(source: Path) -> Path:
    held = source / "region"
    found = held if held.is_dir() else source
    if not any(found.glob("r.*.mca")):
        raise ValueError(
            f"the terraform world names {source}, which holds no region files. "
            "Point it at a save folder or at its region directory"
        )
    return found


@runtime_checkable
class Terrain(Protocol):
    def generated(self, chunk_x: int, chunk_z: int) -> bool: ...

    def column(self, x: int, z: int) -> Sequence[tuple[int, BlockState]]: ...


@dataclass(frozen=True, slots=True)
class EmptyTerrain:
    def generated(self, chunk_x: int, chunk_z: int) -> bool:
        return False

    def column(self, x: int, z: int) -> Sequence[tuple[int, BlockState]]:
        return ()


@dataclass(slots=True)
class FieldTerrain:
    blocks: BlockField
    _columns: dict[tuple[int, int], list[tuple[int, BlockState]]] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        for position, block_id in self.blocks.iter_block_ids():
            state = self.blocks.palette.get_state(block_id)
            self._columns.setdefault((position.x, position.z), []).append(
                (position.y, state)
            )
        for stack in self._columns.values():
            stack.sort()

    def generated(self, chunk_x: int, chunk_z: int) -> bool:
        return any(
            x // CHUNK == chunk_x and z // CHUNK == chunk_z for x, z in self._columns
        )

    def column(self, x: int, z: int) -> Sequence[tuple[int, BlockState]]:
        return self._columns.get((x, z), ())


@dataclass(frozen=True, slots=True)
class Survey:
    height: Field
    available: Field
    state: Mapping[tuple[int, int], BlockState]
    growth: frozenset[BlockPos]
    intruding: frozenset[BlockPos]
    # The growth that is part of trees.
    trees: frozenset[BlockPos] = frozenset()
    water: Field | None = None
    structures: Field | None = None


def survey(
    terrain: Terrain,
    grid: Grid,
    intrudes=None,
    growth: bool = True,
) -> Survey:
    height = grid.filled(ABSENT, dtype=np.int32)
    water = grid.filled(ABSENT, dtype=np.int32)
    structures = grid.filled(False, dtype=bool)
    available = grid.filled(False, dtype=bool)
    states: dict[tuple[int, int], BlockState] = {}
    standing: set[BlockPos] = set()
    trees: set[BlockPos] = set()
    inside: set[BlockPos] = set()
    chunks: dict[tuple[int, int], bool] = {}
    for row, cell, column in grid.by_chunk(CHUNK):
        chunk = (column.x // CHUNK, column.z // CHUNK)
        if chunk not in chunks:
            chunks[chunk] = terrain.generated(*chunk)
        if not chunks[chunk]:
            continue
        available[row, cell] = True
        found = False
        highest_structure = ABSENT
        for y, state in reversed(terrain.column(column.x, column.z)):
            kind = classify(state)
            if kind is BlockClass.STRUCTURE:
                highest_structure = max(highest_structure, y)
            if state.name in ("minecraft:water", "minecraft:bubble_column"):
                water[row, cell] = max(water[row, cell], y)
            if kind is BlockClass.VEGETATION:
                if growth:
                    standing.add(BlockPos(column.x, y, column.z))
                    if is_tree(state):
                        trees.add(BlockPos(column.x, y, column.z))
                continue
            if kind in (*SOLID, BlockClass.FLUID) and intrudes is not None and intrudes(row, cell, y):
                inside.add(BlockPos(column.x, y, column.z))
            if kind is BlockClass.GROUND and not found:
                height[row, cell] = y
                states[(row, cell)] = state
                found = True
        structures[row, cell] = found and highest_structure >= height[row, cell]
    return Survey(
        height, available, states, frozenset(standing), frozenset(inside),
        frozenset(trees), water, structures,
    )


def load_terrain(source: Path | str | None) -> Terrain:
    if source is None:
        return EmptyTerrain()
    source = Path(source).resolve()
    if not source.exists():
        raise ValueError(f"the terraform world names {source}, which is not there")
    if source.is_dir():
        from .region import RegionTerrain

        return RegionTerrain(region_directory(source))
    from ..Format.schematics import load_blocks

    return FieldTerrain(load_blocks(source))
