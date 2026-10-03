from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..Core.Field.block_field import BlockField
from ..Core.block.states import BlockState
from ..Core.coordinates import BlockPos
from .classify import BlockClass, classify, is_tree, is_trunk
from .grid import Field, Grid


CHUNK = 16
ABSENT = -32768
SOLID = (BlockClass.GROUND, BlockClass.STRUCTURE)
# How many blocks under each column's surface the survey keeps.
SUBSURFACE = 16


def region_directory(source: Path, dimension: str | None = None) -> Path:
    if dimension is not None and (not isinstance(dimension, str) or not dimension):
        raise ValueError("dimension must be a name")
    if dimension and dimension not in ("overworld", "minecraft:overworld"):
        namespace, separator, name = dimension.partition(":")
        if not separator:
            namespace, name = "minecraft", namespace
        if (not namespace or not name or ":" in name or "/" in namespace
                or namespace in (".", "..")
                or any(part in ("", ".", "..") for part in name.split("/"))
                or "\\" in dimension):
            raise ValueError(f"invalid dimension {dimension!r}")
        if (namespace, name) == ("minecraft", "the_nether"):
            source /= "DIM-1"
        elif (namespace, name) == ("minecraft", "the_end"):
            source /= "DIM1"
        else:
            source = source / "dimensions" / namespace / name
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
    # The solid blocks under each column's surface, a layer per block of
    # depth (-1 where there is none), as indices into `layer_states`.
    layers: np.ndarray | None = None
    layer_states: tuple[BlockState, ...] = ()
    top: Field | None = None
    trunks: frozenset[BlockPos] = frozenset()


def survey(
    terrain: Terrain,
    grid: Grid,
    intrudes=None,
    growth: bool = True,
    area: Field | None = None,
) -> Survey:
    height = grid.filled(ABSENT, dtype=np.int32)
    top = grid.filled(ABSENT, dtype=np.int32)
    water = grid.filled(ABSENT, dtype=np.int32)
    structures = grid.filled(False, dtype=bool)
    available = grid.filled(False, dtype=bool)
    layers = np.full((SUBSURFACE, grid.depth, grid.width), -1, dtype=np.int16)
    kept: dict[BlockState, int] = {}
    states: dict[tuple[int, int], BlockState] = {}
    standing: set[BlockPos] = set()
    trees: set[BlockPos] = set()
    trunks: set[BlockPos] = set()
    inside: set[BlockPos] = set()
    chunks: dict[tuple[int, int], bool] = {}
    for row, cell, column in grid.by_chunk(CHUNK):
        if area is not None and not area[row, cell]:
            continue
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
            if kind is not BlockClass.AIR:
                top[row, cell] = max(top[row, cell], y)
            if kind is BlockClass.STRUCTURE:
                highest_structure = max(highest_structure, y)
            if state.name in ("minecraft:water", "minecraft:bubble_column"):
                water[row, cell] = max(water[row, cell], y)
            if kind is BlockClass.VEGETATION:
                if growth:
                    standing.add(BlockPos(column.x, y, column.z))
                    if is_tree(state):
                        trees.add(BlockPos(column.x, y, column.z))
                        if is_trunk(state):
                            trunks.add(BlockPos(column.x, y, column.z))
                continue
            if kind in (*SOLID, BlockClass.FLUID) and intrudes is not None and intrudes(row, cell, y):
                inside.add(BlockPos(column.x, y, column.z))
            if kind is BlockClass.GROUND and not found:
                height[row, cell] = y
                states[(row, cell)] = state
                found = True
            if found and kind in SOLID and height[row, cell] - y < SUBSURFACE:
                index = kept.get(state)
                if index is None:
                    index = kept[state] = len(kept)
                layers[height[row, cell] - y, row, cell] = index
        structures[row, cell] = found and highest_structure >= height[row, cell]
    return Survey(
        height, available, states, frozenset(standing), frozenset(inside),
        frozenset(trees), water, structures, layers, tuple(kept), top, frozenset(trunks),
    )


def load_terrain(source: Path | str | None, dimension: str | None = None) -> Terrain:
    if source is None:
        return EmptyTerrain()
    source = Path(source).resolve()
    if not source.exists():
        raise ValueError(f"the terraform world names {source}, which is not there")
    if source.is_dir():
        from .region import RegionTerrain

        return RegionTerrain(region_directory(source, dimension))
    if dimension:
        raise ValueError("dimension can only be selected for a world save")
    from ..Format.schematics import load_blocks

    return FieldTerrain(load_blocks(source))
