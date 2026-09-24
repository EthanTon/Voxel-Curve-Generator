from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..Core.Field.block_field import BlockField
from ..Core.block.palette import BlockPalette
from ..Core.block.states import BlockState
from ..Core.coordinates import BlockPos
from .clearance import Clearance
from .grid import Field, Grid
from .terrain import ABSENT


# The most a cut's new top may stand over its lowest neighbour and still be
# soiled rather than left as a bare face.
SOILED_RISE = 2


@dataclass(frozen=True, slots=True)
class Strata:
    surface: BlockState
    subsurface: BlockState
    filler: BlockState
    depth: int = 3

    def at(self, below: int) -> BlockState:
        if below <= 0:
            return self.surface
        if below <= self.depth:
            return self.subsurface
        return self.filler


@dataclass(frozen=True, slots=True)
class Earthworks:
    fill: BlockField
    cut: frozenset[BlockPos]

    @property
    def changed(self) -> int:
        return len(self.fill) + len(self.cut)


def build(
    grid: Grid,
    natural: Field,
    designed: Field,
    strata: Strata,
    palette: BlockPalette,
    inherited: Mapping[tuple[int, int], BlockState] | None = None,
    water: Field | None = None,
    clearance: Clearance | None = None,
) -> Earthworks:
    fill = BlockField(palette)
    cut: set[BlockPos] = set()
    rounded = np.rint(designed).astype(np.int64)
    rise = rounded - _lowest_neighbour(rounded, natural > ABSENT)
    for row, cell, column in grid.columns():
        was = int(natural[row, cell])
        if was <= ABSENT:
            continue
        now = int(rounded[row, cell])
        waterline = ABSENT if water is None else int(water[row, cell])
        if now > was:
            cap = _cap(strata, inherited, row, cell)
            for y in range(was + 1, now + 1):
                if waterline > was and y <= waterline:
                    # A submerged bed is not the new dry surface.
                    state = (
                        cap.subsurface if cap.surface.name in (
                            "minecraft:grass_block", "minecraft:podzol",
                            "minecraft:mycelium", "minecraft:moss_block",
                        ) else cap.surface
                    )
                elif waterline > was:
                    state = strata.at(now - y)
                else:
                    state = cap.at(now - y)
                fill.set_block(BlockPos(column.x, y, column.z), state)
        elif now < was:
            for y in range(now + 1, was + 1):
                position = BlockPos(column.x, y, column.z)
                if (
                    waterline > was and y <= waterline
                    and (clearance is None or not clearance.holds(row, cell, y))
                ):
                    fill.set_block(position, BlockState.create("minecraft:water"))
                else:
                    cut.add(position)
            # A shallow cut, or a deep one sloping gently, is soiled again:
            # its surface on top and its subsurface in the riser. A steep
            # face of a deep cut shows the filler.
            if waterline <= was and (clearance is None or not clearance.holds(row, cell, now)):
                cap = _cap(strata, inherited, row, cell)
                step = int(rise[row, cell])
                if was - now <= cap.depth:
                    fill.set_block(BlockPos(column.x, now, column.z), cap.surface)
                elif step <= SOILED_RISE:
                    fill.set_block(BlockPos(column.x, now, column.z), cap.surface)
                    for y in range(now - step + 1, now):
                        fill.set_block(BlockPos(column.x, y, column.z), cap.subsurface)
                else:
                    fill.set_block(BlockPos(column.x, now, column.z), cap.filler)
    return Earthworks(fill, frozenset(cut))


def _lowest_neighbour(rounded: np.ndarray, known: Field) -> np.ndarray:
    levels = np.pad(rounded, 1, mode="edge")
    present = np.pad(known, 1, mode="constant")
    lowest = rounded.copy()
    for rows, cells in (
        (slice(None, -2), slice(1, -1)), (slice(2, None), slice(1, -1)),
        (slice(1, -1), slice(None, -2)), (slice(1, -1), slice(2, None)),
    ):
        lowest = np.minimum(lowest, np.where(present[rows, cells], levels[rows, cells], rounded))
    return lowest


def _cap(
    strata: Strata,
    inherited: Mapping[tuple[int, int], BlockState] | None,
    row: int,
    cell: int,
) -> Strata:
    if inherited is None:
        return strata
    found = inherited.get((row, cell))
    if found is None:
        return strata
    # Fill takes the local column's strata, not just its top block.
    if found.name in ("minecraft:sand", "minecraft:red_sand"):
        sandstone = "minecraft:sandstone" if found.name == "minecraft:sand" else "minecraft:red_sandstone"
        return Strata(found, found, BlockState.create(sandstone), strata.depth)
    if found.name in ("minecraft:gravel", "minecraft:clay", "minecraft:mud"):
        return Strata(found, found, strata.filler, strata.depth)
    if found.name in (
        "minecraft:stone", "minecraft:granite", "minecraft:diorite",
        "minecraft:andesite", "minecraft:deepslate", "minecraft:tuff",
    ):
        return Strata(found, found, found, strata.depth)
    return Strata(found, strata.subsurface, strata.filler, strata.depth)
