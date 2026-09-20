from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..Core.Field.block_field import BlockField
from ..Core.block.palette import BlockPalette
from ..Core.block.states import BlockState
from ..Core.coordinates import BlockPos
from .grid import Field, Grid
from .terrain import ABSENT


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
) -> Earthworks:
    fill = BlockField(palette)
    cut: set[BlockPos] = set()
    rounded = np.rint(designed).astype(np.int64)
    for row, cell, column in grid.columns():
        was = int(natural[row, cell])
        if was <= ABSENT:
            continue
        now = int(rounded[row, cell])
        if now > was:
            cap = _cap(strata, inherited, row, cell)
            for y in range(was + 1, now + 1):
                fill.set_block(BlockPos(column.x, y, column.z), cap.at(now - y))
        elif now < was:
            for y in range(now + 1, was + 1):
                cut.add(BlockPos(column.x, y, column.z))
    return Earthworks(fill, frozenset(cut))


def _cap(
    strata: Strata,
    inherited: Mapping[tuple[int, int], BlockState] | None,
    row: int,
    cell: int,
) -> Strata:
    if inherited is None:
        return strata
    found = inherited.get((row, cell))
    return strata if found is None else Strata(
        found, strata.subsurface, strata.filler, strata.depth
    )
