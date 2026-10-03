from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..Core.Field.block_field import BlockField
from ..Core.block.palette import BlockPalette
from ..Core.block.states import BlockState
from ..Core.coordinates import BlockPos

from .clearance import Clearance
from .grid import Field, Grid
from .classify import DEFAULT_TOPS, BlockNames, is_ground
from .subsurface import Layering
from .terrain import ABSENT, Terrain


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
    layering: Layering | None = None,
    water: Field | None = None,
    clearance: Clearance | None = None,
    tops: BlockNames = DEFAULT_TOPS,
    rock_surface: Field | None = None,
    subsurface: np.ndarray | None = None,
) -> Earthworks:
    # A fill is built the way the land round it lies (`layering`, else
    # `strata`), its `tops` never under water; a cut leaves the save's own
    # ground in its face and takes the land's surface on its new top.
    # `subsurface` (solid blocks by depth, -1 for none) keeps that top off
    # a saved cave or overhang the cut opens into.
    fill = BlockField(palette)
    cut: set[BlockPos] = set()
    rounded = np.floor(designed + 0.5).astype(np.int64)
    for row, cell, column in grid.columns():
        was = int(natural[row, cell])
        if was <= ABSENT:
            continue
        now = int(rounded[row, cell])
        waterline = ABSENT if water is None else int(water[row, cell])
        if now > was:
            for y in range(was + 1, now + 1):
                # A surface can rise above a buried bore. Keep its roof
                # cover, but do not rebuild the empty space below it.
                if clearance is not None and clearance.holds(row, cell, y):
                    continue
                under = waterline > was and y <= waterline
                fill.set_block(
                    BlockPos(column.x, y, column.z),
                    _material(strata, layering, tops, row, cell, now - y, under),
                )
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
            depth = was - now
            hollow = (subsurface is not None and depth < subsurface.shape[0]
                      and subsurface[depth, row, cell] < 0)
            if (waterline <= was and (clearance is None or not clearance.holds(row, cell, now))
                    and (rock_surface is None or now > rock_surface[row, cell]) and not hollow):
                # A cut into rock exposes the saved rock. Painting every
                # ledge with grass creates artificial diagonal turf stripes.
                fill.set_block(
                    BlockPos(column.x, now, column.z),
                    _material(strata, layering, tops, row, cell, 0, False),
                )
    return Earthworks(fill, frozenset(cut))


def finish_surfaces(terrain: Terrain, grid: Grid, natural: Field, designed: Field,
                    eligible: Field, fill: BlockField, cut: frozenset[BlockPos],
                    strata: Strata, layering: Layering | None, water: Field,
                    clearance: Clearance, tops: BlockNames = DEFAULT_TOPS,
                    skin: Field | None = None, rock_surface: Field | None = None) -> int:
    """Re-establish local soil on broad gentle cut surfaces, after voxel repair.

    This is replacement of surviving ground, never fill across a cave or
    headroom. Narrow rock ledges and steep faces keep their saved material.
    The neighbouring land supplies the palette and subsurface layering.
    """
    from scipy.ndimage import binary_opening, label
    from . import kernels

    levels = np.floor(designed + .5)
    known = natural > ABSENT
    smooth = kernels.normalized(levels, known, 2.)
    dz = np.gradient(smooth, axis=0) if smooth.shape[0] > 1 else np.zeros_like(smooth)
    dx = np.gradient(smooth, axis=1) if smooth.shape[1] > 1 else np.zeros_like(smooth)
    local_drop = np.zeros(levels.shape)
    for axis in (0, 1):
        a, b = [slice(None)] * 2, [slice(None)] * 2
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        step = np.abs(levels[a] - levels[b])
        local_drop[a] = np.maximum(local_drop[a], step)
        local_drop[b] = np.maximum(local_drop[b], step)
    gentle = known & (np.hypot(dx, dz) <= 1.25) & (local_drop <= 2)
    # A three-block-wide shelf can hold soil. A one-block stair tread
    # cannot: coating those is what produces diagonal grass stripes.
    patches = binary_opening(gentle, structure=np.ones((3, 3), bool))
    components, count = label(patches)
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    patches &= sizes[components] >= 9
    candidates = eligible & patches
    depths = np.full(levels.shape, strata.depth + 1)
    if skin is not None:
        # Sculpted soil takes the land's skin on every tread. A cut into rock
        # stands as bare stone, as a natural cliff does: only a ledge two
        # blocks wide takes the top, with no soil under it.
        candidates |= skin
        if rock_surface is not None:
            level = np.zeros(levels.shape, dtype=np.int8)
            for axis in (0, 1):
                a, b = [slice(None)] * 2, [slice(None)] * 2
                a[axis], b[axis] = slice(None, -1), slice(1, None)
                a, b = tuple(a), tuple(b)
                even = levels[a] == levels[b]
                level[a] += even
                level[b] += even
            bare = skin & (levels <= rock_surface)
            depths = np.where(bare, np.where(level >= 2, 1, 0), depths)
    candidates &= (levels < natural - .5) & (water <= levels) & (depths > 0)
    painted = 0
    for r, c in zip(*np.nonzero(candidates)):
        column = grid.column(r, c)
        original = {y: state for y, state in terrain.column(column.x, column.z)}
        now = int(levels[r, c])

        def surviving(y):
            p = BlockPos(column.x, y, column.z)
            replacement = fill.get_block(p)
            state = replacement if replacement is not None else None if p in cut else original.get(y)
            return state is not None and is_ground(state)

        # Heightmap tops can sit over empty saved caves. Never manufacture
        # a floating skin, nor paint down into a surviving bore.
        if not (surviving(now) and surviving(now - 1)):
            continue
        top = _material(strata, layering, tops, r, c, 0, False)
        if not tops(top.name):
            continue  # Native bare rock/sand/gravel remains that environment.
        for depth in range(depths[r, c]):
            y = now - depth
            if not surviving(y) or clearance.holds(r, c, y):
                break
            fill.set_block(BlockPos(column.x, y, column.z),
                           _material(strata, layering, tops, r, c, depth, False))
            painted += 1
    return painted


def _material(
    strata: Strata, layering: Layering | None, tops: BlockNames,
    row: int, cell: int, depth: int, under: bool,
) -> BlockState:
    found = None if layering is None else layering.at(row, cell, depth)
    state = strata.at(depth) if found is None else found
    if under and tops(state.name):
        # Under water the land's top gives way to the first thing beneath it
        # that is not a top.
        state = strata.subsurface
        if layering is not None:
            for below in range(depth + 1, layering.layers.shape[0]):
                found = layering.at(row, cell, below)
                if found is not None and not tops(found.name):
                    state = found
                    break
    return state
