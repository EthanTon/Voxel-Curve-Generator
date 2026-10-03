from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..Core.block.states import BlockState
from .classify import DEFAULT_LOOSE, BlockNames
from .grid import Field

# The land's layering is the commonest block at each depth under the
# columns within REACH of a column, tallied a tile of TILE by TILE columns
# at a time (the reach rounded up to whole tiles).
TILE = 4
REACH = 12
# The rock rule's defaults: a face is rock where this share of what stands
# in it is, and the rock under a column starts where this many blocks down
# from it are rock, so a thin band over a pocket of soil is not.
STONE_SHARE = 0.66
ROCK_BODY = 6


@dataclass(frozen=True, slots=True)
class RockRule:
    # What is rock: anything solid under the soil that is not `loose`,
    # starting where the `body` blocks down from it are at least `share`
    # rock; a face is rock where `share` of what stands in it is.
    loose: BlockNames = DEFAULT_LOOSE
    body: int = ROCK_BODY
    share: float = STONE_SHARE


DEFAULT_ROCK = RockRule()


def stony(names, rule: RockRule = DEFAULT_ROCK) -> bool:
    # Most of what stands in a face is rock: a pocket of soil, an ore or a
    # cave does not make it soil. `names` holds None where nothing is.
    solid = [name for name in names if name is not None]
    return bool(solid) and sum(not rule.loose(name) for name in solid) >= rule.share * len(solid)


def rock_depth(names, rule: RockRule = DEFAULT_ROCK) -> int | None:
    # How far down a column its rock starts, from its names from the
    # surface down (None where nothing solid is), as rock_tops reckons it.
    hard = [None if name is None else not rule.loose(name) for name in names]
    for depth, here in enumerate(hard):
        if not here:
            continue
        window = [each for each in hard[depth:depth + rule.body] if each is not None]
        if sum(window) >= rule.share * len(window):
            return depth
    return None


@dataclass(frozen=True, slots=True)
class Layering:
    # The land's own layering round each column: its block at each depth
    # under the surface, the deepest carrying on below.
    layers: np.ndarray
    states: tuple[BlockState, ...]

    def at(self, row: int, cell: int, depth: int) -> BlockState | None:
        index = self.layers[min(depth, self.layers.shape[0] - 1), row, cell]
        return None if index < 0 else self.states[index]


@dataclass(frozen=True, slots=True)
class Subsurface:
    # What the save has under each column's surface: `layers` indexes
    # `states` a block of depth at a time, -1 where nothing solid is.
    layers: np.ndarray
    states: tuple[BlockState, ...]
    height: Field
    rule: RockRule = DEFAULT_ROCK

    def rock_tops(self) -> Field:
        # The top of the rock under each column's soil: the first depth from
        # which the rule's body of blocks is rock enough, -inf where none is
        # within the kept layers.
        depth, body = self.layers.shape[0], self.rule.body
        hard = np.append([not self.rule.loose(state.name) for state in self.states], False)
        rocky = hard[self.layers].astype(np.int32)
        solid = (self.layers >= 0).astype(np.int32)
        # Below the kept layers the deepest carries on.
        rocky = np.concatenate([rocky, np.repeat(rocky[-1:], body, axis=0)])
        solid = np.concatenate([solid, np.repeat(solid[-1:], body, axis=0)])
        windows_rock = np.cumsum(np.concatenate([np.zeros_like(rocky[:1]), rocky]), axis=0)
        windows_solid = np.cumsum(np.concatenate([np.zeros_like(solid[:1]), solid]), axis=0)
        in_rock = windows_rock[body:body + depth] - windows_rock[:depth]
        in_solid = windows_solid[body:body + depth] - windows_solid[:depth]
        body = (in_solid > 0) & (in_rock >= self.rule.share * in_solid) & (rocky[:depth] > 0)
        return np.where(body.any(axis=0), self.height - body.argmax(axis=0), -np.inf)

    def face(self, rows, cells, lows, highs) -> np.ndarray:
        # Whether each column's own blocks from `lows` to `highs` stand in
        # rock; below the kept layers the deepest is taken to carry on.
        out = np.zeros(len(rows), dtype=bool)
        deepest = self.layers.shape[0] - 1
        for index, (row, cell, low, high) in enumerate(zip(rows, cells, lows, highs)):
            if high < low:
                out[index] = True
                continue
            top = int(self.height[row, cell])
            found = [self.layers[min(top - y, deepest), row, cell] for y in range(int(low), int(high) + 1) if y <= top]
            out[index] = stony((None if each < 0 else self.states[each].name for each in found), self.rule)
        return out

    def layering(self, known: Field, tile: int = TILE, reach: int = REACH) -> Layering:
        # The commonest block at each depth under the columns round each
        # one, so a fill is built the way the land beside it lies. A block
        # at a time, so a small tile never holds every block's tally at once.
        depth, rows, cells = self.layers.shape
        out = np.full(self.layers.shape, -1, dtype=np.int16)
        if not self.states:
            return Layering(out, self.states)
        tile_rows, tile_cells = -(-rows // tile), -(-cells // tile)
        tiles = -(-reach // tile)
        across = (np.arange(rows) // tile)[:, None]
        along = (np.arange(cells) // tile)[None, :]
        index = across * tile_cells + along
        for level in range(depth):
            found = self.layers[level]
            held = (found >= 0) & known
            kinds, places = found[held], index[held]
            order = np.argsort(kinds, kind="stable")
            kinds, places = kinds[order], places[order]
            breaks = np.flatnonzero(np.diff(kinds)) + 1
            most = np.zeros((tile_rows, tile_cells))
            commonest = np.full((tile_rows, tile_cells), -1, dtype=np.int16)
            for kind, where in zip(np.split(kinds, breaks), np.split(places, breaks)):
                if not len(kind):
                    continue
                tally = np.bincount(where, minlength=tile_rows * tile_cells).reshape(tile_rows, tile_cells)
                around = _box(tally.astype(np.float64), tiles)
                more = around > most
                most = np.where(more, around, most)
                commonest = np.where(more, kind[0], commonest)
            out[level] = commonest[across, along]
        for level in range(1, depth):
            out[level] = np.where(out[level] < 0, out[level - 1], out[level])
        return Layering(out, self.states)


def surface_blocks(subsurface: Subsurface, sample: Field, share: float = 0.9) -> tuple[str, ...]:
    """Blocks the land only ever has on top: at least `share` of where each
    lies in the sampled columns is their surface, on a common enough part of it."""
    if not subsurface.states or not sample.any():
        return ()
    layers = subsurface.layers[:, sample]
    kinds = len(subsurface.states)
    on_top = np.bincount(layers[0][layers[0] >= 0], minlength=kinds)
    anywhere = np.bincount(layers[layers >= 0], minlength=kinds)
    least = max(16, int(0.005 * sample.sum()))
    return tuple(subsurface.states[kind].name for kind in range(kinds)
                 if on_top[kind] >= least and on_top[kind] >= share * anywhere[kind])


def _box(tally: np.ndarray, reach: int) -> np.ndarray:
    # Sums over the (2 reach + 1)^2 tiles (or cells) round each one.
    padded = np.pad(tally, ((reach + 1, reach), (reach + 1, reach)))
    summed = padded.cumsum(axis=0).cumsum(axis=1)
    width = 2 * reach + 1
    return (summed[width:, width:] - summed[:-width, width:]
            - summed[width:, :-width] + summed[:-width, :-width])
