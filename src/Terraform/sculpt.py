"""Shape the ground the works change the way the land itself lies.

Cuts climb from the contacts the build lowers as steeply as their material
stands; banks fall from the contacts it raises as loose fill settles; both
come down to the ground the works hold no steeper than they could stand.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

from .distance import _shift_into, nearest
from .grid import Field, Grid
from .kernels import square_closing, square_opening
from .noise import fbm
from .slopes import STEPS, _gates

# How far ledges may bite below a shaped face, and how far from a fixed
# contact they may not reach below it.
LEDGE_BITE = 4.0
CONTACT_REACH = 4.0
DEAD = 0.5
# A drop in the land this deep is a natural cliff a bank does not spill over;
# one deeper than RIM the bank tapers out at, rather than making it taller.
CLIFF = 2.0
RIM = 4.0
# Bounds are relaxed a tile at a time, each with this much land round it, so
# a winding route never relaxes its whole bounding box.
TILE = 160
HALO = 64


@dataclass(frozen=True, slots=True)
class Rates:
    # Blocks of rise per column cut through rock and through the soil over
    # it, and of fall per column down a loose bank.
    rock: Field
    soil: Field
    bank: Field


def rates_for(grid: Grid, seed: int, rock: tuple[float, float], soil: tuple[float, float],
              bank: tuple[float, float], steep: tuple[Field, Field] | None = None,
              wavelength: float = 18.0) -> Rates:
    """Rates wandering in broad patches with a finer grain over them.

    Treads widen and narrow along a slope instead of running as rings, and a
    rock face steps unevenly. Given `steep` (how steep the land's own
    hillsides run round each column, commonly and at its steepest), soil
    grades follow the first and rock faces the second within their ranges.
    """
    def wandering(offset: int, length: float, octaves: int) -> Field:
        return np.clip((fbm(grid, seed + offset, length, octaves, length / 3) + 1.0) / 2.0, 0.0, 1.0)

    def mixed(offset: int, fine: float) -> Field:
        return 0.6 * wandering(offset, wavelength * 1.5, 3) + 0.4 * wandering(offset + 7, fine, 2)

    def across(span: tuple[float, float], share: Field) -> Field:
        return span[0] + (span[1] - span[0]) * share

    def around(span: tuple[float, float], target: Field, share: Field) -> Field:
        return np.clip(target * (0.6 + 0.8 * share), span[0], span[1])

    rocky, soily, banked = mixed(31, 3.0), mixed(57, wavelength / 3), mixed(83, wavelength / 3)
    if steep is None:
        return Rates(across(rock, rocky), across(soil, soily), across(bank, banked))
    return Rates(around(rock, steep[1], rocky), around(soil, steep[0], soily), across(bank, banked))


def native_steepness(natural: Field, land: Field, quantiles: tuple[float, ...],
                     tile: int = 16, reach: int = 3, least: float = 0.25,
                     origin: tuple[int, int] = (0, 0)) -> tuple[Field, ...]:
    """The given quantiles of the land's hillside slopes round each column.

    Slopes are blocks a column over four columns, on land at least `least`
    steep, tallied `tile` columns square at a time over `reach` tiles round
    each tile. Tiles sit on world coordinates (`origin` is the field's
    first column, x then z), so the same land reads the same in any field;
    land with too few slopes round it reads as gentle.
    """
    level = np.where(land, natural, np.nan)
    along = np.full(level.shape, np.nan)
    across = np.full(level.shape, np.nan)
    along[:, 2:-2] = (level[:, 4:] - level[:, :-4]) / 4.0
    across[2:-2] = (level[4:] - level[:-4]) / 4.0
    with np.errstate(invalid="ignore"):
        slope = np.fmax(np.abs(along), np.abs(across))
    sampled = land & np.isfinite(slope) & (slope >= least)
    bins = np.clip(np.rint(np.where(sampled, slope, 0.0) * 4).astype(np.int64), 0, 63)
    rows, cells = natural.shape
    row_shift, cell_shift = origin[1] % tile, origin[0] % tile
    row_tile = (np.arange(rows) + row_shift) // tile
    cell_tile = (np.arange(cells) + cell_shift) // tile
    tile_rows, tile_cells = int(row_tile[-1]) + 1, int(cell_tile[-1]) + 1
    index = row_tile[:, None] * tile_cells + cell_tile[None, :]
    counts = np.zeros((64, tile_rows, tile_cells))
    for value in np.unique(bins[sampled]):
        chosen = sampled & (bins == value)
        tally = np.bincount(index[chosen], minlength=tile_rows * tile_cells).reshape(tile_rows, tile_cells)
        counts[value] = _box_sum(tally.astype(np.float64), reach)
    total = counts.sum(axis=0)
    running = np.cumsum(counts, axis=0)
    found = []
    for share in quantiles:
        reached = running >= share * np.maximum(total, 1)[None]
        level_bin = np.argmax(reached, axis=0) / 4.0
        level_bin = np.where(total >= 32, level_bin, least)
        found.append(level_bin[row_tile[:, None], cell_tile[None, :]])
    return tuple(found)


def _box_sum(tally: Field, reach: int) -> Field:
    padded = np.pad(tally, ((reach + 1, reach), (reach + 1, reach)))
    summed = padded.cumsum(axis=0).cumsum(axis=1)
    width = 2 * reach + 1
    return (summed[width:, width:] - summed[:-width, width:]
            - summed[width:, :-width] + summed[:-width, :-width])


def fall(height: Field, length: float, rock_top: Field, rock: Field, soil: Field) -> Field:
    # The lowest ground `length` columns on from `height`: down through its
    # soil at the soil rate, then down the rock under it at the rock rate.
    with np.errstate(invalid="ignore"):
        in_soil = height > rock_top
        to_rock = np.divide(height - rock_top, soil, out=np.zeros_like(height), where=in_soil)
        under_soil = rock_top - (length - to_rock) * rock
    return np.where(in_soil, np.where(length <= to_rock, height - length * soil, under_soil),
                    height - length * rock)


def bank_up(height: Field, length: float, bank: Field) -> Field:
    return height + length * bank


def bank_down(height: Field, length: float, bank: Field) -> Field:
    return height - length * bank


def spread(anchors: Field, heights: Field, domain: Field, climb, fields=(), upward: bool = True,
           bound: Field | None = None, carry: Field | None = None, limit: Field | None = None,
           cliff: float | None = None) -> Field:
    """The tightest bound over `domain` that `climb` carries out from the anchors.

    Upward it is the lowest of the bounds climbing from every anchor, downward
    the highest of those falling from them. `climb(height, length, *fields)`
    is the bound `length` columns on. A `bound` already standing on a domain
    column is carried out from it too, and none passes the `limit` (a floor
    upward, a ceiling downward). With `carry` a bound passes on only from
    where it is tighter than that ground -- a cut face stops where it comes
    out of the hill, a bank where it meets the land -- and, given `cliff`,
    never down a drop in it deeper than that.
    """
    absent = np.inf if upward else -np.inf
    seeded = np.where(anchors, heights, absent).astype(np.float64)
    reach = domain & ~anchors
    if bound is not None:
        seeded = np.where(reach, (np.minimum if upward else np.maximum)(seeded, bound), seeded)
    if limit is not None:
        seeded = np.where(reach & np.isfinite(seeded),
                          (np.maximum if upward else np.minimum)(seeded, limit), seeded)
    found = seeded.copy()
    sources = anchors | (reach & np.isfinite(seeded))
    rows, cells = found.shape
    for r0 in range(0, rows, TILE):
        for c0 in range(0, cells, TILE):
            inner = np.s_[r0:r0 + TILE, c0:c0 + TILE]
            if not reach[inner].any():
                continue
            top, left = max(r0 - HALO, 0), max(c0 - HALO, 0)
            outer = np.s_[top:r0 + TILE + HALO, left:c0 + TILE + HALO]
            if not sources[outer].any():
                continue
            local = _relax(anchors[outer], seeded[outer], reach[outer], climb,
                           [each[outer] for each in fields], upward,
                           None if carry is None else carry[outer],
                           None if limit is None else limit[outer], cliff)
            held = local[r0 - top:r0 - top + TILE, c0 - left:c0 - left + TILE]
            found[inner] = np.where(reach[inner], held, found[inner])
    return found


def _relax(anchors: Field, found: Field, reach: Field, climb, fields, upward: bool,
           carry: Field | None, limit: Field | None, cliff: float | None) -> Field:
    absent = np.inf if upward else -np.inf
    found = found.copy()
    gates = _gates(reach | anchors)
    moved = np.empty_like(found)
    passing = np.empty(found.shape, dtype=bool)
    ledges = []
    if carry is not None and cliff is not None:
        # Steps along which ground no deeper than `cliff` below its neighbour.
        behind = np.empty_like(found)
        for rows, cells, *_ in STEPS:
            _shift_into(carry, rows, cells, -np.inf, behind)
            ledges.append(behind - carry <= cliff)
    for _ in range(2 * HALO + TILE):
        settled = True
        if carry is not None:
            active = anchors | ((found < carry) if upward else (found > carry))
        for index, ((rows, cells, length, _, _), gate) in enumerate(zip(STEPS, gates)):
            _shift_into(found, rows, cells, absent, moved)
            candidate = climb(moved, length, *fields)
            if limit is not None:
                candidate = (np.maximum if upward else np.minimum)(candidate, limit)
            better = reach & ((candidate < found) if upward else (candidate > found))
            if gate is not None:
                better &= gate
            if carry is not None:
                _shift_into(active, rows, cells, False, passing)
                better &= passing
                if ledges:
                    better &= ledges[index]
            if better.any():
                np.copyto(found, candidate, where=better)
                settled = False
        if settled:
            break
    return found


@dataclass(frozen=True, slots=True)
class Design:
    heights: Field
    # The lowest a cut may go and stay standing against the held ground.
    cut_floor: Field


def design(base: Field, fixed: Field, heights: Field, free: Field, rock_top: Field,
           rates: Rates, cap: Field | None = None, floor: Field | None = None,
           fill_only: Field | None = None, hold: Field | None = None) -> Design:
    """Shape the `free` columns between the `fixed` ones.

    Fixed columns standing below `base` are cut contacts, those above it fill
    contacts, the rest held ground. A contact cut into the rock under the
    land stands as a cliff, rising through rock and its soil cap at the rock
    rate; a shallower one grades back through the soil. Fill contacts fall
    as loose banks, tapering out at the top of a natural drop deeper than
    RIM. Nothing changes further than lets it meet held ground, and
    `fill_only` columns are never cut. A capped column holds its neighbours
    down at `hold` (its cap unless given).
    """
    lowered = fixed & (heights < base - DEAD)
    raised = fixed & (heights > base + DEAD)
    held = fixed & ~lowered & ~raised
    cap = np.full(base.shape, np.inf) if cap is None else cap
    hold = cap if hold is None else hold
    floor = np.full(base.shape, -np.inf) if floor is None else floor
    rocky = lowered & (heights < rock_top - DEAD)

    # The land's surface with its single steps eased, so measuring from it
    # never stacks a native step on a new slope.
    surface = _eased(base)

    def bounded(anchors, climb, rate, upward, bound, limit, domain, cliff=None):
        # Measured from the land's own surface as well, a face or bank on
        # rising or falling ground runs out within its depth's reach instead
        # of shaving the whole hillside: the change is the smaller of the two.
        flat = np.zeros(base.shape)
        absolute = spread(anchors, heights, domain, climb, (rate,), upward, bound, base, limit, cliff)
        relative = surface + spread(anchors, heights - surface, domain, climb, (rate,), upward,
                                    bound - surface, flat, limit - surface)
        return np.maximum(absolute, relative) if upward else np.minimum(absolute, relative)

    # A cap or floor on a free column holds its neighbours as a contact would;
    # a cut never reaches into, or past, ground that is not to be cut.
    cuttable = free if fill_only is None else free & ~fill_only
    cut = np.minimum(bounded(rocky, bank_up, rates.rock, True, hold, floor, cuttable),
                     bounded(lowered & ~rocky, bank_up, rates.soil, True, hold, floor, cuttable))
    cut = np.where(cuttable, cut, np.inf)
    # A bank stops at the top of a natural cliff rather than filling what lies below it.
    bank = bounded(raised, bank_down, rates.bank, False, floor, cap, free, CLIFF)
    # Held ground bounds only the columns that move, through those columns;
    # the rim of a deep natural drop the bank reaches holds it as held ground does.
    moving = free & ((cut < base - DEAD) | (bank > base + DEAD))
    rims = free & (bank > base + DEAD) & cliff_tops(base, RIM)
    material = (rock_top, rates.rock, rates.soil)
    # Ground that is never cut holds a cut beside it as held ground does.
    holding = held if fill_only is None else held | fill_only
    cut_floor = spread(holding, np.where(held, heights, base), moving, fall, material, False, floor,
                       limit=cap)
    bank_top = spread(held | rims, np.where(rims, base, heights), moving, bank_up, (rates.bank,), True,
                      hold, limit=floor)
    shaped = np.minimum(base, np.maximum(cut, cut_floor))
    shaped = np.maximum(shaped, np.minimum(bank, bank_top))
    shaped = np.maximum(np.minimum(shaped, cap), floor)
    return Design(np.where(free, shaped, np.where(fixed, heights, base)),
                  np.where(free, cut_floor, -np.inf))


def cliff_tops(ground: Field, drop: float) -> Field:
    """Columns standing more than `drop` above a side neighbour."""
    tops = np.zeros(ground.shape, dtype=bool)
    beside = np.empty_like(ground)
    for rows, cells in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        _shift_into(ground, rows, cells, np.inf, beside)
        tops |= ground - beside > drop
    return tops


def _eased(ground: Field, reach: int = 2) -> Field:
    # A box mean of the ground over (2 reach + 1)^2 columns.
    width = 2 * reach + 1
    padded = np.pad(ground, reach, mode="edge")
    summed = np.pad(padded.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    return (summed[width:, width:] - summed[:-width, width:]
            - summed[width:, :-width] + summed[:-width, :-width]) / (width * width)


def texture(shaped: Field, base: Field, fixed: Field, free: Field, heights: Field,
            rock_top: Field, cut_floor: Field, grid: Grid, seed: int,
            ledges: tuple[int, int], rock_drop: float, fillable: Field | None = None,
            cap: Field | None = None, land: Field | None = None) -> Field:
    """Break rock cuts into ledges and level the narrow ridges and grooves left.

    Ridges are levelled wherever the ground moved from `land` (the saved
    ground, where `base` already differs from it). Bank toes are then built
    out under `cap`, and narrow grooves and pits in `fillable` ground
    filled, never cut.
    """
    cut = free & (shaped < base - DEAD)
    textured = strata(shaped, rock_top, cut, grid, seed, ledges)
    # A ledge bites no deeper than LEDGE_BITE, a nearby contact or what the
    # held ground needs, and never builds a cut column up past the land.
    contact = nearest(fixed & binary_dilation(free), (heights,), CONTACT_REACH)
    lowest = np.where(contact.distance < CONTACT_REACH, np.minimum(contact[0], shaped), -np.inf)
    guard = np.minimum(np.maximum(shaped - LEDGE_BITE, np.maximum(lowest, cut_floor)), base)
    textured = np.where(cut, np.maximum(textured, guard), textured)
    textured = _detail(textured, base, free & (textured > rock_top))
    # Narrow remnants left standing at the rim of a change go with it.
    moved = np.abs(textured - base) >= DEAD
    if land is not None:
        moved |= np.abs(base - land) >= DEAD
    native_faces = cliff_tops(base, CLIFF) | cliff_tops(-base, CLIFF)
    held_faces = free & native_faces & (shaped >= base - DEAD)
    changed = free & binary_dilation(moved, iterations=2) & ~held_faces
    textured = level_fins(textured, changed)
    textured = cap_risers(textured, shaped, cut, rock_drop)
    high = np.full(base.shape, np.inf) if cap is None else cap
    textured = settle_soil(textured, base, free & ~cliff_tops(base, RIM), rock_top, high)
    if fillable is not None and fillable.any():
        levels = np.floor(textured + 0.5)
        textured = np.where(fillable, np.maximum(textured, np.minimum(square_closing(levels, 3), high)),
                            textured)
    # Closing construction grooves must not lift a saved ravine ledge or
    # its rim. Actual excavations still receive their rock-face finishing.
    textured = np.where(held_faces, np.minimum(textured, shaped), textured)
    return textured


def settle_soil(heights: Field, base: Field, free: Field, rock_top: Field, high: Field,
                rounds: int = 48) -> Field:
    """Build out a bank's toe where it leaves a step of two or more beside soil.

    The lower soil-topped column beside a raised one rises to a block under
    it, up to `high`, so the bank runs out instead of standing as a wall.
    Nothing spreads past a step the land already had; rock keeps its faces.
    """
    levels = np.floor(heights + 0.5)
    land = np.floor(base + 0.5)
    beside, ground = np.empty(levels.shape), np.empty(levels.shape)
    near_soil, near_raised = np.empty(levels.shape, dtype=bool), np.empty(levels.shape, dtype=bool)
    for _ in range(rounds):
        soil = levels > rock_top
        raised = levels > land
        up = np.full(levels.shape, -np.inf)
        for rows, cells in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            _shift_into(levels, rows, cells, np.nan, beside)
            _shift_into(land, rows, cells, np.nan, ground)
            _shift_into(soil, rows, cells, False, near_soil)
            _shift_into(raised, rows, cells, False, near_raised)
            with np.errstate(invalid="ignore"):
                toe = (soil & near_soil & (raised | near_raised) & (beside - levels >= 2)
                       & (np.abs(ground - land) < 2))
            up = np.where(toe, np.maximum(up, beside - 1), up)
        target = np.minimum(up, np.floor(high))
        rising = free & soil & (target > levels)
        if not rising.any():
            break
        levels = np.where(rising, target, levels)
    return np.where(free & (levels != np.floor(heights + 0.5)), levels, heights)


def smooth_banks(heights: Field, base: Field, free: Field, radius: float = 2.0,
                 passes: int = 3, most: float = 1.0, max_drop: int = 1,
                 *, stations: Field | None = None) -> Field:
    """Fair existing fill contours without shrinking their broad shape.

    Alternating diffusion and a smaller reverse step removes terrace-scale
    ripples while limiting broad mound shrinkage. The footprint and fixed
    contacts stay put; cuts and cliff faces never vote.
    A station field directs the fairing along the route, with only light
    diffusion across the bank so its crossfall is retained.
    The displacement budget is measured from the input over all rounds.
    """
    from .layers import smootherstep
    from .smoothing import guard_steps

    if radius <= 0 or passes <= 0 or most <= 0:
        return heights
    cliffs = (cliff_tops(base, CLIFF) | cliff_tops(-base, CLIFF)
              | cliff_tops(heights, max_drop + .5) | cliff_tops(-heights, max_drop + .5))
    domain = (heights > base + DEAD) & ~cliffs
    region = free & domain
    if not region.any():
        return heights
    # Pin the first column beside structural contacts and cliffs. Measuring
    # from those contacts, rather than the edge of the fill, lets narrow
    # banks and their toes smooth as well. Keep the feather local even when
    # the requested smoothing radius grows. Padding holds the survey edge.
    distance = distance_transform_edt(np.pad(free & ~cliffs, 1))[1:-1, 1:-1]
    weight = np.where(region, smootherstep((distance - 1) / min(radius, 2.0)), 0.0)
    if not (weight > 0).any():
        return heights
    direction = None
    if stations is not None:
        # Ease blockwise station jumps before deriving the route's local
        # direction. This follows curves without averaging opposite banks.
        along = gaussian_filter(stations.astype(float), 2.0, mode="nearest")
        gy = np.gradient(along, axis=0) if along.shape[0] > 1 else np.zeros_like(along)
        gx = np.gradient(along, axis=1) if along.shape[1] > 1 else np.zeros_like(along)
        length = np.hypot(gy, gx)
        directed = length > 1e-6
        direction = (np.divide(gy, length, out=np.zeros_like(gy), where=directed),
                     np.divide(gx, length, out=np.zeros_like(gx), where=directed), directed)
    edges = []
    share = np.ones(heights.shape)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        first = (slice(None, -dr) if dr else slice(None),
                 slice(None, -dc) if dc > 0 else slice(-dc, None) if dc else slice(None))
        second = (slice(dr, None) if dr else slice(None),
                  slice(dc, None) if dc > 0 else slice(None, dc) if dc else slice(None))
        valid = domain[first] & domain[second]
        if dr and dc:
            # Diagonals cannot jump between banks across a masked corner.
            valid &= domain[first[0], second[1]] & domain[second[0], first[1]]
        affinity = valid.astype(float) / (2 if dr and dc else 1)
        if direction is not None:
            gy, gx, directed = direction
            projection = ((gy[first] + gy[second]) * dr
                          + (gx[first] + gx[second]) * dc) / (2 * np.hypot(dr, dc))
            affinity *= np.where(directed[first] & directed[second],
                                 .05 + .95 * projection ** 2, 1.0)
        edges.append((first, second, affinity))
        share[first] += affinity
        share[second] += affinity

    def laplacian(values):
        total = values.copy()
        for first, second, affinity in edges:
            total[first] += affinity * values[second]
            total[second] += affinity * values[first]
        change = total / share - values
        return np.where(np.abs(change) > 1e-12, change, 0.0)

    low, high = np.maximum(base + DEAD, heights - most), heights + most
    result = heights.copy()
    for _ in range(max(1, int(np.ceil(radius * radius))) * passes):
        relaxed = result + .5 * weight * laplacian(result)
        candidate = relaxed - .35 * weight * laplacian(relaxed)
        result = np.where(weight > 0, np.clip(candidate, low, high), heights)
    return guard_steps(heights, result, weight > 0, max_drop)


def _detail(heights: Field, ground: Field, region: Field, reach: int = 3, most: float = 1.5) -> Field:
    # The land's own small rises and dips carried onto a broad new soil
    # surface, in full where it moved three blocks or more, so its contours
    # wander as the land's round it do. A bump that would steepen a step
    # past one block is left off.
    level = np.floor(ground + 0.5)
    bumps = np.clip(level - _eased(level, reach), -most, most)
    weight = np.clip(np.abs(heights - ground) / 3.0, 0.0, 1.0)
    carried = np.where(region, heights + weight * bumps, heights)
    before, after = np.floor(heights + 0.5), np.floor(carried + 0.5)
    beside = np.empty_like(after)
    steep = np.zeros(heights.shape, dtype=bool)
    for rows, cells in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        _shift_into(after, rows, cells, np.nan, beside)
        was = np.empty_like(before)
        _shift_into(before, rows, cells, np.nan, was)
        with np.errstate(invalid="ignore"):
            steep |= np.abs(after - beside) > np.maximum(1.0, np.abs(before - was))
    return np.where(region & ~steep, carried, heights)


def strata(heights: Field, rock_top: Field, region: Field, grid: Grid, seed: int,
           spacing: tuple[int, int], dip: float = 2.5) -> Field:
    """Snap rock faces down to irregular bedding levels, leaving ledges of varied width.

    Levels follow world height with a slow undulation, so a ledge runs along
    the hillside rather than parallel to the track.
    """
    if spacing[1] <= 0 or not region.any():
        return heights
    rng = np.random.default_rng(seed + 977)
    lowest = int(np.floor(heights[region].min())) - 8
    highest = int(np.ceil(heights[region].max())) + 8
    gaps = rng.integers(max(spacing[0], 1), spacing[1] + 1,
                        size=(highest - lowest) // max(spacing[0], 1) + 4)
    levels = lowest + np.concatenate(([0], np.cumsum(gaps)))
    undulation = dip * fbm(grid, seed + 211, 40.0, 2, 12.0)
    index = np.searchsorted(levels, heights + undulation, side="right") - 1
    snapped = levels[np.clip(index, 0, len(levels) - 1)] - undulation
    rock = region & (heights < rock_top) & (snapped < heights)
    return np.where(rock, np.floor(snapped + 0.5), heights)


def level_fins(heights: Field, region: Field, width: int = 2, rounds: int = 8) -> Field:
    """Cut down ridges and spikes no wider than `width` inside `region`; fill such grooves."""
    levels = np.floor(heights + 0.5)
    for _ in range(rounds):
        opened = np.where(region, np.minimum(levels, square_opening(levels, width + 1)), levels)
        closed = np.where(region, np.maximum(opened, square_closing(opened, width + 1)), opened)
        if np.array_equal(closed, levels):
            break
        levels = closed
    return np.where(region, levels, heights)


def cap_risers(heights: Field, ceiling: Field, region: Field, most: float, rounds: int = 16) -> Field:
    """Raise `region` columns toward `ceiling` until no face beside them is taller than `most`."""
    heights = heights.copy()
    beside = np.empty_like(heights)
    for _ in range(rounds):
        bound = np.full(heights.shape, -np.inf)
        for rows, cells in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            _shift_into(heights, rows, cells, -np.inf, beside)
            np.maximum(bound, beside - most, out=bound)
        raised = np.where(region, np.maximum(heights, np.minimum(ceiling, bound)), heights)
        if np.array_equal(raised, heights):
            break
        heights = raised
    return heights
