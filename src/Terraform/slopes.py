from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from . import kernels
from .distance import _shift_into, nearest
from .grid import Field

# The slope a plan that sets none gets: a block per two columns.
DEFAULT_RUN = 2.0
# What a column around the track is to its slopes: kept as it is, ground a
# bridge's bank may spill onto, or shaped.
HELD, SPILL, SHAPED = 0, 1, 2
# How far out from the formation a cut through rock stands as a face, and
# how far behind a trench's walls the ground is held at their top.
ROCK_BAND = 2.0
WALL_BAND = 2.0
# How far out from a tunnel's lining its portal's face may stand.
PORTAL = 4.0
# How many passes each phase of the repair may take.
REPAIR_PASSES = 400
# How far round a drop the repair could not close held ground may keep a
# face against it.
POCKET = 12
# Enough for a slope to wind across any grid; it stops once it settles.
PASSES = 1000
# How many columns across a ridge or a groove in the works may be and
# still be levelled, and enough rounds of levelling for the ground to settle.
RIDGE_WIDTH = 2
RIDGE_ROUNDS = 16

_FACE, _CORNER, _KNIGHT = 1.0, sqrt(2.0), sqrt(5.0)
# (rows, cells, length, gates, every): a step is taken only where its gates,
# counted from the column it leaves, are open -- any of them for a corner
# step, both for a knight step -- so a slope never slips through a wall.
STEPS = (
    (-1, 0, _FACE, (), True), (1, 0, _FACE, (), True),
    (0, -1, _FACE, (), True), (0, 1, _FACE, (), True),
    (-1, -1, _CORNER, ((-1, 0), (0, -1)), False), (-1, 1, _CORNER, ((-1, 0), (0, 1)), False),
    (1, -1, _CORNER, ((1, 0), (0, -1)), False), (1, 1, _CORNER, ((1, 0), (0, 1)), False),
    (-2, -1, _KNIGHT, ((-1, 0), (-1, -1)), True), (-2, 1, _KNIGHT, ((-1, 0), (-1, 1)), True),
    (2, -1, _KNIGHT, ((1, 0), (1, -1)), True), (2, 1, _KNIGHT, ((1, 0), (1, 1)), True),
    (-1, -2, _KNIGHT, ((0, -1), (-1, -1)), True), (1, -2, _KNIGHT, ((0, -1), (1, -1)), True),
    (-1, 2, _KNIGHT, ((0, 1), (-1, 1)), True), (1, 2, _KNIGHT, ((0, 1), (1, 1)), True),
)
# Side neighbours as (lower's cells, higher's cells, higher's offset).
_PAIRS = (
    ((slice(None, -1), slice(None)), (slice(1, None), slice(None)), (1, 0)),
    ((slice(1, None), slice(None)), (slice(None, -1), slice(None)), (0, 0)),
    ((slice(None), slice(None, -1)), (slice(None), slice(1, None)), (0, 1)),
    ((slice(None), slice(1, None)), (slice(None), slice(None, -1)), (0, 0)),
)


def level(designed: Field) -> Field:
    # Half up everywhere, so heights a block apart never round two apart.
    return np.floor(designed + 0.5)


def cone(
    sources: Field, heights: Field, open_ground: Field, run: float, rising: bool, passes: int,
    natural: Field | None = None, floor: Field | None = None,
) -> Field:
    """The surface through `heights` at `sources` changing a block per `run`
    columns, measured through `open_ground`: falling, the least the ground
    may stand (a fill's slope); rising, the most (a cut's). Given `natural`,
    it runs on only through ground it fills or cuts: a cutting ends where it
    meets the ground, however high the hill beyond it rises. A rising one
    goes no more than a block below `floor`: a cut stops where it would reach
    rock."""
    absent = np.inf if rising else -np.inf
    found = np.where(sources, heights, absent).astype(np.float64)
    target = open_ground & ~sources
    gates = _gates(open_ground)
    spread = np.empty_like(found)
    moved = np.empty_like(found)
    better = np.empty(found.shape, dtype=bool)
    for _ in range(passes):
        settled = True
        for (rows, cells, length, _, _), gate in zip(STEPS, gates):
            if natural is None:
                np.copyto(spread, found)
            else:
                beyond = (natural > found) if rising else (natural < found)
                np.copyto(spread, np.where(sources | beyond, found, absent))
            _shift_into(spread, rows, cells, absent, moved)
            moved += (length if rising else -length) / run
            if rising:
                np.less(moved, found, out=better)
            else:
                np.greater(moved, found, out=better)
            better &= target
            if gate is not None:
                better &= gate
            if floor is not None:
                better &= moved >= floor - 1.0
            if better.any():
                np.copyto(found, moved, where=better)
                settled = False
        if settled:
            break
    return found


def _gates(open_ground: Field) -> list[Field | None]:
    out: list[Field | None] = []
    shifted = np.empty(open_ground.shape, dtype=bool)
    for rows, cells, _, gates, every in STEPS:
        if not gates:
            out.append(None)
            continue
        masks = []
        for gate_rows, gate_cells in gates:
            _shift_into(open_ground, rows - gate_rows, cells - gate_cells, False, shifted)
            masks.append(shifted.copy())
        out.append(np.logical_and.reduce(masks) if every else np.logical_or.reduce(masks))
    return out


@dataclass(frozen=True, slots=True)
class Limits:
    # `free` columns stand between `fill` and `cut`; `bed` is the track the
    # ground meets, at `meets`; a drop against an `exempt` column may stay,
    # one against `held` ground is closed from the other side. Any
    # `touchable` column the design has already moved may be repaired.
    free: Field
    fill: Field
    cut: Field
    bed: Field
    meets: Field
    exempt: Field
    held: Field
    cap: Field
    touchable: Field
    # What can tell a rock face (anything with `face(rows, cells, lows,
    # highs)`), and where each column's rock starts (-inf where it does not).
    rock: object | None = None
    stone: Field | None = None
    repair_passes: int = REPAIR_PASSES
    pocket: int = POCKET
    ridge_width: int = RIDGE_WIDTH
    # The least each column may stand (-inf where the works may cut it).
    least: Field | None = None
    # Explicit grading ceiling, separate from the ordinary cutting cone:
    # repair may ease a cutting back, but cannot lift a shoreline taper.
    ceiling: Field | None = None
    # Natural designs use the cones as bounds, not as replacement shapes.
    conform: bool = True
    max_drop: int = 1
    # Local rock grades apply to cuts only; new fill keeps max_drop.
    cut_drop: Field | None = None

    def _allowed_drop(self, heights, natural, low, high):
        if self.cut_drop is None:
            return self.max_drop
        cutting = (heights[low] <= natural[low]) & (heights[high] <= natural[high])
        return np.where(cutting, self.cut_drop[high], self.max_drop)

    def shape(self, designed: Field, natural: Field, kept: Field) -> Field:
        # The bank and the cutting laid on the slopes where the natural
        # ground lies past them, and the ground on the bed's own columns
        # meeting it; elsewhere the design, held between the slopes.
        if self.conform:
            banked = self.free & ~kept & (natural < self.fill)
            cutting = self.free & ~kept & (natural > self.cut)
            designed = np.where(banked, self.fill, designed)
            designed = np.where(cutting, self.cut, designed)
        designed = np.where(self.bed & ~kept, self.meets, designed)
        return self.clamp(designed)

    def clamp(self, designed: Field) -> Field:
        held = np.minimum(np.maximum(designed, self.fill), self.cut)
        return np.where(self.free, held, designed)

    def enforce(self, designed: Field, natural: Field) -> tuple[Field, int]:
        # Every drop beyond max_drop between side neighbours outside the
        # exceptions is closed from whichever side can move: a fill's low
        # side raised, a cut's high side cut back. Where the ground between
        # the track's structures and held ground is too short for a block a
        # column, the drop is left as one face against the held ground.
        # Then a ridge or a groove no wider than `ridge_width` in the works,
        # or where they meet the ground, is levelled -- grooves filled, then
        # ridges cut down -- and the drops that leaves closed again.
        # Returns the levels and the drops left.
        free = self.free | (self.touchable & (designed != natural))
        exempt = self.exempt & ~free
        bounded = designed if self.ceiling is None else np.minimum(designed, self.ceiling)
        heights, exempt = self._settle(np.where(self.bed, self.meets, bounded), natural, free, exempt)
        if self.ridge_width > 0:
            heights = self._level_ridges(heights, natural, free, exempt)
        return np.where(self.bed, designed, heights), self._left(heights, natural, free, exempt)

    def _level_ridges(self, heights: Field, natural: Field, free: Field, exempt: Field) -> Field:
        # Only the works and the columns round them move, and a column whose
        # change would leave a drop keeps its height: closing a drop by moving
        # other ground instead would run a spur out across it.
        moved = kernels.local_maximum((heights != natural).astype(np.float64), 1) > 0
        # Held ground stays natural unless the brush already edited it.
        # Such columns are touchable in enforce, and must also be tidied:
        # otherwise a brush ridge beyond a bank's toe can never come down.
        level = free & moved & ~self.exempt & ~self.bed & (~self.held | (heights != natural))
        size = self.ridge_width + 1
        for _ in range(RIDGE_ROUNDS):
            start = heights
            # Grooves filled, then ridges cut, each tried on its own: a rib
            # whose groove cannot fill may still be cut.
            for operation in (kernels.square_closing, kernels.square_opening):
                levelled = np.where(level, operation(heights, size), heights)
                if self.least is not None:
                    levelled = np.maximum(levelled, self.least)
                if self.ceiling is not None:
                    levelled = np.minimum(levelled, self.ceiling)
                levelled = self._carried(levelled, heights, natural, free, exempt, level)
                while True:
                    clash = self._stuck(levelled, natural, free, exempt) & (levelled != heights)
                    if not clash.any():
                        break
                    levelled = np.where(clash, heights, levelled)
                heights = levelled
            if np.array_equal(heights, start):
                break
        return heights

    def _carried(
        self, levelled: Field, heights: Field, natural: Field, free: Field, exempt: Field, level: Field,
    ) -> Field:
        # A rib down a slope too steep to cut alone is cut whole: a column
        # cut down takes the ground above it down with it, and a groove
        # filled lifts the ground below it.
        least = self.least if self.least is not None else -np.inf
        for _ in range(self.repair_passes):
            carried = levelled.copy()
            for low, high, offset in _PAIRS:
                drops = self._drops(levelled, natural, free, exempt, low, high, offset)
                cut = drops & level[high] & (levelled[low] < heights[low])
                view = carried[high]
                allowed = self._allowed_drop(levelled, natural, low, high)
                np.minimum(view, np.where(cut, levelled[low] + allowed, np.inf), out=view)
                filled = drops & level[low] & (levelled[high] > heights[high])
                view = carried[low]
                np.maximum(view, np.where(filled, levelled[high] - allowed, -np.inf), out=view)
            carried = np.maximum(carried, least)
            if self.ceiling is not None:
                carried = np.minimum(carried, self.ceiling)
            if np.array_equal(carried, levelled):
                break
            levelled = carried
        return levelled

    def _settle(self, heights: Field, natural: Field, free: Field, exempt: Field) -> tuple[Field, Field]:
        heights = self._repair(heights, natural, free, exempt)
        stuck = self._stuck(heights, natural, free, exempt)
        if stuck.any():
            near = kernels.local_maximum(stuck.astype(np.float64), self.pocket) > 0
            exempt = exempt | (self.held & ~free & near)
            heights = self._repair(heights, natural, free, exempt)
        return heights, exempt

    def _stuck(self, heights: Field, natural: Field, free: Field, exempt: Field) -> Field:
        # The columns on either side of a drop the repair could not close.
        found = np.zeros(heights.shape, dtype=bool)
        for low, high, offset in _PAIRS:
            drops = self._drops(heights, natural, free, exempt, low, high, offset)
            found[low] |= drops
            found[high] |= drops
        return found

    def _repair(self, heights: Field, natural: Field, free: Field, exempt: Field) -> Field:
        # Raise first: every low side lifted to within a block of its
        # neighbour, which fills a pit and eases a cut back. Then, where a
        # low side cannot rise -- under the track's headroom, beside its bed
        # or held ground -- the high side is cut back from it. Each pass only
        # ever raises, or only ever lowers, so each settles.
        for _ in range(self.repair_passes):
            ceiling, _ = self._bounds(heights, free, exempt, natural)
            up = np.full(heights.shape, -np.inf)
            for low, high, offset in _PAIRS:
                drops = self._drops(heights, natural, free, exempt, low, high, offset)
                view = up[low]
                allowed = self._allowed_drop(heights, natural, low, high)
                np.maximum(view, np.where(drops & free[low], heights[high] - allowed, -np.inf), out=view)
            now = np.where(free & (up > heights), np.minimum(up, np.maximum(ceiling, heights)), heights)
            if np.array_equal(now, heights):
                break
            heights = now
        for _ in range(self.repair_passes):
            _, floor = self._bounds(heights, free, exempt, natural)
            if self.least is not None:
                floor = np.maximum(floor, self.least)
            down = np.full(heights.shape, np.inf)
            for low, high, offset in _PAIRS:
                drops = self._drops(heights, natural, free, exempt, low, high, offset)
                view = down[high]
                allowed = self._allowed_drop(heights, natural, low, high)
                np.minimum(view, np.where(drops & free[high], heights[low] + allowed, np.inf), out=view)
            now = np.where(free & (down < heights), np.maximum(down, np.minimum(floor, heights)), heights)
            if np.array_equal(now, heights):
                break
            heights = now
        return heights

    def _left(self, heights: Field, natural: Field, free: Field, exempt: Field) -> int:
        return sum(int(self._drops(heights, natural, free, exempt, low, high, offset).sum())
                   for low, high, offset in _PAIRS)

    def _bounds(self, heights: Field, free: Field, exempt: Field,
                natural: Field | None = None) -> tuple[Field, Field]:
        # How high and how low each column may go: under its cap, and
        # within a block of the neighbours that cannot move -- the bed
        # either way, a column at its cap upward.
        fixed = (self.bed | self.held) & ~free
        stuck = (fixed | (free & (self.cap <= heights))) & ~exempt
        ceiling = self.cap.copy()
        floor = np.full(heights.shape, -np.inf)
        beside = np.empty_like(heights)
        above = np.empty(heights.shape, dtype=bool)
        below = np.empty(heights.shape, dtype=bool)
        for rows, cells in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            _shift_into(heights, rows, cells, np.nan, beside)
            _shift_into(stuck, rows, cells, False, above)
            _shift_into(fixed & ~exempt, rows, cells, False, below)
            up_drop = down_drop = self.max_drop
            if self.cut_drop is not None and natural is not None:
                adjacent_native = np.empty_like(natural)
                adjacent_drop = np.empty_like(self.cut_drop)
                _shift_into(natural, rows, cells, np.nan, adjacent_native)
                _shift_into(self.cut_drop, rows, cells, self.max_drop, adjacent_drop)
                cutting = (heights <= natural) & (beside <= adjacent_native)
                up_drop = np.where(cutting, self.cut_drop, self.max_drop)
                down_drop = np.where(cutting, adjacent_drop, self.max_drop)
            ceiling = np.where(above, np.minimum(ceiling, beside + up_drop), ceiling)
            floor = np.where(below, np.maximum(floor, beside - down_drop), floor)
        return ceiling, floor

    def _drops(self, heights, natural, free, exempt, low, high, offset) -> Field:
        lower, higher = heights[low], heights[high]
        drops = (higher - lower > self._allowed_drop(heights, natural, low, high)) & (free[low] | free[high])
        if not drops.any():
            return drops
        drops &= ~exempt[low] & ~exempt[high]
        # Existing faces may stay, but a natural design must slope any
        # face it edits even when the old face happened to be taller.
        existing = (higher - lower <= natural[high] - natural[low])
        if not self.conform:
            existing &= (higher == natural[high]) & (lower == natural[low])
        drops &= ~(~self.bed[high] & ~self.bed[low] & existing)
        if self.rock is not None and drops.any():
            # A face of rock; or an untouched column's rock under its own
            # soil, the way a natural cliff stands.
            rows, cells = np.nonzero(drops)
            at = (rows + offset[0], cells + offset[1])
            base = lower[rows, cells]
            top = heights[at] - 1.0
            if self.stone is not None:
                capped = (heights[at] == natural[at]) & (self.stone[at] - base >= 2.0)
                top = np.where(capped, self.stone[at], top)
            face = ~self.bed[at] & (heights[at] <= natural[at])
            if self.cut_drop is not None:
                # Native rock can remain steep. At a newly graded contact,
                # rock must meet its configured ledge bound as well.
                face &= self.cut_drop[at] <= self.max_drop
            face[face] = self.rock.face(at[0][face], at[1][face], base[face] + 1.0, top[face])
            drops[rows[face], cells[face]] = False
        return drops


def limits(
    natural: Field,
    track: Field,
    role: Field,
    meets: Field,
    retained: Field,
    blocked: Field,
    cap: Field,
    shoulder: float,
    fill_run: float,
    cut_run: float,
    passes: int,
    rock=None,
    portal: float = PORTAL,
    repair_passes: int = REPAIR_PASSES,
    rock_band: float = ROCK_BAND,
    pocket: int = POCKET,
    wall_band: float = WALL_BAND,
    ridge_width: int = RIDGE_WIDTH,
    shore: Field | None = None,
    retained_area: Field | None = None,
    rock_area: Field | None = None,
    hold_walls: bool = False,
) -> Limits:
    """`track` columns with `role` SHAPED hold the ground at `meets` (the top
    of `retained` walls, else the formation) and the slopes run out from
    them; SPILL ground takes them too, HELD and `blocked` columns do not.
    A `shore` column may be filled but never cut below its natural height."""
    open_ground = ~blocked & np.where(track, role == SPILL, role >= SPILL)
    shaped = track & (role == SHAPED) & ~blocked
    edge = shaped & (kernels.local_maximum(open_ground.astype(np.float64), 1) > 0)
    near = nearest(edge, (meets, (edge & retained).astype(np.float64)), shoulder + 1.0)
    strip = open_ground & (near.distance <= shoulder + 1e-6)
    if retained_area is not None:
        strip &= (near[1] <= 0.5) | retained_area
    anchors = edge | strip
    heights = np.where(edge, meets, near[0])
    walled = np.where(edge, retained, near[1] > 0.5) & anchors
    def propagated(sources, permitted, run, *, rising, floor=None):
        if retained_area is None or not (sources & walled).any():
            return cone(sources, heights, permitted, run, rising=rising,
                        passes=passes, natural=natural, floor=floor)
        # Keep a trench's wall height within the trench. Ordinary formation
        # anchors retain their usual spill across other cross sections.
        ordinary = cone(sources & ~walled, heights, permitted, run, rising=rising,
                        passes=passes, natural=natural, floor=floor)
        walls = cone(sources & walled, heights, permitted & retained_area, run,
                     rising=rising, passes=passes, natural=natural, floor=floor)
        walls = np.where(retained_area, walls, np.inf if rising else -np.inf)
        return np.minimum(ordinary, walls) if rising else np.maximum(ordinary, walls)

    fill = propagated(anchors, open_ground, fill_run, rising=False)
    ring = np.zeros(natural.shape, dtype=bool)
    stone = None
    band = nearest(anchors, (heights, walled.astype(np.float64)), max(rock_band, wall_band) + 1.0)
    # Right behind a trench's walls the ground is cut to their top and stays
    # there, rock or not.
    behind = open_ground & (band.distance <= wall_band) & (band[1] > 0.5)
    if retained_area is not None:
        behind &= retained_area
    cap = np.where(behind, np.minimum(cap, band[0]), cap)
    if rock is not None:
        # Rock beside open track stands as a face with the soil set back
        # from its top. Behind the face no cut goes into rock: the hill
        # stands as the save has it.
        stone = np.where(behind | (False if rock_area is None else ~rock_area),
                         -np.inf, rock.rock_tops())
        candidates = (open_ground & ~anchors & (band.distance <= rock_band)
                      & (natural >= band[0] + 2.0) & (band[1] < 0.5))
        rows, cells = np.nonzero(candidates)
        tops = stone[rows, cells]
        stands = tops >= band[0][rows, cells] + 2.0
        stands[stands] = rock.face(rows[stands], cells[stands], band[0][rows, cells][stands] + 1.0, tops[stands])
        ring[rows[stands], cells[stands]] = True
        heights = heights.copy()
        heights[rows[stands], cells[stands]] = np.minimum(
            natural[rows[stands], cells[stands]], tops[stands]
        )
    cut = propagated(
        anchors | ring, open_ground & ~track & ~ring, cut_run, rising=True, floor=stone,
    )
    wall_floor = None
    if hold_walls and walled.any():
        # A low bed's cutting cone must not pass through the higher bank
        # cone behind a retaining wall at a curved section handover.
        # Only authored wall edges are controls. A shoulder copied from a
        # lower neighbouring edge must remain editable at the handover.
        wall_area = open_ground if retained_area is None else open_ground & retained_area
        # This is a cutting floor, so it must cross native ground above it;
        # a fill cone would stop immediately upon meeting that ground.
        wall_floor = cone(edge & retained, heights, wall_area, fill_run,
                          rising=False, passes=passes)
        wall_floor = np.where(wall_area, wall_floor, -np.inf)
        wall_floor = np.where(np.isfinite(wall_floor),
                              np.maximum(wall_floor, fill), wall_floor)
        cut = np.maximum(cut, wall_floor)
        cap = np.maximum(cap, wall_floor)
    bed = shaped & ~retained
    # The track's own structures, protected and unknown ground, ground under
    # a deck the bank cannot reach, and a tunnel's portal keep their faces;
    # the rest of a held stretch's ground is natural ground that stays.
    lining = track & (role == HELD)
    portal = ~track & (nearest(lining, (), portal + 1.0).distance <= portal)
    exempt = (track & ~bed & ~open_ground) | blocked | (open_ground & (cap < fill)) | portal
    held = ~open_ground & ~track & ~exempt
    least = wall_floor
    if shore is not None and shore.any():
        shore_floor = np.where(shore & ~blocked, natural, -np.inf)
        least = shore_floor if least is None else np.maximum(least, shore_floor)
        cut = np.maximum(cut, least)
    return Limits(open_ground, fill, cut, bed, np.where(bed, meets, np.nan), exempt, held, cap,
                  ~blocked & ~track, rock, stone, repair_passes, pocket, ridge_width, least)
