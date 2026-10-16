from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from ...Core.CrossSection.cross_section import MIRRORED, CrossSection, lean_of
from ...Core.CrossSection.turns import (
    CLOCKWISE,
    Rotation,
    SectionTurn,
    Turn,
    forward,
    is_quarter,
    neighbors,
    normalize,
    step_along,
)
from ...Core.Path.path import Path
from ...Core.coordinates import BlockPos, Vec3
from ...Core.line import line
from ...Core.modes import StepMode
from ...Core.plan_index import NearestInPlan

# How far either side of a diagonal to look for the steps that place its
# handover.
SWITCH_REACH = 64

Cell = tuple[int, int]
# An annotation's position in a row: its name, the diagonal it lies on
# counted from the station, and its height.
Piece = tuple[str, int, int]


class Laid(NamedTuple):
    # One position for `FeaturePath.add`, in its argument order.
    position: BlockPos
    gradient: Vec3
    turn: Turn
    path_index: int
    y_offset: int
    direction: Rotation
    layout: Turn
    mirrored: bool


@dataclass(frozen=True, slots=True)
class Corner:
    # A turn through a diagonal: the last station laid in the quarter being
    # left (the next is laid in the one taken up) and the quarter of every
    # station that changes because of it. Two turning back with nothing to
    # turn for keep one quarter (`leaving` is `taking`) and lay no wedge.
    index: int
    leaving: Turn
    taking: Turn
    direction: Rotation
    switch: int
    turns: Mapping[int, SectionTurn]


def corners_of(
    turns: Sequence[SectionTurn], path: Path, section: CrossSection, start: int, stop: int
) -> tuple[Corner, ...]:
    # The diagonals of one span whose quarters lay every column on the same
    # diagonal at the handover, and whose handover lies inside the span. A
    # run of diagonal stations is one corner, handing over no further than
    # the corners either side of it.
    runs = _diagonal_runs(turns, start, stop)
    plans: list[_Plan] = []
    for number, (first, last) in enumerate(runs):
        turn = turns[first]
        lower, upper = neighbors(turn.absolute)
        leaving, taking = (lower, upper) if turn.direction == CLOCKWISE else (upper, lower)
        earliest = runs[number - 1][1] + 1 if number else start
        latest = runs[number + 1][0] - 1 if number + 1 < len(runs) else stop
        left, taken = _handover(path, first, leaving, taking, earliest, latest)
        lowest, highest = max(left, start), min(taken - 1, stop, len(path.path_points) - 2)
        if lowest > highest:
            continue
        # The middle of the stretch between the steps along each quarter. Where
        # a neighbouring corner or the span's end hides one quarter's step,
        # the stretch reaches past it, so the quarter stepped along keeps up
        # to the neighbour.
        if (taken > latest) != (left < earliest):
            left, taken = _handover(path, first, leaving, taking, 1, len(path.path_points) - 1)
        switch = min(highest, max(lowest, (left + taken - 1) // 2))
        plans.append(_Plan(first, last, leaving, taking, turn.direction, lowest, highest, switch))
    found = []
    for plan in _ordered(plans, path):
        switch, kept = plan.switch, plan.leaving == plan.taking
        relaid = {
            k: SectionTurn(
                normalize((plan.leaving if k <= switch else plan.taking) * plan.direction),
                plan.direction, turns[k].angle,
            )
            for k in range(min(plan.first, switch), (plan.last if kept else max(plan.last, switch + 1)) + 1)
        }
        corner = Corner(plan.first, plan.leaving, plan.taking, plan.direction, switch, relaid)
        if kept:
            found.append(corner)
            continue
        left, right = _ends(section, path, corner)
        if set(left) == set(right):
            found.append(corner)
    return tuple(
        corner for corner in found
        if not any(other is not corner and _clash(corner, other) for other in found)
    )


class _Plan(NamedTuple):
    first: int
    last: int
    leaving: Turn
    taking: Turn
    direction: Rotation
    lowest: int
    highest: int
    switch: int


def _ordered(plans: Sequence[_Plan], path: Path) -> list[_Plan]:
    # Each corner's switch after the one before it. Neighbours reaching into
    # one gap split it at its middle, unless they turn back to the quarter
    # both would keep across it and the other is never stepped along there:
    # then neither turns, and the stretch stays in that quarter.
    ordered: list[_Plan] = []
    for plan in plans:
        before = ordered[-1] if ordered else None
        if before is None or before.leaving == before.taking or before.switch < plan.switch:
            ordered.append(plan)
        elif before.leaving == plan.taking and not _stepped(path, before.taking, before.first, plan.last):
            ordered[-1] = before._replace(taking=before.leaving, last=plan.last, switch=plan.last)
        else:
            middle = (before.last + plan.first - 1) // 2
            ordered[-1] = before._replace(switch=max(before.lowest, min(before.switch, middle)))
            ordered.append(plan._replace(switch=min(plan.highest, max(plan.switch, middle + 1))))
    return ordered


def _stepped(path: Path, quarter: Turn, first: int, last: int) -> bool:
    points = path.path_points
    fx, fz = forward(quarter)
    return any(
        (points[k].position.x - points[k - 1].position.x, points[k].position.z - points[k - 1].position.z)
        == (fx, fz)
        for k in range(max(1, first), last + 1)
    )


def _diagonal_runs(turns: Sequence[SectionTurn], start: int, stop: int) -> list[tuple[int, int]]:
    # (first, last) of each run of stations on one diagonal, turning one way.
    runs: list[tuple[int, int]] = []
    for index in range(start, stop + 1):
        turn = turns[index]
        if is_quarter(turn.absolute):
            continue
        if runs and runs[-1][1] == index - 1 and (
            turns[index - 1].absolute, turns[index - 1].direction
        ) == (turn.absolute, turn.direction):
            runs[-1] = (runs[-1][0], index)
        else:
            runs.append((index, index))
    return runs


def _clash(corner: Corner, other: Corner) -> bool:
    # Corners relaying a station in different quarters. Neighbours handing
    # over in one gap share a station laid in the quarter between them.
    return any(
        k in other.turns and other.turns[k].absolute != turn.absolute
        for k, turn in corner.turns.items()
    )


def _handover(
    path: Path, index: int, leaving: Turn, taking: Turn, earliest: int, latest: int
) -> tuple[int, int]:
    # The last step along the quarter being left and the first along the one
    # taken up, looking no further than `earliest` and `latest`; the quarters
    # hand over between them. A row laid across a step sideways to it lands
    # on the row before, so each quarter must see only steps along it or
    # diagonal ones.
    points = path.path_points

    def step(k: int) -> tuple[int, int]:
        here, before = points[k].position, points[k - 1].position
        return here.x - before.x, here.z - before.z

    first = max(1, index - SWITCH_REACH, earliest)
    last = min(len(points) - 1, index + SWITCH_REACH, latest)
    taken = next((k for k in range(index, last + 1) if step(k) == forward(taking)), last + 1)
    left = next((k for k in range(taken - 1, first - 1, -1) if step(k) == forward(leaving)), first - 1)
    return left, taken


def _diagonal(corner: Corner) -> Turn:
    lower, upper = sorted((corner.leaving, corner.taking))
    return lower + 45 if upper - lower == 90 else (upper + 45) % 360


def _ends(
    section: CrossSection, path: Path, corner: Corner
) -> tuple[dict[Piece, tuple[Cell, int]], dict[Piece, tuple[Cell, int]]]:
    # Each piece's cell in the last row of the quarter being left and the
    # first row of the one taken up, with the station that lays it.
    sx, sz = step_along(_diagonal(corner))
    ends = []
    for k, quarter in ((corner.switch, corner.leaving), (corner.switch + 1, corner.taking)):
        point = path.path_points[k]
        lean = lean_of(quarter, Vec3.coerce(point.gradient).horizontal)
        row = {}
        for offset in section.orient(normalize(quarter * corner.direction), corner.direction, lean):
            for p in offset.offsets:
                piece = (offset.annotation.name, p.x * sz - p.z * sx, p.y)
                row[piece] = ((point.position.x + p.x, point.position.z + p.z), k)
        ends.append(row)
    return ends[0], ends[1]


def fill_corner(
    corner: Corner,
    section: CrossSection,
    path: Path,
    placeable: Callable[[int, int, int], bool],
    level: Callable[[int, int, int], int],
) -> dict[str, list[Laid]]:
    """The wedge between a corner's quarters, by annotation. At the handover
    the two quarters lay each column on the same diagonal, but one's row may
    stop short of the other's along it; each column is carried straight on
    from where the quarter being left last laid it to where the one taken up
    first does.
    """
    if corner.leaving == corner.taking:
        return {}
    step = step_along(_diagonal(corner))
    last, first = _ends(section, path, corner)
    filled: dict[str, list[Laid]] = {}
    for piece, ((x0, z0), leaving) in last.items():
        (x1, z1), taking = first[piece]
        if (x1 - x0) * step[0] + (z1 - z0) * step[1] <= 2:
            continue
        name, _, height = piece
        between = line((x0, 0, z0), (x1, 0, z1))[1:-1]
        for number, cell in enumerate(between):
            k = leaving if 2 * number < len(between) else taking
            if not placeable(cell.x, cell.z, k):
                continue
            point = path.path_points[k]
            turn = corner.turns[k]
            lean = lean_of(turn.absolute, Vec3.coerce(point.gradient).horizontal)
            filled.setdefault(name, []).append(Laid(
                BlockPos(cell.x, height, cell.z), point.gradient, turn.turn, k,
                level(cell.x, cell.z, k), turn.direction, turn.absolute, lean == MIRRORED,
            ))
    return filled



@dataclass(frozen=True, slots=True)
class Flip:
    # A quarter's staircase turning over as the heading crosses the quarter.
    # The stations of `window` lay both leans, each column taking the old
    # lean's cells up to `cut` along the quarter and the new lean's after it;
    # `leans` sets the stations between the natural flip and the window.
    quarter: Turn
    before: int
    after: int
    window: range
    cut: int
    leans: Mapping[int, int]

    def keeps(self, lean: int, x: int, z: int) -> bool:
        fx, fz = forward(self.quarter)
        return (x * fx + z * fz <= self.cut) == (lean == self.before)


def flips_of(
    leans: Mapping[int, int | None], turns: Sequence[SectionTurn], path: Path,
    section: CrossSection, start: int, stop: int,
) -> tuple[Flip, ...]:
    """The lean flips of one span that can be laid as one cut. Turning over
    moves every pushed column from behind its station to ahead of it, and
    the rows either side overlap. A step sideways among the stations that
    lay that overlap puts each pushed column on its neighbour's line, so the
    flip is moved to where every station within the deepest push of it is
    reached without one, and every column changes lean at the same place
    along the road. Where there is no such place the flip stays.
    """
    push = _push(section)
    found = []
    for k in range(start, stop):
        before, after = leans.get(k), leans.get(k + 1)
        quarter = turns[k].absolute
        if (before is None or after is None or before == after
                or turns[k + 1].absolute != quarter or not is_quarter(quarter)):
            continue
        fits = [
            centre
            for low, high in {_straight(path, k, quarter), _straight(path, k + 1, quarter)}
            for centre in range(low + push, high - push + 1)
        ]
        if fits:
            switch = min(fits, key=lambda centre: (abs(centre - k), centre))
            window = range(switch - push, switch + push + 1)
        else:
            # No straight run holds the deepest push: turn over at the flip,
            # across as much of its own straight run as there is.
            low, high = _straight(path, k + 1, quarter)
            switch = k
            reach = min(push, k - low, high - k)
            window = range(switch - reach, switch + reach + 1)
        if window.start < start or window.stop - 1 > stop:
            continue
        point = path.path_points[switch].position
        fx, fz = forward(quarter)
        found.append(Flip(
            quarter, before, after, window, point.x * fx + point.z * fz,
            {s: before if s <= switch else after
             for s in range(min(k, window.start), max(k + 1, window.stop - 1) + 1)},
        ))
    return tuple(
        flip for flip in found
        if not any(other is not flip and set(other.leans) & set(flip.leans) for other in found)
    )


def fill_flip(
    flip: Flip,
    path: Path,
    laid: Mapping[Hashable, Sequence[Laid]],
    push: int,
    placeable: Callable[[int, int, int], bool],
    level: Callable[[int, int, int], int],
) -> dict[Hashable, list[Laid]]:
    """Each column carried across a flip's cut. Where no straight run holds
    the deepest push, the old lean's cells of an outer column stop short of
    the cut and the new lean's start past it; within the flip's straight run
    the column is carried straight on between them along the quarter.
    """
    fx, fz = forward(flip.quarter)
    low, high = _straight(path, flip.window.start + len(flip.window) // 2, flip.quarter)
    near = range(low, high + 1)
    filled: dict[Hashable, list[Laid]] = {}
    for key, positions in laid.items():
        lines: dict[tuple[int, int], list[Laid]] = {}
        for each in positions:
            if each.path_index in near:
                p = each.position
                lines.setdefault((p.x * -fz + p.z * fx, p.y), []).append(each)
        for cells in lines.values():
            cells.sort(key=lambda each: each.position.x * fx + each.position.z * fz)
            for before, after in zip(cells, cells[1:]):
                start = before.position.x * fx + before.position.z * fz
                end = after.position.x * fx + after.position.z * fz
                if not 1 < end - start <= 2 * push + 1 or not start <= flip.cut < end:
                    continue
                for ahead in range(1, end - start):
                    x, z = before.position.x + fx * ahead, before.position.z + fz * ahead
                    if placeable(x, z, before.path_index):
                        filled.setdefault(key, []).append(before._replace(
                            position=BlockPos(x, before.position.y, z), y_offset=level(x, z, before.path_index),
                        ))
    return filled


def _push(section: CrossSection) -> int:
    # The furthest any column is laid along travel from its station.
    columns = {annotation.position.x - section.origin.x for annotation in section.points}
    return max(
        (abs(a) for x in columns for a in section.along(x, section.step_mode_at(x))),
        default=0,
    )


def _straight(path: Path, index: int, quarter: Turn) -> tuple[int, int]:
    # The stations either side of `index` joined by steps with nothing
    # sideways to the quarter.
    fx, fz = forward(quarter)
    points = path.path_points

    def sideways(k: int) -> bool:
        here, before = points[k].position, points[k - 1].position
        return (here.x - before.x) * -fz + (here.z - before.z) * fx != 0

    low = high = index
    while low > 1 and not sideways(low):
        low -= 1
    while high + 1 < len(points) and not sideways(high + 1):
        high += 1
    return low, high

# Stations either side of a handover whose columns the search may hand to
# another of the columns laid there, the stations past them it must meet,
# and how many choices it may try.
SETTLED_REACH = 6
HELD_REACH = 8
SEARCH_BUDGET = 20_000

_FOUR = ((1, 0), (-1, 0), (0, 1), (0, -1))
_EIGHT = _FOUR + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def settle_corners(
    owners: dict[tuple[str, int], dict[Cell, str]],
    runs: Sequence,
    path: Path,
    silhouette=None,
) -> None:
    """Hand the world columns round each handover between quarters (see
    `fill_corner`) to the section columns by search, in place of `owners`'
    one-by-one choice. A column that is one run either side of a corner
    stays one run through it (face-connected if it steps), and columns keep
    their order across the road. Where no such handing is found the choice
    stands.
    """
    groups: dict[tuple[str, int], list] = {}
    for run in runs:
        groups.setdefault((run.section_name, run.span), []).append(run)
    nearest = NearestInPlan(
        (point.position.x, point.position.z, index)
        for index, point in enumerate(path.path_points)
    )
    for key, group in groups.items():
        held = owners.get(key)
        handovers = _handovers(group)
        if not held or not handovers:
            continue
        lanes = _Lanes.of(group, silhouette)
        # Another pass over a column keeps the choice it was given.
        home = {cell: nearest.nearest(*cell) for cell in held if len(cell) == 2}
        by_station: dict[int, list[Cell]] = {}
        for cell, station in home.items():
            by_station.setdefault(station, []).append(cell)
        for handover in handovers:
            _settle(held, lanes, path, home, by_station, handover)


def _handovers(group: Sequence) -> list[int]:
    # Stations after which a span's quarter changes with no join near.
    quarters: dict[int, set[Turn]] = {}
    joins = set()
    for run in group:
        for point in run.path:
            quarters.setdefault(point.path_index, set()).add(point.laid_at)
            if not is_quarter(normalize(point.turn * point.direction)):
                joins.add(point.path_index)
    return [
        k for k in sorted(quarters)
        if k + 1 in quarters and len(quarters[k]) == 1 and len(quarters[k + 1]) == 1
        and quarters[k] != quarters[k + 1]
        and not any(abs(join - k) <= SETTLED_REACH + HELD_REACH for join in joins)
    ]


@dataclass(frozen=True, slots=True)
class _Lanes:
    # Every world column a span's columns were laid in, with the stations
    # that laid it, and each column's place across and step.
    laid: Mapping[Cell, frozenset[str]]
    stations: Mapping[Cell, frozenset[int]]
    across: Mapping[str, int]
    stepping: Mapping[str, bool]

    @classmethod
    def of(cls, group: Sequence, silhouette=None) -> _Lanes:
        laid: dict[Cell, set[str]] = {}
        stations: dict[Cell, set[int]] = {}
        across: dict[str, int] = {}
        stepping: dict[str, bool] = {}
        for run in group:
            name = run.annotation.column_name
            across[name] = run.annotation.position.x
            stepping[name] = stepping.get(name, False) or run.annotation.step_mode is StepMode.MANHATTAN
            for point in run.path:
                cell = (point.position.x, point.position.z)
                if silhouette is not None and silhouette.column(*cell, point.path_index) != cell:
                    continue
                laid.setdefault(cell, set()).add(name)
                stations.setdefault(cell, set()).add(point.path_index)
        return cls(
            {cell: frozenset(each) for cell, each in laid.items()},
            {cell: frozenset(each) for cell, each in stations.items()},
            across, stepping,
        )


def _settle(
    held: dict[Cell, str], lanes: _Lanes, path: Path, home: Mapping[Cell, int],
    by_station: Mapping[int, list[Cell]], handover: int,
) -> None:
    settled = range(handover - SETTLED_REACH, handover + SETTLED_REACH + 2)
    reach = range(settled.start - HELD_REACH, settled.stop + HELD_REACH)
    near = {
        cell for station in range(reach.start - HELD_REACH, reach.stop + HELD_REACH)
        for cell in by_station.get(station, ())
    }
    free = {
        cell for cell in near
        if len(lanes.laid.get(cell, ())) > 1
        and (home[cell] in settled or any(k in settled for k in lanes.stations.get(cell, ())))
    }
    cells = free | {cell for cell in near if home[cell] in reach}
    owner = {cell: held[cell] for cell in cells - free}
    domain = {
        cell: sorted(lanes.laid[cell], key=lambda lane: (lane != held[cell], lane))
        for cell in free
    }
    sides = {
        lane: _sides(lane, held, free, cells, near, home, settled, reach, lanes.stepping.get(lane))
        for lane in lanes.across
    }
    required = [lane for lane, (before, after) in sides.items() if before and after]

    def across_road(cell: Cell) -> float:
        point = path.path_points[home[cell]]
        g = point.gradient
        length = (g.x * g.x + g.z * g.z) ** 0.5 or 1.0
        return ((cell[0] - point.position.x) * -g.z + (cell[1] - point.position.z) * g.x) / length

    sideways = {cell: across_road(cell) for cell in cells}
    pairs: dict[Cell, list[tuple[Cell, Cell]]] = {cell: [] for cell in cells}
    for cell in cells:
        for dx, dz in ((1, 0), (0, 1)):
            other = (cell[0] + dx, cell[1] + dz)
            if other in cells and abs(sideways[other] - sideways[cell]) > 0.5:
                pair = (cell, other) if sideways[cell] < sideways[other] else (other, cell)
                pairs[cell].append(pair)
                pairs[other].append(pair)
    order = sorted(free, key=lambda cell: (home[cell], sideways[cell]))
    tried = 0

    def ordered(cell: Cell) -> bool:
        for left, right in pairs[cell]:
            a, b = owner.get(left), owner.get(right)
            if a is not None and b is not None and lanes.across[b] < lanes.across[a]:
                return False
        return True

    def connectable(lane: str) -> bool:
        # Each side is one run outside, so the corner need only join them.
        steps = _FOUR if lanes.stepping.get(lane) else _EIGHT
        before, after = sides[lane]
        seen, stack = set(before), list(before)
        while stack:
            x, z = stack.pop()
            for dx, dz in steps:
                cell = (x + dx, z + dz)
                if cell in seen or cell not in cells:
                    continue
                if owner.get(cell) == lane or (cell not in owner and lane in domain.get(cell, ())):
                    seen.add(cell)
                    stack.append(cell)
        return bool(seen & after)

    def place(position: int) -> bool:
        nonlocal tried
        if position == len(order):
            return True
        tried += 1
        if tried > SEARCH_BUDGET:
            return False
        cell = order[position]
        for lane in domain[cell]:
            owner[cell] = lane
            if ordered(cell) and all(
                connectable(each) for each in domain[cell] if each in required
            ) and place(position + 1):
                return True
        del owner[cell]
        return False

    # A column no handing could join is left to the one-by-one choice.
    required = [lane for lane in required if connectable(lane)]
    if place(0):
        for cell in free:
            held[cell] = owner[cell]


def _sides(
    lane: str, held: Mapping[Cell, str], free: set[Cell], cells: set[Cell], near: set[Cell],
    home: Mapping[Cell, int], settled: range, reach: range, stepping: bool | None,
) -> tuple[set[Cell], set[Cell]]:
    # The column's run before a corner and after it, as far as the search
    # sees them: the run near the corner, outside the handed-out cells, that
    # holds most of the column's cells on that side, or nothing if one run
    # holds both sides.
    steps = _FOUR if stepping else _EIGHT
    runs = []
    for side in (range(reach.start, settled.start), range(settled.stop, reach.stop)):
        ring = {cell for cell in cells - free if held[cell] == lane and home[cell] in side}
        best: set[Cell] = set()
        left = set(ring)
        while left:
            start = left.pop()
            seen, stack = {start}, [start]
            while stack:
                x, z = stack.pop()
                for dx, dz in steps:
                    cell = (x + dx, z + dz)
                    if cell in near and cell not in seen and cell not in free and held[cell] == lane:
                        seen.add(cell)
                        stack.append(cell)
            left -= seen
            if len(seen & ring) > len(best & ring):
                best = seen
        runs.append(best)
    if runs[0] & runs[1]:
        return set(), set()
    return runs[0] & cells, runs[1] & cells
