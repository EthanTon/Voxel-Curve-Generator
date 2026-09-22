from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence, Set
from itertools import groupby
from math import hypot
from typing import NamedTuple

from ...Core.CrossSection.annotations import FeatureAnnotation
from ...Core.CrossSection.schedule import SectionSchedule
from ...Core.CrossSection.turns import SectionTurn, forward, quarter_of
from ...Core.Feature.feature_path import (
    FeaturePathPoint,
    DEFAULT_RULES,
    FeaturePath,
    FeaturePathRule,
    either,
    ending_at,
)
from ...Core.Path.path import Path
from ...Core.line import line
from ...Core.modes import LineMode
from ...Core.coordinates import BlockPos
from ...Core.Path.path_frame import PathFrame
from .delivery import deliver
from .schedule import walk_schedule


class DeliveredPath(NamedTuple):
    # One run of delivered points: everything one annotation put down over one
    # span of the schedule. A run crosses frames without breaking, so the
    # rules see the whole of it; a new span starts a new run and fresh rules.
    annotation: FeatureAnnotation
    span: int
    path: FeaturePath


def deliver_paths(
    schedule: SectionSchedule,
    path: Path,
    frames: Sequence[PathFrame],
    turns: Sequence[SectionTurn],
    lead_in: int = 0,
    lead_out: int = 0,
    rules: Iterable[FeaturePathRule] = DEFAULT_RULES,
) -> tuple[DeliveredPath, ...]:
    # The schedule swept into runs, one per annotation per span.
    #
    # A point whose lead reaches into the frame ahead is delivered into both,
    # clipped to each: frames partition columns, so the two deliveries never
    # land on the same one. Each point takes the level of the frame it was
    # delivered into, which is what a rigid feature is built at.
    runs: dict[tuple[str, int], DeliveredPath] = {}
    for step in walk_schedule(schedule, path, frames, lead_in, lead_out):
        span = step.span
        layout = turns[step.index]
        oriented = step.section.orient(layout.turn, layout.direction)
        for frame in step.frames:
            for delivery in deliver(step.point, oriented, frame):
                run = runs.get((delivery.annotation.name, span))
                if run is None:
                    run = DeliveredPath(delivery.annotation, span, FeaturePath(rules))
                    runs[delivery.annotation.name, span] = run
                for position in delivery.positions:
                    # The layout that placed the position is the one it
                    # carries, not the one the path point happens to be at.
                    # A diagonal is stamped as the two quarters either side
                    # of it, so half its positions were placed in one and
                    # half in the other, and a position turned by the layout
                    # it did not come from lands off its column.
                    run.path.add(
                        position,
                        step.point.gradient,
                        layout.turn,
                        step.index,
                        frame.y_offset,
                        layout.direction,
                        delivery.layout,
                        delivery.mirrored,
                    )
    return tuple(run for run in runs.values() if len(run.path))


# Every point is delivered in its own layout and no other. Where the path
# changes layout it does so through a diagonal, and the section stamps that
# diagonal as the join itself -- the quarter being left carried forward and
# the quarter taking up carried back, through each other. Nothing has to be
# walked back over the change on top of that.


def along_travel(path: FeaturePath, offsets: Sequence[int]) -> FeaturePath:
    # The same run with every point moved along travel by each of `offsets`,
    # which is what `CrossSection.along` gives for a local column.
    #
    # This is how a fitting is put where the section actually laid the column
    # it rides, rather than where that column was authored. It is the same
    # mechanism the section uses on its own points -- a stepping column
    # delivers twice and a column outboard of one is carried -- applied to
    # something that is not a point of the section.
    if tuple(offsets) == (0,):
        return path
    moved = FeaturePath(rules=())
    for point in path:
        forward_x, forward_z = forward(quarter_of(point.laid_at))
        # A mirrored layout runs its depth the other way, and `along` is
        # measured in the section's own terms, so the walk turns over with
        # it. Without this a stepping fitting lands a block the wrong side of
        # the column it rides at every mirrored quarter -- half of them.
        turned = -1 if point.mirrored else 1
        for offset in offsets:
            moved.add(
                BlockPos(
                    point.position.x + forward_x * turned * offset,
                    point.position.y,
                    point.position.z + forward_z * turned * offset,
                ),
                point.gradient,
                point.turn,
                point.path_index,
                point.y_offset,
                point.direction,
                point.layout,
                point.mirrored,
            )
    return moved


def one_line(path: FeaturePath) -> FeaturePath:
    # One position a path point, which is what a line painted along a swept
    # section wants: the section delivers more than one wherever it would
    # otherwise leave a gap in the deck, and a line inheriting that filling
    # comes out four-connected instead of stepping diagonally.
    #
    # A join is the exception. It is stamped at one path point but reaches
    # over several, in both of the quarters either side of it, and the points
    # it reaches over deliver nothing of their own -- a run drops a position
    # it already holds. Taking one position there leaves a hole four points
    # wide and a line drawn straight across it, which on a bend cuts over the
    # deck and into whatever else is painted on it.
    #
    # So at a join the line follows the arm carried forward, the quarter it
    # is already in, and runs on over the points the join covers. The arm
    # carried back is the quarter it is about to take up, laid over points
    # already passed; following that instead jumps the line to the far side
    # of the joint and then back.
    kept = FeaturePath(rules=())
    following: int | None = None
    for _, delivered in groupby(path, key=lambda point: point.path_index):
        delivered = list(delivered)
        layouts = {point.layout for point in delivered}
        if len(layouts) > 1 and following in layouts:
            chosen = [point for point in delivered if point.layout == following]
        elif delivered[0].turn % 90 == 45:
            # The run may have dropped one arm of a join as a duplicate of
            # the preceding delivery. Its remaining positions then all have
            # one layout, but still cover several path points.
            chosen = delivered
        else:
            chosen = delivered[:1]
        if len(chosen) > 1 and len(kept):
            # The first position of a join can lie behind the last ordinary
            # point. Following it would double back before taking the arm
            # forward and make a rail cut across the deck.
            previous = kept[-1].position
            ahead = chosen[0].gradient
            chosen = [
                point for point in chosen
                if (point.position.x - previous.x) * ahead.x
                + (point.position.z - previous.z) * ahead.z > 0
            ]
        for point in chosen:
            kept.add(
                point.position,
                point.gradient,
                point.turn,
                point.path_index,
                point.y_offset,
                point.direction,
                point.layout,
                point.mirrored,
            )
        if chosen:
            following = chosen[-1].layout
    return kept


def at_their_own_level(
    path: FeaturePath,
    level: Callable[[float, float], int],
    offset: Callable[[FeaturePathPoint], BlockPos] | None = None,
) -> FeaturePath:
    # The same run with every point at the level of the column its block
    # actually lands in, rather than the level of the point it was delivered
    # at.
    #
    # A fitting rides the centreline and is only then moved to the side, and
    # the column it lands in is not always in the same frame: a change of
    # level cuts across the deck, so a rail two to the side of a point just
    # before that cut belongs to the frame after it. Given the centreline's
    # level it climbs a point early and floats a block over the deck, which
    # is a rail with nothing under it and no slope into it.
    nowhere = BlockPos(0, 0, 0)
    sits = (lambda point: nowhere) if offset is None else offset
    relevelled = FeaturePath(rules=())
    for point in path:
        here = sits(point)
        relevelled.add(
            point.position,
            point.gradient,
            point.turn,
            point.path_index,
            level(point.position.x + here.x, point.position.z + here.z),
            point.direction,
            point.layout,
            point.mirrored,
        )
    return relevelled


def face_connected(
    path: FeaturePath,
    offset: Callable[[FeaturePathPoint], BlockPos] | None = None,
    continuous: bool = False,
    level: Callable[[float, float], int] | None = None,
    support: Set[BlockPos] | None = None,
) -> FeaturePath:
    # The same run with every diagonal step broken into single-axis moves, so
    # consecutive blocks share a face.
    #
    # A rail only resolves against a neighbour it shares a face with, and the
    # path is eight-connected: wherever it steps diagonally the two blocks a
    # lane paints touch at a corner and the track breaks there.
    #
    # By default only steps between neighbouring path points are filled:
    # either side of a gap in a dash is not consecutive, and that gap is
    # meant to be there. A run with no dashes in it is one unbroken line and
    # every step it took is filled, however far apart the points are. A join
    # lays the next few points' positions itself, and a run drops a position
    # it already holds, so those points deliver nothing and drop out of the
    # run -- on the full curve the rails step straight from point 594 to 599.
    nowhere = BlockPos(0, 0, 0)
    sits = (lambda point: nowhere) if offset is None else offset
    filled = FeaturePath(rules=())
    previous = None
    for point in path:
        if previous is not None and (
            continuous or abs(point.path_index - previous.path_index) <= 1
        ):
            # Lined up where the blocks actually land: the point's height
            # above the path plus the frame's level, plus wherever the
            # feature's own offset puts it in that layout.
            #
            # A run that climbs steps a block along and a block up at once,
            # and one crossing a layout change steps sideways by as much as
            # the offset moves. Lining up the bare positions would bridge
            # neither, because the height is carried by the frame and the
            # sideways jump only happens once the offset is applied.
            here = sits(point)
            start = _world(previous) + sits(previous)
            end = _world(point) + here
            # At a quarter handover, walk the new quarter's travel axis
            # first. The other order cuts the corner across the trackbed.
            along_x, along_z = forward(quarter_of(point.laid_at))
            bridge = _bridge(
                start, end,
                z_first=previous.layout != point.layout and abs(along_z) > abs(along_x),
            )
            # A frame boundary can cross the bridge between its ends. Give
            # each inserted block the level of its own column; interpolating
            # height can put a rail inside the raised deck or on another rail.
            if level is not None:
                bridge = tuple(
                    BlockPos(
                        p.x,
                        point.position.y + here.y + level(p.x, p.z),
                        p.z,
                    )
                    for p in bridge
                )
            if support is not None and any(p not in support for p in bridge):
                # At a quarter join either fixed axis order can leave the
                # sleeper column. Follow its actual cells across the join.
                bridge = _supported_bridge(start, end, support) or bridge
            for between in bridge:
                landed = between - here
                filled.add(
                    BlockPos(landed.x, point.position.y, landed.z),
                    point.gradient,
                    point.turn,
                    point.path_index,
                    landed.y - point.position.y,
                    point.direction,
                    point.layout,
                    point.mirrored,
                )
        filled.add(
            point.position,
            point.gradient,
            point.turn,
            point.path_index,
            point.y_offset,
            point.direction,
            point.layout,
            point.mirrored,
        )
        previous = point
    return filled


def _bridge(
    start: BlockPos, end: BlockPos, *, z_first: bool = False
) -> tuple[BlockPos, ...]:
    # The blocks between two positions that a face-connected run needs.
    #
    # A step that is one along and one up is left alone: that is a slope, and
    # a rail climbs one by being laid one along and one up. Filling it puts a
    # block in the corner instead, which is a step the rail cannot express --
    # it comes out flat and cuts into whatever it is running over.
    along = abs(end.x - start.x) + abs(end.z - start.z)
    if along <= 1 and abs(end.y - start.y) <= 1:
        return ()
    # A manhattan line is face-connected in plan and carries its climb on the
    # moves across, so the bridge climbs the way a rail does without being
    # flattened to one height first.
    if z_first:
        # Transpose the horizontal axes so the same connected line walks z
        # before x. The switch is needed only where a join changes quarters.
        swapped = line(
            BlockPos(start.z, start.y, start.x),
            BlockPos(end.z, end.y, end.x),
            LineMode.MANHATTAN,
        )
        return tuple(BlockPos(p.z, p.y, p.x) for p in swapped[1:-1])
    return tuple(line(start, end, LineMode.MANHATTAN)[1:-1])


def _supported_bridge(
    start: BlockPos, end: BlockPos, support: Set[BlockPos]
) -> tuple[BlockPos, ...] | None:
    if start not in support or end not in support:
        return None
    margin = 1
    left, right = min(start.x, end.x) - margin, max(start.x, end.x) + margin
    back, front = min(start.z, end.z) - margin, max(start.z, end.z) + margin
    queue = deque((start,))
    preceding: dict[BlockPos, BlockPos | None] = {start: None}
    while queue:
        here = queue.popleft()
        if here == end:
            route = []
            while here != start:
                route.append(here)
                here = preceding[here]
            return tuple(reversed(route))[0:-1]
        for dx, dz in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            x, z = here.x + dx, here.z + dz
            if not (left <= x <= right and back <= z <= front):
                continue
            for dy in (0, 1, -1):
                next_pos = BlockPos(x, here.y + dy, z)
                if next_pos in support and next_pos not in preceding:
                    preceding[next_pos] = here
                    queue.append(next_pos)
    return None


def _world(point: FeaturePathPoint) -> BlockPos:
    # Where a point's blocks land: its height above the path plus the level
    # of the frame it was delivered into.
    return BlockPos(
        point.position.x, point.position.y + point.y_offset, point.position.z
    )


def span_of(schedule: SectionSchedule, path_index: int) -> int:
    # Which span of the schedule a path point falls in. Spans are what breaks
    # one run from the next, so every delivered point carries its span.
    for index, span in enumerate(schedule.spans):
        if span.contains(path_index):
            return index
    raise ValueError(f"Path point {path_index} is in no span of the schedule")


def rewalk(path: FeaturePath, rules: Iterable[FeaturePathRule]) -> FeaturePath:
    # The same run walked again under different rules, keeping what they
    # allow. What rides a path -- a pole, a wire -- decides where it sits this
    # way, rather than sitting at every point the section was delivered at.
    #
    # The point the run ends on is kept whatever the rules say, so a wire
    # reaches the end of what it spans.
    rules = tuple(rules)
    if rules and len(path):
        rules = (either(*rules, ending_at(path[-1].position)),)
    walked = FeaturePath(rules)
    for point in path:
        walked.add(
            point.position,
            point.gradient,
            point.turn,
            point.path_index,
            point.y_offset,
            point.direction,
            point.layout,
            point.mirrored,
        )
    return walked


def dashed(
    path: FeaturePath, on: int, off: int, by_index: bool = True
) -> FeaturePath:
    # A run broken into dashes: `on` points or blocks kept, then `off`
    # skipped, over and over. What a lane line is -- three white, five gone.
    #
    # Counted in points by default; by distance it is measured in plan along
    # the run, so a climb does not shorten a dash.
    if on <= 0:
        raise ValueError(f"a dash keeps at least one point, not {on}")
    if off < 0:
        raise ValueError(f"a gap skips no points at all, not {off}")
    dashes = FeaturePath(rules=())
    cycle = on + off
    travelled = 0.0
    previous: BlockPos | None = None
    for index, point in enumerate(path):
        if previous is not None:
            travelled += _apart(previous, point.position)
        previous = point.position
        along = index if by_index else travelled
        if along % cycle < on:
            dashes.add(
                point.position,
                point.gradient,
                point.turn,
                point.path_index,
                point.y_offset,
                point.direction,
                point.layout,
                point.mirrored,
            )
    return dashes


def _apart(first: BlockPos, second: BlockPos) -> float:
    # Distance in plan: a run climbs, but a dash is measured along the ground.
    return hypot(second.x - first.x, second.z - first.z)
