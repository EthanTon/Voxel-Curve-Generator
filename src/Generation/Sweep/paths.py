from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable, Iterable, Sequence, Set
from itertools import groupby
from math import hypot
from typing import NamedTuple, TypeVar

from ...Core.CrossSection.annotations import FeatureAnnotation
from ...Core.CrossSection.schedule import SectionSchedule
from ...Core.CrossSection.cross_section import lean_of
from ...Core.CrossSection.turns import SectionTurn, forward, is_quarter, quarter_of
from ...Core.coordinates import Vec3
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
from ..Geometry.silhouette import PathSilhouette
from .delivery import deliver
from .schedule import walk_schedule

Level = Callable[[float, float, int], int]
OffsetRule = Callable[[FeaturePathPoint], BlockPos]
_Item = TypeVar("_Item")

_NOWHERE = BlockPos(0, 0, 0)


class DeliveredPath(NamedTuple):
    # Everything one annotation delivered over one span of the schedule.
    annotation: FeatureAnnotation
    span: int
    path: FeaturePath
    start: int = 0

    @property
    def section_name(self) -> str:
        # A profile's per-column annotations carry their section's name.
        return self.annotation.section or self.annotation.name


def deliver_paths(
    schedule: SectionSchedule,
    path: Path,
    frames: Sequence[PathFrame],
    turns: Sequence[SectionTurn],
    lead_in: int = 0,
    lead_out: int = 0,
    rules: Iterable[FeaturePathRule] = DEFAULT_RULES,
    silhouette: PathSilhouette | None = None,
) -> tuple[DeliveredPath, ...]:
    # Sweeps the schedule into one run per annotation per span. A point is
    # delivered into every frame its lead reaches, clipped to that frame and
    # taking its level. Where a quarter's lean flips between neighbouring
    # points, both points are laid with both leans so the column level with
    # the flip is covered.
    runs: dict[tuple[str, int], DeliveredPath] = {}
    if silhouette is None:
        silhouette = PathSilhouette.for_sections(
            path, (span.section for span in schedule.spans), frames
        )
    span_masks = tuple(silhouette.for_span(span) for span in schedule.spans)
    steps = list(walk_schedule(
        schedule, path, frames, lead_in, lead_out, extend_spans=True
    ))
    leans = _leans(steps, turns)
    for number, step in enumerate(steps):
        span = step.span
        owned = schedule.spans[span]
        span_mask = span_masks[span]
        layout = turns[step.index]
        laid = [leans[number]]
        for near in (number - 1, number + 1):
            if (
                0 <= near < len(steps)
                and steps[near].span == span
                and abs(steps[near].index - step.index) == 1
                and turns[steps[near].index].absolute == layout.absolute
                and leans[near] not in laid
            ):
                laid.append(leans[near])
        for oriented in (
            step.section.orient(layout.turn, layout.direction, lean) for lean in laid
        ):
            for frame in step.frames:
                for delivery in deliver(step.point, oriented, frame):
                    # Points beyond the span's own (extended lead) carry only
                    # section material: never the centreline that fittings
                    # anchor to, and clip-to-span details only for three
                    # points past the end.
                    extra = not owned.contains(step.index)
                    centerline = (
                        delivery.annotation.position.x == 0
                        and delivery.annotation.position.z == 0
                    )
                    if extra and (
                        centerline or delivery.annotation.clip_to_span
                        and step.index > owned.path_stop + 3
                    ):
                        continue
                    if (
                        frame.path_start > owned.path_stop
                        or frame.path_stop < owned.path_start
                        # Interior frames end before their boundary station;
                        # the final frame includes the path's last station.
                        or (frame.path_stop == owned.path_start
                            and frame.path_stop < path.last_index)
                    ):
                        continue
                    run = runs.get((delivery.annotation.name, span))
                    if run is None:
                        run = DeliveredPath(
                            delivery.annotation, span, FeaturePath(rules),
                            owned.path_start,
                        )
                        runs[delivery.annotation.name, span] = run
                    for position in delivery.positions:
                        if not span_mask.contains(position.x, position.z):
                            continue
                        if silhouette.frames and silhouette.frame_at(
                            position.x, position.z
                        ) != frame:
                            continue
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


def _leans(steps, turns: Sequence[SectionTurn]) -> list[int | None]:
    # The lean of each walked point (None on diagonals), kept while the path
    # heads straight along its quarter.
    leans: list[int | None] = []
    held: tuple[int, int] | None = None
    for step in steps:
        travelling = turns[step.index].absolute
        if not is_quarter(travelling):
            leans.append(None)
            held = None
            continue
        lean = lean_of(
            travelling,
            Vec3.coerce(step.point.gradient).horizontal,
            held[1] if held and held[0] == travelling else None,
        )
        leans.append(lean)
        held = (travelling, lean)
    return leans


def along_travel(path: FeaturePath, offsets: Sequence[int]) -> FeaturePath:
    # Each point moved along travel by each of `offsets` (from
    # `CrossSection.along`), back to front, so a fitting rides the column
    # where the section actually laid it. Mirrored points walk the other way.
    if tuple(offsets) == (0,):
        return path
    moved = FeaturePath(rules=())
    for point in path:
        forward_x, forward_z = forward(quarter_of(point.laid_at))
        turned = -1 if point.mirrored else 1
        for offset in sorted(offsets, key=lambda each: turned * each):
            moved.add_point(
                point,
                position=BlockPos(
                    point.position.x + forward_x * turned * offset,
                    point.position.y,
                    point.position.z + forward_z * turned * offset,
                ),
            )
    return moved


def one_line(path: FeaturePath) -> FeaturePath:
    # One position per path point, for something painted along a section.
    # Through a join it follows the arm in the current layout across all the
    # points the join covers, and it never steps backwards along travel.
    kept = FeaturePath(rules=())
    following: int | None = None
    for _, delivered in groupby(path, key=lambda point: point.path_index):
        delivered = list(delivered)
        layouts = {point.layout for point in delivered}
        if len(layouts) > 1 and following in layouts:
            chosen = [point for point in delivered if point.layout == following]
        elif delivered[0].turn % 90 == 45:
            # A join whose other arm was dropped as a duplicate.
            chosen = delivered
        else:
            chosen = delivered[:1]
        if len(kept):
            previous = kept[-1].position
            ahead = chosen[0].gradient
            chosen = [
                point for point in chosen
                if (point.position.x - previous.x) * ahead.x
                + (point.position.z - previous.z) * ahead.z > 0
            ]
        for point in chosen:
            kept.add_point(point)
        if chosen:
            following = chosen[-1].layout
    return kept


def at_their_own_level(
    path: FeaturePath,
    level: Level,
    offset: OffsetRule | None = None,
) -> FeaturePath:
    # Each point at the level of the column its block lands in once `offset`
    # moves it, which can be in a different frame from the point itself.
    sits = _offset_rule(offset)
    relevelled = FeaturePath(rules=())
    for point in path:
        here = sits(point)
        relevelled.add_point(
            point,
            y_offset=level(point.position.x + here.x, point.position.z + here.z, point.path_index),
        )
    return relevelled


def face_connected(
    path: FeaturePath,
    offset: OffsetRule | None = None,
    continuous: bool = False,
    level: Level | None = None,
    support: Set[BlockPos] | None = None,
) -> FeaturePath:
    # Breaks every diagonal step into single-axis moves so consecutive blocks
    # share a face, as rails need. Only neighbouring path points are bridged
    # unless `continuous`. A bridge through `support` is preferred when the
    # straight one leaves it, and loops a join doubles back over are removed.
    sits = _offset_rule(offset)

    def between(previous, point, start, end, here):
        # At a quarter handover, walk the new quarter's travel axis first.
        along_x, along_z = forward(quarter_of(point.laid_at))
        bridge = _relevelled(
            _bridge(
                start, end,
                z_first=previous.layout != point.layout and abs(along_z) > abs(along_x),
            ),
            point, here, level,
        )
        if support is not None and any(p not in support for p in bridge):
            bridge = _supported_bridge(start, end, support) or bridge
        return bridge

    filled = _filled(path, sits, continuous, between)
    simple = FeaturePath(rules=())
    for point in without_detours(filled, lambda point: _world(point) + sits(point)):
        simple.add_point(point)
    return simple


def eight_connected(
    path: FeaturePath,
    offset: OffsetRule | None = None,
    continuous: bool = False,
    level: Level | None = None,
) -> FeaturePath:
    # Fills every gap wider than a block with a diagonal line, so a painted
    # line touches itself at least at corners.
    def between(previous, point, start, end, here):
        return _relevelled(line(start, end, LineMode.CHEBYSHEV)[1:-1], point, here, level)

    return _filled(path, _offset_rule(offset), continuous, between)


def _filled(
    path: FeaturePath,
    sits: OffsetRule,
    continuous: bool,
    between: Callable[..., Sequence[BlockPos]],
) -> FeaturePath:
    # `path` with the blocks `between` returns inserted before each point,
    # lined up where the blocks land (level plus offset) and given back as
    # points of the later one.
    filled = FeaturePath(rules=())
    previous = None
    for point in path:
        if previous is not None and (
            continuous or abs(point.path_index - previous.path_index) <= 1
        ):
            here = sits(point)
            start = _world(previous) + sits(previous)
            end = _world(point) + here
            for block in between(previous, point, start, end, here):
                landed = block - here
                filled.add_point(
                    point,
                    position=BlockPos(landed.x, point.position.y, landed.z),
                    y_offset=landed.y - point.position.y,
                )
        filled.add_point(point)
        previous = point
    return filled


def _relevelled(
    blocks: Sequence[BlockPos],
    point: FeaturePathPoint,
    here: BlockPos,
    level: Level | None,
) -> tuple[BlockPos, ...]:
    # Each block at its own column's level, since a frame boundary can cross
    # between the two ends.
    if level is None:
        return tuple(blocks)
    return tuple(
        BlockPos(p.x, point.position.y + here.y + level(p.x, p.z, point.path_index), p.z)
        for p in blocks
    )


def without_detours(
    items: Iterable[_Item], key: Callable[[_Item], Hashable] = lambda item: item
) -> list[_Item]:
    # On revisiting a key, drops everything since its first visit.
    kept: list[_Item] = []
    seen: dict[Hashable, int] = {}
    for item in items:
        cell = key(item)
        if cell in seen:
            first = seen[cell]
            for removed in kept[first + 1:]:
                seen.pop(key(removed), None)
            del kept[first + 1:]
            continue
        seen[cell] = len(kept)
        kept.append(item)
    return kept


def _bridge(
    start: BlockPos, end: BlockPos, *, z_first: bool = False
) -> tuple[BlockPos, ...]:
    # The blocks a face-connected run needs between two positions. One along
    # and one up is a rail slope and needs none.
    along = abs(end.x - start.x) + abs(end.z - start.z)
    if along <= 1 and abs(end.y - start.y) <= 1:
        return ()
    if z_first:
        swapped = line(
            BlockPos(start.z, start.y, start.x),
            BlockPos(end.z, end.y, end.x),
            LineMode.RAIL,
        )
        return tuple(BlockPos(p.z, p.y, p.x) for p in swapped[1:-1])
    return tuple(line(start, end, LineMode.RAIL)[1:-1])


def _supported_bridge(
    start: BlockPos, end: BlockPos, support: Set[BlockPos]
) -> tuple[BlockPos, ...] | None:
    # The shortest face-connected walk through `support` cells, within a
    # block of the box the ends span.
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


def _offset_rule(offset: OffsetRule | None) -> OffsetRule:
    return (lambda point: _NOWHERE) if offset is None else offset


def _world(point: FeaturePathPoint) -> BlockPos:
    # Where a point's blocks land: its height above the path plus its level.
    return BlockPos(
        point.position.x, point.position.y + point.y_offset, point.position.z
    )


def span_of(schedule: SectionSchedule, path_index: int) -> int:
    for index, span in enumerate(schedule.spans):
        if span.contains(path_index):
            return index
    raise ValueError(f"Path point {path_index} is in no span of the schedule")


def rewalk(path: FeaturePath, rules: Iterable[FeaturePathRule]) -> FeaturePath:
    # The run re-filtered by `rules`, always keeping its last position.
    rules = tuple(rules)
    if rules and len(path):
        rules = (either(*rules, ending_at(path[-1].position)),)
    walked = FeaturePath(rules)
    for point in path:
        walked.add_point(point)
    return walked


def dashed(
    path: FeaturePath, on: int, off: int, by_index: bool = True
) -> FeaturePath:
    # Keeps `on` points (or blocks, measured in plan) then skips `off`.
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
            travelled += hypot(point.position.x - previous.x, point.position.z - previous.z)
        previous = point.position
        along = index if by_index else travelled
        if along % cycle < on:
            dashes.add_point(point)
    return dashes
