from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import hypot
from typing import NamedTuple

from ...Core.CrossSection.annotations import FeatureAnnotation
from ...Core.CrossSection.schedule import SectionSchedule
from ...Core.CrossSection.turns import SectionTurn
from ...Core.Feature.feature_path import (
    DEFAULT_RULES,
    FeaturePath,
    FeaturePathRule,
    either,
    ending_at,
)
from ...Core.Path.path import Path
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
        turn = turns[step.index]
        oriented = step.section.orient(turn.turn, turn.direction)
        span = span_of(schedule, step.index)
        for frame in step.frames:
            for delivery in deliver(step.point, oriented, frame):
                run = runs.get((delivery.annotation.name, span))
                if run is None:
                    run = DeliveredPath(
                        delivery.annotation, span, FeaturePath(rules)
                    )
                    runs[delivery.annotation.name, span] = run
                for position in delivery.positions:
                    run.path.add(
                        position,
                        step.point.gradient,
                        turn.turn,
                        step.index,
                        frame.y_offset,
                        turn.direction,
                    )
    return tuple(run for run in runs.values() if len(run.path))


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
            )
    return dashes


def _apart(first: BlockPos, second: BlockPos) -> float:
    # Distance in plan: a run climbs, but a dash is measured along the ground.
    return hypot(second.x - first.x, second.z - first.z)
