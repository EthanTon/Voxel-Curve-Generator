from __future__ import annotations

from collections.abc import Iterator, Sequence
from math import ceil, hypot
from typing import NamedTuple

from ...Core.CrossSection.cross_section import CrossSection
from ...Core.CrossSection.schedule import SectionSchedule, SectionSpan
from ...Core.Path.path import Path, PathPoint
from ...Core.Path.path_frame import PathFrame, frames_for_index


class ScheduledPoint(NamedTuple):
    index: int
    point: PathPoint
    section: CrossSection
    # The frames this point's section belongs in; empty when walked without frames.
    frames: tuple[PathFrame, ...]
    # Which span of the schedule put it here. Carried rather than looked up:
    # a point in the path's padding is walked by the span beside it and is
    # inside no span at all.
    span: int = 0


def walk_schedule(
    schedule: SectionSchedule,
    path: Path,
    frames: Sequence[PathFrame] | None = None,
    lead_in: int = 0,
    lead_out: int = 0,
) -> Iterator[ScheduledPoint]:
    # Sequential access, for generation. Gaps in the schedule are skipped
    # entirely. With frames, each point comes with the frames its lead
    # reaches, and points that reach none -- padding beyond the lead -- are
    # skipped too.
    #
    # A span that reaches an end of the path runs on into the padding there.
    # That is what the padding is for: a section at the last point reaches
    # past it, and without points to paste it at, the run stops a section
    # short of where it was asked to and the end comes out unfinished. It
    # runs no further than the lead, since `frames_for_index` widens a frame
    # by exactly that much and a point past every frame is dropped.
    points = path.path_points
    for number, span in enumerate(schedule.spans):
        for index in range(*_walked(span, path, lead_in, lead_out)):
            if frames is None:
                yield ScheduledPoint(index, points[index], span.section, (), number)
                continue
            reached = tuple(frames_for_index(frames, index, lead_in, lead_out))
            if reached:
                yield ScheduledPoint(
                    index, points[index], span.section, reached, number
                )


def _walked(
    span: SectionSpan, path: Path, lead_in: int, lead_out: int
) -> tuple[int, int]:
    # The half-open range of points a span is walked over: its own, and the
    # padding beyond whichever end of the path it reaches.
    #
    # Only the padding. A span that stops inside the path stops there, or it
    # would paste itself over the span after it.
    start, stop = span.path_start, span.path_stop
    if start <= path.first_index:
        start = max(0, start - lead_in)
    if stop >= path.last_index:
        stop = min(len(path.path_points) - 1, stop + lead_out)
    return start, stop + 1


def lead_for(path: Path, reach: float) -> int:
    # Points of lead needed for sections reaching `reach` blocks along the
    # path, measured against the tightest spacing between consecutive points.
    if reach < 0:
        raise ValueError("reach must be non-negative")
    points = path.path_points
    steps = [
        hypot(b.position.x - a.position.x, b.position.z - a.position.z)
        for a, b in zip(points, points[1:])
    ]
    steps = [step for step in steps if step > 0.0]
    if not steps or reach == 0.0:
        return 0
    return ceil(reach / min(steps))
