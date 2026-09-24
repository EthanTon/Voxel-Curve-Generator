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
    # The walking span; padding points belong to no span of their own.
    span: int = 0


def walk_schedule(
    schedule: SectionSchedule,
    path: Path,
    frames: Sequence[PathFrame] | None = None,
    lead_in: int = 0,
    lead_out: int = 0,
    *,
    extend_spans: bool = False,
) -> Iterator[ScheduledPoint]:
    # Each scheduled point with its section and the frames its lead reaches.
    # Spans at the path ends run on into the padding by the lead; points in
    # no frame are skipped.
    points = path.path_points
    for number, span in enumerate(schedule.spans):
        for index in range(*_walked(span, path, lead_in, lead_out, extend_spans)):
            if frames is None:
                yield ScheduledPoint(index, points[index], span.section, (), number)
                continue
            reached = tuple(frames_for_index(frames, index, lead_in, lead_out))
            if reached:
                yield ScheduledPoint(
                    index, points[index], span.section, reached, number
                )


def _walked(
    span: SectionSpan, path: Path, lead_in: int, lead_out: int,
    extend_spans: bool = False,
) -> tuple[int, int]:
    # The half-open range of points a span walks; `extend_spans` extends
    # interior handovers by the lead too.
    start, stop = span.path_start, span.path_stop
    if extend_spans or start <= path.first_index:
        start = max(0, start - lead_in)
    if extend_spans or stop >= path.last_index:
        stop = min(len(path.path_points) - 1, stop + lead_out)
    return start, stop + 1


def lead_for(path: Path, reach: float) -> int:
    # Points of lead covering `reach` blocks at the tightest point spacing.
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
