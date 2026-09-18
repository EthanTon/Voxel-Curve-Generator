from __future__ import annotations

from collections.abc import Iterator, Sequence
from math import ceil, hypot
from typing import NamedTuple

from ...Core.CrossSection.cross_section import CrossSection
from ...Core.CrossSection.schedule import SectionSchedule
from ...Core.Path.path import Path, PathPoint
from ...Core.Path.path_frame import PathFrame, frames_for_index


class ScheduledPoint(NamedTuple):
    index: int
    point: PathPoint
    section: CrossSection
    # The frames this point's section belongs in; empty when walked without frames.
    frames: tuple[PathFrame, ...]


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
    points = path.path_points
    for span in schedule.spans:
        for index in range(span.path_start, span.path_stop + 1):
            if frames is None:
                yield ScheduledPoint(index, points[index], span.section, ())
                continue
            reached = tuple(frames_for_index(frames, index, lead_in, lead_out))
            if reached:
                yield ScheduledPoint(index, points[index], span.section, reached)


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
