from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...Core.CrossSection.schedule import SectionSchedule
from ...Core.Path.boundary import Boundary
from ...Core.Path.elevation import ConstantElevation, MultiElevationStep
from ...Core.Path.path import Path
from ...Core.Path.path_frame import PathFrame
from .boundaries import (
    LateralReach,
    Orientation,
    boundary_at,
    orthogonal,
    validate_frames,
)


@dataclass(frozen=True, slots=True)
class ElevationBreak:
    # From `path_index` onward the path is held at `y`, until the next break.
    path_index: int
    y: int


def elevation_breaks(path: Path) -> list[ElevationBreak]:
    # A break wherever the path proper changes height. No frame can open at the
    # last point, so a change there keeps the level before it.
    points = path.path_points
    y = points[path.first_index].position.y
    breaks = []
    for index in range(path.first_index + 1, path.last_index):
        point_y = points[index].position.y
        if point_y != y:
            breaks.append(ElevationBreak(index, point_y))
            y = point_y
    return breaks


def span_stations(schedule: SectionSchedule, path: Path) -> list[int]:
    # Points on the path proper where the delivered section changes: where a
    # span starts, or where one ends before a gap. Section frames need a
    # boundary at each, so different sections meet on a seam.
    first = path.first_index
    last = path.last_index
    stations = set()
    for span in schedule.spans:
        stations.add(span.path_start)
        if schedule.at(span.path_stop + 1) is None:
            stations.add(span.path_stop + 1)
    return sorted(index for index in stations if first < index < last)


def check_padding(path: Path, lead_in: int, lead_out: int) -> None:
    # Pasting with a lead borrows points beyond each frame. The outer frames
    # borrow them from the path's padding, so the padding must be that long.
    if lead_in < 0 or lead_out < 0:
        raise ValueError("lead_in and lead_out must be non-negative")
    if path.lead_in < lead_in or path.lead_out < lead_out:
        raise ValueError(
            f"A lead of {lead_in} in and {lead_out} out needs at least that much "
            f"padding; the path has {path.lead_in} in and {path.lead_out} out"
        )


def build_frames(
    path: Path,
    start_y: int,
    breaks: Sequence[ElevationBreak] = (),
    *,
    orientation: Orientation = orthogonal,
    reach: LateralReach | None = None,
    lead_in: int = 0,
    lead_out: int = 0,
) -> list[PathFrame]:
    # Frames tile the path proper. The first starts at its first point, the
    # last ends at its last point, and every break closes one frame and opens
    # the next on a single boundary object they share.
    #
    # Each frame is the clip for whatever is pasted in it, so anything thrown
    # past its boundaries never exists in the world. `lead_in` and `lead_out`
    # are the lead that will be pasted with (see `frames_for_index` in
    # path_frame.py); they are only checked against the path's padding here.
    if not path.bounded_points:
        raise ValueError("Cannot frame a path with no points")
    check_padding(path, lead_in, lead_out)

    first = path.first_index
    last = path.last_index

    # (path index a frame starts at, its level)
    starts: list[tuple[int, int]] = [(first, start_y)]
    previous_index = first
    for step in breaks:
        if step.path_index <= previous_index:
            raise ValueError(
                f"Elevation break at point {step.path_index} does not come after "
                f"point {previous_index}"
            )
        if step.path_index >= last:
            raise ValueError(
                f"Elevation break at point {step.path_index} must come before the "
                f"path's last point {last}"
            )
        previous_index = step.path_index
        # A break that keeps the current level needs no boundary.
        if step.y != starts[-1][1]:
            starts.append((step.path_index, step.y))

    stops = [index for index, _ in starts[1:]] + [last]
    boundaries = {
        index: boundary_at(path, index, orientation)
        for index in {first, last, *stops}
    }
    frames = _frames_between(
        boundaries,
        [(start, stop, y) for (start, y), stop in zip(starts, stops)],
    )
    validate_frames(path, frames, reach)
    return frames


def elevation_from_frames(
    frames: Sequence[PathFrame],
) -> ConstantElevation | MultiElevationStep:
    # The frames' levels as one column-wise elevation, for anything placed as
    # a single piece across several frames. It takes the frames' boundaries as
    # they are, so it is orthogonal exactly when the frames were built so.
    #
    # It has no ends: past the path it holds the first and last levels. Clip
    # whatever uses it to the path's bounds.
    if not frames:
        raise ValueError("No frames to take an elevation from")
    if len(frames) == 1:
        return ConstantElevation(frames[0].y_offset)
    return MultiElevationStep(
        y_first=frames[0].y_offset,
        steps=tuple((frame.start_boundary, frame.y_offset) for frame in frames[1:]),
    )


def build_section_frames(
    path: Path,
    elevation_frames: Sequence[PathFrame],
    stations: Sequence[int] | None = None,
    *,
    orientation: Orientation = orthogonal,
    reach: LateralReach | None = None,
) -> list[PathFrame]:
    # Frames with a boundary at every station, so consecutive sections meet on
    # a boundary instead of overlapping. Stations default to every point on
    # the path proper; the path's first and last points are always stations.
    #
    # Every elevation boundary must be a station. Each section frame then lies
    # inside exactly one elevation frame and takes its level, so a single clip
    # is enough. Boundaries at elevation breaks are the elevation frames' own
    # objects; the rest are oriented with `orientation`.
    validate_frames(path, elevation_frames)
    first = path.first_index
    last = path.last_index
    if elevation_frames[0].path_start != first or elevation_frames[-1].path_stop != last:
        raise ValueError("Elevation frames do not cover the path proper")

    if stations is None:
        stations = range(first, last + 1)
    points = list(stations)
    for before, after in zip(points, points[1:]):
        if after <= before:
            raise ValueError(f"Station {after} does not come after station {before}")
    if points and (points[0] < first or points[-1] > last):
        raise ValueError(
            f"Stations must lie on the path proper, points {first}-{last}"
        )
    if not points or points[0] != first:
        points.insert(0, first)
    if points[-1] != last:
        points.append(last)
    if len(points) < 2:
        raise ValueError("Cannot frame a path with a single point")

    station_set = set(points)
    missing = [
        frame.path_start for frame in elevation_frames[1:]
        if frame.path_start not in station_set
    ]
    if missing:
        raise ValueError(f"Elevation breaks at points {missing} are not stations")

    boundaries: dict[int, Boundary] = {}
    for frame in elevation_frames:
        boundaries[frame.path_start] = frame.start_boundary
        boundaries[frame.path_stop] = frame.end_boundary
    for station in points:
        if station not in boundaries:
            boundaries[station] = boundary_at(path, station, orientation)

    spans = []
    level = iter(elevation_frames)
    current = next(level)
    for start, stop in zip(points, points[1:]):
        while start >= current.path_stop:
            current = next(level)
        spans.append((start, stop, current.y_offset))

    frames = _frames_between(boundaries, spans)
    validate_frames(path, frames, reach)
    return frames


def _frames_between(
    boundaries: dict[int, Boundary],
    spans: Sequence[tuple[int, int, int]],
) -> list[PathFrame]:
    # Frames over consecutive (start, stop, level) spans, sharing boundaries.
    return [
        PathFrame(
            index=position,
            path_start=start,
            path_stop=stop,
            start_boundary=boundaries[start],
            end_boundary=boundaries[stop],
            y_offset=y,
        )
        for position, (start, stop, y) in enumerate(spans)
    ]
