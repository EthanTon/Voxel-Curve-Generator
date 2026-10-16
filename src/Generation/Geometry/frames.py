from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Sequence
from math import sqrt
from typing import NamedTuple

from ...Core.CrossSection.schedule import SectionSchedule
from ...Core.Path.elevation import ConstantElevation, MultiElevationStep
from ...Core.Path.path import Path
from ...Core.Path.path_frame import PathFrame, frames_for_index
from .boundaries import (
    LateralReach,
    Orientation,
    boundary_at,
    crosses_within,
    orthogonal,
    path_bounds,
    validate_frames,
)

__all__ = [
    "ElevationBreak",
    "build_frames",
    "build_section_frames",
    "check_padding",
    "elevation_breaks",
    "elevation_from_frames",
    "level_near",
    "frames_for_index",
    "span_stations",
    "heading_stations",
]


class ElevationBreak(NamedTuple):
    # The point where the level changes (it belongs to the frame after).

    path_index: int
    y_offset: int


def check_padding(path: Path, lead_in: int = 0, lead_out: int = 0) -> None:
    # Leads are pasted from padding, so the padding must be long enough.
    if lead_in < 0 or lead_out < 0:
        raise ValueError(
            f"Leads must be non-negative, got lead_in={lead_in}, lead_out={lead_out}"
        )
    before = path.first_index
    after = len(path.path_points) - 1 - path.last_index
    if lead_in > before:
        raise ValueError(
            f"A lead of {lead_in} needs {lead_in} points of padding before the "
            f"path, and there are {before}"
        )
    if lead_out > after:
        raise ValueError(
            f"A lead of {lead_out} needs {lead_out} points of padding after the "
            f"path, and there are {after}"
        )


def elevation_breaks(path: Path) -> list[ElevationBreak]:
    # Where the path proper changes height, except at its last point.
    points = path.bounded_points
    breaks = []
    for offset, (before, after) in enumerate(zip(points, points[1:])):
        index = path.first_index + offset + 1
        if after.position.y != before.position.y and index != path.last_index:
            breaks.append(ElevationBreak(index, after.position.y))
    return breaks


def span_stations(schedule: SectionSchedule, path: Path) -> list[int]:
    # Interior points where a span starts or the one before has ended.
    stations = set()
    for span in schedule.spans:
        stations.add(span.path_start)
        stations.add(span.path_stop + 1)
    return sorted(
        station
        for station in stations
        if path.first_index < station < path.last_index
    )


def heading_stations(
    path: Path, existing: Iterable[int] = (),
    half_width: float | LateralReach | None = None,
) -> list[int]:
    # Long level spans may turn back toward their start. Split their planes
    # as the heading turns, before a far end plane can clip an earlier leg.
    facing = path.path_points[path.first_index].heading.unit()
    bounds = path_bounds(path)
    indices = [path.first_index, *sorted(existing), path.last_index]
    planes = [bounds.start, *(boundary_at(path, i) for i in indices[1:-1]), bounds.end]
    stations = []
    for index in range(path.first_index + 1, path.last_index):
        direction = path.path_points[index].heading.unit()
        if facing.x * direction.x + facing.z * direction.z <= sqrt(0.5):
            at = bisect_right(indices, index)
            if indices[at - 1] == index:
                facing = direction
                continue
            plane = boundary_at(path, index)
            # A sharp corner needs room for its two layouts. Wait until
            # the added plane no longer crosses its neighbours in the bed.
            if half_width is not None and (
                crosses_within(planes[at - 1], plane, half_width)
                or crosses_within(plane, planes[at], half_width)
            ):
                continue
            stations.append(index)
            indices.insert(at, index)
            planes.insert(at, plane)
            facing = direction
    return stations


def build_frames(
    path: Path,
    y_offset: int,
    breaks: Iterable[ElevationBreak] = (),
    *,
    orientation: Orientation = orthogonal,
    half_width: float | LateralReach | None = None,
    reach: LateralReach | None = None,
    lead_in: int = 0,
    lead_out: int = 0,
) -> list[PathFrame]:
    # One frame per level, split at each break that changes the level.
    if len(path.bounded_points) < 2:
        raise ValueError("A path needs at least two points to be framed")
    check_padding(path, lead_in, lead_out)

    bounds = path_bounds(path, orientation)
    kept: list[ElevationBreak] = []
    level = y_offset
    previous: int | None = None
    for each in breaks:
        if not path.first_index < each.path_index < path.last_index:
            raise ValueError(
                f"Break at point {each.path_index} is not inside the path proper "
                f"{path.first_index}-{path.last_index}"
            )
        if previous is not None and each.path_index <= previous:
            raise ValueError(
                f"Breaks must come in order along the path; {each.path_index} "
                f"follows {previous}"
            )
        previous = each.path_index
        if each.y_offset != level:
            kept.append(each)
            level = each.y_offset

    indices = [path.first_index, *(each.path_index for each in kept), path.last_index]
    levels = [y_offset, *(each.y_offset for each in kept)]
    planes = [
        bounds.start,
        *(boundary_at(path, each.path_index, orientation) for each in kept),
        bounds.end,
    ]
    frames = [
        PathFrame(index, indices[index], indices[index + 1], *planes[index : index + 2], levels[index])
        for index in range(len(levels))
    ]
    validate_frames(path, frames, _half_width(half_width, reach))
    return frames


def build_section_frames(
    path: Path,
    elevation_frames: Sequence[PathFrame],
    stations: Iterable[int] | None = None,
    *,
    orientation: Orientation = orthogonal,
    half_width: float | LateralReach | None = None,
    reach: LateralReach | None = None,
) -> list[PathFrame]:
    # Elevation frames split further at every station (default: every
    # interior point), reusing the elevation frames' boundaries.
    validate_frames(path, elevation_frames, _half_width(half_width, reach))
    shared = {elevation_frames[0].path_start: elevation_frames[0].start_boundary}
    for frame in elevation_frames:
        shared[frame.path_stop] = frame.end_boundary

    if stations is None:
        stations = range(path.first_index + 1, path.last_index)
    stations = list(stations)

    previous = path.first_index
    for station in stations:
        if not previous < station < path.last_index:
            raise ValueError(
                f"Station {station} is not between {previous} and the path's "
                f"last point {path.last_index}"
            )
        previous = station
    missing = {frame.path_start for frame in elevation_frames[1:]} - set(stations)
    if missing:
        raise ValueError(
            f"The path changes level at {sorted(missing)}, which must be "
            "stations too, or a section frame would span two levels"
        )

    indices = [path.first_index, *stations, path.last_index]
    planes = [
        shared[index] if index in shared else boundary_at(path, index, orientation)
        for index in indices
    ]
    frames = [
        PathFrame(
            index,
            indices[index],
            indices[index + 1],
            *planes[index : index + 2],
            _level_of(elevation_frames, indices[index], indices[index + 1]),
        )
        for index in range(len(indices) - 1)
    ]
    validate_frames(path, frames, _half_width(half_width, reach), holding=True)
    return frames


class LevelNear:
    # A column's level, found by walking from the frame of the path point it
    # belongs to across the boundaries it is past. Asking every boundary
    # would misread columns on a loop, where far planes face the same way.

    def __init__(self, frames: Sequence[PathFrame]) -> None:
        if not frames:
            raise ValueError("No frames to take a level from")
        self.frames = tuple(sorted(frames, key=lambda frame: frame.path_start))
        self._starts = [frame.path_start for frame in self.frames]

    def __call__(self, x: float, z: float, index: int) -> int:
        frames = self.frames
        at = max(0, bisect_right(self._starts, index) - 1)
        while at + 1 < len(frames) and frames[at + 1].start_boundary.after(x, z):
            at += 1
        while at > 0 and not frames[at].start_boundary.after(x, z):
            at -= 1
        return frames[at].y_offset


def level_near(frames: Sequence[PathFrame]) -> LevelNear:
    return LevelNear(frames)


def elevation_from_frames(
    frames: Sequence[PathFrame],
) -> ConstantElevation | MultiElevationStep:
    # The frames' levels as one elevation over every column; unbounded, so
    # clip its users to the path.
    if not frames:
        raise ValueError("No frames to take an elevation from")
    if len(frames) == 1:
        return ConstantElevation(frames[0].y_offset)
    return MultiElevationStep(
        y_first=frames[0].y_offset,
        steps=tuple((frame.start_boundary, frame.y_offset) for frame in frames[1:]),
    )


def _half_width(
    half_width: float | None,
    reach: LateralReach | None,
) -> float | LateralReach | None:
    return reach if half_width is None else half_width


def _level_of(frames: Sequence[PathFrame], start: int, stop: int) -> int:
    for frame in frames:
        if frame.path_start <= start and stop <= frame.path_stop:
            return frame.y_offset
    raise ValueError(f"Points {start}-{stop} lie in no elevation frame")
