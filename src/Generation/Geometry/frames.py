from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import NamedTuple

from ...Core.CrossSection.schedule import SectionSchedule
from ...Core.Path.elevation import ConstantElevation, MultiElevationStep
from ...Core.Path.path import Path
from ...Core.Path.path_frame import PathFrame, frames_for_index
from .boundaries import (
    LateralReach,
    Orientation,
    boundary_at,
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
    "frames_for_index",
    "span_stations",
]


class ElevationBreak(NamedTuple):
    # The path point a level changes at, and the level it changes to. The
    # point belongs to the frame after the break, as every boundary's does.

    path_index: int
    y_offset: int


def check_padding(path: Path, lead_in: int = 0, lead_out: int = 0) -> None:
    # A lead is pasted from padding, so there has to be padding to paste from.
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
    # Where the path proper changes height. A change at its last point has no
    # frame after it to belong to, so it is not a break.
    points = path.bounded_points
    breaks = []
    for offset, (before, after) in enumerate(zip(points, points[1:])):
        index = path.first_index + offset + 1
        if after.position.y != before.position.y and index != path.last_index:
            breaks.append(ElevationBreak(index, after.position.y))
    return breaks


def span_stations(schedule: SectionSchedule, path: Path) -> list[int]:
    # The points on the path proper where the section changes: where a span
    # starts, and where the path runs on past one that has ended. The path's
    # own ends are not changes, since nothing is pasted beyond them.
    stations = set()
    for span in schedule.spans:
        stations.add(span.path_start)
        stations.add(span.path_stop + 1)
    return sorted(
        station
        for station in stations
        if path.first_index < station < path.last_index
    )


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
    # One frame per level, split wherever the path proper changes height. The
    # outer boundaries are the path's own bounds, and each break contributes
    # the one boundary the frames either side of it share.
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
        # A break back to the level already held changes nothing, so it adds no
        # boundary -- but it is still a break, and the ones after it follow it.
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
    # Frames split at every station as well as at every change of level, each
    # at the level of the elevation frame it lies in. Stations default to every
    # point of the path proper, which is the finest a path can be cut.
    #
    # A boundary an elevation frame already placed is used as it is, so the two
    # runs of frames seam in the same places.
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
    validate_frames(path, frames, _half_width(half_width, reach))
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


def _half_width(
    half_width: float | None,
    reach: LateralReach | None,
) -> float | LateralReach | None:
    # A reach is kept as it is rather than flattened to its widest side: which
    # side a section reaches on is what decides whether a bend is too tight.
    return reach if half_width is None else half_width


def _level_of(frames: Sequence[PathFrame], start: int, stop: int) -> int:
    for frame in frames:
        if frame.path_start <= start and stop <= frame.path_stop:
            return frame.y_offset
    raise ValueError(f"Points {start}-{stop} lie in no elevation frame")
