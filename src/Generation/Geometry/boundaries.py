from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

from ...Core.Path.boundary import Boundary, PathBounds
from ...Core.Path.path import Path
from ...Core.Path.path_frame import PathFrame
from ...Core.coordinates import Vec2

# How the boundary at a path point is faced. Core holds the plane; which way it
# looks is Generation's decision, so every builder here takes one of these.
Orientation = Callable[[Path, int], Vec2]


class LateralReach(NamedTuple):
    # How far a section reaches either side of the path, in blocks. Boundaries
    # only have to agree over the columns a section can occupy, so this is what
    # sizes the crossing checks.

    left: float = 0.0
    right: float = 0.0

    @property
    def half_width(self) -> float:
        return max(self.left, self.right)


def orthogonal(path: Path, index: int) -> Vec2:
    # Square across the path: the point's own heading. Grade does not tilt the
    # plane, since a boundary is vertical whatever the path is climbing.
    point = path.path_points[index]
    if point.is_vertical:
        raise ValueError(
            f"Path point {index} has no horizontal heading to face a boundary along"
        )
    return point.heading.unit()


def boundary_at(
    path: Path,
    index: int,
    orientation: Orientation = orthogonal,
) -> Boundary:
    # The plane through one path point, facing the way `orientation` says.
    point = path.path_points[index]
    return Boundary.from_direction(
        (point.position.x, point.position.z),
        orientation(path, index),
        path_index=index,
    )


def path_bounds(path: Path, orientation: Orientation = orthogonal) -> PathBounds:
    # The columns the path proper owns: from the plane at its first point to
    # the plane at its last, so padding falls outside at either end.
    if not path.path_points:
        raise ValueError("An empty path has no bounds")
    return PathBounds(
        boundary_at(path, path.first_index, orientation),
        boundary_at(path, path.last_index, orientation),
    )


def crossing_offsets(first: Boundary, second: Boundary) -> tuple[float, float] | None:
    # Where two planes meet, as the distance along each from its own origin,
    # signed: a plane runs along the right of travel, so a positive offset is
    # a meeting to the right of the path and a negative one to its left.
    # Parallel planes never meet and have no offsets.
    if first.is_parallel_to(second):
        return None
    along_first = Vec2(-first.normal.z, first.normal.x)
    along_second = Vec2(-second.normal.z, second.normal.x)
    between = Vec2(
        second.origin.x - first.origin.x, second.origin.z - first.origin.z
    )
    turned = along_first.cross(along_second)
    return between.cross(along_second) / turned, between.cross(along_first) / turned


def reach_of(width: float | LateralReach) -> LateralReach:
    # A plain width reaches the same distance either side.
    return width if isinstance(width, LateralReach) else LateralReach(width, width)


def crosses_within(
    first: Boundary,
    second: Boundary,
    half_width: float | LateralReach,
) -> bool:
    # Whether two planes meet close enough to either origin to matter: a
    # meeting further out than a section reaches is not one it can see.
    #
    # The planes on a bend meet on its inside, so a section is measured by the
    # reach it has on the side the meeting is on. One reaching out over the
    # outside of a bend never meets anything, however far it reaches.
    reach = reach_of(half_width)
    offsets = crossing_offsets(first, second)
    if offsets is None:
        return False
    return all(
        abs(offset) <= (reach.right if offset > 0 else reach.left)
        for offset in offsets
    )


def find_crossings(
    boundaries: Sequence[Boundary],
    half_width: float | LateralReach,
) -> list[tuple[int, int]]:
    # The neighbouring pairs that meet within `half_width`. Only neighbours are
    # checked: planes further apart than that have frames between them.
    return [
        (index, index + 1)
        for index, (first, second) in enumerate(zip(boundaries, boundaries[1:]))
        if crosses_within(first, second, half_width)
    ]


def validate_frames(
    path: Path,
    frames: Sequence[PathFrame],
    half_width: float | LateralReach | None = None,
) -> None:
    # Everything a run of frames has to be before generation uses it: numbered
    # in order, tiling the path proper end to end, sharing each boundary with
    # its neighbour, and none of them empty.
    #
    # A boundary that came from a path point has to sit at the point its frame
    # starts or stops at; one built any other way is taken as given. With a
    # `half_width`, neighbouring boundaries must also stay apart over the
    # columns a section that wide can occupy.
    if not frames:
        raise ValueError("A path needs at least one frame")

    for position, frame in enumerate(frames):
        if frame.index != position:
            raise ValueError(
                f"Frame {position} is numbered {frame.index}; frames are numbered "
                "in the order they are given"
            )
        if frame.is_degenerate:
            raise ValueError(f"Frame {frame.index} spans no columns at all")
        if frame.path_start < path.first_index or frame.path_stop > path.last_index:
            raise ValueError(
                f"Frame {frame.index} covers points {frame.path_start}-"
                f"{frame.path_stop}, outside the path proper "
                f"{path.first_index}-{path.last_index}"
            )
        _check_boundary(frame, frame.start_boundary, frame.path_start, "starts")
        _check_boundary(frame, frame.end_boundary, frame.path_stop, "stops")

    if frames[0].path_start != path.first_index:
        raise ValueError(
            f"Frame 0 starts at point {frames[0].path_start}, not at the path's "
            f"first point {path.first_index}"
        )
    if frames[-1].path_stop != path.last_index:
        raise ValueError(
            f"Frame {frames[-1].index} stops at point {frames[-1].path_stop}, not "
            f"at the path's last point {path.last_index}"
        )

    for before, after in zip(frames, frames[1:]):
        if before.path_stop != after.path_start:
            raise ValueError(
                f"Frames {before.index} and {after.index} meet at points "
                f"{before.path_stop} and {after.path_start}, leaving a gap"
            )
        if before.end_boundary != after.start_boundary:
            raise ValueError(
                f"Frames {before.index} and {after.index} do not share the "
                "boundary between them"
            )

    if half_width is None:
        return
    boundaries = [frames[0].start_boundary, *(frame.end_boundary for frame in frames)]
    crossings = find_crossings(boundaries, half_width)
    if crossings:
        first, second = crossings[0]
        raise ValueError(
            f"Boundaries {first} and {second} cross within {half_width} blocks "
            "of the path, so the frames either side of them overlap"
        )


def _check_boundary(
    frame: PathFrame,
    boundary: Boundary,
    index: int,
    end: str,
) -> None:
    if boundary.path_index is not None and boundary.path_index != index:
        raise ValueError(
            f"Frame {frame.index} {end} at point {index}, but its boundary was "
            f"placed at point {boundary.path_index}"
        )
