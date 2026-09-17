from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ...Core.Path.boundary import Boundary, PathBounds
from ...Core.Path.path import Path
from ...Core.Path.path_frame import PathFrame
from ...Core.coordinates import Vec2


# Chooses the horizontal direction a boundary faces at a path point. Core
# boundaries accept any direction; the choice of orientation is made here.
Orientation = Callable[[Path, int], Vec2]


@dataclass(frozen=True, slots=True)
class LateralReach:
    # How far a cross section extends either side of the path, in blocks.
    # Left and right are relative to the direction a boundary faces, which is
    # the direction of travel for every orientation here. Sections need not be
    # symmetric, and a bend only crosses planes on its inside, so each side is
    # checked against its own reach.
    left: float
    right: float

    def __post_init__(self) -> None:
        if self.left < 0 or self.right < 0:
            raise ValueError(
                f"Lateral reach must be non-negative, got left {self.left} "
                f"and right {self.right}"
            )

    def covers(self, offset: float) -> bool:
        # Whether a signed offset along a boundary, positive to the right,
        # lies within the section.
        return -self.left <= offset <= self.right


def orthogonal(path: Path, index: int) -> Vec2:
    # Face along travel, so the plane is perpendicular to the path in plan
    # view. Only the horizontal projection of the gradient is used: its
    # vertical component is grade, which belongs to elevation, not the plane.
    point = path.path_points[index]
    if point.is_vertical:
        raise ValueError(
            f"Path point {index} at {point.position} has no horizontal heading; "
            "a boundary cannot be oriented orthogonally to it"
        )
    return point.heading


def boundary_at(path: Path, index: int, orientation: Orientation = orthogonal) -> Boundary:
    if not 0 <= index < len(path.path_points):
        raise IndexError(f"Path has no point {index}")
    position = path.path_points[index].position
    return Boundary.from_direction(
        Vec2(position.x, position.z),
        orientation(path, index),
        path_index=index,
    )


def path_bounds(path: Path, orientation: Orientation = orthogonal) -> PathBounds:
    # Boundaries at the first and last points of the path proper. Padding
    # points are neighbours when their gradients are computed, which is what
    # keeps these planes well oriented at the very ends.
    if not path.bounded_points:
        raise ValueError("Cannot bound a path with no points")
    return PathBounds(
        start=boundary_at(path, path.first_index, orientation),
        end=boundary_at(path, path.last_index, orientation),
    )


def crossing_offsets(a: Boundary, b: Boundary) -> tuple[float, float] | None:
    # Where two planes meet in plan view, as signed distances along each
    # plane from its own origin, positive to the right of the direction the
    # plane faces. None when they are parallel and never meet.
    if a.is_parallel_to(b):
        return None
    ax, az = -a.normal.z, a.normal.x
    bx, bz = -b.normal.z, b.normal.x
    wx = b.origin.x - a.origin.x
    wz = b.origin.z - a.origin.z
    det = bx * az - ax * bz
    along_a = (bx * wz - bz * wx) / det
    along_b = (ax * wz - az * wx) / det
    return along_a, along_b


def crosses_within(a: Boundary, b: Boundary, reach: LateralReach) -> bool:
    # True when the planes meet inside a cross-section of `reach` at either
    # boundary. On a bend tighter than the section reaches toward its inside,
    # the columns past that point fall in no frame, or in the wrong one.
    offsets = crossing_offsets(a, b)
    if offsets is None:
        return False
    along_a, along_b = offsets
    return reach.covers(along_a) or reach.covers(along_b)


def find_crossings(
    boundaries: Sequence[Boundary],
    reach: LateralReach,
) -> list[tuple[int, int]]:
    # Positions of consecutive boundaries whose planes cross within the section.
    return [
        (i, i + 1)
        for i in range(len(boundaries) - 1)
        if crosses_within(boundaries[i], boundaries[i + 1], reach)
    ]


def validate_frames(
    path: Path,
    frames: Sequence[PathFrame],
    reach: LateralReach | None = None,
) -> None:
    # Path-aware checks no Core type can make on its own. Raises on the first
    # problem found; pass `reach` to also reject planes crossing on bends.
    if not frames:
        raise ValueError("A path needs at least one frame")

    count = len(path.path_points)
    for position, frame in enumerate(frames):
        label = f"Frame {frame.index}"
        if frame.index != position:
            raise ValueError(f"{label} is at position {position} in the sequence")
        if frame.path_start < 0 or frame.path_stop >= count:
            raise ValueError(
                f"{label} covers points {frame.path_start}-{frame.path_stop}, "
                f"outside the path's {count} points"
            )
        for end, boundary, expected in (
            ("start", frame.start_boundary, frame.path_start),
            ("end", frame.end_boundary, frame.path_stop),
        ):
            if boundary.path_index is not None and boundary.path_index != expected:
                raise ValueError(
                    f"{label}: {end} boundary sits at point {boundary.path_index}, "
                    f"not {expected}"
                )
        if frame.is_degenerate:
            raise ValueError(f"{label} spans no columns")

    for before, after in zip(frames, frames[1:]):
        if before.path_stop != after.path_start:
            raise ValueError(
                f"Frames {before.index} and {after.index} do not meet: "
                f"{before.path_stop} != {after.path_start}"
            )
        if before.end_boundary != after.start_boundary:
            raise ValueError(
                f"Frames {before.index} and {after.index} do not share a boundary"
            )

    if reach is not None:
        planes = [frames[0].start_boundary, *(frame.end_boundary for frame in frames)]
        crossings = find_crossings(planes, reach)
        if crossings:
            first, second = crossings[0]
            raise ValueError(
                f"Boundaries {first} and {second} of the frame sequence cross within "
                f"{reach.left} left or {reach.right} right of the path; "
                f"{len(crossings)} crossing(s) in total"
            )
