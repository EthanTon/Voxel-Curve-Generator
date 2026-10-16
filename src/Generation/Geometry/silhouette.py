from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import ceil, hypot
from typing import TYPE_CHECKING

from ...Core.CrossSection.cross_section import CrossSection
from ...Core.CrossSection.schedule import SectionSchedule, SectionSpan
from ...Core.CrossSection.turns import heading
from ...Core.Feature.feature_path import FeaturePathPoint
from ...Core.Path.boundary import Boundary
from ...Core.Path.path import Path
from ...Core.Path.path_frame import PathFrame
from ...Core.World.block_world import ColumnMask
from ...Core.modes import StepMode
from ...Core.plan_index import NearestInPlan
from .boundaries import boundary_at

if TYPE_CHECKING:
    from ..Sweep.paths import DeliveredPath


class PathSilhouette:
    """Columns within `radius` of the path proper, each owned by its nearest
    station. One radius for every section keeps handovers consistent.

    Where the path passes over itself further apart than `height` (the
    sections' own height), each pass owns the column for itself: a station
    competes only with the stations of its own pass, those near it along the
    path or level with it.
    """

    def __init__(
        self, path: Path, radius: int,
        frames: Iterable[PathFrame] = (),
        height: int | None = None,
    ) -> None:
        if radius < 0 or not path.bounded_points:
            raise ValueError("a silhouette needs a path and a non-negative radius")
        self.path = path
        self.radius = radius
        self.frames = tuple(frames)
        self.height = height
        self._nearest = NearestInPlan(
            (point.position.x, point.position.z, index)
            for index, point in enumerate(path.path_points)
            if path.first_index <= index <= path.last_index
        )
        self._owners: dict[tuple[float, ...], int | None] = {}
        self._layers: dict[tuple[float, float], tuple[int, ...]] = {}
        self._frame_owners: dict[tuple[float, ...], PathFrame | None] = {}
        self._frame_candidates: dict[int, tuple[PathFrame, ...]] = {}
        self._span_masks: dict[tuple[int, int], SectionSilhouette] = {}
        self._frame_masks: dict[int, FrameSilhouette] = {}
        self._boundaries: dict[int, Boundary] = {}

    @classmethod
    def for_sections(
        cls, path: Path, sections: Iterable[CrossSection],
        frames: Iterable[PathFrame] = (),
        fittings: Iterable[int] = (),
    ) -> PathSilhouette:
        # `fittings`: the heights above the path of what rides the sections,
        # which a pass over another must clear as well.
        # Deferred: reach imports the sweep, which imports this module.
        from .reach import lateral_reach, longitudinal_reach

        sections = tuple(sections)
        lateral = lateral_reach(sections).half_width
        longitudinal = longitudinal_reach(sections)
        return cls(path, ceil(hypot(lateral, longitudinal)) + 1, frames, _height(sections, fittings))

    def owner(self, x: float, z: float, station: int | None = None) -> int | None:
        # The nearest station, or with `station` the nearest of its pass.
        column = (x, z)
        if column not in self._owners:
            index = self._nearest.nearest(x, z)
            point = self.path.path_points[index].position
            distance = (x - point.x) ** 2 + (z - point.z) ** 2
            self._owners[column] = index if distance <= self.radius ** 2 else None
        nearest = self._owners[column]
        if station is None or nearest is None or self.same_pass(nearest, station):
            return nearest
        key = (x, z, station)
        if key not in self._owners:
            self._owners[key] = self._nearest.nearest(
                x, z, lambda index: self.same_pass(index, station), self.radius ** 2
            )
        return self._owners[key]

    def same_pass(self, first: int, second: int) -> bool:
        # Near each other along the path, or close enough in height that
        # their sections would meet.
        if self.height is None or abs(first - second) <= 2 * self.radius + 2:
            return True
        points = self.path.path_points
        return abs(points[first].position.y - points[second].position.y) < self.height

    def layers(self, x: float, z: float) -> tuple[int, ...]:
        # The owner of each pass over the column, the nearest pass first.
        column = (x, z)
        if column not in self._layers:
            owners = []
            found = self.owner(x, z)
            while found is not None:
                owners.append(found)
                found = self._nearest.nearest(
                    x, z, lambda index: not any(self.same_pass(index, each) for each in owners),
                    self.radius ** 2,
                )
            self._layers[column] = tuple(owners)
        return self._layers[column]

    def column(self, x: int, z: int, station: int) -> tuple[int, ...]:
        # A column's key in per-column tables as `station`'s pass sees it:
        # the column itself on the nearest pass, else with its pass's owner.
        owner = self.owner(x, z, station)
        return (x, z) if owner == self.owner(x, z) else (x, z, owner)

    def for_span(self, span: SectionSpan) -> SectionSilhouette:
        key = (span.path_start, span.path_stop)
        if key not in self._span_masks:
            self._span_masks[key] = SectionSilhouette(self, *key)
        return self._span_masks[key]

    def for_frame(self, frame: PathFrame) -> FrameSilhouette:
        if frame.index not in self._frame_masks:
            self._frame_masks[frame.index] = FrameSilhouette(self, frame)
        return self._frame_masks[frame.index]

    def clip(self, point: FeaturePathPoint) -> ColumnMask:
        frame = self.frame_at(point.position.x, point.position.z, point.path_index)
        if frame is None:
            raise ValueError(
                f"{tuple(point.position)} is outside the path's framed silhouette"
            )
        return self.for_frame(frame)

    def frame_at(self, x: float, z: float, station: int | None = None) -> PathFrame | None:
        """The frame owning a column: the nearest frame to the column's station
        whose planes contain it, so a far turn of the path cannot claim it.
        With `station`, the frame owning it on that station's pass.
        """
        owner = self.owner(x, z, station)
        column = (x, z) if owner == self.owner(x, z) else (x, z, owner)
        if column not in self._frame_owners:
            station = owner
            chosen = None
            if station is not None:
                if station not in self._frame_candidates:

                    def gap(frame: PathFrame) -> int:
                        return max(frame.path_start - station, 0, station - frame.path_stop)

                    self._frame_candidates[station] = tuple(sorted(
                        (frame for frame in self.frames if gap(frame) <= self.radius + 1),
                        key=lambda frame: (gap(frame), frame.index),
                    ))
                for frame in self._frame_candidates[station]:
                    if frame.contains(x, z):
                        chosen = frame
                        break
            self._frame_owners[column] = chosen
        return self._frame_owners[column]

    def boundary(self, station: int) -> Boundary:
        if station not in self._boundaries:
            self._boundaries[station] = boundary_at(self.path, station)
        return self._boundaries[station]


def _height(sections: Sequence[CrossSection], fittings: Iterable[int] = ()) -> int | None:
    # How tall the sections and their fittings stand, bottom block to top.
    heights = [
        local.y + annotation.position.y - section.origin.y
        for section in sections
        for annotation in section.points
        if getattr(annotation.feature, "blocks", None) is not None
        for local, _ in annotation.feature.blocks.iter_block_ids()
    ]
    if not heights:
        return None
    heights.extend(fittings)
    return max(heights) - min(heights) + 1


@dataclass(frozen=True, slots=True)
class SectionSilhouette:
    path: PathSilhouette
    start: int
    stop: int
    _cache: dict[tuple[float, float], bool] = field(default_factory=dict, compare=False, repr=False)

    def contains(self, x: float, z: float) -> bool:
        column = (x, z)
        if column not in self._cache:
            self._cache[column] = self._contains(*column)
        return self._cache[column]

    def _contains(self, x: float, z: float) -> bool:
        # On any pass over the column.
        return any(self._on(x, z, owner) for owner in self.path.layers(x, z))

    def _on(self, x: float, z: float, nearest: int) -> bool:
        if self.path.frames:
            frame = self.path.frame_at(x, z, nearest)
            if frame is None:
                return False
            if frame.path_start > self.stop or frame.path_stop < self.start:
                return False
            # A frame not cut at this span's ends falls back to the planes.
            if frame.path_start >= self.start and (
                frame.path_stop <= self.stop
                or (frame.path_stop == self.stop + 1
                    and frame.path_stop < self.path.path.last_index)
            ):
                return True
        return (
            self._after(self.start, nearest, x, z)
            and not self._after(self.stop + 1, nearest, x, z)
        )

    def _after(self, station: int, nearest: int, x: float, z: float) -> bool:
        path = self.path.path
        if station <= path.first_index:
            return True
        if station > path.last_index:
            return False
        # The plane decides only near the station; elsewhere the nearest
        # station does, so a later part of a loop is not reassigned.
        origin = path.path_points[station].position
        local = (
            abs(nearest - station) <= self.path.radius + 1
            and (x - origin.x) ** 2 + (z - origin.z) ** 2
            <= (self.path.radius + 1) ** 2
        )
        return self.path.boundary(station).after(x, z) if local else nearest >= station


@dataclass(frozen=True, slots=True)
class FrameSilhouette:
    path: PathSilhouette
    frame: PathFrame

    def contains(self, x: float, z: float) -> bool:
        return self.path.frame_at(x, z, self.frame.path_start) == self.frame


@dataclass(frozen=True, slots=True)
class IntersectMask:
    first: ColumnMask
    second: ColumnMask

    def contains(self, x: float, z: float) -> bool:
        return self.first.contains(x, z) and self.second.contains(x, z)


@dataclass(frozen=True, slots=True)
class OutsideMask:
    # Columns in none of `masks`.
    masks: tuple[ColumnMask, ...]

    def contains(self, x: float, z: float) -> bool:
        return not any(mask.contains(x, z) for mask in self.masks)


@dataclass(frozen=True, slots=True)
class SectionColumnMask:
    # Only the columns owned by (section, span); `key` reads a column as one
    # pass of the path sees it (see `PathSilhouette.column`).
    owners: Mapping[tuple[int, ...], tuple[str, int]]
    section: str
    span: int
    key: Callable[[int, int], tuple[int, ...]] | None = None

    def contains(self, x: float, z: float) -> bool:
        column = (x, z) if self.key is None else self.key(x, z)
        return self.owners.get(column) == (self.section, self.span)


@dataclass(frozen=True, slots=True)
class AnnotationColumnMask:
    # Only the world columns `annotation` owns among its section's columns.
    owners: Mapping[tuple[int, ...], str]
    annotation: str
    key: Callable[[int, int], tuple[int, ...]] | None = None

    def contains(self, x: float, z: float) -> bool:
        column = (x, z) if self.key is None else self.key(x, z)
        return self.owners.get(column, self.annotation) == self.annotation


def annotation_column_owners(
    runs: Sequence[DeliveredPath], path: Path, silhouette: PathSilhouette | None = None,
) -> dict[tuple[str, int], dict[tuple[int, ...], str]]:
    # Where a span's columns deliver to one world column -- a join's arms
    # laid through each other, or over the rows of the points beside it --
    # it goes to a column laid in the layout of the path point nearest it;
    # among those to a stepping column, whose step holds a rail or a wall
    # together; then to the one whose place across the section, as its
    # layout lays it, is nearest the world column's offset from the point
    # that laid it; a tie to the later laid.
    laid = sorted(
        runs,
        key=lambda each: (each.start, each.annotation.step_mode is StepMode.MANHATTAN),
    )
    layouts: dict[int, set] = {}
    for run in laid:
        for point in run.path:
            layouts.setdefault(point.path_index, set()).add(_layout(point))
    nearest = NearestInPlan(
        (point.position.x, point.position.z, index)
        for index, point in enumerate(path.path_points)
    )
    preferred: dict[tuple[int, int], set] = {}
    owners: dict[tuple[str, int], dict[tuple[int, int], str]] = {}
    ranks: dict[tuple[str, int], dict[tuple[int, int], tuple[int, float]]] = {}
    for run in laid:
        key = (run.section_name, run.span)
        held, best = owners.setdefault(key, {}), ranks.setdefault(key, {})
        column = run.annotation.position.x
        for point in run.path:
            cell = (point.position.x, point.position.z)
            if silhouette is not None:
                # A pass over another keeps its own columns.
                cell = silhouette.column(*cell, point.path_index)
            wanted = preferred.get(cell)
            if wanted is None:
                home = cell[2] if len(cell) > 2 else nearest.nearest(*cell)
                wanted = preferred[cell] = layouts.get(home, set())
            travel = (point.gradient.x, point.gradient.z)
            length = hypot(*travel) or 1.0
            tx, tz = travel[0] / length, travel[1] / length
            centre = path.path_points[point.path_index].position
            across = (point.position.x - centre.x) * -tz + (point.position.z - centre.z) * tx
            axis = heading(_layout(point))
            rank = (
                0 if _layout(point) in wanted else 1,
                0 if run.annotation.step_mode is StepMode.MANHATTAN else 1,
                abs(across - column * (axis.x * tx + axis.z * tz)),
            )
            if cell not in best or rank[:2] < best[cell][:2] or (
                rank[:2] == best[cell][:2] and rank[2] <= best[cell][2] + 1e-9
            ):
                held[cell], best[cell] = run.annotation.column_name, rank
    return owners


def _layout(point: FeaturePathPoint):
    return point.turn if point.layout is None else point.layout


def section_column_owners(
    runs: Sequence[DeliveredPath],
    schedules: Mapping[str, SectionSchedule],
    silhouette: PathSilhouette | None = None,
) -> dict[tuple[int, ...], tuple[str, int]]:
    # One section per world column: the later span wins a handover, and
    # among simultaneous sections the first declared.
    declared = {name: index for index, name in enumerate(schedules)}
    owners: dict[tuple[int, int], tuple[str, int]] = {}
    ranks: dict[tuple[int, int], tuple[int, int]] = {}
    for run in runs:
        if run.annotation.feature.blocks.bounds is None:
            continue
        section = run.section_name
        key = (section, run.span)
        rank = (schedules[section].spans[run.span].path_start, -declared[section])
        for point in run.path:
            column = (point.position.x, point.position.z)
            if silhouette is not None:
                column = silhouette.column(*column, point.path_index)
            if column not in ranks or rank > ranks[column]:
                owners[column] = key
                ranks[column] = rank
    return owners
