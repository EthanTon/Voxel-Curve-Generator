from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from math import hypot

from ..Core.CrossSection.cross_section import CrossSection
from ..Core.Feature.feature_path import (
    FeaturePath,
    FeaturePathPoint,
    either,
    ending_at,
    one_per_delivery,
    spacing_rules,
)
from ..Core.coordinates import BlockPos
from ..Generation.Sweep.paths import (
    DeliveredPath,
    Level,
    along_travel,
    at_their_own_level,
    one_line,
    rewalk,
)
from ..Parser.spec import BuildSpec, LaneSpec, Measure, StructureSpec, WireSpec
from ..Stamping import laid_offset


WireRoute = tuple[tuple[DeliveredPath, FeaturePath], ...]


def strand_source(build: BuildSpec, wire: WireSpec) -> str | None:
    # Relative offsets can be identical on separate rail dependencies.
    # Only paths from the same source can continue one another at a handover.
    if wire.feature_path is None:
        return None
    source = f"feature_path:{wire.feature_path}"
    while source.startswith("feature_path:"):
        path = build.feature_path(source.removeprefix("feature_path:"))
        source = (path.projection.target
                  if path.projection is not None and path.projection.placement == "intersection"
                  else path.source)
    return source


def runs_under(
    spec: BuildSpec,
    fitting: StructureSpec | WireSpec | LaneSpec,
    runs: Sequence[DeliveredPath],
) -> tuple[DeliveredPath, ...]:
    # The runs of the sections a fitting rides (a wire between two others
    # rides theirs).
    sections = fitted_sections(spec, fitting)
    return tuple(run for run in runs if run.annotation.name in sections)


def fitted_sections(
    spec: BuildSpec, fitting: StructureSpec | WireSpec | LaneSpec
) -> tuple[str, ...]:
    between = getattr(fitting, "between", ())
    if not between:
        return fitting.sections
    return tuple(
        dict.fromkeys(name for wire in between for name in spec.wire(wire).sections)
    )


def points_for(spec: StructureSpec | WireSpec, run: DeliveredPath) -> FeaturePath:
    return anchor_path(spec, run.path)


def anchor_path(spec: StructureSpec | WireSpec, path: FeaturePath) -> FeaturePath:
    rules = spacing_rules(spec.spacing, spec.angle, spec.measure is Measure.INDEX)
    if not spec.min_distance or len(path) < 2:
        return rewalk(path, rules)

    def distance(first, second):
        a = first.position + laid_offset(spec.offset, first.laid_at, first.mirrored)
        b = second.position + laid_offset(spec.offset, second.laid_at, second.mirrored)
        return hypot(a.x - b.x, a.z - b.z)

    chosen = FeaturePath(rules)
    for point in path.points[:-1]:
        if not len(chosen) or distance(chosen[-1], point) >= spec.min_distance:
            chosen.add_point(point)
    kept = list(chosen)
    end = path[-1]
    # Reserve the real endpoint, removing a nearby interior anchor rather
    # than forcing a close extra pole. A whole run shorter than the minimum
    # still needs its two endpoints.
    while len(kept) > 1 and distance(kept[-1], end) < spec.min_distance:
        kept.pop()
    if not kept or kept[-1].position != end.position:
        kept.append(end)
    selected = FeaturePath(rules=())
    for point in kept:
        selected.add_point(point)
    return selected


def wire_anchors(wire: WireSpec, run: DeliveredPath) -> FeaturePath:
    # Shared by the strand and its poles: never thin the two independently.
    return points_for(wire, run._replace(path=one_line(run.path)))


def wire_routes(
    wire: WireSpec, carried: Sequence[DeliveredPath],
) -> tuple[WireRoute, ...]:
    # A section handover does not end the strand. Share one anchor sequence
    # through every contiguous run; reset only across an unwired gap.
    groups: list[list[DeliveredPath]] = []
    stop = -1
    for run in sorted(carried, key=lambda each: (each.start, each.path[0].path_index)):
        start = min(point.path_index for point in run.path)
        if not groups or start > stop + 1:
            groups.append([])
            stop = -1
        groups[-1].append(run)
        stop = max(stop, max(point.path_index for point in run.path))
    routes = []
    for group in groups:
        owners = {}
        path = FeaturePath(rules=())
        for _, number, _, point in sorted(
            (point.path_index, number, point.point_index, point)
            for number, run in enumerate(group) for point in run.path
        ):
            key = (point.path_index, point.position, point.layout, point.mirrored)
            owners.setdefault(key, number)
            path.add_point(point)
        single = FeaturePath(rules=(one_per_delivery,))
        for point in one_line(path):
            single.add_point(point)
        anchors = anchor_path(wire, single)
        separated = [FeaturePath(rules=()) for _ in group]
        for point in anchors:
            key = (point.path_index, point.position, point.layout, point.mirrored)
            separated[owners[key]].add_point(point)
        routes.append(tuple(
            (run, selected) for run, selected in zip(group, separated) if len(selected)
        ))
    return tuple(routes)


def structure_runs(
    build: BuildSpec,
    structure: StructureSpec,
    runs: Sequence[DeliveredPath],
    routes: Sequence[WireRoute] | None = None,
) -> tuple[tuple[DeliveredPath, FeaturePath], ...]:
    # A wire-bound structure uses each strand's anchors, including its ends.
    # Independent structures carry spacing across section swaps, and those
    # sharing a feature path walk it together, keeping their own sections.
    if structure.feature_path is not None:
        walked = structure_runs(build, replace(
            structure, feature_path=None, sections=shared_sections(build, structure),
        ), runs)
        return tuple(
            (run, anchors) for run, anchors in walked
            if run.annotation.name in structure.sections
        )
    carried = runs_under(build, structure, runs)
    if structure.wire is not None:
        wire = build.wire(structure.wire)
        return tuple(
            (run, anchors)
            for route in (wire_routes(wire, runs_under(build, wire, runs)) if routes is None else routes)
            for run, anchors in route if run.annotation.name in structure.sections
        )
    candidates = sorted(
        (
            (point.path_index, run_number, point.point_index, point)
            for run_number, run in enumerate(carried)
            for point in run.path
        ),
        key=lambda item: item[:3],
    )
    if not candidates:
        return ()
    if structure.min_distance:
        complete = FeaturePath(rules=())
        owners = {}
        for _, run_number, _, point in candidates:
            complete.add_point(point)
            owners.setdefault(
                (point.path_index, point.position, point.layout, point.mirrored), run_number
            )
        separated = [FeaturePath(rules=()) for _ in carried]
        for point in anchor_path(structure, complete):
            key = (point.path_index, point.position, point.layout, point.mirrored)
            separated[owners[key]].add_point(point)
        return tuple((run, anchors) for run, anchors in zip(carried, separated) if len(anchors))
    rules = spacing_rules(
        structure.spacing, structure.angle, structure.measure is Measure.INDEX
    )
    if rules:
        rules = (either(*rules, ending_at(candidates[-1][3].position)),)
    chosen = FeaturePath(rules)
    separated = [FeaturePath(rules=()) for _ in carried]
    for _, run_number, _, point in candidates:
        if chosen.add_point(point) is not None:
            separated[run_number].add_point(point)
    return tuple(
        (run, anchors)
        for run, anchors in zip(carried, separated)
        if anchors
    )


def shared_sections(build: BuildSpec, structure: StructureSpec) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        name for each in build.structures
        if each.feature_path == structure.feature_path
        for name in each.sections
    ))


def structure_points(
    structure: StructureSpec,
    anchors: FeaturePath,
    section: CrossSection,
    level: Level,
) -> FeaturePath:
    # A layout-turned structure spans the section's columns, so it rides the
    # centreline.
    column = 0 if structure.turn == "layout" else structure.offset.x
    return at_their_own_level(
        along_travel(anchors, section.along(column)), level, as_laid(structure.offset)
    )


def as_laid(offset: BlockPos) -> Callable[[FeaturePathPoint], BlockPos]:
    return lambda point: laid_offset(offset, point.laid_at, point.mirrored)


def column_cells(
    centre: DeliveredPath, runs: Sequence[DeliveredPath], x: int, z: int = 0
) -> frozenset[tuple[int, int]]:
    # The plan cells of the section's own column (x, z) over `centre`'s span.
    return frozenset(
        (point.position.x, point.position.z)
        for run in column_runs(centre, runs, x, z)
        for point in run.path
    )


def column_runs(
    centre: DeliveredPath, runs: Sequence[DeliveredPath], x: int, z: int
) -> Iterator[DeliveredPath]:
    return (
        run for run in runs
        if run.span == centre.span
        and run.annotation.section == centre.annotation.section
        and run.annotation.position.x == x
        and run.annotation.position.z == z
    )
