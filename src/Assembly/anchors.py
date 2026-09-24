from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

from ..Core.CrossSection.cross_section import CrossSection
from ..Core.Feature.feature_path import (
    FeaturePath,
    FeaturePathPoint,
    either,
    ending_at,
    spacing_rules,
)
from ..Core.coordinates import BlockPos
from ..Generation.Sweep.paths import (
    DeliveredPath,
    Level,
    along_travel,
    at_their_own_level,
    rewalk,
)
from ..Parser.spec import BuildSpec, LaneSpec, Measure, StructureSpec, WireSpec
from ..Stamping import laid_offset


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
    return rewalk(
        run.path,
        spacing_rules(spec.spacing, spec.angle, spec.measure is Measure.INDEX),
    )


def structure_runs(
    build: BuildSpec,
    structure: StructureSpec,
    runs: Sequence[DeliveredPath],
) -> tuple[tuple[DeliveredPath, FeaturePath], ...]:
    # One spacing sequence across all the structure's runs, forcing only the
    # final end, so masts are not doubled up where sections swap.
    carried = runs_under(build, structure, runs)
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
