from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from ..Core.CrossSection.cross_section import CrossSection
from ..Core.CrossSection.turns import snap
from ..Core.Feature.dependent_path import DependentFeaturePath, PathRuns, resolve_feature_paths
from ..Core.Feature.feature_path import FeaturePath, one_per_delivery, spacing_rules
from ..Core.Path.path import Path
from ..Core.coordinates import BlockPos
from ..Generation.Sweep.paths import DeliveredPath, Level, along_travel, one_line
from ..Parser.spec import BuildSpec, Measure, WireGeometry, WireSpec
from ..Stamping import laid_offset
from .anchors import runs_under, strand_source, structure_points, structure_runs, wire_routes
from .lanes import lane_support, mark_lane
from .wires import attachment


def world_path(path: FeaturePath, offset: BlockPos = BlockPos(0, 0, 0)) -> FeaturePath:
    # Dependencies use world heights, encoded as local y=0 plus y_offset.
    result = FeaturePath(rules=())
    for point in path:
        moved = laid_offset(offset, point.laid_at)
        result.add_point(point, position=BlockPos(point.position.x + moved.x, 0, point.position.z + moved.z),
                         y_offset=point.position.y + point.y_offset + moved.y)
    return result


def _groups(runs: Sequence[DeliveredPath]) -> tuple[tuple[DeliveredPath, ...], ...]:
    groups: list[list[DeliveredPath]] = []
    stop = -1
    for run in sorted(runs, key=lambda each: (each.start, each.path[0].path_index)):
        start = min(point.path_index for point in run.path)
        if not groups or start > stop + 1:
            groups.append([])
            stop = -1
        groups[-1].append(run)
        stop = max(stop, max(point.path_index for point in run.path))
    return tuple(tuple(group) for group in groups)


def _merge(paths: Sequence[FeaturePath], *, centre: bool = False) -> FeaturePath:
    merged = FeaturePath(rules=())
    seen = set()
    for _, _, _, point in sorted(
        (point.path_index, number, point.point_index, point)
        for number, path in enumerate(paths) for point in path
    ):
        key = (point.path_index, point.position, point.y_offset)
        if key not in seen:
            merged.add_point(point)
            seen.add(key)
    if not centre:
        return merged
    single = FeaturePath(rules=(one_per_delivery,))
    for point in one_line(merged):
        single.add_point(point)
    return single


def _split_gaps(path: FeaturePath) -> PathRuns:
    groups = []
    current = FeaturePath(rules=())
    for point in path:
        if len(current) and point.path_index > current[-1].path_index + 1:
            groups.append(current)
            current = FeaturePath(rules=())
        current.add_point(point)
    if len(current):
        groups.append(current)
    return tuple(groups)


def _clipped(path: FeaturePath, clip) -> FeaturePath:
    if clip is None:
        return path
    kept = FeaturePath(rules=())
    for point in path:
        if clip(point).contains(point.position.x, point.position.z):
            kept.add_point(point)
    return kept


def _supported(path: FeaturePath, carried: Sequence[DeliveredPath], offset: BlockPos) -> PathRuns:
    # Restrict consumers to their own sections without drawing across gaps.
    result = []
    for group in _groups(carried):
        first = min(point.path_index for run in group for point in run.path)
        last = max(point.path_index for run in group for point in run.path)
        selected = FeaturePath(rules=())
        for point in path:
            if first <= point.path_index <= last:
                selected.add_point(point)
        if len(selected):
            result.append(world_path(selected, offset))
    return tuple(result)


def dependent_intervals(spec: BuildSpec, wire: WireSpec, runs: Sequence[DeliveredPath]) -> tuple[tuple[int, int], ...]:
    # Stepping cells may carry a neighboring station's metadata at a portal.
    # Continuous rail dependencies continue by their assigned section bounds,
    # preserving handovers without connecting across an authored source gap.
    source = strand_source(spec, wire)
    if source is None or not source.startswith("lane:"):
        return ()
    lane = spec.lane(source.removeprefix("lane:"))
    if lane.off:
        return ()
    names = set(wire.sections) & set(lane.sections)
    path = spec.feature_path(wire.feature_path)
    while True:
        if path.sections:
            names.intersection_update(path.sections)
        if not path.source.startswith("feature_path:"):
            break
        path = spec.feature_path(path.source.removeprefix("feature_path:"))
    carried = tuple(run for run in runs if run.annotation.name in names)
    return tuple((min(point.path_index for run in group for point in run.path),
                  max(point.path_index for run in group for point in run.path))
                 for group in _groups(carried))


def feature_path_graph(
    spec: BuildSpec, runs: Sequence[DeliveredPath], sections: Mapping[str, CrossSection],
    path: Path, level: Level, clip_for: Callable | None = None,
):
    initial: dict[str, PathRuns] = {}
    dependencies = {}
    named = {each.name for each in spec.feature_paths}
    authored = FeaturePath(rules=())
    for index in range(path.first_index, path.last_index + 1):
        point = path.path_points[index]
        authored.add((point.position.x, 0, point.position.z), point.gradient, snap(point.gradient),
                     index, point.position.y)
    initial["path"] = (authored,) if len(authored) else ()
    for name in (section.name for section in spec.sections):
        carried = tuple(run for run in runs if run.annotation.name == name)
        initial[f"section:{name}"] = tuple(
            world_path(_merge([run.path for run in group], centre=True))
            for group in _groups(carried)
        )
    for lane in spec.lanes:
        paths = []
        for group in _groups(runs_under(spec, lane, runs)):
            merged = _merge([
                _clipped(world_path(mark_lane(lane, run, sections[run.section_name], level,
                                             lane_support(lane, run, runs)), lane.offset),
                         None if clip_for is None else clip_for(lane))
                for run in group
            ])
            # Connected stepping lanes can skip source station deliveries at
            # a layout join while their physical cells remain continuous.
            # Only authored dash gaps split such a run.
            if len(merged):
                paths.extend(_split_gaps(merged) if lane.off else (merged,))
        initial[f"lane:{lane.name}"] = tuple(paths)
    for wire in spec.wires:
        if wire.between:
            continue
        carried = runs_under(spec, wire, runs)
        if wire.feature_path is not None:
            dependencies[f"wire:{wire.name}"] = DependentFeaturePath(
                f"feature_path:{wire.feature_path}", rules=(),
                transform=lambda selected, carried=carried, offset=wire.offset: _supported(selected, carried, offset),
            )
            continue
        paths = []
        if wire.geometry is WireGeometry.LINE:
            routes = wire_routes(wire, carried)
        else:
            routes = tuple(tuple((run, one_line(run.path)) for run in group) for group in _groups(carried))
        for route in routes:
            anchors = FeaturePath(rules=())
            for run, selected in route:
                for point in along_travel(selected, sections[run.section_name].along(wire.offset.x)):
                    position = attachment(wire, point, level)
                    anchors.add_point(point, position=BlockPos(position.x, 0, position.z), y_offset=position.y)
            if len(anchors):
                paths.append(anchors)
        initial[f"wire:{wire.name}"] = tuple(paths)
    for structure in spec.structures:
        carried = runs_under(spec, structure, runs)
        if structure.feature_path in named:
            source, offset = f"feature_path:{structure.feature_path}", structure.offset
        elif structure.wire is not None:
            source = f"wire:{structure.wire}"
            # The wire path already includes its across/along offset.
            offset = BlockPos(0, structure.offset.y - spec.wire(structure.wire).offset.y, 0)
        else:
            placements = tuple(
                (run, world_path(structure_points(structure, anchors, sections[run.section_name], level), structure.offset))
                for run, anchors in structure_runs(spec, structure, runs)
            )
            initial[f"structure:{structure.name}"] = tuple(
                _merge([points for run, points in placements if run in group])
                for group in _groups(carried)
                if any(run in group for run, _ in placements)
            )
            continue
        dependencies[f"structure:{structure.name}"] = DependentFeaturePath(
            source, rules=(), transform=lambda selected, carried=carried, offset=offset: _supported(selected, carried, offset),
        )
    for each in spec.feature_paths:
        carried = tuple(run for run in runs if run.annotation.name in each.sections)
        source_clip = None if not each.sections or clip_for is None else clip_for(each)
        dependencies[f"feature_path:{each.name}"] = DependentFeaturePath(
            each.source, spacing_rules(each.spacing, each.angle, each.measure is Measure.INDEX),
            each.projection, each.include_end,
            min_distance=each.min_distance,
            source_transform=None if not each.sections else (
                lambda source, carried=carried, clip=source_clip: _supported(
                    _clipped(source, clip), carried, BlockPos(0, 0, 0),
                )
            ),
        )
    return initial, dependencies


def build_feature_paths(*args, **kwargs) -> dict[str, PathRuns]:
    return resolve_feature_paths(*feature_path_graph(*args, **kwargs))
