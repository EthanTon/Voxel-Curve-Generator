from dataclasses import replace

from ..Core.CrossSection.annotations import FeatureAnnotation
from ..Core.CrossSection.cross_section import CrossSection, PointField
from ..Core.Feature.feature_path import FeaturePath
from ..Core.coordinates import BlockPos
from ..Core.modes import StepMode
from .anchors import runs_under
from .dependencies import world_path
from .lanes import mark_lane


def formation_section(section, formation, lane):
    wool = tuple(FeatureAnnotation(f"{formation.name}_wool[{x}]",
        BlockPos(lane.offset.x + x, 0, lane.offset.z), next(iter(section.points)).feature,
        StepMode.MANHATTAN) for x in (-formation.clearance, 0, formation.clearance))
    return CrossSection(PointField((*section.points, *wool)), section.origin, section.step_rules)


def extend_run(run, path, lead):
    extended = FeaturePath(rules=())

    def append(endpoint, indices):
        for index in indices:
            point = path.path_points[index]
            extended.add_point(replace(endpoint, path_index=index, gradient=point.gradient,
                position=BlockPos(point.position.x, endpoint.position.y, point.position.z), y_offset=point.position.y))

    first, last = run.path[0], run.path[-1]
    append(first, range(max(0, first.path_index - lead), first.path_index))
    for point in run.path:
        extended.add_point(point)
    append(last, range(last.path_index + 1, min(len(path.path_points), last.path_index + lead + 1)))
    return run._replace(path=extended)


def formation_paths(spec, runs, sections, level, path):
    result = {}
    for formation in spec.formations:
        strips = {name: set() for name in ("flanks", "shoulders", "edges")}
        for lane in spec.lanes:
            if lane.domain not in formation.rail_domains:
                continue
            for run in runs_under(spec, lane, runs):
                section = formation_section(sections[run.section_name], formation, lane)
                carried = extend_run(run, path, spec.path.lead)
                for role, reach, step in (("flanks", 1, StepMode.CHEBYSHEV),
                                          ("shoulders", formation.clearance, StepMode.MANHATTAN),
                                          ("edges", formation.clearance + 1, StepMode.CHEBYSHEV)):
                    for side in (-1, 1):
                        offset = lane.offset + (side * reach, -1, 0)
                        strip = replace(lane, offset=offset, step=step)
                        traced = world_path(mark_lane(strip, carried, section, level), offset)
                        strips[role].update(BlockPos(p.position.x, p.y_offset, p.position.z) for p in traced)
        result[formation.name] = strips
    return result
