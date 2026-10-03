from __future__ import annotations

from collections.abc import Mapping

from ..Core.Feature.dependent_path import DependentFeaturePath, PathProjection, resolve_feature_paths
from .fittings import read_min_distance, read_where_it_sits
from .spec import BuildSpec, FeaturePathSpec, Measure
from .values import require


def read_feature_path(data: Mapping) -> FeaturePathSpec:
    name, source = require(data, "name"), require(data, "source")
    if not isinstance(name, str) or not name:
        raise ValueError("feature_path name must be a nonempty string")
    if not isinstance(source, str) or not source:
        raise ValueError(f"feature_path {name!r} source must name a path")
    allowed = {"name", "source", "spacing", "angle", "measure", "min_distance", "sections", "include_end", "project"}
    if set(data) - allowed:
        raise ValueError(f"feature_path {name!r} has unknown options: {sorted(set(data) - allowed)}")
    spacing, angle, measure = read_where_it_sits(data, f"feature_path {name!r}")
    if measure is Measure.INDEX and spacing != int(spacing):
        raise ValueError(f"feature_path {name!r} index spacing must be a whole number")
    include_end = data.get("include_end", True)
    if not isinstance(include_end, bool):
        raise ValueError(f"feature_path {name!r} include_end must be a boolean")
    sections = data.get("sections", ())
    if (not isinstance(sections, (list, tuple)) or any(not isinstance(name, str) or not name for name in sections)
            or "sections" in data and not sections):
        raise ValueError(f"feature_path {name!r} sections must be a nonempty list of section names")
    return FeaturePathSpec(name, source, spacing, angle, measure, include_end,
                           read_projection(data.get("project")), read_min_distance(data, f"feature_path {name!r}"),
                           tuple(sections))


def read_projection(data: Mapping | None) -> PathProjection | None:
    if data is None:
        return None
    allowed = {"target", "direction", "angle", "max_distance", "min_distance", "height_tolerance", "placement"}
    if not isinstance(data, Mapping) or set(data) - allowed:
        raise ValueError("feature_path.project accepts target, direction or angle, max_distance, min_distance, height_tolerance and placement")
    if "direction" in data and "angle" in data:
        raise ValueError("feature_path.project uses direction or angle, not both")
    directions = {"left": -90.0, "right": 90.0, "forward": 0.0, "backward": 180.0}
    direction = data.get("direction", "left")
    if not isinstance(direction, str) or direction not in directions:
        raise ValueError("feature_path.project direction must be left, right, forward or backward")
    return PathProjection(
        target=require(data, "target"),
        angle=data.get("angle", directions[direction]),
        max_distance=data.get("max_distance", 32.0),
        min_distance=data.get("min_distance", 0.0),
        height_tolerance=data.get("height_tolerance"),
        placement=data.get("placement", "intersection"),
    )


def check_feature_paths(spec: BuildSpec) -> None:
    # Empty paths validate the graph without generating any geometry.
    initial = {"path": ()}
    dependencies = {}
    names = set()
    for each in spec.feature_paths:
        if each.name in names:
            raise ValueError(f"duplicate feature_path name {each.name!r}")
        names.add(each.name)
        for name in each.sections:
            spec.section(name)
        dependencies[f"feature_path:{each.name}"] = DependentFeaturePath(each.source, projection=each.projection)
    for kind, items in (("section", spec.sections), ("lane", spec.lanes),
                        ("wire", spec.wires), ("structure", spec.structures)):
        for fitting in items:
            key = f"{kind}:{fitting.name}"
            if key in initial or key in dependencies:
                raise ValueError(f"duplicate {kind} name {fitting.name!r}")
            referenced = getattr(fitting, "feature_path", None)
            if kind == "wire" and referenced is not None and referenced not in names:
                raise ValueError(f"wire {fitting.name!r} references unknown feature_path {referenced!r}")
            if referenced in names:
                if any(getattr(fitting, option) for option in ("spacing", "angle", "min_distance")):
                    raise ValueError(f"{kind} {fitting.name!r} uses feature_path {referenced!r}; put spacing and angle on that path")
                dependencies[key] = DependentFeaturePath(f"feature_path:{referenced}")
            elif kind == "structure" and fitting.wire is not None:
                dependencies[key] = DependentFeaturePath(f"wire:{fitting.wire}")
            elif kind == "wire" and fitting.between:
                # Droppers have two paths and do not supply one longitudinal path.
                continue
            else:
                initial[key] = ()
    resolve_feature_paths(initial, dependencies)
