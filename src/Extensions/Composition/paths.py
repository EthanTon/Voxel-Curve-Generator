from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import cKDTree

from ...Core.Feature.dependent_path import DependentFeaturePath, resolve_feature_paths
from ...Core.Feature.feature_path import FeaturePath, spacing_rules
from ...Parser.spec import Measure


@dataclass(frozen=True, slots=True)
class JoinedPaths:
    dependencies: tuple[str, ...]

    def derive(self, paths):
        return tuple(run for name in self.dependencies for run in paths[name])


def qualify(dependency, prefix):
    projection = dependency.projection
    if projection is not None:
        projection = replace(projection, target=prefix + projection.target)
    return replace(dependency, source=prefix + dependency.source, projection=projection)


def reindex(path):
    points = path.path_points[path.first_index:path.last_index + 1]
    positions = np.asarray([point.position for point in points], float)
    tree = cKDTree(positions[:, (0, 2)])

    def map_stations(source):
        mapped = FeaturePath(rules=())
        for point in source:
            position = np.asarray((point.position.x, point.y_offset, point.position.z), float)
            distance, index = tree.query(position[(0, 2),])
            if distance > 32:
                continue
            if index in {0, len(points) - 1}:
                along = np.dot((position - positions[index])[(0, 2),], np.asarray(points[index].gradient)[(0, 2),])
                if index == 0 and along < -.5 or index == len(points) - 1 and along > .5:
                    continue
            mapped.add_point(replace(point, path_index=int(index) + path.first_index))
        return mapped

    return map_stations


def resolve_paths(spec, prepared):
    initial, dependencies, aliases = {}, {}, {}
    for component, each in zip(spec.components, prepared):
        prefix = component.name + "/"
        initial.update({prefix + name: paths for name, paths in each.feature_paths.items()})
        dependencies.update({prefix + name: qualify(dependency, prefix) for name, dependency in each.dependencies.items()})
        branch = component.name.split("/", 1)[0]
        if branch != component.name:
            for name in (*each.feature_paths, *each.dependencies):
                aliases.setdefault(branch + "/" + name, []).append(prefix + name)
    dependencies.update({name: JoinedPaths(tuple(paths)) for name, paths in aliases.items()})
    for each in spec.feature_paths:
        key = f"feature_path:{each.name}"
        if key in dependencies:
            raise ValueError(f"duplicate composition feature path {each.name!r}")
        dependencies[key] = DependentFeaturePath(
            each.source, spacing_rules(each.spacing, each.angle, each.measure is Measure.INDEX),
            each.projection, each.include_end, min_distance=each.min_distance,
        )
    by_name = {component.name: each for component, each in zip(spec.components, prepared)}
    seen = set()
    for binding in spec.bindings:
        key = f"{binding.branch}/{binding.target}"
        if key in seen:
            raise ValueError(f"composition binds {key!r} more than once")
        seen.add(key)
        dependencies[key] = DependentFeaturePath(binding.source, rules=(), source_transform=reindex(by_name[binding.branch].path))
    resolved = resolve_feature_paths(initial, dependencies)
    for component, each in zip(spec.components, prepared):
        prefix = component.name + "/"
        each.feature_paths.update({key[len(prefix):]: value for key, value in resolved.items() if key.startswith(prefix)})
    return [replace(each, layings=tuple(each.layings)) for each in prepared]
