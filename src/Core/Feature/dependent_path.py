from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from math import cos, hypot, isfinite, radians, sin

from ..coordinates import BlockPos, Vec2
from .feature_path import DEFAULT_RULES, FeaturePath, FeaturePathPoint, FeaturePathRule


# Separate runs never imply a segment across a gap in the supporting feature.
PathRuns = tuple[FeaturePath, ...]
PathTransform = Callable[[FeaturePath], FeaturePath | PathRuns]
ProjectionDirection = Callable[[FeaturePathPoint], Vec2]
_EPSILON = 1e-9
_Segment = tuple[int, FeaturePathPoint, FeaturePathPoint]


@dataclass(frozen=True, slots=True)
class _SegmentTree:
    # Segment bounds, rather than endpoint proximity, retain crossings in
    # sparse paths. Compile once per derivation and prune bounded ray queries.
    bounds: tuple[int, int, int, int]
    segments: tuple[_Segment, ...] = ()
    children: tuple[_SegmentTree, ...] = ()

    @classmethod
    def build(cls, segments: list[_Segment]) -> _SegmentTree | None:
        if not segments:
            return None
        xs = [p.position.x for _, start, end in segments for p in (start, end)]
        zs = [p.position.z for _, start, end in segments for p in (start, end)]
        bounds = min(xs), max(xs), min(zs), max(zs)
        if len(segments) <= 8:
            return cls(bounds, tuple(segments))
        axis = "x" if bounds[1] - bounds[0] >= bounds[3] - bounds[2] else "z"
        ordered = sorted(segments, key=lambda segment: getattr(segment[1].position, axis) + getattr(segment[2].position, axis))
        middle = len(ordered) // 2
        return cls(bounds, children=(cls.build(ordered[:middle]), cls.build(ordered[middle:])))

    def query(self, px, pz, dx, dz, minimum, maximum) -> Iterator[_Segment]:
        for origin, direction, low, high in ((px, dx, self.bounds[0], self.bounds[1]),
                                              (pz, dz, self.bounds[2], self.bounds[3])):
            if abs(direction) <= _EPSILON:
                if not low - _EPSILON <= origin <= high + _EPSILON:
                    return
            else:
                first, last = sorted(((low - origin) / direction, (high - origin) / direction))
                minimum, maximum = max(minimum, first), min(maximum, last)
                if minimum > maximum + _EPSILON:
                    return
        yield from self.segments
        for child in self.children:
            yield from child.query(px, pz, dx, dz, minimum, maximum)


@dataclass(frozen=True, slots=True)
class PathProjection:
    """Cast a horizontal ray and place at its nearest intersection with a path.

    Angles are relative to travel in the x/z plane: -90 is left, +90 right.
    A direction callback may supply another rule in world coordinates. The hit
    inherits the source station and heading, and the target's interpolated height.
    """

    target: str
    angle: float = -90.0
    max_distance: float = 32.0
    min_distance: float = 0.0
    height_tolerance: float | None = None
    direction: ProjectionDirection | None = None
    placement: str = "intersection"

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("projection target must name a path")
        for name in ("angle", "max_distance", "min_distance", "height_tolerance"):
            value = getattr(self, name)
            if name == "height_tolerance" and value is None:
                continue
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not isfinite(value)):
                raise ValueError(f"projection {name} must be finite")
        if self.max_distance <= 0 or not 0 <= self.min_distance <= self.max_distance:
            raise ValueError("projection requires 0 <= min_distance <= max_distance and max_distance > 0")
        if self.height_tolerance is not None and self.height_tolerance < 0:
            raise ValueError("projection height_tolerance must be nonnegative")
        if self.placement not in ("intersection", "source"):
            raise ValueError("projection placement must be intersection or source")

    def intersect(
        self, point: FeaturePathPoint, targets: Sequence[FeaturePath],
    ) -> FeaturePathPoint | None:
        return self.projector(targets)(point)

    def projector(self, targets: Sequence[FeaturePath]) -> Callable[[FeaturePathPoint], FeaturePathPoint | None]:
        segments = []
        for path in targets:
            pairs = zip(path, path.points[1:]) if len(path) > 1 else ((p, p) for p in path)
            first = len(segments)
            segments.extend((first + index, start, end) for index, (start, end) in enumerate(pairs))
        tree = _SegmentTree.build(segments)
        return lambda point: self._intersect(point, tree)

    def _intersect(self, point: FeaturePathPoint, tree: _SegmentTree | None) -> FeaturePathPoint | None:
        if self.direction is None:
            gx, gz = point.gradient.x, point.gradient.z
            theta = radians(self.angle)
            dx, dz = gx * cos(theta) - gz * sin(theta), gx * sin(theta) + gz * cos(theta)
        else:
            direction = Vec2.coerce(self.direction(point))
            dx, dz = direction.x, direction.z
        length = hypot(dx, dz)
        if not isfinite(length) or length <= _EPSILON:
            raise ValueError("projection requires a nonzero finite horizontal direction")
        dx, dz = dx / length, dz / length
        nearest: tuple[float, float] | None = None
        order = -1
        candidates = () if tree is None else tree.query(point.position.x, point.position.z, dx, dz,
                                                         self.min_distance, self.max_distance)
        for index, start, end in candidates:
            hit = self._segment(point, dx, dz, start, end)
            if hit is not None and (nearest is None or hit[0] < nearest[0] - _EPSILON
                                    or abs(hit[0] - nearest[0]) <= _EPSILON and index < order):
                nearest, order = hit, index
        if nearest is None:
            return None
        if self.placement == "source":
            return point
        distance, height = nearest
        return replace(
            point,
            position=BlockPos(round(point.position.x + dx * distance), 0,
                              round(point.position.z + dz * distance)),
            y_offset=round(height),
        )

    def _segment(self, point, dx, dz, start, end) -> tuple[float, float] | None:
        px, pz = point.position.x, point.position.z
        ax, az = start.position.x - px, start.position.z - pz
        sx, sz = end.position.x - start.position.x, end.position.z - start.position.z
        denominator = dx * sz - dz * sx
        first_y = start.position.y + start.y_offset
        last_y = end.position.y + end.y_offset
        source_y = point.position.y + point.y_offset
        if abs(denominator) > _EPSILON:
            distance = (ax * sz - az * sx) / denominator
            fraction = (ax * dz - az * dx) / denominator
            if not -_EPSILON <= fraction <= 1 + _EPSILON:
                return None
            height = first_y + (last_y - first_y) * min(1.0, max(0.0, fraction))
        else:
            # Parallel rays only intersect collinear segments. Choose the
            # nearest point in their overlap, including degenerate segments.
            if abs(ax * dz - az * dx) > _EPSILON:
                return None
            before = ax * dx + az * dz
            after = before + sx * dx + sz * dz
            low, high = max(self.min_distance, min(before, after)), min(self.max_distance, max(before, after))
            if low > high + _EPSILON:
                return None
            if self.height_tolerance is not None and abs(last_y - first_y) > _EPSILON:
                # Restrict an overlapping sloped segment to the allowed height.
                fractions = sorted(((source_y - self.height_tolerance - first_y) / (last_y - first_y),
                                    (source_y + self.height_tolerance - first_y) / (last_y - first_y)))
                bounds = sorted((before + (after - before) * fractions[0],
                                 before + (after - before) * fractions[1]))
                low, high = max(low, bounds[0]), min(high, bounds[1])
                if low > high + _EPSILON:
                    return None
            distance = low
            if abs(after - before) <= _EPSILON:
                # A vertical target has one plan position but a range of
                # possible heights; meet it closest to the source level.
                height = min(max(source_y, min(first_y, last_y)), max(first_y, last_y))
            else:
                fraction = (distance - before) / (after - before)
                height = first_y + (last_y - first_y) * fraction
        if not self.min_distance - _EPSILON <= distance <= self.max_distance + _EPSILON:
            return None
        if self.height_tolerance is not None and abs(height - source_y) > self.height_tolerance + _EPSILON:
            return None
        return max(self.min_distance, min(self.max_distance, distance)), height


@dataclass(frozen=True, slots=True)
class DependentFeaturePath:
    """Select points on a named source, then optionally project or transform them.

    Rules inspect the selected source points, before projection. A missing hit
    therefore does not move the next spacing/angle anchor. Callers may supply
    any FeaturePathRule or a transform for other placement rules.
    """

    source: str
    rules: tuple[FeaturePathRule, ...] = DEFAULT_RULES
    projection: PathProjection | None = None
    include_end: bool = False
    transform: PathTransform | None = None
    min_distance: float = 0.0
    source_transform: PathTransform | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("dependent feature source must name a path")
        if not isinstance(self.include_end, bool):
            raise ValueError("dependent feature include_end must be a boolean")
        if (not isinstance(self.min_distance, (int, float)) or isinstance(self.min_distance, bool)
                or not isfinite(self.min_distance) or self.min_distance < 0):
            raise ValueError("dependent feature min_distance must be finite and nonnegative")

    @property
    def dependencies(self) -> tuple[str, ...]:
        return (self.source,) if self.projection is None else (self.source, self.projection.target)

    def derive(self, paths: Mapping[str, PathRuns]) -> PathRuns:
        targets = () if self.projection is None else paths[self.projection.target]
        project = None if self.projection is None else self.projection.projector(targets)
        derived = []
        sources = paths[self.source]
        if self.source_transform is not None:
            prepared = []
            for source in sources:
                transformed = self.source_transform(source)
                prepared.extend((transformed,) if isinstance(transformed, FeaturePath) else transformed)
            sources = tuple(prepared)
        for source in sources:
            selected = FeaturePath(self.rules)
            candidates = source.points[:-1] if self.min_distance and self.include_end else source.points
            for point in candidates:
                if (not len(selected) or not self.min_distance or
                        hypot(point.position.x - selected[-1].position.x,
                              point.position.z - selected[-1].position.z) >= self.min_distance):
                    selected.add_point(point)
            if self.include_end and len(source) and (not len(selected) or
                    replace(selected[-1], point_index=source[-1].point_index) != source[-1]):
                # Keep the boundary even when the selection rule refuses it.
                points = FeaturePath(rules=())
                kept = list(selected)
                while (len(kept) > 1 and self.min_distance and
                       hypot(source[-1].position.x - kept[-1].position.x,
                             source[-1].position.z - kept[-1].position.z) < self.min_distance):
                    kept.pop()
                for point in kept:
                    points.add_point(point)
                points.add_point(source[-1])
                selected = points
            if project is not None:
                projected = FeaturePath(rules=())
                for point in selected:
                    hit = project(point)
                    if hit is not None:
                        projected.add_point(hit)
                selected = projected
            if self.transform is not None:
                transformed = self.transform(selected)
                results = (transformed,) if isinstance(transformed, FeaturePath) else transformed
            else:
                results = (selected,)
            derived.extend(result for result in results if len(result))
        return tuple(derived)


def resolve_feature_paths(
    paths: Mapping[str, PathRuns], dependencies: Mapping[str, DependentFeaturePath],
) -> dict[str, PathRuns]:
    """Resolve dependencies regardless of declaration order; reject cycles."""
    resolved = dict(paths)
    duplicates = resolved.keys() & dependencies.keys()
    if duplicates:
        raise ValueError(f"feature paths defined twice: {', '.join(sorted(duplicates))}")
    visiting: list[str] = []

    def resolve(name: str) -> PathRuns:
        if name in resolved:
            return resolved[name]
        if name in visiting:
            raise ValueError(f"feature path dependency cycle: {' -> '.join((*visiting, name))}")
        if name not in dependencies:
            raise ValueError(f"unknown feature path {name!r}")
        visiting.append(name)
        dependency = dependencies[name]
        for required in dependency.dependencies:
            resolve(required)
        resolved[name] = dependency.derive(resolved)
        visiting.pop()
        return resolved[name]

    for name in dependencies:
        resolve(name)
    return resolved
