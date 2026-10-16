from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import dist
from pathlib import Path

from ...Core.coordinates import BlockPos
from ...Parser import parse as parse_build
from ...Parser.domains import read_domain, read_interaction
from ...Parser.paths import read_output
from ...Parser.values import read_toml, require
from .regions import contains
from .spec import ComponentSpec, CompositionSpec, ConnectionSpec, OwnershipSpec, RegionSpec
from .values import check_keys, coordinates, integer, number


def polygon(value, name: str):
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ValueError(f"{name} needs at least three vertices")
    points = tuple(coordinates(each, 2, name) for each in value)
    if points[0] == points[-1]:
        points = points[:-1]
    edges = tuple(zip(points, (*points[1:], points[0])))
    area = sum(ax * bz - bx * az for (ax, az), (bx, bz) in edges)
    if len(set(points)) != len(points) or abs(area) < 1e-9:
        raise ValueError(f"{name} must be a simple polygon with nonzero area")
    for index, (a, b) in enumerate(edges):
        for other, (c, d) in enumerate(edges[index + 1:], index + 1):
            if other == index + 1 or (index == 0 and other == len(edges) - 1):
                continue
            def turn(p, q, r):
                return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
            if (turn(a, b, c) * turn(a, b, d) <= 0 and turn(c, d, a) * turn(c, d, b) <= 0
                    and max(min(a[0], b[0]), min(c[0], d[0])) <= min(max(a[0], b[0]), max(c[0], d[0]))
                    and max(min(a[1], b[1]), min(c[1], d[1])) <= min(max(a[1], b[1]), max(c[1], d[1]))):
                raise ValueError(f"{name} must not intersect itself")
    return points


def read_region(data: Mapping) -> RegionSpec:
    check_keys(data, "name domain boundary block top grade thickness holes triangles terrain_component", "region")
    boundary = polygon(require(data, "boundary"), "region.boundary")
    holes = tuple(polygon(each, "region.holes") for each in data.get("holes", ()))
    if any(not contains(boundary, x, z) for hole in holes for x, z in hole):
        raise ValueError("region holes must lie inside its boundary")
    triangles = []
    for triangle in data.get("triangles", ()):
        if len(triangle) != 3:
            raise ValueError("region.triangles entries need three [x,y,z] vertices")
        triangle = tuple(coordinates(each, 3, "region.triangles") for each in triangle)
        polygon([(x, z) for x, _, z in triangle], "region.triangles")
        triangles.append(triangle)
    if triangles and ("top" in data or "grade" in data):
        raise ValueError("region uses triangles or top/grade, not both")
    thickness = integer(data.get("thickness", 1), "region.thickness")
    if thickness < 1:
        raise ValueError("region.thickness must be positive")
    return RegionSpec(
        require(data, "name"), require(data, "domain"), boundary, require(data, "block"),
        number(data.get("top", 0), "region.top"), coordinates(data.get("grade", [0, 0]), 2, "region.grade"),
        thickness, holes, tuple(triangles), data.get("terrain_component"),
    )


def read_ownership(data: Mapping) -> OwnershipSpec:
    check_keys(data, "owner boundary heights components domains priority", "ownership")
    heights = data.get("heights")
    if heights is not None:
        heights = coordinates(heights, 2, "ownership.heights")
        heights = tuple(integer(each, "ownership.heights") for each in heights)
        if heights[0] > heights[1]:
            raise ValueError("ownership.heights needs minimum <= maximum")
    return OwnershipSpec(
        require(data, "owner"), polygon(require(data, "boundary"), "ownership.boundary"), heights,
        tuple(data.get("components", ())), tuple(data.get("domains", ())),
        integer(data.get("priority", 0), "ownership.priority"),
    )


def load(path: str | Path) -> CompositionSpec:
    path = Path(path)
    return parse(read_toml(path, "composition"), path.parent)


def parse(data: Mapping, base: Path = Path(".")) -> CompositionSpec:
    check_keys(data, "composition output component region ownership connection domain interaction terraform", "composition")
    settings = require(data, "composition")
    check_keys(settings, "origin", "composition settings")
    components = []
    for entry in data.get("component", ()):
        check_keys(entry, "name build path terraform terrain_priority", "component")
        source = (base / require(entry, "build")).resolve()
        build_data = dict(read_toml(source, "build"))
        if "path" in entry:
            build_data["path"] = {"path": str((base / entry["path"]).resolve())}
        if "terraform" in entry:
            build_data["terraform"] = str((base / entry["terraform"]).resolve())
        elif "terraform" in data:
            build_data["terraform"] = str((base / data["terraform"]).resolve())
        build = parse_build(build_data, source.parent)
        components.append(ComponentSpec(require(entry, "name"), build, source,
                                        integer(entry.get("terrain_priority", 0), "component.terrain_priority")))
    if not components:
        raise ValueError("composition needs at least one component")
    domains = {}
    interactions = []
    for domain in (*[each for component in components for each in component.build.domains],
                   *[read_domain(each) for each in data.get("domain", ())]):
        if domain.name in domains and domains[domain.name] != domain:
            raise ValueError(f"conflicting definitions of domain {domain.name!r}")
        domains[domain.name] = domain
    for interaction in (*[each for component in components for each in component.build.interactions],
                        *[read_interaction(each) for each in data.get("interaction", ())]):
        if interaction not in interactions:
            interactions.append(interaction)
    terrain = [each.build.terraform for each in components if each.build.terraform is not None]
    if terrain and len({(each.terrain_world, each.dimension, each.waterlog) for each in terrain}) != 1:
        raise ValueError("component terrain plans need the same source world, dimension, and waterlog setting")
    for index, component in enumerate(components):
        build = replace(component.build, domains=tuple(domains.values()), interactions=tuple(interactions))
        if terrain and build.surface_world is None:
            build = replace(build, surface_world=terrain[0].terrain_world, surface_dimension=terrain[0].dimension)
        components[index] = replace(component, build=build)
    connections = []
    for entry in data.get("connection", ()):
        check_keys(entry, "from to from_station to_station tolerance", "connection")
        connections.append(ConnectionSpec(require(entry, "from"), require(entry, "to"),
            integer(require(entry, "from_station"), "connection.from_station"),
            integer(require(entry, "to_station"), "connection.to_station"),
            number(entry.get("tolerance", 1), "connection.tolerance")))
    origin = settings.get("origin")
    if origin is not None:
        origin = BlockPos(*(integer(each, "composition.origin") for each in coordinates(origin, 3, "composition.origin")))
    spec = CompositionSpec(read_output(require(data, "output"), base), tuple(components),
        tuple(read_region(each) for each in data.get("region", ())),
        tuple(read_ownership(each) for each in data.get("ownership", ())), tuple(connections), origin)
    validate(spec)
    return spec


def validate(spec: CompositionSpec) -> None:
    if not spec.components:
        raise ValueError("composition needs at least one component")
    if any(each.build.composition is not None for each in spec.components):
        raise ValueError("composition components cannot reference another composition; run its root build instead")
    names = [each.name for each in (*spec.components, *spec.regions)]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("component and region names must be nonempty and unique")
    components = {each.name: each for each in spec.components}
    domains = {each.name for each in spec.components[0].build.domains}
    for rule in spec.formations:
        if not rule.components or set(rule.components) - components.keys():
            raise ValueError("formation needs known components")
        for component in rule.components:
            if components[component].build.formation(rule.formation.name) != rule.formation:
                raise ValueError("composition formation must use its root build definition")
    for region in spec.regions:
        if region.domain not in domains or (region.terrain_component is not None and region.terrain_component not in components):
            raise ValueError(f"region {region.name!r} references an unknown domain or terrain component")
    for rule in spec.ownership:
        if rule.owner not in names or set(rule.components) - set(names) or set(rule.domains) - domains:
            raise ValueError("ownership references an unknown owner, component, or domain")
    for connection in spec.connections:
        if connection.source not in components or connection.target not in components or connection.tolerance < 0:
            raise ValueError("connection needs known components and a nonnegative tolerance")
        paths = components[connection.source].build.path, components[connection.target].build.path
        indices = connection.source_station, connection.target_station
        if any(not 0 <= index < len(path.points) for path, index in zip(paths, indices)):
            raise ValueError("connection station lies outside its component path")
        if dist(paths[0].points[indices[0]], paths[1].points[indices[1]]) > connection.tolerance:
            raise ValueError(f"connection {connection.source!r} to {connection.target!r} exceeds its tolerance")
