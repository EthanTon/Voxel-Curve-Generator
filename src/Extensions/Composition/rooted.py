from dataclasses import replace
from pathlib import Path

from ...Assembly.structure import build_path, resolve_spans
from ...Core.coordinates import BlockPos
from ...Parser import check_references
from ...Parser.assignments import read_assignments
from ...Parser.feature_paths import read_feature_path
from ...Parser.paths import read_path
from ...Parser.spec import AssignmentSpec, StationSpanSpec
from ...Parser.values import read_toml
from .graph import read_graph
from .routes import split_runs
from .parser import validate
from .rules import apply_rules, reservations
from .spec import ComponentSpec, CompositionSpec
from .values import check_keys, coordinates, integer, names


def station_assignments(build):
    path = build_path(build)
    return tuple(AssignmentSpec(StationSpanSpec(first - path.first_index, last - path.first_index), (name,))
                 for name, spans in resolve_spans(build, path).items() for first, last in spans)


def crop_assignments(assignments, run):
    first, last = run.source_stations[0], run.source_stations[-1]
    return tuple(replace(each, span=StationSpanSpec(max(first, each.span.start) - first,
                                                   min(last, each.span.stop) - first))
                 for each in assignments if each.span.start <= last and each.span.stop >= first)


def compose(build, source: Path | None = None):
    if build.composition is None:
        raise ValueError("build needs composition = a file name")
    config = build.composition
    data = read_toml(config, "composition")
    check_keys(data, "origin branch node transition feature_path composition_rule rule_application", "composition")
    branches, sources = {}, [config]
    for entry in data.get("branch", ()):
        check_keys(entry, "name path stations path_assignments priority terrain_priority jump max_step sections structures wires lanes feature_paths", "branch")
        name = entry.get("name")
        if not isinstance(name, str) or not name or "/" in name or ":" in name or name in branches:
            raise ValueError("branch names must be nonempty, unique, and contain no / or :")
        path = build.path
        reference = entry.get("path", "build:path")
        if reference != "build:path":
            if not isinstance(reference, str) or not reference:
                raise ValueError("branch.path must name a path file or build:path")
            path_source = (config.parent / reference).resolve()
            sources.append(path_source)
            path = read_path({"path": str(path_source), "lead": build.path.lead,
                              "diagonal_degrees": build.path.diagonal_degrees})
        base = replace(build, path=path, composition=None)
        if "path_assignments" in entry:
            base = replace(base, assignments=read_assignments({"assignment": entry["path_assignments"]}, build.markers, config.parent))
        assignments = station_assignments(base)
        if "sections" in entry:
            selected = names(entry["sections"], "branch.sections")
            for section in selected:
                build.section(section)
            assignments = tuple(replace(each, sections=tuple(name for name in each.sections if name in selected))
                                for each in assignments if set(each.sections) & set(selected))
        for attribute in ("structures", "wires", "lanes", "feature_paths"):
            if attribute in entry:
                selected = names(entry[attribute], f"branch.{attribute}")
                items = getattr(base, attribute)
                if set(selected) - {item.name for item in items}:
                    raise ValueError(f"branch.{attribute} references an unknown build feature")
                base = replace(base, **{attribute: tuple(item for item in items if item.name in selected)})
        runs = split_runs(path, entry)
        branches[name] = []
        for index, run in enumerate(runs):
            resolved = replace(base, path=run.path, assignments=crop_assignments(assignments, run))
            check_references(resolved)
            priority = integer(entry.get("priority", 0), "branch.priority")
            component = ComponentSpec(name=name if len(runs) == 1 else f"{name}/{index}", build=resolved,
                source=source or config, priority=priority,
                terrain_priority=integer(entry.get("terrain_priority", priority), "branch.terrain_priority"))
            branches[name].append((component, run))
    if not branches:
        raise ValueError("composition needs at least one branch")
    connections, transitions = read_graph(data, branches)
    bindings, ownership, formations = apply_rules(data, branches, transitions)
    for transition in transitions.values():
        flags = transition["flags"]
        if "owner" in flags:
            options = {key: transition[key] for key in ("boundary", "heights", "priority") if key in transition}
            options.update({key: flags[key] for key in ("owner", "domains") if key in flags})
            ownership += reservations(options, transition["branches"], branches)
    components = tuple(component for parts in branches.values() for component, _ in parts)
    for component in components:
        check_references(component.build)
    feature_paths = tuple(read_feature_path(each) for each in data.get("feature_path", ()))
    if any(each.sections for each in feature_paths):
        raise ValueError("composition feature paths select branches through their source, not sections")
    origin = data.get("origin")
    if origin is not None:
        origin = BlockPos(*(integer(v, "composition.origin") for v in coordinates(origin, 3, "composition.origin")))
    spec = CompositionSpec(build.output, components, ownership=ownership, connections=connections,
                           origin=origin, feature_paths=feature_paths, bindings=bindings, sources=tuple(sources),
                           formations=formations)
    validate(spec)
    return spec
