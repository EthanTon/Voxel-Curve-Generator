from math import dist

from .spec import ConnectionSpec
from .values import check_keys, coordinates, integer, number


def endpoint(reference, branches):
    if not isinstance(reference, str) or ":" not in reference:
        raise ValueError("node endpoints use branch:start, branch:end, or branch:station")
    name, station = reference.rsplit(":", 1)
    if name not in branches:
        raise ValueError(f"unknown branch {name!r}")
    runs = branches[name]
    index = runs[0][1].source_stations[0] if station == "start" else (
        runs[-1][1].source_stations[-1] if station == "end" else int(station)
    )
    for component, run in runs:
        if index in run.source_stations:
            return component, run, run.source_stations.index(index)
    raise ValueError(f"endpoint {reference!r} is outside its branch")


def read_graph(data, branches):
    nodes, transitions, connections = {}, {}, []
    for entry in data.get("node", ()):
        check_keys(entry, "name connections", "node")
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in nodes:
            raise ValueError("node names must be nonempty and unique")
        pairs = entry.get("connections", ())
        if not isinstance(pairs, (list, tuple)) or not pairs:
            raise ValueError("a node needs connections")
        seen = set()
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError("node connections need two distinct endpoints")
            for reference in pair:
                endpoint(reference, branches)
            resolved = tuple((component.name, index) for component, _, index in (endpoint(ref, branches) for ref in pair))
            if resolved in seen or resolved[0] == resolved[1]:
                raise ValueError("node connections must be distinct")
            seen.add(resolved)
        nodes[name] = pairs
    handled = set()
    for entry in data.get("transition", ()):
        check_keys(entry, "name node handler spans priority boundary heights flags", "transition")
        name, node = entry.get("name"), entry.get("node")
        if not isinstance(name, str) or not name or name in transitions:
            raise ValueError("transition names must be nonempty and unique")
        if node not in nodes or node in handled:
            raise ValueError("each transition needs a known, distinct node")
        handled.add(node)
        handler = entry.get("handler", "continue")
        if handler not in {"continue", "merge", "split", "cross"}:
            raise ValueError(f"unknown transition handler {handler!r}")
        flags = entry.get("flags", {})
        check_keys(flags, "lane_mode tolerance owner domains", "transition.flags")
        mode = flags.get("lane_mode", "gap" if handler == "cross" else "connect")
        if mode not in {"connect", "gap"} or handler == "cross" and mode != "gap":
            raise ValueError("transition lane_mode must be connect or gap; crossings use gap")
        tolerance = number(flags.get("tolerance", 1), "transition.tolerance")
        if tolerance < 0:
            raise ValueError("transition tolerance must be nonnegative")
        pairs = nodes[node]
        if handler == "continue" and len(pairs) != 1:
            raise ValueError("continue transitions need one connection")
        if handler in {"merge", "split"}:
            side = 1 if handler == "merge" else 0
            if len(pairs) < 2 or len({pair[side] for pair in pairs}) != 1:
                raise ValueError(f"{handler} transitions need a common {'target' if side else 'source'}")
        involved = {reference.rsplit(":", 1)[0] for pair in pairs for reference in pair}
        spans = {}
        for span in entry.get("spans", ()):
            check_keys(span, "branch stations", "transition.span")
            branch = span.get("branch")
            if branch not in involved or branch in spans:
                raise ValueError("transition spans need distinct branches connected by its node")
            first, last = (integer(v, "transition.stations") for v in coordinates(span.get("stations"), 2, "transition.stations"))
            if first > last:
                raise ValueError("transition span needs start <= stop")
            endpoint(f"{branch}:{first}", branches)
            endpoint(f"{branch}:{last}", branches)
            spans[branch] = (first, last)
        for source, target in pairs:
            a, ar, ai = endpoint(source, branches)
            b, br, bi = endpoint(target, branches)
            if mode == "connect":
                if dist(ar.path.points[ai], br.path.points[bi]) > tolerance:
                    raise ValueError(f"transition {name!r} endpoints exceed tolerance; supply connecting geometry")
                connections.append(ConnectionSpec(a.name, b.name, ai, bi, tolerance))
            for reference in (source, target):
                branch, station = reference.rsplit(":", 1)
                _, run, index = endpoint(reference, branches)
                if branch in spans and not spans[branch][0] <= run.source_stations[index] <= spans[branch][1]:
                    raise ValueError("transition span must include its node endpoint")
        transitions[name] = {**entry, "branches": involved, "spans": spans, "flags": flags}
    if handled != nodes.keys():
        raise ValueError("every node needs a transition handler")
    return tuple(connections), transitions
