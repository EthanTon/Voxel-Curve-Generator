from dataclasses import dataclass, replace
from math import dist

from ...Core.coordinates import BlockPos
from ...Parser.spec import PathSpec
from .values import check_keys, coordinates, integer, number, stations


@dataclass(frozen=True, slots=True)
class PathRun:
    path: PathSpec
    source_stations: tuple[int, ...]


def split_runs(path: PathSpec, entry) -> tuple[PathRun, ...]:
    count = len(path.points)
    if not count:
        raise ValueError("a composition branch needs path points")
    start, stop = stations(entry.get("stations", [0, count - 1]), count, "branch.stations")
    points = [BlockPos(*(integer(v, "path.points") for v in point)) for point in path.points]
    for gradient in path.gradients:
        for value in gradient:
            number(value, "path.gradients")
        if gradient.x == gradient.z == 0:
            raise ValueError("composition paths need nonzero horizontal gradients")
    cuts, seen = {start, stop + 1}, set()
    for jump in entry.get("jump", ()):
        check_keys(jump, "station mode delta", "branch.jump")
        index = integer(jump.get("station"), "jump.station")
        if not 0 < index < count or index in seen:
            raise ValueError("jump stations must be distinct interior path stations")
        seen.add(index)
        if jump.get("mode") == "rebase":
            delta = tuple(integer(v, "jump.delta") for v in coordinates(jump.get("delta"), 3, "jump.delta"))
            points[index:] = [BlockPos(*(v - shift for v, shift in zip(point, delta))) for point in points[index:]]
        elif jump.get("mode") == "gap" and "delta" not in jump:
            if start < index <= stop:
                cuts.add(index)
        else:
            raise ValueError("jump mode must be rebase with delta, or gap without delta")
    maximum = number(entry.get("max_step", 2), "branch.max_step")
    if maximum <= 0:
        raise ValueError("branch.max_step must be positive")
    cuts = sorted(cuts)
    runs = []
    for first, last in zip(cuts, cuts[1:]):
        selected = tuple(points[first:last])
        if any(dist(a, b) > maximum for a, b in zip(selected, selected[1:])):
            raise ValueError("undeclared path jump; declare a rebase/gap or supply a connecting path")
        resolved = replace(path, points=selected, gradients=path.gradients[first:last])
        runs.append(PathRun(resolved, tuple(range(first, last))))
    return tuple(runs)
