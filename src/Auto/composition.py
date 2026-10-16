from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import dist, isfinite

from ..Parser.stations import resolve_station


def section_names(raw: object, defined: set[str], where: str) -> tuple[str, ...]:
    if (not isinstance(raw, list) or not raw
            or any(not isinstance(name, str) or name not in defined for name in raw)
            or len(set(raw)) != len(raw)):
        raise ValueError(f"{where}.sections needs distinct, defined section names")
    return tuple(raw)


def station_range(raw: object, count: int, where: str) -> tuple[int, int]:
    message = (f"{where}.stations needs inclusive indices 0 <= start <= stop < {count}; "
               "-1 means the last station")
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(message)
    try:
        start, stop = (resolve_station(index, count, f"{where}.stations") for index in raw)
    except ValueError:
        raise ValueError(message) from None
    if stop < start:
        raise ValueError(message)
    return start, stop


def endpoint_index(raw: object, points: Sequence, markers: Mapping, where: str) -> int:
    if isinstance(raw, str):
        if raw not in markers:
            raise ValueError(f"{where} names undefined marker {raw!r}")
        raw = markers[raw]
    if (not isinstance(raw, (list, tuple)) or len(raw) != 2
            or any(not isinstance(v, int) or isinstance(v, bool) for v in raw)):
        raise ValueError(f"{where} needs a marker or an exact [x, z] path column")
    matches = [i for i, p in enumerate(points) if (p.x, p.z) == tuple(raw)]
    if len(matches) != 1:
        raise ValueError(f"{where} must identify one path point; use stations for repeated columns")
    return matches[0]


def world_point(raw: object, where: str) -> tuple[float, ...]:
    if (not isinstance(raw, list) or len(raw) not in (2, 3)
            or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                   or not isfinite(v) for v in raw)):
        raise ValueError(f"{where} needs finite [x, z] or [x, y, z] coordinates")
    return tuple(raw)


# 3D distance for [x, y, z], plan distance for [x, z]; ties go to the first station.
def nearest_station(points: Sequence, point: Sequence[float]) -> tuple[int, float]:
    coordinates = [(p.x, p.z) if len(point) == 2 else (p.x, p.y, p.z) for p in points]
    index = min(range(len(points)), key=lambda i: (dist(coordinates[i], point), i))
    return index, dist(coordinates[index], point)


@dataclass(frozen=True)
class Override:
    sections: tuple[str, ...]
    source: str
    label: str


@dataclass(frozen=True)
class ControlPoint:
    point: tuple[float, ...]
    station: int
    distance: float
    override: Override | None

    def as_dict(self) -> dict:
        return dict(point=list(self.point), station=self.station, distance=self.distance,
                    sections=list(self.override.sections) if self.override else None,
                    mode="sections" if self.override else "auto")


@dataclass(frozen=True)
class Composition:
    automatic: bool
    overrides: tuple[Override | None, ...]
    controls: tuple[ControlPoint, ...]

    @classmethod
    def read(cls, data: object, points: Sequence, defined: set[str], markers: Mapping) -> Composition:
        if not isinstance(data, Mapping):
            raise ValueError("selection must be a table")
        extra = set(data) - {"auto", "manual", "control_point"}
        if extra:
            raise ValueError(f"unknown selection fields: {sorted(extra)}")
        automatic = data.get("auto", True)
        if not isinstance(automatic, bool):
            raise ValueError("selection.auto must be true or false")
        controls = []
        for number, raw in enumerate(tables(data, "control_point"), 1):
            label = f"control_point[{number}]"
            if set(raw) - {"point", "station", "sections", "auto"}:
                raise ValueError(f"{label} accepts point or station and either sections or auto = true")
            if "station" in raw:
                if "point" in raw:
                    raise ValueError(f"{label} must use point or station, not both")
                index = resolve_station(raw["station"], len(points), f"{label}.station")
                point, distance = tuple(points[index]), 0.0
            else:
                point = world_point(raw.get("point"), f"{label}.point")
                index, distance = nearest_station(points, point)
            if "sections" in raw:
                if "auto" in raw:
                    raise ValueError(f"{label} must set either sections or auto = true")
                override = Override(section_names(raw["sections"], defined, label), "control_point", label)
            elif raw.get("auto") is True:
                if not automatic:
                    raise ValueError(f"{label} cannot resume auto when selection.auto = false")
                override = None
            else:
                raise ValueError(f"{label} needs sections or auto = true")
            controls.append(ControlPoint(point, index, distance, override))
        controls.sort(key=lambda control: control.station)
        if any(a.station == b.station for a, b in zip(controls, controls[1:])):
            raise ValueError("control points snap to the same station; specify only one switch there")
        overrides: list[Override | None] = [None] * len(points)
        for position, control in enumerate(controls):
            stop = controls[position + 1].station if position + 1 < len(controls) else len(points)
            overrides[control.station:stop] = [control.override] * (stop - control.station)

        manual: dict[int, Override] = {}
        for number, raw in enumerate(tables(data, "manual"), 1):
            label = f"manual[{number}]"
            if set(raw) - {"from", "to", "stations", "sections"}:
                raise ValueError(f"{label} accepts stations or from/to, and sections")
            sections = section_names(raw.get("sections"), defined, label)
            if "stations" in raw:
                if "from" in raw or "to" in raw:
                    raise ValueError(f"{label} must use stations or from/to, not both")
                start, stop = station_range(raw["stations"], len(points), label)
            else:
                start = endpoint_index(raw.get("from"), points, markers, f"{label}.from")
                stop = endpoint_index(raw.get("to"), points, markers, f"{label}.to")
                if stop < start:
                    raise ValueError(f"{label} runs backwards along the path")
            override = Override(sections, "manual", label)
            for index in range(start, stop + 1):
                if index in manual and manual[index].sections != sections:
                    raise ValueError(f"conflicting manual assignments at station {index}")
                manual[index] = override
                overrides[index] = override
        return cls(automatic, tuple(overrides), tuple(controls))


def tables(data: Mapping, key: str) -> list[Mapping]:
    raw = data.get(key, [])
    if not isinstance(raw, list) or any(not isinstance(each, Mapping) for each in raw):
        raise ValueError(f"selection.{key} must be a list of tables")
    return raw
