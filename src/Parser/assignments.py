from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..Core.coordinates import ColumnPos
from .sections import read_span
from .spec import AssignmentSpec, StationSpanSpec
from .values import listed_entries, require


def read_assignments(
    data: Mapping, markers: Mapping[str, ColumnPos], base: Path = Path(".")
) -> tuple[AssignmentSpec, ...]:
    if "auto" in data:
        entries = read_auto_assignments(data, base)
    else:
        entries, _ = listed_entries(
            data, "path_assignments", "assignment", "path assignments", base
        )
    return tuple(read_assignment(each, markers) for each in entries)


def read_auto_assignments(data: Mapping, base: Path) -> list[dict]:
    # Auto reads the build through this Parser, so it is imported only when used.
    from ..Auto import auto_assignments

    if "path_assignments" in data or "assignment" in data:
        raise ValueError(
            "[auto] chooses the assignments; remove path_assignments and [[assignment]]"
        )
    return auto_assignments(data, base)


def read_assignment(
    data: Mapping, markers: Mapping[str, ColumnPos]
) -> AssignmentSpec:
    sections = tuple(require(data, "sections"))
    if "stations" in data:
        if "from" in data or "to" in data:
            raise ValueError("an assignment uses stations or from/to, not both")
        indices = data["stations"]
        if not isinstance(indices, (list, tuple)) or len(indices) != 2:
            raise ValueError("assignment stations needs [start, stop] inclusive indices")
        span = StationSpanSpec(*indices)
    else:
        span = read_span(
            [require(data, "from"), require(data, "to")], markers, sections
        )
    if not sections:
        raise ValueError(f"the stretch {span} is assigned no sections")
    return AssignmentSpec(span=span, sections=sections)
