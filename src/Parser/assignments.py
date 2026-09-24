from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..Core.coordinates import ColumnPos
from .sections import read_span
from .spec import AssignmentSpec
from .values import listed_entries, require


def read_assignments(
    data: Mapping, markers: Mapping[str, ColumnPos], base: Path = Path(".")
) -> tuple[AssignmentSpec, ...]:
    entries, _ = listed_entries(
        data, "path_assignments", "assignment", "path assignments", base
    )
    return tuple(read_assignment(each, markers) for each in entries)


def read_assignment(
    data: Mapping, markers: Mapping[str, ColumnPos]
) -> AssignmentSpec:
    sections = tuple(require(data, "sections"))
    span = read_span(
        [require(data, "from"), require(data, "to")], markers, sections
    )
    if not sections:
        raise ValueError(f"the stretch {span} is assigned no sections")
    return AssignmentSpec(span=span, sections=sections)
