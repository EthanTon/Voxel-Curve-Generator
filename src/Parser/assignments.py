from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ..Core.coordinates import ColumnPos
from .sections import read_span, read_toml
from .spec import AssignmentSpec
from .values import require


def read_assignments(
    data: Mapping, markers: Mapping[str, ColumnPos], base: Path = Path(".")
) -> tuple[AssignmentSpec, ...]:
    named = data.get("path_assignments")
    entries = data.get("assignment", ())
    if named is not None and entries:
        raise ValueError(
            "path assignments are listed in the build or in a file of their "
            "own, not both"
        )
    if named is not None:
        entries = read_toml(base / named, "path assignments").get("assignment", ())
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
