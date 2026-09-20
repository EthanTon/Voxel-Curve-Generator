from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

from ..Core.coordinates import ColumnPos
from ..Core.modes import PlacementMode
from .spec import SectionElevation, SectionSpec, SpanSpec
from .values import read_column, read_enum, require


def read_sections(
    data: Mapping, markers: Mapping[str, ColumnPos], base: Path = Path(".")
) -> tuple[SectionSpec, ...]:
    named = data.get("sections")
    entries = data.get("section", ())
    if named is not None and entries:
        raise ValueError(
            "sections are listed in the build or in a file of their own, "
            "not both"
        )
    if named is not None:
        listing = base / named
        entries = read_toml(listing, "sections").get("section", ())
        base = listing.parent
    return tuple(read_section(each, markers, base) for each in entries)


def read_section(
    data: Mapping, markers: Mapping[str, ColumnPos], base: Path = Path(".")
) -> SectionSpec:
    name = require(data, "name")
    defined = with_definition_from_file(data, base)
    return SectionSpec(
        name=name,
        domain=require(defined, "domain"),
        schematic=base / require(defined, "schematic"),
        placement_mode=read_enum(PlacementMode, defined.get("placement_mode")),
        width=defined.get("width", 0),
        depth=defined.get("depth", 0),
        elevation=read_enum(
            SectionElevation, defined.get("elevation"), SectionElevation.FOLLOW
        ),
    )


def with_definition_from_file(data: Mapping, base: Path) -> Mapping:
    # What the assignment says wins, so one definition serves several uses.
    if "file" not in data:
        return data
    held = read_toml(base / data["file"], "section")
    return {**held.get("section", held), **data}


def read_toml(path: Path, kind: str) -> Mapping:
    if not path.exists():
        raise ValueError(f"no {kind} file at {path}")
    return tomllib.loads(path.read_text())


def read_span(
    value: Sequence, markers: Mapping[str, ColumnPos], used_by: Sequence[str]
) -> SpanSpec:
    where = "a stretch assigned " + (", ".join(used_by) or "nothing")
    if len(value) != 2:
        raise ValueError(f"{where} needs a start and a stop, got {list(value)}")
    start, stop = value
    span = SpanSpec(
        start=resolve_marker(start, markers, where),
        stop=resolve_marker(stop, markers, where),
        start_marker=start if isinstance(start, str) else None,
        stop_marker=stop if isinstance(stop, str) else None,
    )
    if span.start == span.stop:
        raise ValueError(f"{where} ends where it starts: {span}")
    return span


def resolve_marker(
    value, markers: Mapping[str, ColumnPos], where: str
) -> ColumnPos:
    if not isinstance(value, str):
        return read_column(value)
    if value not in markers:
        known = ", ".join(sorted(markers)) or "none"
        raise ValueError(
            f"{where} names marker {value!r}, which is not defined. "
            f"Known markers: {known}"
        )
    return markers[value]
