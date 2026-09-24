from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ..Core.coordinates import ColumnPos
from ..Core.modes import PlacementMode, StepMode
from .spec import SectionElevation, SectionSpec, SpanSpec
from .values import listed_entries, read_column, read_enum, read_toml, require
from ..Terraform.surface_mask import read_surface_mask


def read_sections(
    data: Mapping, markers: Mapping[str, ColumnPos], base: Path = Path(".")
) -> tuple[SectionSpec, ...]:
    entries, base = listed_entries(data, "sections", "section", "sections", base)
    return tuple(read_section(each, markers, base) for each in entries)


def read_section(
    data: Mapping, markers: Mapping[str, ColumnPos], base: Path = Path(".")
) -> SectionSpec:
    name = require(data, "name")
    defined = with_definition_from_file(data, base)
    edge_domains = read_edge_domains(defined.get("edge_domains"), name)
    edge_from = read_edge_from(defined.get("edge_from"), name)
    if edge_from is not None and edge_domains is None:
        raise ValueError(f"section {name!r} edge_from requires edge_domains")
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
        step=read_enum(StepMode, defined.get("step"), StepMode.CHEBYSHEV),
        steps=read_steps(defined.get("steps", {}), name),
        edge_domains=edge_domains,
        edge_from=edge_from,
        surface_mask=read_surface_mask(defined.get("surface_mask")),
    )


def read_edge_domains(data: object, name: str) -> tuple[str, str] | None:
    if data is None:
        return None
    if not isinstance(data, (list, tuple)) or len(data) != 2 or not all(
        isinstance(domain, str) and domain for domain in data
    ):
        raise ValueError(
            f"section {name!r} edge_domains must name left and right domains"
        )
    return data[0], data[1]


def read_edge_from(data: object, name: str) -> int | None:
    if data is None:
        return None
    if not isinstance(data, int) or isinstance(data, bool) or data <= 0:
        raise ValueError(f"section {name!r} edge_from must be a positive column")
    return data


def read_steps(data: Mapping, name: str) -> dict[tuple[int, int], StepMode]:
    # Per-column step modes, keyed "x,z" in the profile's own local axes.
    steps = {}
    for column, mode in data.items():
        try:
            x, _, z = column.partition(",")
            where = (int(x), int(z))
        except ValueError:
            raise ValueError(
                f"section {name!r} keys a step by {column!r}; a column is "
                'written "x,z", as in "0,0"'
            ) from None
        steps[where] = read_enum(StepMode, mode)
    return steps


def with_definition_from_file(data: Mapping, base: Path) -> Mapping:
    # Keys given inline override the file's.
    if "file" not in data:
        return data
    held = read_toml(base / data["file"], "section")
    return {**held.get("section", held), **data}


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
