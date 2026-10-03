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
        formation=read_formation(defined.get("formation"), name),
        terrain_overrides=read_block_names(defined.get("terrain_overrides", ()), name),
        terrain_overrides_columns=read_override_columns(defined.get("terrain_overrides_columns")),
        schematic_turns=read_local_height(defined.get("schematic_turns", 0), "schematic_turns"),
        left_schematic=None if "left_schematic" not in defined else base / defined["left_schematic"],
        right_schematic=None if "right_schematic" not in defined else base / defined["right_schematic"],
        half_schematic_turns=read_local_height(defined.get("half_schematic_turns", 0), "half_schematic_turns"),
        surface_mask_min_y=read_local_height(defined.get("surface_mask_min_y"), "surface_mask_min_y"),
        surface_mask_columns=read_mask_columns(defined.get("surface_mask_columns")),
    )


def read_local_height(value, name):
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(f"{name} must be a whole number")
    return value


def read_mask_columns(value):
    if value is None:
        return None
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(not isinstance(v, int) or isinstance(v, bool) for v in value)
            or value[0] > value[1]):
        raise ValueError("surface_mask_columns needs inclusive [minimum, maximum] columns")
    return tuple(value)


def read_override_columns(value):
    if value is None:
        return None
    if (not isinstance(value, (list, tuple))
            or any(not isinstance(v, int) or isinstance(v, bool) for v in value)):
        raise ValueError("terrain_overrides_columns must list whole local columns")
    return frozenset(value)


def read_block_names(data: object, name: str) -> frozenset[str]:
    if not isinstance(data, (list, tuple)) or not all(
        isinstance(each, str) and ":" in each for each in data
    ):
        raise ValueError(
            f"section {name!r} terrain_overrides must list block names such as "
            '"minecraft:andesite"'
        )
    return frozenset(data)


def read_formation(data: object, name: str) -> int | None:
    if data is None:
        return None
    if not isinstance(data, int) or isinstance(data, bool):
        raise ValueError(f"section {name!r} formation must be a whole local height")
    return data


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


def read_steps(data: Mapping, name: str) -> dict[tuple[int, ...], StepMode]:
    # Step modes in the profile's own local axes: a whole column keyed "x,z",
    # or one height of it keyed "x,y,z".
    steps = {}
    for key, mode in data.items():
        try:
            where = tuple(int(part) for part in key.split(","))
        except ValueError:
            where = ()
        if len(where) not in (2, 3):
            raise ValueError(
                f"section {name!r} keys a step by {key!r}; a column is "
                'written "x,z", as in "0,0", and one height of it "x,y,z"'
            )
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
