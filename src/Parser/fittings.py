from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from pathlib import Path

from ..Core.coordinates import BlockPos
from ..Core.modes import ElevationMode, PlacementMode, StepMode
from ..Stamping import authored_headings, mirrorable_headings
from .spec import LaneSpec, Measure, StructureSpec, WireGeometry, WireSpec
from .values import read_enum, require
from .sections import read_local_height
from ..Terraform.surface_mask import read_surface_mask


def read_structure(data: Mapping, base: Path = Path("."), named_paths: frozenset[str] = frozenset()) -> StructureSpec:
    name = require(data, "name")
    assembly = data.get("assembly")
    if assembly is not None and (not isinstance(assembly, str) or not assembly):
        raise ValueError(f"structure {name!r} assembly must name a complete fitting")
    turn = data.get("turn", "heading")
    if turn not in ("heading", "layout"):
        raise ValueError(
            f'structure {name!r} turns by "heading" or "layout", not {turn!r}'
        )
    # A layout-turned structure has a single heading, turned by quarters.
    resolution = 90.0 if turn == "layout" else data.get("resolution", 90.0)
    headings = {
        float(key): base / read_quoted_heading(name, key, value)
        for key, value in require(data, "headings").items()
    }
    check_headings(name, headings, resolution)
    sections = read_fitted_to(data, f"structure {name!r}")
    spacing, angle, measure = read_where_it_sits(data, f"structure {name!r}")
    wire = data.get("wire")
    if wire is not None:
        if not isinstance(wire, str) or not wire:
            raise ValueError(f"structure {name!r} wire must name a wire")
        if any(key in data for key in ("spacing", "angle", "measure", "min_distance")):
            raise ValueError(
                f"structure {name!r} uses its wire's anchors; "
                "omit spacing, angle, measure and min_distance"
            )
    feature_path = data.get("feature_path")
    if feature_path is not None:
        if not isinstance(feature_path, str) or not feature_path:
            raise ValueError(f"structure {name!r} feature_path must be a name")
        if wire is not None:
            raise ValueError(
                f"structure {name!r} uses its wire's anchors; omit feature_path"
            )
        if feature_path in named_paths and any(key in data for key in ("spacing", "angle", "measure", "min_distance")):
            raise ValueError(f"structure {name!r} uses feature_path anchors; put spacing and angle on that path and omit measure and min_distance")
    return StructureSpec(
        name=name,
        domain=require(data, "domain"),
        sections=sections,
        headings=headings,
        resolution=resolution,
        spacing=spacing,
        angle=angle,
        measure=measure,
        offset=BlockPos.coerce(data.get("offset", (0, 0, 0))),
        placement_mode=read_enum(PlacementMode, data.get("placement_mode")),
        elevation_mode=read_enum(
            ElevationMode, data.get("elevation_mode"), ElevationMode.RIGID
        ),
        turn=turn,
        surface_mask=read_surface_mask(data.get("surface_mask")),
        wire=wire,
        min_distance=read_min_distance(data, f"structure {name!r}"),
        feature_path=feature_path,
        schematic_turns=read_local_height(data.get("schematic_turns", 0), "schematic_turns"),
        assembly=assembly,
    )


def read_wire(data: Mapping, base: Path = Path(".")) -> WireSpec:
    name = require(data, "name")
    sections, between = read_what_a_wire_spans(data, name)
    geometry = read_how_a_wire_runs(data, name, between)
    spacing, angle, measure = read_where_it_sits(data, f"wire {name!r}")
    feature_path = data.get("feature_path")
    if feature_path is not None:
        if not isinstance(feature_path, str) or not feature_path:
            raise ValueError(f"wire {name!r} feature_path must be a name")
        if between:
            raise ValueError(f"wire {name!r} feature_path needs a longitudinal wire along sections")
        if any(key in data for key in ("spacing", "angle", "measure", "min_distance")):
            raise ValueError(f"wire {name!r} uses feature_path anchors; omit spacing, angle, measure and min_distance")
    return WireSpec(
        name=name,
        domain=require(data, "domain"),
        sections=sections,
        between=between,
        schematic=base / require(data, "schematic"),
        offset=BlockPos.coerce(data.get("offset", (0, 0, 0))),
        geometry=geometry,
        spacing=spacing,
        angle=angle,
        measure=measure,
        step=read_enum(StepMode, data.get("step"), StepMode.MANHATTAN),
        placement_mode=read_enum(PlacementMode, data.get("placement_mode")),
        surface_mask=read_surface_mask(data.get("surface_mask")),
        min_distance=read_min_distance(data, f"wire {name!r}"),
        feature_path=feature_path,
    )


def read_lane(data: Mapping, base: Path = Path(".")) -> LaneSpec:
    name = require(data, "name")
    on, off = read_dash(data, name)
    schematic, block = read_what_a_lane_paints(data, name)
    return LaneSpec(
        name=name,
        domain=require(data, "domain"),
        sections=read_fitted_to(data, f"lane {name!r}"),
        on=on,
        off=off,
        schematic=None if schematic is None else base / schematic,
        block=block,
        measure=read_enum(Measure, data.get("measure"), Measure.INDEX),
        offset=BlockPos.coerce(data.get("offset", (0, 0, 0))),
        step=read_enum(StepMode, data.get("step"), StepMode.MANHATTAN),
        placement_mode=read_enum(PlacementMode, data.get("placement_mode")),
        surface_mask=read_surface_mask(data.get("surface_mask")),
    )


def read_dash(data: Mapping, name: str) -> tuple[int, int]:
    on, off = require(data, "on"), require(data, "off")
    if on <= 0:
        raise ValueError(f"lane {name!r} keeps at least one block, not {on}")
    if off < 0:
        raise ValueError(f"lane {name!r} skips no blocks at all, not {off}")
    return on, off


def read_what_a_lane_paints(
    data: Mapping, name: str
) -> tuple[str | None, str | None]:
    schematic, block = data.get("schematic"), data.get("block")
    if (schematic is None) == (block is None):
        raise ValueError(
            f"lane {name!r} takes a schematic or a block, not both and not neither"
        )
    return schematic, block


def read_what_a_wire_spans(
    data: Mapping, name: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    sections = tuple(data.get("sections", ()))
    between = tuple(data.get("between", ()))
    if bool(sections) == bool(between):
        raise ValueError(
            f"wire {name!r} runs along sections or between two other wires, "
            "not both and not neither"
        )
    if between and len(between) != 2:
        raise ValueError(
            f"wire {name!r} stretches between two wires, and names "
            f"{len(between)}: {list(between)}"
        )
    return sections, between


def read_how_a_wire_runs(
    data: Mapping, name: str, between: tuple[str, ...]
) -> WireGeometry:
    # A dropper is always a line. A followed wire lies on every point, so it
    # has no anchors to place.
    geometry = read_enum(
        WireGeometry,
        data.get("geometry"),
        WireGeometry.LINE if between else WireGeometry.FOLLOW,
    )
    if geometry is WireGeometry.LINE:
        return geometry
    if between:
        raise ValueError(
            f"wire {name!r} stretches between two wires, which only a line can"
        )
    anchoring = [key for key in ("spacing", "angle", "measure", "min_distance") if key in data]
    if anchoring:
        raise ValueError(
            f"wire {name!r} follows its column through every point; "
            f"{', '.join(anchoring)} only place the anchors of a line "
            f'(geometry = "line")'
        )
    return geometry


def read_fitted_to(data: Mapping, where: str) -> tuple[str, ...]:
    sections = tuple(require(data, "sections"))
    if not sections:
        raise ValueError(f"{where} is assigned to no sections")
    return sections


def read_where_it_sits(data: Mapping, where: str) -> tuple[float, float, Measure]:
    spacing = data.get("spacing", 0.0)
    angle = data.get("angle", 0.0)
    measure = read_enum(Measure, data.get("measure"), Measure.DISTANCE)
    if (not isinstance(spacing, (int, float)) or isinstance(spacing, bool)
            or not isfinite(spacing)):
        raise ValueError(f"{where} spacing must be a finite nonnegative number, got {spacing!r}")
    if spacing < 0:
        raise ValueError(f"{where} has a negative spacing of {spacing}")
    if (not isinstance(angle, (int, float)) or isinstance(angle, bool)
            or not isfinite(angle) or not 0 <= angle <= 180):
        raise ValueError(
            f"{where} has an angle of {angle}; a turn is between 0 and 180 degrees"
        )
    return spacing, angle, measure


def read_min_distance(data: Mapping, where: str) -> float:
    distance = data.get("min_distance", 0.0)
    if (not isinstance(distance, (int, float)) or isinstance(distance, bool)
            or not isfinite(distance) or distance < 0):
        raise ValueError(f"{where} min_distance must be a nonnegative number")
    if (data.get("measure", "distance") == "distance"
            and data.get("spacing", 0) > 0 and distance > data["spacing"]):
        raise ValueError(f"{where} min_distance cannot exceed spacing")
    return distance


def read_quoted_heading(name: str, key: str, value: object) -> str:
    # An unquoted 0.0 key is a dotted TOML key and parses as a table.
    if not isinstance(value, str):
        raise ValueError(
            f"structure {name!r} has a heading written as {key}, which TOML "
            f'reads as a nested key; write it quoted, as "{key}.0"'
        )
    return value


def check_headings(
    name: str, headings: Mapping[float, object], resolution: float
) -> None:
    # The first octant is required; headings past the diagonal are optional
    # and otherwise mirrored from it.
    wanted = authored_headings(resolution)
    allowed = (*wanted, *mirrorable_headings(resolution))
    missing = [heading for heading in wanted if heading not in headings]
    extra = [heading for heading in headings if heading not in allowed]
    if missing or extra:
        raise ValueError(
            f"structure {name!r} at a resolution of {resolution} needs "
            f"{list(wanted)} and may also have {list(mirrorable_headings(resolution))}; "
            f"missing {missing} and given extra {extra}"
        )
