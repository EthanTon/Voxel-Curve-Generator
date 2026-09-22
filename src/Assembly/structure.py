from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path as FilePath

from ..Core.CrossSection.annotations import FeatureAnnotation
from ..Core.CrossSection.cross_section import CrossSection, PointField
from ..Core.CrossSection.schedule import SectionSchedule, SectionSpan
from ..Core.Domain import DomainRegistry
from ..Core.Feature.feature import (
    ContinuousFeature,
    Feature,
    PointFeature,
)
from ..Core.Field.block_field import BlockField
from ..Core.Path.path import Path
from ..Core.Path.path_frame import PathFrame
from ..Core.block.palette import BlockPalette
from ..Core.block.registry import BlockRegistry
from ..Core.coordinates import BlockPos, ColumnPos
from ..Core.modes import ElevationMode, PlacementMode
from ..Format.region import AIR
from ..Format.schematics import load_schematic, parse_state_key
from ..Generation.Geometry.frames import (
    build_frames,
    build_section_frames,
    elevation_breaks,
    span_stations,
)
from ..Generation.Geometry.padding import padded_path
from ..Generation.Geometry.reach import lateral_reach, required_lead
from ..Stamping import HeadingStamp
from ..Parser.spec import (
    BuildSpec,
    LaneSpec,
    SectionSpec,
    SpanSpec,
    StructureSpec,
    WireSpec,
)


def build_domains(spec: BuildSpec) -> DomainRegistry:
    domains = DomainRegistry()
    for each in order_by_parents(spec):
        domains.create(
            each.name,
            parents=[domains.get(name) for name in each.parents],
            properties=each.properties,
            connectivity=each.connectivity,
            placement=each.placement,
            terrain=each.terrain,
        )
    for interaction in spec.interactions:
        domains.interact(
            interaction.source,
            interaction.target,
            connect=interaction.connect,
            overwrite=interaction.overwrite,
            overrides=interaction.overrides,
            set_properties=interaction.set_properties,
            properties=interaction.properties,
            grants=interaction.grants,
            symmetric=interaction.symmetric,
        )
    return domains


def order_by_parents(spec: BuildSpec):
    placed: list[str] = []
    remaining = list(spec.domains)
    while remaining:
        ready = [each for each in remaining if set(each.parents) <= set(placed)]
        if not ready:
            unplaced = ", ".join(sorted(each.name for each in remaining))
            raise ValueError(f"domains have a parent cycle: {unplaced}")
        for each in ready:
            placed.append(each.name)
            remaining.remove(each)
        yield from ready


def build_path(spec: BuildSpec) -> Path:
    return padded_path(spec.path.points, spec.path.gradients, spec.path.lead)


def resolve_spans(spec: BuildSpec, path: Path) -> dict[str, tuple[tuple[int, int], ...]]:
    # Sections may cover the same stretch as each other -- they stack -- but a
    # section may not cover its own twice, which is what SectionSchedule refuses.
    return {
        name: section_spans(spec.section(name), spans, path)
        for name, spans in stretches_by_section(spec).items()
    }


def stretches_by_section(spec: BuildSpec) -> dict[str, list[SpanSpec]]:
    stretches: dict[str, list[SpanSpec]] = {}
    for assignment in spec.assignments:
        for name in assignment.sections:
            stretches.setdefault(name, []).append(assignment.span)
    return stretches


def section_spans(
    section: SectionSpec, spans: Sequence[SpanSpec], path: Path
) -> tuple[tuple[int, int], ...]:
    # Stretches that meet or overlap become one. Assignments are written
    # stretch by stretch, so a section carried along several in a row is one
    # sweep, and a schedule takes no two spans that touch.
    ranges = sorted(span_indices(section, span, path) for span in spans)
    merged: list[tuple[int, int]] = []
    for start, stop in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return tuple(merged)


def span_indices(section: SectionSpec, span: SpanSpec, path: Path) -> tuple[int, int]:
    start = marker_index(section, span, span.start, path)
    stop = marker_index(section, span, span.stop, path)
    if stop < start:
        raise ValueError(
            f"section {section.name!r} sweeps {span}, which runs backwards "
            f"along the path"
        )
    return start, stop


def marker_index(
    section: SectionSpec, span: SpanSpec, column: ColumnPos, path: Path
) -> int:
    for index, point in enumerate(path.path_points):
        if (point.position.x, point.position.z) == (column.x, column.z):
            return index
    raise ValueError(
        f"section {section.name!r} sweeps {span}, but {tuple(column)} "
        "is not a point on the path"
    )


def build_sections(
    spec: BuildSpec,
    registry: BlockRegistry,
    domains: DomainRegistry,
    spans: Mapping[str, object],
) -> dict[str, CrossSection]:
    # Only the sections an assignment names, so a file of definitions costs
    # nothing for the ones this build does not use.
    return {
        name: build_section(spec.section(name), registry, domains)
        for name in spans
    }


def build_section(
    spec: SectionSpec, registry: BlockRegistry, domains: DomainRegistry
) -> CrossSection:
    # A section records where features sit, not blocks. The schematic is the
    # swept profile, and it becomes a point per column of that profile rather
    # than one point carrying the whole row.
    #
    # That is what makes a diagonal work. A diagonal layout is the two
    # quarters either side of it, staggered half the section's length apart
    # and joined at their outermost column. Joining them needs the columns to
    # be separate points: one point carrying the whole row has nothing to
    # split between the two halves, so the row is painted twice, once in each
    # half, and the halves neither meet nor tile.
    #
    # The column on the centreline is named for the section itself. It is the
    # one a fitting rides, so a lane or a wire sees the path rather than
    # eleven copies of it, and it is there whether the profile paints
    # anything on the centreline or not.
    blocks = load_field(spec.schematic, registry.palette, spec.name)
    check_extent(spec, blocks)
    mode = placing_mode(spec, domains)
    domain = domains.get(spec.domain)
    return CrossSection(
        points=PointField(
            FeatureAnnotation(
                name=spec.name if column == (0, 0) else f"{spec.name}[{column[0]},{column[1]}]",
                position=BlockPos(column[0], 0, column[1]),
                feature=PointFeature(
                    blocks=stack,
                    placement_mode=mode,
                    domain=domain,
                    offset=None,
                    # Each delivered point carries the level of the frame it
                    # landed in, so a column sits at that level.
                    elevation_mode=ElevationMode.RIGID,
                ),
                step_mode=spec.steps.get(column, spec.step),
                section=spec.name,
            )
            for column, stack in columns_of(blocks, registry).items()
        )
    )


def columns_of(
    blocks: BlockField, registry: BlockRegistry
) -> dict[tuple[int, int], BlockField]:
    # The profile split into one field per column, each holding that column's
    # blocks at its own local heights and nothing else. The centreline column
    # is always there, empty if the profile paints nothing on it.
    columns: dict[tuple[int, int], BlockField] = {
        (0, 0): BlockField(registry.palette)
    }
    for position, block_id in blocks.iter_block_ids():
        column = columns.setdefault(
            (position.x, position.z), BlockField(registry.palette)
        )
        column.set_block(
            BlockPos(0, position.y, 0), blocks.palette.get_state(block_id)
        )
    return columns


def check_extent(spec: SectionSpec, field: BlockField) -> None:
    bounds = field.bounds
    measured = {
        "width": bounds.maximum.x - bounds.minimum.x + 1,
        "depth": bounds.maximum.z - bounds.minimum.z + 1,
    }
    for name, declared in (("width", spec.width), ("depth", spec.depth)):
        if declared and declared != measured[name]:
            raise ValueError(
                f"section {spec.name!r} declares a {name} of {declared}, but "
                f"its schematic is {measured[name]} across"
            )


def build_schedules(
    sections: dict[str, CrossSection],
    spans: dict[str, tuple[tuple[int, int], ...]],
    length: int,
) -> dict[str, SectionSchedule]:
    # One schedule per section, not one for all of them. Sections stack: two
    # may cover the same stretch, and a schedule refuses spans that overlap.
    return {
        name: SectionSchedule(
            [SectionSpan(start, stop, sections[name]) for start, stop in ranges],
            length,
        )
        for name, ranges in spans.items()
    }


def build_path_frames(
    spec: BuildSpec,
    schedules: dict[str, SectionSchedule],
    path: Path,
) -> list[PathFrame]:
    # The frames a stamp is clipped to: cut at every change of level and at
    # every point a section starts or stops, so no frame spans two levels or
    # two sections.
    sections = [span.section for schedule in schedules.values() for span in schedule.spans]
    check_lead(spec, path, sections)
    elevation = build_elevation_frames(spec, path, sections)
    stations = sorted(
        {
            *(
                index
                for schedule in schedules.values()
                for index in span_stations(schedule, path)
            ),
            *(frame.path_start for frame in elevation[1:]),
        }
    )
    return build_section_frames(
        path, elevation, stations, half_width=lateral_reach(sections)
    )


def build_elevation_frames(
    spec: BuildSpec, path: Path, sections: Sequence[CrossSection]
) -> list[PathFrame]:
    # One frame per level. The path's own points carry the heights, so a
    # change between two of them is a break and nothing else declares one.
    return build_frames(
        path,
        path.bounded_points[0].position.y,
        elevation_breaks(path),
        half_width=lateral_reach(sections),
        lead_in=spec.path.lead,
        lead_out=spec.path.lead,
    )


def check_lead(spec: BuildSpec, path: Path, sections: Sequence[CrossSection]) -> None:
    # A section near a boundary reaches past it, and that reach is pasted from
    # the lead. Too short a lead and the ends come out unfinished.
    needed = required_lead(path, sections)
    if needed > spec.path.lead:
        raise ValueError(
            f"the sections reach far enough to need a lead of {needed} points "
            f"at each end, and the path gives {spec.path.lead}"
        )


def build_structure(
    spec: StructureSpec, registry: BlockRegistry, domains: DomainRegistry
) -> HeadingStamp:
    return HeadingStamp(
        {
            heading: build_structure_feature(spec, schematic, registry, domains)
            for heading, schematic in spec.headings.items()
        },
        spec.resolution,
        domains,
    )


def build_structure_feature(
    spec: StructureSpec,
    schematic: FilePath,
    registry: BlockRegistry,
    domains: DomainRegistry,
) -> Feature:
    return PointFeature(
        blocks=load_field(schematic, registry.palette, spec.name),
        placement_mode=placing_mode(spec, domains),
        domain=domains.get(spec.domain),
        offset=spec.offset,
        elevation_mode=spec.elevation_mode,
    )


def build_lane(
    spec: LaneSpec, registry: BlockRegistry, domains: DomainRegistry
) -> Feature:
    return PointFeature(
        blocks=lane_blocks(spec, registry),
        placement_mode=placing_mode(spec, domains),
        domain=domains.get(spec.domain),
        offset=spec.offset,
        elevation_mode=ElevationMode.RIGID,
    )


def lane_blocks(spec: LaneSpec, registry: BlockRegistry) -> BlockField:
    if spec.schematic is not None:
        return load_field(spec.schematic, registry.palette, spec.name)
    blocks = BlockField(registry.palette)
    blocks.set_block(BlockPos(0, 0, 0), parse_state_key(spec.block))
    return blocks


def build_wire(
    spec: WireSpec,
    positions: Sequence[BlockPos],
    registry: BlockRegistry,
    domains: DomainRegistry,
) -> ContinuousFeature:
    strand = load_field(spec.schematic, registry.palette, spec.name)
    feature = ContinuousFeature(
        blocks=BlockField(registry.palette),
        placement_mode=placing_mode(spec, domains),
        domain=domains.get(spec.domain),
        offset=None,
        elevation_mode=ElevationMode.RIGID,
    )
    for position in positions:
        for local, block_id in strand.iter_block_ids():
            feature.add(position + local, strand.palette.get_state(block_id))
    return feature


def placing_mode(
    spec: SectionSpec | StructureSpec | WireSpec | LaneSpec,
    domains: DomainRegistry,
) -> PlacementMode:
    if spec.placement_mode is not None:
        return spec.placement_mode
    return domains.profile(spec.domain).placement_mode


def load_field(path: FilePath, palette: BlockPalette, section: str) -> BlockField:
    if not path.exists():
        raise ValueError(f"section {section!r} has no schematic at {path}")
    region, _ = load_schematic(path)
    field = BlockField(palette)
    for position, state in region.iter_occupied():
        if state != AIR:
            field.set_block(position, state)
    if field.bounds is None:
        raise ValueError(f"section {section!r} has an empty schematic at {path}")
    return field
