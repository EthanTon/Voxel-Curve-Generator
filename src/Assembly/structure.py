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
from ..Format.schematics import load_blocks, parse_state_key
from ..Generation.Geometry.frames import (
    build_frames,
    build_section_frames,
    elevation_breaks,
    span_stations,
)
from ..Generation.Geometry.padding import padded_path
from ..Generation.Geometry.reach import lateral_reach, required_lead
from ..Stamping import HeadingStamp, TurnStamp
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
    # Each section's merged path-index spans.
    starts = {
        (assignment.span.start.x, assignment.span.start.z)
        for assignment in spec.assignments
    }
    return {
        name: section_spans(spec.section(name), spans, path, starts)
        for name, spans in stretches_by_section(spec).items()
    }


def stretches_by_section(spec: BuildSpec) -> dict[str, list[SpanSpec]]:
    stretches: dict[str, list[SpanSpec]] = {}
    for assignment in spec.assignments:
        for name in assignment.sections:
            stretches.setdefault(name, []).append(assignment.span)
    return stretches


def section_spans(
    section: SectionSpec, spans: Sequence[SpanSpec], path: Path,
    starts: set[tuple[int, int]] | None = None,
) -> tuple[tuple[int, int], ...]:
    # Merges stretches that meet or overlap.
    ranges = sorted(span_indices(section, span, path, starts) for span in spans)
    merged: list[tuple[int, int]] = []
    for start, stop in ranges:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return tuple(merged)


def span_indices(
    section: SectionSpec, span: SpanSpec, path: Path,
    starts: set[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    start = marker_index(section, span, span.start, path)
    stop = marker_index(section, span, span.stop, path)
    # A marker shared with the next assignment belongs to that one.
    if (
        starts is not None
        and (span.stop.x, span.stop.z) in starts
        and stop < path.last_index
    ):
        stop -= 1
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
    # Only the sections some assignment uses.
    return {
        name: build_section(spec.section(name), registry, domains)
        for name in spans
    }


def build_section(
    spec: SectionSpec, registry: BlockRegistry, domains: DomainRegistry
) -> CrossSection:
    # One annotation per column of the profile schematic (with every block
    # of that column), so a diagonal join can stagger the columns. The
    # centreline column, named after the section, is what fittings ride.
    blocks = load_field(spec.schematic, registry.palette, spec.name)
    check_extent(spec, blocks)
    mode = placing_mode(spec, domains)
    domain = domains.get(spec.domain)
    annotations: list[FeatureAnnotation] = []
    for column, stack in columns_of(blocks, registry).items():
        edge = spec.edge_domains is not None and (
            spec.edge_from is not None and abs(column[0]) >= spec.edge_from
            or any(local.y > 0 for local, _ in stack.iter_block_ids())
        )
        if edge and column[0] == 0:
            raise ValueError(
                f"section {spec.name!r} has a raised edge on its centreline"
            )
        position = BlockPos(column[0], 0, column[1])
        step = spec.steps.get(column, spec.step)
        if stack.bounds is None and column != (0, 0):
            continue
        placed_domain = (
            domains.get(spec.edge_domains[column[0] > 0])
            if edge else domain
        )
        annotations.append(FeatureAnnotation(
            name=(
                f"{spec.name}_edge[{column[0]},{column[1]}]" if edge
                else spec.name if column == (0, 0)
                else f"{spec.name}[{column[0]},{column[1]}]"
            ),
            position=position,
            feature=PointFeature(
                blocks=stack,
                placement_mode=(
                    spec.placement_mode
                    or domains.profile(placed_domain).placement_mode
                ) if edge else mode,
                domain=placed_domain, offset=None,
                elevation_mode=ElevationMode.RIGID,
                surface_mask=spec.surface_mask,
            ),
            step_mode=step,
            section=spec.name,
            clip_to_span=edge,
        ))
    return CrossSection(points=PointField(annotations))


def columns_of(
    blocks: BlockField, registry: BlockRegistry
) -> dict[tuple[int, int], BlockField]:
    # One field per (x, z) column of the profile; the centreline always exists.
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
    # One schedule per section, since sections may overlap each other.
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
    # Frames cut at every level change and every section start or stop.
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
    return build_frames(
        path,
        path.bounded_points[0].position.y,
        elevation_breaks(path),
        half_width=lateral_reach(sections),
        lead_in=spec.path.lead,
        lead_out=spec.path.lead,
    )


def check_lead(spec: BuildSpec, path: Path, sections: Sequence[CrossSection]) -> None:
    needed = required_lead(path, sections)
    if needed > spec.path.lead:
        raise ValueError(
            f"the sections reach far enough to need a lead of {needed} points "
            f"at each end, and the path gives {spec.path.lead}"
        )


def build_structure(
    spec: StructureSpec, registry: BlockRegistry, domains: DomainRegistry
) -> HeadingStamp | TurnStamp:
    if spec.turn == "layout":
        [schematic] = spec.headings.values()
        return TurnStamp(
            build_structure_feature(spec, schematic, registry, domains), domains
        )
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
        surface_mask=spec.surface_mask,
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
        surface_mask=spec.surface_mask,
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
        surface_mask=spec.surface_mask,
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
    field = load_blocks(path, palette)
    if field.bounds is None:
        raise ValueError(f"section {section!r} has an empty schematic at {path}")
    return field
