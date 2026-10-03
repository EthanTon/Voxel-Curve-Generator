from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path as FilePath

from ..Core.CrossSection.annotations import FeatureAnnotation
from ..Core.CrossSection.cross_section import CrossSection, PointField
from ..Core.CrossSection.schedule import SectionSchedule, SectionSpan
from ..Core.Domain import DomainRegistry
from ..Core.Feature.feature import (
    ContinuousFeature,
    DependentFeature,
    Feature,
    PointFeature,
)
from ..Core.Field.block_field import BlockField
from ..Core.Feature.dependent_path import DependentFeaturePath
from ..Core.World.block_world import BlockTransform
from ..Core.Path.path import Path
from ..Core.Path.path_frame import PathFrame
from ..Core.block.palette import BlockPalette
from ..Core.block.registry import BlockRegistry
from ..Core.coordinates import BlockPos, ColumnPos
from ..Core.modes import ElevationMode, PlacementMode, StepMode
from ..Format.schematics import load_blocks, parse_state_key
from ..Generation.Geometry.frames import (
    build_frames,
    build_section_frames,
    elevation_breaks,
    heading_stations,
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
    StationSpanSpec,
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
            replacement=each.replacement,
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
        if isinstance(assignment.span, SpanSpec)
    }
    return {
        name: section_spans(spec.section(name), spans, path, starts)
        for name, spans in stretches_by_section(spec).items()
    }


def stretches_by_section(spec: BuildSpec) -> dict[str, list[SpanSpec | StationSpanSpec]]:
    stretches: dict[str, list[SpanSpec | StationSpanSpec]] = {}
    for assignment in spec.assignments:
        for name in assignment.sections:
            stretches.setdefault(name, []).append(assignment.span)
    return stretches


def section_spans(
    section: SectionSpec, spans: Sequence[SpanSpec | StationSpanSpec], path: Path,
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
    section: SectionSpec, span: SpanSpec | StationSpanSpec, path: Path,
    starts: set[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    if isinstance(span, StationSpanSpec):
        start, stop = span.start + path.first_index, span.stop + path.first_index
        if stop > path.last_index:
            raise ValueError(
                f"section {section.name!r} sweeps {span}, past the last authored path station "
                f"({path.last_index - path.first_index})"
            )
        return start, stop
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
    blocks = authored_field(spec.schematic, registry, spec.name, spec.schematic_turns)
    for side, schematic in ((-1, spec.left_schematic), (1, spec.right_schematic)):
        if schematic is None:
            continue
        half = authored_field(schematic, registry, spec.name, spec.half_schematic_turns)
        combined = BlockField(registry.palette)
        for source, replacing in ((blocks, False), (half, True)):
            for position, block_id in source.iter_block_ids():
                on_side = position.x < 0 if side < 0 else position.x > 0
                if on_side == replacing:
                    combined.set_block(position, source.palette.get_state(block_id))
        blocks = combined
    check_extent(spec, blocks)
    placing = placing_mode(spec, domains)
    domain = domains.get(spec.domain)
    annotations: list[FeatureAnnotation] = []
    columns = columns_of(blocks, registry)
    depth_counts: dict[int, int] = {}
    for x, _ in columns:
        depth_counts[x] = depth_counts.get(x, 0) + 1
    for column, stack in columns.items():
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
        pieces = pieces_by_step(stack, column, spec.steps, step, registry)
        placed_domain = (
            domains.get(spec.edge_domains[column[0] > 0])
            if edge else domain
        )
        name = (
            f"{spec.name}_edge[{column[0]},{column[1]}]" if edge
            else spec.name if column == (0, 0)
            else f"{spec.name}[{column[0]},{column[1]}]"
        )
        for mode, piece in pieces.items():
            if piece.bounds is None and (column != (0, 0) or mode is not step):
                continue
            annotations.append(FeatureAnnotation(
                name=name if mode is step else f"{name}@{mode.value}",
                position=position,
                feature=PointFeature(
                    blocks=piece,
                    placement_mode=(
                        spec.placement_mode
                        or domains.profile(placed_domain).placement_mode
                    ) if edge else placing,
                    domain=placed_domain, offset=None,
                    elevation_mode=ElevationMode.RIGID,
                    surface_mask=(spec.surface_mask if spec.surface_mask_columns is None
                                  or spec.surface_mask_columns[0] <= column[0] <= spec.surface_mask_columns[1]
                                  else None),
                    surface_mask_min_y=spec.surface_mask_min_y,
                    yielding=(spec.terrain_overrides if spec.terrain_overrides_columns is None
                              or column[0] in spec.terrain_overrides_columns else frozenset()),
                ),
                step_mode=mode,
                section=spec.name,
                clip_to_span=edge,
                # Depth slices of one lateral column overlap when swept.
                # Share ownership so a recessed roof does not clip away
                # the floor or panels in the adjacent slice.
                column=(
                    f"{spec.name}[{column[0]},*]" if depth_counts[column[0]] > 1
                    else name if len(pieces) > 1 else None
                ),
            ))
    return CrossSection(points=PointField(annotations))


def pieces_by_step(
    stack: BlockField,
    column: tuple[int, int],
    steps: Mapping[tuple[int, ...], StepMode],
    step: StepMode,
    registry: BlockRegistry,
) -> dict[StepMode, BlockField]:
    # A column's blocks by step mode: heights keyed "x,y,z" step their own
    # way, the rest as the column does.
    if not any(len(key) == 3 and (key[0], key[2]) == column for key in steps):
        return {step: stack}
    pieces = {step: BlockField(registry.palette)}
    for local, block_id in stack.iter_block_ids():
        mode = steps.get((column[0], local.y, column[1]), step)
        pieces.setdefault(mode, BlockField(registry.palette)).set_block(
            local, stack.palette.get_state(block_id)
        )
    return pieces


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
    # Frames cut at levels, section ends and bends that turn back on a leg.
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
    stations = sorted({*stations, *heading_stations(path, stations, lateral_reach(sections))})
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
    spec: StructureSpec, registry: BlockRegistry, domains: DomainRegistry,
    dependency: DependentFeaturePath | None = None,
) -> HeadingStamp | TurnStamp:
    if spec.turn == "layout":
        [schematic] = spec.headings.values()
        return TurnStamp(
            build_structure_feature(spec, schematic, registry, domains, dependency), domains
        )
    return HeadingStamp(
        {
            heading: build_structure_feature(spec, schematic, registry, domains, dependency)
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
    dependency: DependentFeaturePath | None = None,
) -> Feature:
    kind = PointFeature if dependency is None else DependentFeature
    return kind(
        blocks=authored_field(schematic, registry, spec.name, spec.schematic_turns),
        placement_mode=placing_mode(spec, domains),
        domain=domains.get(spec.domain),
        offset=spec.offset,
        elevation_mode=spec.elevation_mode,
        surface_mask=spec.surface_mask,
        **({} if dependency is None else {"dependency": dependency}),
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


def authored_field(path: FilePath, registry: BlockRegistry, name: str, turns: int) -> BlockField:
    field = load_field(path, registry.palette, name)
    if turns % 4 == 0:
        return field
    from ..Core.block.rotation import rotate_block_state
    transform = BlockTransform(quarter_turns=turns)
    rotated = BlockField(registry.palette)
    for position, block_id in field.iter_block_ids():
        state = field.palette.get_state(block_id)
        rotated.set_block(transform.apply(position), rotate_block_state(state, registry.get_info(block_id), turns))
    return rotated


def load_field(path: FilePath, palette: BlockPalette, section: str) -> BlockField:
    if not path.exists():
        raise ValueError(f"section {section!r} has no schematic at {path}")
    field = load_blocks(path, palette)
    if field.bounds is None:
        raise ValueError(f"section {section!r} has an empty schematic at {path}")
    return field
