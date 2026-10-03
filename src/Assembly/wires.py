from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import cache

from ..Core.CrossSection.cross_section import CrossSection
from ..Core.Feature.feature_path import FeaturePath, FeaturePathPoint
from ..Core.Domain import DomainRegistry
from ..Core.Path.path_frame import PathFrame
from ..Core.World.block_world import BlockWorld, ColumnMask
from ..Core.block.registry import BlockRegistry
from ..Core.block.connectivity_options import ConnectionAction
from ..Core.block.states import BlockState
from ..Core.coordinates import BlockPos
from ..Core.line import line
from ..Core.modes import LineMode, StepMode
from ..Core.plan_index import NearestInPlan
from ..Generation.Sweep.delivery import cover
from ..Generation.Sweep.paths import (
    DeliveredPath,
    Level,
    along_travel,
    one_line,
    without_detours,
)
from ..Parser.spec import BuildSpec, WireGeometry, WireSpec
from ..Stamping import laid_offset
from .anchors import column_cells, points_for, wire_anchors
from .structure import build_wire


WIRE_LINES = {
    StepMode.MANHATTAN: LineMode.MANHATTAN,
    StepMode.CHEBYSHEV: LineMode.CHEBYSHEV,
}


def dependent_strand(path: FeaturePath, step: StepMode, clip=None) -> tuple[BlockPos, ...]:
    # Join world anchors without selecting or offsetting them again.
    cells = []
    previous = None
    for point in path:
        here = BlockPos(point.position.x, point.position.y + point.y_offset, point.position.z)
        segment = (here,) if previous is None else line(previous[0], here, WIRE_LINES[step])
        for cell in segment:
            if clip is None or clip(point).contains(cell.x, cell.z) or (previous is not None and clip(previous[1]).contains(cell.x, cell.z)):
                if not cells or cells[-1] != cell:
                    cells.append(cell)
        previous = here, point
    return tuple(cells)


def lay_wire(
    spec: WireSpec,
    positions: Sequence[BlockPos],
    world: BlockWorld,
    registry: BlockRegistry,
    domains: DomainRegistry,
    frames: Sequence[PathFrame],
) -> None:
    # One placement with the strand's own world heights.
    if not positions:
        return
    feature = build_wire(spec, positions, registry, domains)
    profile = domains.profile(feature.domain)
    centre = feature.blocks.get_block_id(positions[0])
    if centre is not None and spec.step is StepMode.MANHATTAN and not spec.between:
        routed = _around_fittings(
            spec, positions, world, registry, domains, registry.get_state(centre), cover(frames)
        )
        if routed != tuple(positions):
            feature = build_wire(spec, routed, registry, domains)
    world.add_field(
        feature.blocks,
        placement_mode=feature.placement_mode,
        terrain_mask_mode=profile.terrain_mask_mode,
        replace=profile.replace,
        block_replacement=profile.replacement,
        connectivity_domain=profile.domain_id,
        clip=cover(frames),
        surface_mask=(
            None if feature.surface_mask is None else
            world.surface_reference.mask(feature.surface_mask)
        ),
    )


def _around_fittings(
    spec: WireSpec, positions: Sequence[BlockPos], world: BlockWorld,
    registry: BlockRegistry, domains: DomainRegistry, state: BlockState, clip: ColumnMask,
) -> tuple[BlockPos, ...]:
    # The stamped fitting owns its cells. An explicit connection override
    # lets the strand meet an attachment there, while the rest is an obstacle.
    policy = domains.compile()
    domain = policy.id_of(spec.domain)
    info = registry.get_info_for_state(state)

    @cache
    def attachment_at(position):
        source = world.source_at(position)
        if source is None or policy.can_overwrite(domain, source.domain):
            return False
        target = registry.get_state(source.block_id)
        target_info = registry.get_info(source.block_id)
        return bool(policy.options and any(
            policy.options.decide(state, domain, target, source.domain, direction,
                                  info.behavior, target_info.behavior, info.tags, target_info.tags).action
            is ConnectionAction.CONNECT
            for direction in ("north", "south", "east", "west")
        ))

    @cache
    def open_at(position):
        source = world.source_at(position)
        return (source is None or policy.can_overwrite(domain, source.domain)
                or attachment_at(position))

    positions = list(positions)
    for index in (0, len(positions) - 1):
        point = positions[index]
        if open_at(point):
            continue
        # A neighbouring strand may finish in an arm cell at a section
        # handover. Meet the nearby fitting's attachment instead.
        nearby = [point + (dx, 0, dz) for dx in range(-2, 3) for dz in range(-2, 3)
                  if attachment_at(point + (dx, 0, dz))]
        if nearby:
            positions[index] = min(nearby, key=lambda cell:
                                   (cell.x - point.x) ** 2 + (cell.z - point.z) ** 2)

    def bridge(start, end):
        # Local face-connected detour, preferring the strand's own height.
        # Enter and leave an attachment horizontally, so panes can join it.
        bounds = tuple((min(start[axis], end[axis]) - (2 if axis != 1 else 0),
                        max(start[axis], end[axis]) + (2 if axis != 1 else 0))
                       for axis in range(3))
        queue = deque((start,))
        came = {start: None}
        while queue:
            here = queue.popleft()
            if here == end:
                path = []
                while here is not None:
                    path.append(here)
                    here = came[here]
                return tuple(reversed(path))
            for offset in ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1),
                           (0, 1, 0), (0, -1, 0)):
                step = here + offset
                if step in came or not all(low <= step[axis] <= high
                                           for axis, (low, high) in enumerate(bounds)):
                    continue
                if offset[1] and ((here == start and attachment_at(start))
                                  or (step == end and attachment_at(end))):
                    continue
                if clip.contains(step.x, step.z) and open_at(step):
                    came[step] = here
                    queue.append(step)
        raise ValueError(f"wire {spec.name!r} cannot pass its fitting between {start} and {end}")

    walked = []
    index = 0
    while index < len(positions):
        point = positions[index]
        if open_at(point):
            if walked and sum(abs(a - b) for a, b in zip(walked[-1], point)) > 1:
                walked.extend(bridge(walked[-1], point)[1:])
            else:
                walked.append(point)
            index += 1
            continue
        stop = index
        while stop < len(positions) and not open_at(positions[stop]):
            stop += 1
        if not walked or stop == len(positions):
            raise ValueError(f"wire {spec.name!r} has a blocked endpoint at {point}")
        walked.extend(bridge(walked[-1], positions[stop])[1:])
        index = stop + 1
    return tuple(dict.fromkeys(walked))


def strand_of(
    spec: BuildSpec,
    wire: WireSpec,
    run: DeliveredPath,
    section: CrossSection,
    level: Level,
    clip: Callable[[object], ColumnMask] | None = None,
    runs: Sequence[DeliveredPath] = (),
    dependent_paths: Mapping[str, Sequence[FeaturePath]] | None = None,
) -> tuple[BlockPos, ...]:
    if wire.between:
        return tuple(drop_between(spec, wire, run, section, level, clip, runs, dependent_paths))
    return tuple(draw_wire(wire, run, section, level, clip, runs))


def draw_wire(
    spec: WireSpec,
    run: DeliveredPath,
    section: CrossSection,
    level: Level,
    clip: Callable[[object], ColumnMask] | None = None,
    runs: Sequence[DeliveredPath] = (),
) -> Iterator[BlockPos]:
    # Followed, the column the wire hangs over at every point and each
    # point's own level, walking the column's cells between them; as a line,
    # straight from anchor to anchor whatever the track does between. Each
    # span is held to the frames near its ends.
    column = spec.offset.x
    if spec.geometry is WireGeometry.FOLLOW:
        single = run._replace(path=one_line(run.path))
        points = list(along_travel(single.path, section.along(column, spec.step)))
        cells = column_cells(run, runs, column, spec.offset.z)
    else:
        points = list(along_travel(wire_anchors(spec, run), section.along(column)))
        cells = frozenset()
    yield from _draw_points(spec, points, level, clip, cells)


def line_strand(
    spec: WireSpec,
    route: Sequence[tuple[DeliveredPath, FeaturePath]],
    sections: Mapping[str, CrossSection],
    level: Level,
    clip: Callable[[object], ColumnMask] | None = None,
) -> tuple[BlockPos, ...]:
    points = [
        point for run, anchors in route
        for point in along_travel(anchors, sections[run.section_name].along(spec.offset.x))
    ]
    return tuple(_draw_points(spec, points, level, clip, frozenset()))


def _draw_points(
    spec: WireSpec,
    points: Sequence[FeaturePathPoint],
    level: Level,
    clip: Callable[[object], ColumnMask] | None,
    cells: frozenset[tuple[int, int]],
) -> Iterator[BlockPos]:
    ends = [(attachment(spec, point, level), point) for point in points]
    # A run of one point, such as a short taper, still lays its block.
    spans = list(zip(ends, ends[1:])) or [(end, end) for end in ends]
    walked: list[BlockPos] = []
    for (before, first), (after, second) in spans:
        masks = () if clip is None else (clip(first), clip(second))
        for block in _run_along_column(before, after, WIRE_LINES[spec.step], cells):
            if not masks or any(mask.contains(block.x, block.z) for mask in masks):
                walked.append(block)
    if spec.geometry is WireGeometry.LINE:
        # A line's anchors carry stamped fittings. Removing a loop across
        # two spans can erase the anchor itself and leave its pole off-wire.
        yield from dict.fromkeys(walked)
    else:
        yield from without_detours(walked)


def drop_between(
    spec: BuildSpec,
    wire: WireSpec,
    run: DeliveredPath,
    section: CrossSection,
    level: Level,
    clip: Callable[[object], ColumnMask] | None = None,
    runs: Sequence[DeliveredPath] = (),
    dependent_paths: Mapping[str, Sequence[FeaturePath]] | None = None,
) -> Iterator[BlockPos]:
    # Drops between the two wires' actual blocks, not their offsets.
    first, second = (spec.wire(name) for name in wire.between)
    def strand(wire):
        if wire.feature_path is None:
            return tuple(draw_wire(wire, run, section, level, clip, runs))
        if dependent_paths is None:
            raise ValueError(f"wire {wire.name!r} needs its resolved feature_path for droppers")
        start, stop = min(p.path_index for p in run.path), max(p.path_index for p in run.path)
        return tuple(cell for path in dependent_paths[f"wire:{wire.name}"]
                     if path[0].path_index <= stop and path[-1].path_index >= start
                     for cell in dependent_strand(path, wire.step, clip))

    over, under = strand(first), strand(second)
    if not over or not under:
        return
    over_index = NearestInPlan((p.x, p.z, p) for p in over)
    under_index = NearestInPlan((p.x, p.z, p) for p in under)
    for point in points_for(wire, run):
        yield from line(
            over_index.nearest(point.position.x, point.position.z),
            under_index.nearest(point.position.x, point.position.z),
            WIRE_LINES[wire.step],
        )


def _run_along_column(
    start: BlockPos,
    end: BlockPos,
    mode: LineMode,
    cells: frozenset[tuple[int, int]],
) -> tuple[BlockPos, ...]:
    # The run between two anchors. Where a face-connected straight line
    # would leave the column's cells, it walks those cells instead (as the
    # rail below does) and climbs on the last step.
    straight = line(start, end, mode)
    if mode is LineMode.CHEBYSHEV or not cells or all(
        (block.x, block.z) in cells for block in straight
    ):
        return straight
    route = _route_through(cells, (start.x, start.z), (end.x, end.z))
    if route is None:
        return straight
    laid = [start]
    for x, z in route[1:-1]:
        laid.append(BlockPos(x, start.y, z))
    laid.append(end)
    blocks: list[BlockPos] = [start]
    for before, after in zip(laid, laid[1:]):
        blocks.extend(line(before, after, mode)[1:])
    return tuple(blocks)


def _route_through(
    cells: frozenset[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]] | None:
    # The shortest face-connected walk over `cells`, within a block of the
    # box the ends span.
    if start not in cells or end not in cells:
        return None
    left, right = min(start[0], end[0]) - 1, max(start[0], end[0]) + 1
    back, front = min(start[1], end[1]) - 1, max(start[1], end[1]) + 1
    came: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue = deque((start,))
    while queue:
        here = queue.popleft()
        if here == end:
            route = [here]
            while came[route[-1]] is not None:
                route.append(came[route[-1]])
            return route[::-1]
        for dx, dz in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            step = (here[0] + dx, here[1] + dz)
            if (
                step in cells
                and step not in came
                and left <= step[0] <= right
                and back <= step[1] <= front
            ):
                came[step] = here
                queue.append(step)
    return None


def attachment(
    spec: WireSpec, point, level: Level
) -> BlockPos:
    # An anchor at the track's level in the column it lands in.
    local = BlockPos.coerce(point.position) + laid_offset(
        spec.offset, point.laid_at, point.mirrored
    )
    return BlockPos(
        local.x, local.y + level(local.x, local.z, point.path_index), local.z
    )
