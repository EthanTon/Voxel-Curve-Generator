from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Sequence

from ..Core.CrossSection.cross_section import CrossSection
from ..Core.Domain import DomainRegistry
from ..Core.Path.path_frame import PathFrame
from ..Core.World.block_world import BlockWorld, ColumnMask
from ..Core.block.registry import BlockRegistry
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
from .anchors import column_cells, points_for
from .structure import build_wire


WIRE_LINES = {
    StepMode.MANHATTAN: LineMode.MANHATTAN,
    StepMode.CHEBYSHEV: LineMode.CHEBYSHEV,
}


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
    world.add_field(
        feature.blocks,
        placement_mode=feature.placement_mode,
        terrain_mask_mode=profile.terrain_mask_mode,
        replace=profile.replace,
        connectivity_domain=profile.domain_id,
        clip=cover(frames),
        surface_mask=(
            None if feature.surface_mask is None else
            world.surface_reference.mask(feature.surface_mask)
        ),
    )


def strand_of(
    spec: BuildSpec,
    wire: WireSpec,
    run: DeliveredPath,
    section: CrossSection,
    level: Level,
    clip: Callable[[object], ColumnMask] | None = None,
    runs: Sequence[DeliveredPath] = (),
) -> tuple[BlockPos, ...]:
    if wire.between:
        return tuple(drop_between(spec, wire, run, section, level, clip, runs))
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
    single = run._replace(path=one_line(run.path))
    if spec.geometry is WireGeometry.FOLLOW:
        points = list(along_travel(single.path, section.along(column, spec.step)))
        cells = column_cells(run, runs, column, spec.offset.z)
    else:
        points = list(along_travel(points_for(spec, single), section.along(column)))
        cells = frozenset()
    ends = [(attachment(spec, point, level), point) for point in points]
    # A run of one point, such as a short taper, still lays its block.
    spans = list(zip(ends, ends[1:])) or [(end, end) for end in ends]
    walked: list[BlockPos] = []
    for (before, first), (after, second) in spans:
        masks = () if clip is None else (clip(first), clip(second))
        for block in _run_along_column(before, after, WIRE_LINES[spec.step], cells):
            if not masks or any(mask.contains(block.x, block.z) for mask in masks):
                walked.append(block)
    # Rails drop the loops a join walks, so the wire does too.
    yield from without_detours(walked)


def drop_between(
    spec: BuildSpec,
    wire: WireSpec,
    run: DeliveredPath,
    section: CrossSection,
    level: Level,
    clip: Callable[[object], ColumnMask] | None = None,
    runs: Sequence[DeliveredPath] = (),
) -> Iterator[BlockPos]:
    # Drops between the two wires' actual blocks, not their offsets.
    first, second = (spec.wire(name) for name in wire.between)
    over = tuple(draw_wire(first, run, section, level, clip, runs))
    under = tuple(draw_wire(second, run, section, level, clip, runs))
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
