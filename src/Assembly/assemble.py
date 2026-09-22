from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path as FilePath

from ..Core.CrossSection.cross_section import CrossSection
from ..Core.CrossSection.schedule import SectionSchedule
from ..Core.CrossSection.turns import tolerance_for
from ..Core.Domain import DomainRegistry
from ..Core.Feature.feature_path import (
    FeaturePath,
    spacing_rules,
)
from ..Core.Path.path import Path
from ..Core.Path.path_frame import PathFrame
from ..Core.World.block_world import BlockWorld
from ..Core.block.registry import BlockRegistry
from ..Core.coordinates import BlockPos
from ..Core.modes import PlacementMode, StepMode
from ..Core.line import line
from ..Format.litematic import write_litematic
from ..Format.region import BlockRegion
from ..Format.schematics import SCHEM_SUFFIX, write_schematic
from ..Generation.Geometry.frames import elevation_from_frames
from ..Generation.Geometry.orientation import SectionOrientation
from ..Generation.Sweep.delivery import cover
from ..Generation.Sweep.paths import (
    DeliveredPath,
    along_travel,
    at_their_own_level,
    one_line,
    dashed,
    deliver_paths,
    face_connected,
    rewalk,
)
from ..Terraform import Alignment, Terraforming, load_terrain, terraform
from ..Stamping import (
    ClipRule,
    TurnStamp,
    in_frames,
    laid_offset,
    resolve,
    stamp_along,
)
from ..Parser.spec import (
    BuildSpec,
    LaneSpec,
    Measure,
    StructureSpec,
    WireSpec,
)
from .structure import (
    build_domains,
    build_lane,
    build_path,
    build_path_frames,
    build_schedules,
    build_sections,
    build_structure,
    build_wire,
    resolve_spans,
)


def assemble(spec: BuildSpec) -> BlockWorld:
    # The catalog is what makes connectivity mean anything: without it a
    # rail is an anonymous block and never takes a shape.
    registry = BlockRegistry.for_vanilla_26_2()
    domains = build_domains(spec)
    world = BlockWorld(registry)
    domains.bind(world)

    path = build_path(spec)
    spans = resolve_spans(spec, path)
    sections = build_sections(spec, registry, domains, spans)
    schedules = build_schedules(sections, spans, len(path.path_points))
    frames = build_path_frames(spec, schedules, path)
    clip = in_frames(frames)

    runs = sweep(spec, schedules, path, frames)
    fittings: dict[str, list[BlockPos]] = {}
    for laying in in_build_order(
        spec, everything(spec, runs, sections, world, registry, domains, frames, clip)
    ):
        placed = laying.lay()
        if laying.fitting:
            fittings.setdefault(laying.fitting, []).extend(
                blocks_of(world, placed or ())
            )

    reshape(spec, world, registry, domains, path, fittings)
    resolve(world, registry, domains)
    return world


def reshape(
    spec: BuildSpec,
    world: BlockWorld,
    registry: BlockRegistry,
    domains: DomainRegistry,
    path: Path,
    fittings: Mapping[str, Sequence[BlockPos]],
) -> Terraforming | None:
    if spec.terraform is None:
        return None
    shaped = terraform(
        spec.terraform,
        load_terrain(spec.terraform.world),
        alignment_of(world, path),
        registry.palette,
        fittings=fittings,
    )
    if len(shaped.fill):
        profile = domains.profile(spec.terraform.domain)
        world.add_field(
            shaped.fill,
            placement_mode=PlacementMode.REPLACE_AIR_ONLY,
            connectivity_domain=profile.domain_id,
        )
    if shaped.cut:
        world.base_blocks = Excavated(world.base_blocks, shaped.cut)
        world.invalidate_connectivity_policy()
    return shaped


def alignment_of(world: BlockWorld, path: Path) -> Alignment:
    stations = {
        (point.position.x, point.position.z): index
        for index, point in enumerate(path.path_points)
    }
    laid = tuple(world.iter_source_positions())
    return Alignment.of(laid, [nearest_station(stations, each) for each in laid])


def nearest_station(stations: Mapping[tuple[int, int], int], at: BlockPos) -> int:
    return min(
        stations.items(),
        key=lambda each: (each[0][0] - at.x) ** 2 + (each[0][1] - at.z) ** 2,
    )[1]


@dataclass(frozen=True, slots=True)
class Excavated(Mapping):
    held: Mapping
    removed: frozenset[BlockPos]

    def __getitem__(self, position):
        if position in self.removed:
            raise KeyError(position)
        return self.held[position]

    def __iter__(self):
        return (each for each in self.held if each not in self.removed)

    def __len__(self) -> int:
        return sum(1 for _ in iter(self))


def blocks_of(world: BlockWorld, placements: Iterable[int]) -> Iterator[BlockPos]:
    for placement_id in placements:
        placement = world.get_placement(placement_id)
        for local, _ in placement.field.iter_block_ids():
            yield placement.transform.apply(local)


@dataclass(frozen=True, slots=True)
class Laying:
    domain: str
    lay: Callable[[], object]
    fitting: str | None = None


def in_build_order(spec: BuildSpec, layings: Iterable[Laying]) -> list[Laying]:
    # A domain's order is when its blocks go in, so a later one covers an
    # earlier. Equal orders keep the order they were declared in.
    return sorted(layings, key=lambda laying: spec.domain(laying.domain).order)


# A fitting sits beside the path rather than tiling frame to frame, and the
# frame a point's own column is in is not the frame its blocks land in: a rail
# two to the side of a point near a boundary belongs to the frame beyond it,
# and clipping it to the point's frame drops it and breaks the track. Only the
# section is held to its frame, because only the section has to tile.
def everything(
    spec: BuildSpec,
    runs: Sequence[DeliveredPath],
    sections: Mapping[str, CrossSection],
    world: BlockWorld,
    registry: BlockRegistry,
    domains: DomainRegistry,
    frames: Sequence[PathFrame],
    clip: ClipRule,
) -> Iterator[Laying]:
    for run in runs:
        yield Laying(
            spec.section(run.annotation.section or run.annotation.name).domain,
            partial(
                stamp_along,
                TurnStamp(run.annotation.feature, domains),
                world,
                run.path,
                clip=clip,
            ),
        )
    fitted = cover(frames)
    # A fitting sits beside the path, and the column it lands in is not
    # always in the frame its point was delivered into. This is the level of
    # the column it actually lands in.
    level = elevation_from_frames(frames)
    for structure in spec.structures:
        stamp = build_structure(structure, registry, domains)
        for run in runs_under(spec, structure, runs):
            yield Laying(
                structure.domain,
                partial(
                    stamp_along,
                    stamp,
                    world,
                    at_their_own_level(
                        along_travel(
                            points_for(structure, run),
                            section_of(sections, run).along(structure.offset.x),
                        ),
                        level,
                        lambda point: laid_offset(structure.offset, point.laid_at, point.mirrored),
                    ),
                    clip=lambda point: fitted,
                ),
                structure.name,
            )
    for wire in spec.wires:
        for run in runs_under(spec, wire, runs):
            yield Laying(
                wire.domain,
                partial(
                    lay,
                    wire,
                    strand_of(spec, wire, run, section_of(sections, run), level),
                    world,
                    registry,
                    domains,
                    frames,
                ),
            )
    for lane in spec.lanes:
        stamp = TurnStamp(build_lane(lane, registry, domains), domains)
        for run in runs_under(spec, lane, runs):
            yield Laying(
                lane.domain,
                partial(
                    stamp_along,
                    stamp,
                    world,
                    mark_lane(
                        lane,
                        run,
                        section_of(sections, run),
                        level,
                        lane_support(lane, run, runs),
                    ),
                    clip=lambda point: fitted,
                ),
                lane.name,
            )


def sweep(
    spec: BuildSpec,
    schedules: dict[str, SectionSchedule],
    path: Path,
    frames: Sequence[PathFrame],
) -> tuple[DeliveredPath, ...]:
    # Sections stack, so each is swept on its own schedule and the runs of all
    # of them land in the same world.
    turns = SectionOrientation(
        tolerance=tolerance_for(spec.path.diagonal_degrees)
    ).turns(path)
    return tuple(
        run
        for schedule in schedules.values()
        for run in deliver_paths(
            schedule, path, frames, turns, spec.path.lead, spec.path.lead
        )
    )


def section_of(
    sections: Mapping[str, CrossSection], run: DeliveredPath
) -> CrossSection:
    # The section a run was swept from. Every point of a profile carries the
    # section's name; a section that is one point carries only its own.
    return sections[run.annotation.section or run.annotation.name]


def runs_under(
    spec: BuildSpec,
    fitting: StructureSpec | WireSpec | LaneSpec,
    runs: Sequence[DeliveredPath],
) -> tuple[DeliveredPath, ...]:
    # The runs a fitting is carried by: one per span of each section it is
    # assigned to, so a section that comes and goes carries it each time. A
    # wire stretched between two others is carried by whatever carries them.
    sections = fitted_sections(spec, fitting)
    return tuple(run for run in runs if run.annotation.name in sections)


def fitted_sections(
    spec: BuildSpec, fitting: StructureSpec | WireSpec | LaneSpec
) -> tuple[str, ...]:
    between = getattr(fitting, "between", ())
    if not between:
        return fitting.sections
    return tuple(
        dict.fromkeys(name for wire in between for name in spec.wire(wire).sections)
    )


def points_for(spec: StructureSpec | WireSpec, run: DeliveredPath) -> FeaturePath:
    # A fitting sits under its own spacing rules rather than at every point the
    # section was delivered at.
    return rewalk(
        run.path,
        spacing_rules(spec.spacing, spec.angle, spec.measure is Measure.INDEX),
    )


def mark_lane(
    spec: LaneSpec,
    run: DeliveredPath,
    section: CrossSection,
    level: Callable[[float, float], int],
    support: frozenset[BlockPos] | None = None,
) -> FeaturePath:
    # A lane is a line, so it takes one position a path point whatever the
    # section delivered there -- a point near a layout change is delivered in
    # both, and painting both would draw two lines side by side, each of them
    # broken where it crossed to the other.
    #
    # It then goes where the section put the column it rides. A lane sits at
    # a local column like any point of the section, so it answers the same
    # step question: CHEBYSHEV is one block a point, MANHATTAN is that block
    # and one along travel, and either way a column inboard of it that steps
    # carries it along. That is what lands a rail on the sleeper column
    # reserved for it and the kerb line on the kerb, rather than a block off
    # each. Filling the steps afterwards cannot do it: it joins the blocks up
    # but leaves them beside the column.
    #
    # Thinned before it is dashed, so a dash counted by index counts the
    # points of the line rather than the section's own filling.
    path = one_line(run.path)
    marked = dashed(path, spec.on, spec.off, spec.measure is Measure.INDEX)
    laid = at_their_own_level(
        along_travel(marked, section.along(spec.offset.x, spec.step)),
        level,
        lambda point: laid_offset(spec.offset, point.laid_at, point.mirrored),
    )
    if spec.step is StepMode.MANHATTAN:
        # Stepping leaves the run four-connected within a layout. The join
        # between two of them still moves the offset sideways in one step,
        # and this bridges that.
        return face_connected(
            laid,
            lambda point: laid_offset(spec.offset, point.laid_at, point.mirrored),
            continuous=not spec.off,
            level=level,
            support=support,
        )
    return laid


def lane_support(
    lane: LaneSpec, centre: DeliveredPath, runs: Sequence[DeliveredPath]
) -> frozenset[BlockPos]:
    # The section column a lane rides supplies its actual cells across joins
    # and frame boundaries. A connected lane can choose a bridge on that
    # column instead of cutting across the surrounding deck.
    columns = (
        run for run in runs
        if run.span == centre.span
        and run.annotation.section == centre.annotation.section
        and run.annotation.position.x == lane.offset.x
        and run.annotation.position.z == lane.offset.z
    )
    return frozenset(
        BlockPos(point.position.x, point.position.y + point.y_offset + lane.offset.y,
                 point.position.z)
        for run in columns
        if any(
            local.y == lane.offset.y - 1
            and run.annotation.feature.blocks.palette.get_state(block).name != "minecraft:air"
            for local, block in run.annotation.feature.blocks.iter_block_ids()
        )
        for point in run.path
    )


def lay(
    spec: WireSpec,
    positions: Sequence[BlockPos],
    world: BlockWorld,
    registry: BlockRegistry,
    domains: DomainRegistry,
    frames: Sequence[PathFrame],
) -> None:
    # The strand already has world heights determined by its anchors. Keep it
    # in one placement across frames without bending it to each frame's level.
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
    )


def strand_of(
    spec: BuildSpec,
    wire: WireSpec,
    run: DeliveredPath,
    section: CrossSection,
    level: Callable[[float, float], int],
) -> tuple[BlockPos, ...]:
    if wire.between:
        return tuple(drop_between(spec, wire, run, section, level))
    return tuple(draw_wire(wire, run, section, level))


def draw_wire(
    spec: WireSpec,
    run: DeliveredPath,
    section: CrossSection,
    level: Callable[[float, float], int],
) -> Iterator[BlockPos]:
    poles = [
        attachment(spec, point, level)
        for point in along_travel(points_for(spec, run), section.along(spec.offset.x))
    ]
    for before, after in zip(poles, poles[1:]):
        yield from line(before, after, spec.step)


def drop_between(
    spec: BuildSpec,
    wire: WireSpec,
    run: DeliveredPath,
    section: CrossSection,
    level: Callable[[float, float], int],
) -> Iterator[BlockPos]:
    # Where the wires actually are is taken from the blocks they run through
    # rather than from their offsets: between poles a wire lies on the straight
    # line between them, which on a bend is not where it would sit.
    first, second = (spec.wire(name) for name in wire.between)
    over = tuple(draw_wire(first, run, section, level))
    under = tuple(draw_wire(second, run, section, level))
    if not over or not under:
        return
    for point in points_for(wire, run):
        yield from line(
            nearest(over, point.position), nearest(under, point.position), wire.step
        )


def nearest(positions: Sequence[BlockPos], to: BlockPos) -> BlockPos:
    # Nearest in plan rather than the one in the same column, so a wire cutting
    # the corner of a bend still has something to hang from.
    return min(positions, key=lambda each: (each.x - to.x) ** 2 + (each.z - to.z) ** 2)


def attachment(
    spec: WireSpec, point, level: Callable[[float, float], int]
) -> BlockPos:
    # Only an anchor takes the track's height, at the column where it lands.
    # The blocks between anchors keep the line's own interpolated height.
    local = BlockPos.coerce(point.position) + laid_offset(
        spec.offset, point.laid_at, point.mirrored
    )
    return BlockPos(local.x, local.y + level(local.x, local.z), local.z)


def write(world: BlockWorld, spec: BuildSpec) -> FilePath:
    # The suffix picks the format, so a build says what it wants by naming
    # its output rather than by carrying a setting that could disagree with
    # the name it was given.
    region = BlockRegion.from_world(world, name=spec.output.name)
    if spec.output.path.suffix == SCHEM_SUFFIX:
        # A schematic's Offset is where its corner sits relative to the paste
        # point, so writing the region's own origin pastes the build at the
        # coordinates it was laid out at.
        return write_schematic(
            region,
            spec.output.path,
            data_version=spec.output.data_version,
            offset=region.origin,
        )
    return write_litematic(
        [region],
        spec.output.path,
        data_version=spec.output.data_version,
        name=spec.output.name,
    )
