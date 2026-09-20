from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path as FilePath

from ..Core.CrossSection.schedule import SectionSchedule
from ..Core.Domain import DomainRegistry
from ..Core.Feature.feature_path import FeaturePath, spacing_rules
from ..Core.Path.path import Path
from ..Core.Path.path_frame import PathFrame
from ..Core.World.block_world import BlockTransform, BlockWorld
from ..Core.block.registry import BlockRegistry
from ..Core.coordinates import BlockPos
from ..Core.line import line
from ..Format.litematic import write_litematic
from ..Format.region import BlockRegion
from ..Generation.Geometry.frames import elevation_from_frames
from ..Generation.Geometry.orientation import SectionOrientation
from ..Generation.Sweep.delivery import cover
from ..Generation.Sweep.paths import DeliveredPath, dashed, deliver_paths, rewalk
from ..Stamping import (
    ClipRule,
    TurnStamp,
    in_frames,
    lateral_turns,
    resolve,
    stamp_along,
    turned_offset,
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
    registry = BlockRegistry()
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
    for laying in in_build_order(spec, everything(spec, runs, world, registry, domains, frames, clip)):
        laying.lay()

    resolve(world, registry, domains)
    return world


@dataclass(frozen=True, slots=True)
class Laying:
    domain: str
    lay: Callable[[], None]


def in_build_order(spec: BuildSpec, layings: Iterable[Laying]) -> list[Laying]:
    # A domain's order is when its blocks go in, so a later one covers an
    # earlier. Equal orders keep the order they were declared in.
    return sorted(layings, key=lambda laying: spec.domain(laying.domain).order)


def everything(
    spec: BuildSpec,
    runs: Sequence[DeliveredPath],
    world: BlockWorld,
    registry: BlockRegistry,
    domains: DomainRegistry,
    frames: Sequence[PathFrame],
    clip: ClipRule,
) -> Iterator[Laying]:
    for run in runs:
        yield Laying(
            spec.section(run.annotation.name).domain,
            partial(
                stamp_along,
                TurnStamp(run.annotation.feature, domains),
                world,
                run.path,
                clip=clip,
            ),
        )
    for structure in spec.structures:
        stamp = build_structure(structure, registry, domains)
        for run in runs_under(spec, structure, runs):
            yield Laying(
                structure.domain,
                partial(
                    stamp_along, stamp, world, points_for(structure, run), clip=clip
                ),
            )
    for wire in spec.wires:
        for run in runs_under(spec, wire, runs):
            yield Laying(
                wire.domain,
                partial(
                    lay,
                    wire,
                    strand_of(spec, wire, run),
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
                partial(stamp_along, stamp, world, mark_lane(lane, run), clip=clip),
            )


def sweep(
    spec: BuildSpec,
    schedules: dict[str, SectionSchedule],
    path: Path,
    frames: Sequence[PathFrame],
) -> tuple[DeliveredPath, ...]:
    # Sections stack, so each is swept on its own schedule and the runs of all
    # of them land in the same world.
    turns = SectionOrientation().turns(path)
    return tuple(
        run
        for schedule in schedules.values()
        for run in deliver_paths(
            schedule, path, frames, turns, spec.path.lead, spec.path.lead
        )
    )


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


def mark_lane(spec: LaneSpec, run: DeliveredPath) -> FeaturePath:
    return dashed(run.path, spec.on, spec.off, spec.measure is Measure.INDEX)


def lay(
    spec: WireSpec,
    positions: Sequence[BlockPos],
    world: BlockWorld,
    registry: BlockRegistry,
    domains: DomainRegistry,
    frames: Sequence[PathFrame],
) -> None:
    # One placement for the whole wire, so connectivity settles it as one
    # piece and it may cross frames without being cut. Its columns take their
    # own frame's level, which is what carries a wire over a change of height.
    if not positions:
        return
    feature = build_wire(spec, positions, registry, domains)
    profile = domains.profile(feature.domain)
    world.add_field(
        feature.blocks,
        placement_mode=feature.placement_mode,
        terrain_mask_mode=profile.terrain_mask_mode,
        replace=profile.replace,
        transform=BlockTransform(y_offset=elevation_from_frames(frames)),
        connectivity_domain=profile.domain_id,
        clip=cover(frames),
    )


def strand_of(
    spec: BuildSpec, wire: WireSpec, run: DeliveredPath
) -> tuple[BlockPos, ...]:
    if wire.between:
        return tuple(drop_between(spec, wire, run))
    return tuple(draw_wire(wire, run))


def draw_wire(spec: WireSpec, run: DeliveredPath) -> Iterator[BlockPos]:
    poles = [attachment(spec, point) for point in points_for(spec, run)]
    for before, after in zip(poles, poles[1:]):
        yield from line(before, after, spec.step)


def drop_between(
    spec: BuildSpec, wire: WireSpec, run: DeliveredPath
) -> Iterator[BlockPos]:
    # Where the wires actually are is taken from the blocks they run through
    # rather than from their offsets: between poles a wire lies on the straight
    # line between them, which on a bend is not where it would sit.
    first, second = (spec.wire(name) for name in wire.between)
    over = tuple(draw_wire(first, run))
    under = tuple(draw_wire(second, run))
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


def attachment(spec: WireSpec, point) -> BlockPos:
    turns = lateral_turns(point.turn * point.direction)
    return BlockPos.coerce(point.position) + turned_offset(spec.offset, turns)


def write(world: BlockWorld, spec: BuildSpec) -> FilePath:
    region = BlockRegion.from_world(world, name=spec.output.name)
    return write_litematic(
        [region],
        spec.output.path,
        data_version=spec.output.data_version,
        name=spec.output.name,
    )
