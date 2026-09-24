from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial

from ..Core.CrossSection.cross_section import CrossSection
from ..Core.CrossSection.schedule import SectionSchedule
from ..Core.CrossSection.turns import tolerance_for
from ..Core.Domain import DomainRegistry
from ..Core.Path.path import Path
from ..Core.Path.path_frame import PathFrame
from ..Core.World.block_world import BlockWorld
from ..Core.block.registry import BlockRegistry
from ..Core.coordinates import BlockPos
from ..Core.modes import StepMode
from ..Generation.Geometry.frames import level_near
from ..Generation.Geometry.orientation import SectionOrientation
from ..Generation.Geometry.silhouette import (
    IntersectMask,
    PathSilhouette,
    SectionColumnMask,
    section_column_owners,
)
from ..Generation.Sweep.delivery import nearby_frames
from ..Generation.Sweep.paths import DeliveredPath, deliver_paths
from ..Terraform import Terraforming, load_terrain
from ..Terraform.surface_mask import SurfaceReference
from ..Stamping import ClipRule, TurnStamp, resolve, stamp_along
from ..Parser.spec import BuildSpec
from .anchors import runs_under, structure_points, structure_runs
from .earthworks import feet_of, reshape
from .lanes import lane_support, mark_lane
from .structure import (
    build_domains,
    build_lane,
    build_path,
    build_path_frames,
    build_schedules,
    build_sections,
    build_structure,
    resolve_spans,
)
from .wires import lay_wire, strand_of


# How many path points from its own a fitting may land in a frame of: past
# any sideways offset, well short of the far side of a loop.
FITTING_REACH = 32


@dataclass(frozen=True, slots=True)
class Built:
    world: BlockWorld
    terraforming: Terraforming | None = None


def assemble(spec: BuildSpec) -> BlockWorld:
    return build(spec).world


def build(spec: BuildSpec) -> Built:
    registry = BlockRegistry.for_vanilla_26_2()
    domains = build_domains(spec)
    masked = _surface_masks(spec)
    designed = any(mask.reference == "designed" for mask in masked)
    surface_world = spec.surface_world or (
        None if spec.terraform is None else spec.terraform.world
    )
    if masked and surface_world is None:
        raise ValueError(
            "surface_mask needs [surface].world or a terraform plan with a world"
        )
    if spec.terraform is None and designed:
        raise ValueError("surface_mask.reference = 'designed' needs a terraform plan")
    surface = SurfaceReference(load_terrain(surface_world)) if masked else None
    world = BlockWorld(registry, surface_reference=surface)
    domains.bind(world)

    path = build_path(spec)
    spans = resolve_spans(spec, path)
    sections = build_sections(spec, registry, domains, spans)
    schedules = build_schedules(sections, spans, len(path.path_points))
    frames = build_path_frames(spec, schedules, path)
    silhouette = PathSilhouette.for_sections(path, sections.values(), frames)
    clip = silhouette.clip

    runs = sweep(spec, schedules, path, frames, silhouette)
    fittings: dict[str, list[BlockPos]] = {}
    fitting_ids: dict[str, list[int]] = {}
    for laying in in_build_order(
        spec, all_layings(
            spec, runs, sections, world, registry, domains, frames, clip,
            schedules, silhouette,
        )
    ):
        placed = laying.lay()
        if laying.fitting:
            fitting_ids.setdefault(laying.fitting, []).extend(placed or ())
            fittings.setdefault(laying.fitting, []).extend(
                feet_of(world, placed or ())
            )
    designed_masks = surface is not None and designed
    if designed_masks:
        # Features masked by the designed surface and the earthworks depend
        # on each other: preview the earthworks, re-mask, and repeat until
        # the masked footprints settle.
        previous: tuple[tuple[int, tuple[BlockPos, ...]], ...] | None = None
        for _ in range(4):
            preview = reshape(
                spec, world, registry, domains, path, fittings, silhouette,
                spans, apply=False, terrain=surface.terrain,
            )
            surface.update(preview)
            world.refresh_surface_masks()
            fittings = {
                name: list(feet_of(world, ids))
                for name, ids in fitting_ids.items()
            }
            footprint = tuple(
                (each.placement_id, world._footprints[each.placement_id])
                for each in world.instances if each.surface_mask is not None
            )
            if footprint == previous:
                break
            previous = footprint
        else:
            raise ValueError("designed surface masks did not settle after four passes")
    shaped = reshape(
        spec, world, registry, domains, path, fittings, silhouette, spans,
        terrain=None if surface is None else surface.terrain,
    )
    if designed_masks and shaped is not None:
        surface.update(shaped)
        world.refresh_surface_masks()
    resolve(world, registry, domains)
    return Built(world, shaped)


def _surface_masks(spec: BuildSpec) -> tuple:
    return tuple(
        each.surface_mask
        for each in (*spec.sections, *spec.structures, *spec.lanes, *spec.wires)
        if each.surface_mask is not None
    )


def sweep(
    spec: BuildSpec,
    schedules: dict[str, SectionSchedule],
    path: Path,
    frames: Sequence[PathFrame],
    silhouette: PathSilhouette | None = None,
) -> tuple[DeliveredPath, ...]:
    turns = SectionOrientation(
        tolerance=tolerance_for(spec.path.diagonal_degrees)
    ).turns(path)
    return tuple(
        run
        for schedule in schedules.values()
        for run in deliver_paths(
            schedule, path, frames, turns, spec.path.lead, spec.path.lead,
            silhouette=silhouette,
        )
    )


@dataclass(frozen=True, slots=True)
class Laying:
    domain: str
    lay: Callable[[], object]
    fitting: str | None = None


def in_build_order(spec: BuildSpec, layings: Iterable[Laying]) -> list[Laying]:
    # By domain order; a later domain covers an earlier one. Stable.
    return sorted(layings, key=lambda laying: spec.domain(laying.domain).order)


def all_layings(
    spec: BuildSpec,
    runs: Sequence[DeliveredPath],
    sections: Mapping[str, CrossSection],
    world: BlockWorld,
    registry: BlockRegistry,
    domains: DomainRegistry,
    frames: Sequence[PathFrame],
    clip: ClipRule,
    schedules: Mapping[str, SectionSchedule],
    silhouette: PathSilhouette,
) -> Iterator[Laying]:
    # Every placement of the build, lazily. Sections are clipped to their own
    # frames; fittings may land in frames near their point. Later spans and
    # then stepping columns go in last, so they keep contested cells.
    owners = section_column_owners(runs, schedules)
    for run in sorted(
        runs,
        key=lambda run: (run.start, run.annotation.step_mode is StepMode.MANHATTAN),
    ):
        span_mask = silhouette.for_span(schedules[run.section_name].spans[run.span])
        stack_mask = SectionColumnMask(owners, run.section_name, run.span)
        section_clip = lambda point, mask=span_mask, stack=stack_mask: IntersectMask(
            IntersectMask(clip(point), mask), stack
        )
        feature = run.annotation.feature
        yield Laying(
            feature.domain.name,
            partial(
                stamp_along, TurnStamp(feature, domains), world, run.path,
                clip=section_clip,
            ),
        )
    fitted = nearby_frames(frames, max(spec.path.lead, FITTING_REACH))
    level = level_near(frames)
    for structure in spec.structures:
        stamp = build_structure(structure, registry, domains)
        for run, anchors in structure_runs(spec, structure, runs):
            points = structure_points(
                structure, anchors, sections[run.section_name], level
            )
            yield Laying(
                structure.domain,
                partial(stamp_along, stamp, world, points, clip=fitted),
                structure.name,
            )
    for wire in spec.wires:
        for run in runs_under(spec, wire, runs):
            strand = strand_of(
                spec, wire, run, sections[run.section_name], level, fitted, runs
            )
            yield Laying(
                wire.domain,
                partial(lay_wire, wire, strand, world, registry, domains, frames),
            )
    for lane in spec.lanes:
        stamp = TurnStamp(build_lane(lane, registry, domains), domains)
        for run in runs_under(spec, lane, runs):
            marked = mark_lane(
                lane, run, sections[run.section_name], level,
                lane_support(lane, run, runs),
            )
            yield Laying(
                lane.domain,
                partial(stamp_along, stamp, world, marked, clip=fitted),
                lane.name,
            )
