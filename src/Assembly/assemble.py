from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path as FilePath

from ..Core.CrossSection.cross_section import CrossSection
from ..Core.CrossSection.schedule import SectionSchedule
from ..Core.CrossSection.turns import tolerance_for
from ..Core.Domain import DomainRegistry
from ..Core.Path.path import Path
from ..Core.Path.path_frame import PathFrame
from ..Core.World.block_world import BlockWorld
from ..Core.block.registry import BlockRegistry
from ..Core.coordinates import BlockPos
from ..Core.Feature.dependent_path import DependentFeaturePath, PathRuns
from ..Core.modes import StepMode
from ..Core.line import line
from ..Generation.Geometry.frames import level_near
from ..Generation.Geometry.orientation import SectionOrientation
from ..Generation.Geometry.silhouette import (
    AnnotationColumnMask,
    IntersectMask,
    OutsideMask,
    PathSilhouette,
    SectionColumnMask,
    annotation_column_owners,
    section_column_owners,
)
from ..Generation.Sweep.corners import settle_corners
from ..Generation.Sweep.delivery import nearby_frames
from ..Generation.Sweep.paths import DeliveredPath, deliver_paths
from ..Terraform import Terraforming, load_terrain
from ..Terraform.surface_mask import SurfaceReference
from ..Stamping import ClipRule, TurnStamp, resolve, stamp_along
from ..Parser.spec import BuildSpec, WireGeometry, WireSpec
from .anchors import fitted_sections, runs_under, strand_source, structure_points, structure_runs, wire_routes
from .earthworks import feet_of, reshape
from .lanes import lane_support, mark_lane, support_rail_joints
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
from .wires import WIRE_LINES, dependent_strand, lay_wire, line_strand, strand_of
from .dependencies import build_feature_paths, dependent_intervals, feature_path_graph
from .progress import Progress, report
from .fittings import retain_assemblies
from .formation_paths import formation_paths


# How many path points from its own a fitting may land in a frame of: past
# any sideways offset, well short of the far side of a loop.
FITTING_REACH = 32


@dataclass(frozen=True, slots=True)
class Built:
    world: BlockWorld
    terraforming: Terraforming | None = None


@dataclass(frozen=True, slots=True)
class PreparedBuild:
    spec: BuildSpec
    world: BlockWorld
    domains: DomainRegistry
    path: Path
    spans: Mapping[str, tuple[tuple[int, int], ...]]
    silhouette: PathSilhouette
    layings: Sequence[Laying] | Iterator[Laying]
    feature_paths: dict[str, PathRuns] = field(default_factory=dict)
    dependencies: dict[str, DependentFeaturePath] = field(default_factory=dict)
    formations: dict[str, dict[str, set[BlockPos]]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlacedBuild:
    components: tuple[PreparedBuild, ...]
    fitting_ids: tuple[dict[str, list[int]], ...]


def assemble(spec: BuildSpec) -> BlockWorld:
    return build(spec).world


def build(spec: BuildSpec, *, progress: Progress | None = None) -> Built:
    """Build without UI; optionally report progress within each assembly stage."""
    if spec.composition is not None:
        from ..Extensions.Composition.rooted import compose
        from ..Extensions.Composition.run import build as build_composition
        return build_composition(compose(spec), progress=progress)
    prepared = prepare_build(spec, progress=progress)
    placed = place_features(prepared, progress=progress)
    return finalize_build(placed, progress=progress)


def prepare_build(
    spec: BuildSpec,
    *,
    shared: PreparedBuild | None = None,
    world: BlockWorld | None = None,
    defer_features: bool = False,
    progress: Progress | None = None,
) -> PreparedBuild:
    """Prepare placements without emitting blocks; shared reuses a world and its domains."""
    report(progress, "Preparing build", 0)
    if shared is not None:
        if (spec.domains, spec.interactions) != (shared.spec.domains, shared.spec.interactions):
            raise ValueError("shared builds need identical domain and interaction definitions")
        if _surface_source(spec) != _surface_source(shared.spec):
            raise ValueError("shared builds need the same surface world and dimension")
    domains = build_domains(spec) if shared is None else shared.domains
    masked = _surface_masks(spec)
    designed = any(mask.reference == "designed" for mask in masked)
    surface_world, surface_dimension = _surface_source(spec)
    if masked and surface_world is None:
        raise ValueError(
            "surface_mask needs [surface].world or a terraform plan with a world"
        )
    if spec.terraform is None and designed:
        raise ValueError("surface_mask.reference = 'designed' needs a terraform plan")
    if shared is not None:
        if world is not None and world is not shared.world:
            raise ValueError("world must match the shared build's world")
        world = shared.world
    elif world is None:
        world = BlockWorld(BlockRegistry.for_vanilla_26_2())
    registry = world.registry
    if masked and world.surface_reference is None:
        world.surface_reference = SurfaceReference(load_terrain(surface_world, surface_dimension))
    if shared is None:
        domains.bind(world)
    report(progress, "Preparing build", 1)

    report(progress, "Preparing path", 0)
    path = build_path(spec)
    spans = resolve_spans(spec, path)
    sections = build_sections(spec, registry, domains, spans)
    schedules = build_schedules(sections, spans, len(path.path_points))
    frames = build_path_frames(spec, schedules, path)
    silhouette = PathSilhouette.for_sections(path, sections.values(), frames, (
        fitting.offset.y for fitting in (*spec.lanes, *spec.wires, *spec.structures)
    ))
    clip = silhouette.clip
    report(progress, "Preparing path", 1)

    report(progress, "Preparing placements", 0)
    runs = sweep(spec, schedules, path, frames, silhouette)
    paths, dependencies = {}, {}
    if defer_features:
        paths, dependencies = feature_path_graph(
            spec, runs, sections, path, level_near(frames),
            fitting_clip(spec, frames, schedules, silhouette),
        )
    layings = all_layings(spec, runs, sections, world, registry, domains, frames, clip,
                          schedules, silhouette, paths if defer_features else None)
    if not defer_features:
        layings = tuple(layings)
    report(progress, "Preparing placements", 1)
    formations = formation_paths(spec, runs, sections, level_near(frames), path) if spec.formations else {}
    return PreparedBuild(spec, world, domains, path, spans, silhouette, layings, paths, dependencies, formations)


def place_features(
    prepared: PreparedBuild,
    *additional: PreparedBuild,
    before_place: Callable[[PreparedBuild], None] | None = None,
    priority: Callable[[PreparedBuild], tuple] | None = None,
    defer_terrain: bool = False,
    progress: Progress | None = None,
) -> PlacedBuild:
    """Place all components in domain order without shaping terrain or resolving blocks."""
    components = (prepared, *additional)
    if any(
        each.world is not prepared.world or each.domains is not prepared.domains
        for each in additional
    ):
        raise ValueError("prepare additional components with shared=first_prepared_build")
    if additional and not defer_terrain and any(each.spec.terraform is not None for each in components):
        raise ValueError("shared terrain shaping needs a composition layer; combined builds must omit terraform")
    # Sort across components so a later component's foundation cannot cover an earlier fitting.
    layings = sorted(
        ((index, laying) for index, each in enumerate(components) for laying in each.layings),
        key=lambda item: (prepared.spec.domain(item[1].domain).order,
                          () if priority is None else priority(components[item[0]])),
    )
    fitting_ids: tuple[dict[str, list[int]], ...] = tuple({} for _ in components)
    report(progress, "Placing features", 0, len(layings))
    for number, (index, laying) in enumerate(layings, 1):
        if before_place is not None:
            before_place(components[index])
        placed = laying.lay()
        if laying.fitting:
            fitting_ids[index].setdefault(laying.fitting, []).extend(placed or ())
        report(progress, "Placing features", number, len(layings))
    placed = PlacedBuild(components, fitting_ids)
    retain_assemblies(placed)
    return placed


def finalize_build(
    placed: PlacedBuild,
    *,
    terrain_builder: Callable[[PlacedBuild], Terraforming | None] | None = None,
    progress: Progress | None = None,
) -> Built:
    """Support joints, shape terrain through the selected builder, and resolve once."""
    prepared = placed.components[0]
    spec, world, domains = prepared.spec, prepared.world, prepared.domains
    if terrain_builder is None and len(placed.components) > 1 and any(each.spec.terraform for each in placed.components):
        raise ValueError("deferred shared terrain needs a terrain_builder")
    report(progress, "Supporting rail joints", 0)
    support_rail_joints(world, {domains.profile(section.domain).domain_id
                              for each in placed.components for section in each.spec.sections})
    report(progress, "Supporting rail joints", 1)
    report(progress, "Shaping terrain", 0)
    shaped = _shape_terrain(prepared, placed.fitting_ids[0]) if terrain_builder is None else terrain_builder(placed)
    report(progress, "Shaping terrain", 1)
    report(progress, "Resolving blocks", 0)
    resolve(world, world.registry, domains)
    report(progress, "Resolving blocks", 1)
    return Built(world, shaped)


def _shape_terrain(prepared: PreparedBuild, fitting_ids: Mapping[str, Sequence[int]]) -> Terraforming | None:
    spec, world, domains = prepared.spec, prepared.world, prepared.domains
    path, spans, silhouette = prepared.path, prepared.spans, prepared.silhouette
    registry, surface = world.registry, world.surface_reference
    fittings = {name: list(feet_of(world, ids)) for name, ids in fitting_ids.items()}
    designed_masks = surface is not None and any(mask.reference == "designed" for mask in _surface_masks(spec))
    if designed_masks:
        # Re-mask against previewed earthworks until the dependent footprints settle.
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
    return shaped


def _surface_source(spec: BuildSpec) -> tuple[FilePath | None, str | None]:
    world = spec.surface_world or (None if spec.terraform is None else spec.terraform.terrain_world)
    dimension = spec.surface_dimension or (
        spec.terraform.dimension if spec.surface_world is None and spec.terraform else None
    )
    return (None if world is None else world.resolve()), dimension


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
            silhouette=silhouette, solve_corners=True,
        )
    )


@dataclass(frozen=True, slots=True)
class Laying:
    domain: str
    lay: Callable[[], object]
    fitting: str | None = None
    feature: str | None = None


def fitting_clip(spec, frames, schedules, silhouette):
    fitted = nearby_frames(frames, max(spec.path.lead, FITTING_REACH))

    def kept_to(fitting):
        riding = fitted_sections(spec, fitting)
        if isinstance(fitting, WireSpec) and not fitting.between:
            riding = tuple(dict.fromkeys(
                name for wire in spec.wires
                if not wire.between
                and (wire.domain, wire.offset, wire.schematic) ==
                    (fitting.domain, fitting.offset, fitting.schematic)
                and strand_source(spec, wire) == strand_source(spec, fitting)
                for name in wire.sections
            ))
        ridden = [span for name, schedule in schedules.items() if name in riding for span in schedule.spans]
        others = tuple(
            silhouette.for_span(span)
            for name, schedule in schedules.items() if name not in riding
            for span in schedule.spans
            if not any(min(each.path_stop, span.path_stop) - max(each.path_start, span.path_start) > 1
                       for each in ridden)
        )
        if not others:
            return fitted
        return lambda point: IntersectMask(fitted(point), OutsideMask(others))

    return kept_to


def _column_on(silhouette: PathSilhouette, station: int, x: int, z: int) -> tuple[int, ...]:
    return silhouette.column(x, z, station)


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
    feature_paths: dict[str, PathRuns] | None = None,
) -> Iterator[Laying]:
    # Every placement of the build, lazily. Sections are clipped to their own
    # frames, and a world column two of a span's columns reach goes to one
    # of them; fittings may land in frames near their point.
    owners = section_column_owners(runs, schedules, silhouette)
    columns = annotation_column_owners(runs, silhouette.path, silhouette)
    settle_corners(columns, runs, silhouette.path, silhouette)
    # Slices of a deep section overlap along travel; the slice furthest from
    # the path is laid last, so it shows wherever it has a block.
    for run in sorted(
        runs,
        key=lambda run: (
            run.start,
            run.annotation.step_mode is StepMode.MANHATTAN,
            abs(run.annotation.position.z),
            run.annotation.position.z,
        ),
    ):
        span_mask = silhouette.for_span(schedules[run.section_name].spans[run.span])
        held = columns.get((run.section_name, run.span))

        def section_clip(point, mask=span_mask, run=run, held=held):
            # Each point reads the tables as its own pass of the path sees them.
            key = partial(_column_on, silhouette, point.path_index)
            stack = SectionColumnMask(owners, run.section_name, run.span, key)
            if held is not None:
                stack = IntersectMask(stack, AnnotationColumnMask(held, run.annotation.column_name, key))
            return IntersectMask(IntersectMask(clip(point), mask), stack)

        feature = run.annotation.feature
        yield Laying(
            feature.domain.name,
            partial(
                stamp_along, TurnStamp(feature, domains), world, run.path,
                clip=section_clip,
            ),
        )
    level = level_near(frames)
    kept_to = fitting_clip(spec, frames, schedules, silhouette)

    wire_paths = {
        wire.name: wire_routes(wire, runs_under(spec, wire, runs))
        for wire in spec.wires if wire.geometry is WireGeometry.LINE and not wire.between and wire.feature_path is None
    }
    named_paths = {each.name for each in spec.feature_paths}
    dependent_paths = feature_paths if feature_paths is not None else (
        build_feature_paths(spec, runs, sections, silhouette.path, level, kept_to) if named_paths else {}
    )
    for structure in spec.structures:
        structure_clip = kept_to(structure)
        dependent = structure.feature_path in named_paths or (
            structure.wire is not None and spec.wire(structure.wire).feature_path is not None
        )
        if dependent:
            stamp = build_structure(replace(structure, offset=BlockPos(0, 0, 0)), registry, domains,
                                    DependentFeaturePath(f"structure:{structure.name}", rules=()))
            for points in dependent_paths[f"structure:{structure.name}"]:
                yield Laying(structure.domain, partial(stamp_along, stamp, world, points, clip=structure_clip),
                             structure.name, f"structure:{structure.name}")
            continue
        stamp = build_structure(structure, registry, domains)
        for run, anchors in structure_runs(spec, structure, runs, wire_paths.get(structure.wire)):
            points = structure_points(
                structure, anchors, sections[run.section_name], level
            )
            yield Laying(
                structure.domain,
                partial(stamp_along, stamp, world, points, clip=structure_clip),
                structure.name,
                f"structure:{structure.name}",
            )
    continuations = {}

    def remember(wire, start, stop, strand):
        if strand:
            key = (wire.domain, wire.offset, wire.schematic, wire.step, strand_source(spec, wire))
            continuations.setdefault(key, []).append((start, stop, strand[0], strand[-1], wire))

    for wire in spec.wires:
        if wire.feature_path is not None:
            intervals = dependent_intervals(spec, wire, runs)
            for anchors in dependent_paths[f"wire:{wire.name}"]:
                strand = dependent_strand(anchors, wire.step, kept_to(wire))
                start, stop = anchors[0].path_index, anchors[-1].path_index
                start, stop = next(((first, last) for first, last in intervals
                                    if first <= start <= stop <= last), (start, stop))
                remember(wire, start, stop, strand)
                yield Laying(wire.domain, partial(lay_wire, wire, strand, world, registry, domains, frames),
                             feature=f"wire:{wire.name}")
            continue
        if wire.geometry is WireGeometry.LINE and not wire.between:
            for route in wire_paths[wire.name]:
                strand = line_strand(wire, route, sections, level, kept_to(wire))
                remember(wire, route[0][1][0].path_index, route[-1][1][-1].path_index, strand)
                yield Laying(
                    wire.domain,
                    partial(lay_wire, wire, strand, world, registry, domains, frames),
                    feature=f"wire:{wire.name}",
                )
            continue
        for run in runs_under(spec, wire, runs):
            strand = strand_of(
                spec, wire, run, sections[run.section_name], level, kept_to(wire), runs,
                dependent_paths,
            )
            if not wire.between:
                remember(wire, min(point.path_index for point in run.path),
                         max(point.path_index for point in run.path), strand)
            yield Laying(
                wire.domain,
                partial(lay_wire, wire, strand, world, registry, domains, frames),
                feature=f"wire:{wire.name}",
            )
    for pieces in continuations.values():
        pieces.sort(key=lambda piece: piece[:2])
        for previous, following in zip(pieces, pieces[1:]):
            if following[0] != previous[1] + 1:
                continue
            wire = following[-1]
            # Preserve the face-connected handover between two compatible
            # strands, including a change between line and follow geometry.
            strand = line(previous[3], following[2], WIRE_LINES[wire.step])
            yield Laying(
                wire.domain,
                partial(lay_wire, wire, strand, world, registry, domains, frames),
                feature=f"wire:{wire.name}",
            )
    for lane in spec.lanes:
        stamp = TurnStamp(build_lane(lane, registry, domains), domains)
        clip_lane = kept_to(lane)
        for run in runs_under(spec, lane, runs):
            marked = mark_lane(
                lane, run, sections[run.section_name], level,
                lane_support(lane, run, runs),
            )
            yield Laying(
                lane.domain,
                partial(stamp_along, stamp, world, marked, clip=clip_lane),
                lane.name,
            )
