from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace

from ..Core.Domain import DomainRegistry
from ..Core.Path.path import Path
from ..Core.World.block_world import BlockWorld
from ..Core.block.info import SolidFace
from ..Core.block.registry import BlockRegistry
from ..Core.coordinates import BlockPos
from ..Core.modes import PlacementMode
from ..Core.plan_index import NearestInPlan
from ..Generation.Geometry.frames import level_near
from ..Generation.Geometry.silhouette import PathSilhouette
from ..Parser.spec import BuildSpec
from ..Terraform import Alignment, Terraforming, load_terrain, terraform


RAILS = frozenset({
    "minecraft:rail", "minecraft:powered_rail",
    "minecraft:detector_rail", "minecraft:activator_rail",
})


def reshape(
    spec: BuildSpec,
    world: BlockWorld,
    registry: BlockRegistry,
    domains: DomainRegistry,
    path: Path,
    fittings: Mapping[str, Sequence[BlockPos]],
    silhouette: PathSilhouette | None = None,
    section_spans: Mapping[str, Sequence[tuple[int, int]]] | None = None,
    *,
    apply: bool = True,
    terrain=None,
    positions: Sequence[BlockPos] | None = None,
) -> Terraforming | None:
    if spec.terraform is None:
        return None
    # The sections define the ground's alignment, with protruding rail
    # columns included on request; all build blocks define clearance.
    track = {
        domains.profile(section.domain).domain_id for section in spec.sections
    }
    alignment = alignment_of(
        world, path, track, silhouette, formation_of(spec, silhouette, section_spans),
        include_rails=spec.terraform.include_rails,
        positions=positions,
    )
    if not alignment.columns:
        raise ValueError(
            "no track blocks remain for terraforming; check section surface masks "
            "and placement order"
        )
    if section_spans is not None:
        # Auto selection can leave a declared section unused in this world.
        # Its terrain rule then has no stretches; a misspelled section still
        # fails in Plan.stretches.
        section_spans = {section.name: section_spans.get(section.name, ())
                         for section in spec.sections}
    shaped = terraform(
        spec.terraform,
        terrain if terrain is not None else load_terrain(spec.terraform.terrain_world, spec.terraform.dimension),
        alignment,
        registry.palette,
        locate=marker_locator(spec, path),
        fittings=fittings,
        envelope=envelope_of(world, positions),
        section_spans=section_spans,
        running=running_of(world, alignment, positions),
    )
    if not apply:
        return shaped
    if len(shaped.fill):
        profile = domains.profile(spec.terraform.domain)
        world.add_field(
            shaped.fill,
            placement_mode=PlacementMode.REPLACE_AIR_ONLY,
            connectivity_domain=profile.domain_id,
            block_replacement=profile.replacement,
            earthworks=True,
        )
    if shaped.cut:
        world.base_blocks = Excavated(world.base_blocks, shaped.cut)
        world.invalidate_connectivity_policy()
    return shaped


def running_of(world: BlockWorld, alignment: Alignment, positions: Sequence[BlockPos] | None = None) -> dict[tuple[int, int], int]:
    """Carry actual rail level across the section's occupied track columns.

    Clearance over just the rails leaves soil between lanes and in curved
    joints. Restrict the corridor to the section footprint; enclosing then
    clips each interval against that column's actual wall or roof blocks.
    """
    rails = [position for position in (world.iter_source_positions() if positions is None else positions)
             if (state := world.get_block(position, resolved=False)) is not None
             and state.name in RAILS]
    if not rails:
        return {}
    nearest = NearestInPlan((p.x, p.z, i) for i, p in enumerate(rails))
    running = {(x, z): rails[nearest.nearest(x, z)].y for x, z in alignment.footing}
    running.update(((p.x, p.z), p.y) for p in rails)
    return running


def formation_of(
    spec: BuildSpec,
    silhouette: PathSilhouette | None,
    section_spans: Mapping[str, Sequence[tuple[int, int]]] | None,
) -> Callable[[int, int, int], int | None] | None:
    # A column's formation: its level plus the formation of the section laid
    # at its station, the first declared where several are.
    declared = [
        (section.formation, section_spans.get(section.name, ()))
        for section in spec.sections
        if section.formation is not None
    ] if section_spans else []
    if not declared or silhouette is None or not silhouette.frames:
        return None
    level = level_near(silhouette.frames)

    def at(x: int, z: int, station: int) -> int | None:
        for formation, spans in declared:
            if any(start <= station <= stop for start, stop in spans):
                return level(x, z, station) + formation
        return None

    return at


def marker_locator(spec: BuildSpec, path: Path) -> Callable[[object], int]:
    # Path index of a marker name or path column.
    stations: dict[tuple[int, int], int] = {}
    for index, point in enumerate(path.path_points):
        stations.setdefault((point.position.x, point.position.z), index)

    def locate(where) -> int:
        column = spec.marker(where) if isinstance(where, str) else where
        found = stations.get((column[0], column[1]))
        if found is None:
            raise KeyError(f"terraform span end {where!r} is not a point of the path")
        return found

    return locate


def envelope_of(world: BlockWorld, positions: Sequence[BlockPos] | None = None) -> dict[tuple[int, int], list[int]]:
    # The heights of every block the build lays, by column, but those the
    # earthworks may cover.
    heights: dict[tuple[int, int], list[int]] = {}
    for position in (world.iter_source_positions() if positions is None else positions):
        if world.yields(world.source_at(position)):
            continue
        heights.setdefault((position.x, position.z), []).append(position.y)
    return heights


def alignment_of(
    world: BlockWorld,
    path: Path,
    domains: set[int] | None = None,
    silhouette: PathSilhouette | None = None,
    formation: Callable[[int, int, int], int | None] | None = None,
    *,
    include_rails: bool = False,
    positions: Sequence[BlockPos] | None = None,
) -> Alignment:
    # The firm blocks of `domains` (all if None), each at the station of its
    # nearest path point, held to the points of the frame owning its column
    # so a slanted handover does not take the next stretch's terraforming.
    # Replaceable side foundations and railings are not formation contacts:
    # including them would extend the flat shoulder beyond the actual bed.
    stations = {
        (point.position.x, point.position.z): index
        for index, point in enumerate(path.path_points)
    }
    source_positions = world.iter_source_positions if positions is None else lambda: iter(positions)
    laid = tuple(
        position
        for position in source_positions()
        if ((source := world.source_at(position)) is not None
            and (domains is None or source.domain in domains)
            and not world.yields(source))
    )
    section_columns = {(position.x, position.z) for position in laid}
    rails = tuple(
        position for position in source_positions()
        if (state := world.get_block(position, resolved=False)) is not None
        and state.name in RAILS
        and ((position.x, position.z) not in section_columns
             or world.source_at(position + (0, -1, 0)) is None)
    ) if include_rails else ()
    laid += rails
    nearest = NearestInPlan(
        (x, z, index) for (x, z), index in stations.items()
    )
    owned: dict[tuple[int, int], int] = {}

    def station(at: BlockPos) -> int:
        # The station of the path point that laid the column, else the
        # nearest point within the frame that owns it.
        column = (at.x, at.z)
        found = owned.get(column)
        if found is None:
            source = world.source_at(at)
            laid_by = None if source is None else source.sequence_index
            if laid_by is not None and 0 <= laid_by < len(path.path_points):
                found = owned[column] = laid_by
        if found is None:
            found = owned[column] = _within_frame(
                nearest.nearest(at.x, at.z), at, path, silhouette
            )
        return found

    alignment = Alignment.of(
        laid, [station(each) for each in laid], formation,
        [world.yields(world.source_at(each)) for each in laid],
        [not _full(world, each) for each in laid],
    )
    alignment = replace(alignment, centres={
        index: (point.position.x, point.position.z)
        for index, point in enumerate(path.path_points)
    })
    if rails:
        support = {
            (position.x, position.z): position.y - 1
            for position in rails
        }
        alignment = replace(
            alignment, formation={**alignment.formation, **support}, rail_support=support,
        )
    return alignment


def _full(world: BlockWorld, position: BlockPos) -> bool:
    block_id = world.get_block_id(position)
    return block_id is not None and world.registry.get_info(block_id).solid_faces == SolidFace.ALL


def _within_frame(
    found: int, at: BlockPos, path: Path, silhouette: PathSilhouette | None
) -> int:
    frame = None if silhouette is None else silhouette.frame_at(at.x, at.z)
    if frame is None:
        return found
    # A frame's last point is the next frame's first, except at the path end.
    last = frame.path_stop if frame.path_stop >= path.last_index else frame.path_stop - 1
    return min(max(found, frame.path_start), max(frame.path_start, last))


@dataclass(frozen=True, slots=True)
class Excavated(Mapping):
    # `held` without the `removed` positions.
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


def feet_of(world: BlockWorld, placements: Iterable[int]) -> Iterator[BlockPos]:
    # Each placement's blocks in the columns that reach its lowest level, so
    # an overhanging arm is not treated as standing on the ground.
    for placement_id in placements:
        blocks = list(blocks_of(world, (placement_id,)))
        if not blocks:
            continue
        lowest: dict[tuple[int, int], int] = {}
        for position in blocks:
            column = (position.x, position.z)
            lowest[column] = min(lowest.get(column, position.y), position.y)
        bottom = min(lowest.values())
        yield from (
            position for position in blocks
            if lowest[(position.x, position.z)] == bottom
        )
