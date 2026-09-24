from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from ..Core.Domain import DomainRegistry
from ..Core.Path.path import Path
from ..Core.World.block_world import BlockWorld
from ..Core.block.registry import BlockRegistry
from ..Core.coordinates import BlockPos
from ..Core.modes import PlacementMode
from ..Core.plan_index import NearestInPlan
from ..Generation.Geometry.silhouette import PathSilhouette
from ..Parser.spec import BuildSpec
from ..Terraform import Alignment, Terraforming, load_terrain, terraform


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
) -> Terraforming | None:
    if spec.terraform is None:
        return None
    # The ground is shaped to the sections' blocks alone; everything the
    # build lays defines the space to clear.
    track = {
        domains.profile(section.domain).domain_id for section in spec.sections
    }
    alignment = alignment_of(world, path, track, silhouette)
    if not alignment.columns:
        raise ValueError(
            "no track blocks remain for terraforming; check section surface masks "
            "and placement order"
        )
    shaped = terraform(
        spec.terraform,
        terrain if terrain is not None else load_terrain(spec.terraform.world),
        alignment,
        registry.palette,
        locate=marker_locator(spec, path),
        fittings=fittings,
        envelope=envelope_of(world),
        section_spans=section_spans,
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
        )
    if shaped.cut:
        world.base_blocks = Excavated(world.base_blocks, shaped.cut)
        world.invalidate_connectivity_policy()
    return shaped


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


def envelope_of(world: BlockWorld) -> dict[tuple[int, int], list[int]]:
    # The heights of every block the build lays, by column.
    heights: dict[tuple[int, int], list[int]] = {}
    for position in world.iter_source_positions():
        heights.setdefault((position.x, position.z), []).append(position.y)
    return heights


def alignment_of(
    world: BlockWorld,
    path: Path,
    domains: set[int] | None = None,
    silhouette: PathSilhouette | None = None,
) -> Alignment:
    # The blocks of `domains` (all if None), each at the station of its
    # nearest path point, held to the points of the frame owning its column
    # so a slanted handover does not take the next stretch's terraforming.
    stations = {
        (point.position.x, point.position.z): index
        for index, point in enumerate(path.path_points)
    }
    laid = tuple(
        position
        for position in world.iter_source_positions()
        if domains is None or world.source_at(position).domain in domains
    )
    nearest = NearestInPlan(
        (x, z, index) for (x, z), index in stations.items()
    )
    owned: dict[tuple[int, int], int] = {}

    def station(at: BlockPos) -> int:
        column = (at.x, at.z)
        found = owned.get(column)
        if found is None:
            found = owned[column] = _within_frame(
                nearest.nearest(at.x, at.z), at, path, silhouette
            )
        return found

    return Alignment.of(laid, [station(each) for each in laid])


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
