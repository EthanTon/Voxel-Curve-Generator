from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ...Assembly.assemble import PlacedBuild, _surface_masks
from ...Assembly.earthworks import Excavated, feet_of, reshape
from ...Core.Field.block_field import BlockField
from ...Core.coordinates import ColumnPos
from ...Core.modes import PlacementMode
from ...Terraform import Earthworks, Terraforming, load_terrain
from ...Terraform.clearance import OPEN, Clearance
from ...Terraform.grid import Grid
from ...Terraform.terrain import ABSENT
from .ownership import OwnedWorld
from .spec import CompositionSpec


@dataclass(frozen=True)
class LayeredClearance(Clearance):
    grid: Grid
    layers: tuple[tuple[Grid, Clearance], ...]

    def holds(self, row: int, cell: int, y: int) -> bool:
        x, z = self.grid.column(row, cell)
        # Keep separate vertical intervals so stacked routes leave the ground between them intact.
        return any(grid.holds(x, z) and clearance.holds(*grid.at(x, z), y)
                   for grid, clearance in self.layers)


def combined_clearance(parts, grid: Grid, attribute: str) -> LayeredClearance | None:
    layers = tuple((part.grid, clearance) for part in parts if (clearance := getattr(part, attribute)) is not None)
    if not layers:
        return None
    floor, ceiling = grid.filled(OPEN), grid.filled(-OPEN)
    for source_grid, clearance in layers:
        window = grid_window(source_grid, grid)
        floor[window] = np.minimum(floor[window], clearance.floor)
        ceiling[window] = np.maximum(ceiling[window], clearance.ceiling)
    return LayeredClearance(floor, ceiling, grid, layers)


def grid_window(source: Grid, target: Grid):
    row, column = target.at(source.origin.x, source.origin.z)
    return slice(row, row + source.depth), slice(column, column + source.width)


def combine_terrain(contributions, world: OwnedWorld):
    contributions = sorted(contributions, key=lambda each: each[1])
    parts = [each[3] for each in contributions]
    if not parts:
        return None, {}
    grid = Grid.around([column for part in parts for column in (
        part.grid.origin, ColumnPos(part.grid.origin.x + part.grid.width - 1, part.grid.origin.z + part.grid.depth - 1))])
    natural, designed = grid.filled(ABSENT), grid.filled(ABSENT)
    coverage, terrain_top, water = grid.filled(False, dtype=bool), grid.filled(ABSENT), grid.filled(ABSENT)
    changes, chunks = {}, {}
    for name, priority, domain, part in contributions:
        window = grid_window(part.grid, grid)
        available = part.natural > ABSENT
        natural[window] = np.where(available, part.natural, natural[window])
        changed = available & (part.designed != part.natural)
        designed[window] = np.where(changed | (designed[window] == ABSENT), part.designed, designed[window])
        if part.coverage is not None:
            coverage[window] |= part.coverage
        if part.terrain_top is not None:
            terrain_top[window] = np.maximum(terrain_top[window], part.terrain_top)
        if part.water is not None:
            water[window] = np.maximum(water[window], part.water)
        chunks.update(part.terrain_chunks or {})
        changes.update((position, (None, domain)) for position in part.cut)
        changes.update((position, (part.fill.palette.get_state(block), domain)) for position, block in part.fill.iter_block_ids())
    clearance = combined_clearance(parts, grid, "clearance")
    track_clearance = combined_clearance(parts, grid, "track_clearance")
    fill = BlockField(world.registry.palette)
    fields, cut = {}, set()
    for position, (state, domain) in changes.items():
        if clearance is not None and clearance.holds(*grid.at(position.x, position.z), position.y):
            state = None
        source = world.source_at(position)
        if source is not None and not world.yields(source):
            continue
        if state is None:
            cut.add(position)
        else:
            fill.set_block(position, state)
            fields.setdefault(domain, BlockField(world.registry.palette)).set_block(position, state)
    report = replace(parts[0].report, filled=len(fill), carved=len(cut),
        stretches=tuple((f"{name}:{profile}", start, stop, count)
                        for name, _, _, part in contributions for profile, start, stop, count in part.report.stretches),
        followed=tuple((f"{name}:{fitting}", count) for name, _, _, part in contributions for fitting, count in part.report.followed))
    combined = Terraforming(grid, natural, designed, Earthworks(fill, frozenset(cut)), report,
        clearance=clearance, coverage=coverage, terrain_top=terrain_top, terrain_chunks=chunks or None,
        water=water, track_clearance=track_clearance, dry_blocks=frozenset().union(*(part.dry_blocks for part in parts)))
    return combined, fields


def shape_terrain(placed: PlacedBuild, spec: CompositionSpec) -> Terraforming | None:
    world: OwnedWorld = placed.components[0].world
    plans = [each.build.terraform for each in spec.components if each.build.terraform is not None]
    if not plans:
        return None
    terrain = load_terrain(plans[0].terrain_world, plans[0].dimension)
    region_owners = {each.name: each.terrain_component for each in spec.regions}
    designed_masks = any(mask.reference == "designed" for each in spec.components for mask in _surface_masks(each.build))
    previous = None
    for _ in range(4 if designed_masks else 1):
        positions = {each.name: [] for each in spec.components}
        for position in world.iter_source_positions():
            owner = world.owner_at(position)
            owner = region_owners.get(owner, owner)
            if owner in positions:
                positions[owner].append(position)
        contributions = []
        for index, component in enumerate(spec.components):
            plan = component.build.terraform
            owned = positions[component.name]
            if plan is None or not owned:
                continue
            prepared = placed.components[index]
            track_domains = {prepared.domains.id_of(section.domain) for section in component.build.sections}
            if not plan.include_rails and not any(world.source_at(position).domain in track_domains for position in owned):
                continue
            fittings = {name: [position for position in feet_of(world, ids) if world.owner_at(position) == component.name]
                        for name, ids in placed.fitting_ids[index].items()}
            shaped = reshape(component.build, world, world.registry, prepared.domains, prepared.path,
                fittings, prepared.silhouette, prepared.spans, apply=False, terrain=terrain, positions=owned)
            contributions.append((component.name, component.terrain_priority, plan.domain, shaped))
        combined, fields = combine_terrain(contributions, world)
        if combined is None:
            return None
        if not designed_masks:
            break
        world.surface_reference.update(combined)
        world.refresh_surface_masks()
        footprint = tuple((each.placement_id, world._footprints[each.placement_id]) for each in world.instances)
        if footprint == previous:
            break
        previous = footprint
    else:
        if designed_masks:
            raise ValueError("shared designed surface masks did not settle after four passes")
    for domain, field in fields.items():
        profile = world.domains.profile(domain)
        world.add_field(field, placement_mode=PlacementMode.REPLACE_AIR_ONLY,
                        connectivity_domain=profile.domain_id, block_replacement=profile.replacement, earthworks=True)
    if combined.cut:
        world.base_blocks = Excavated(world.base_blocks, combined.cut)
        world.invalidate_connectivity_policy()
    if world.surface_reference is not None:
        world.surface_reference.update(combined)
        world.refresh_surface_masks()
    return combined
