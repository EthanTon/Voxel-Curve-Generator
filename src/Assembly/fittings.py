from ..Core.constants import AIR_NAMES


def retain_assemblies(placed):
    world = placed.components[0].world
    protected, foundations = set(), {}
    formations = {formation for component in placed.components for formation in getattr(component.spec, "formations", ())}
    for formation in formations:
        domains = {placed.components[0].domains.id_of(name) for name in formation.rail_domains}
        for p in world.iter_source_positions():
            if world.source_at(p).domain in domains:
                margin = formation.clearance + 1
                for dx in range(-margin, margin + 1):
                    for dz in range(-margin, margin + 1):
                        for dy in (-2, -1):
                            foundations.setdefault(p + (dx, dy, dz), set()).add(p.y - 1)
                protected.update(p + (dx, dy, dz) for dx in range(-formation.clearance, formation.clearance + 1)
                                 for dz in range(-formation.clearance, formation.clearance + 1) for dy in range(-1, 5))
    groups = {}
    for index, (component, fittings) in enumerate(zip(placed.components, placed.fitting_ids)):
        for structure in component.spec.structures:
            if structure.assembly is None:
                continue
            for placement_id in fittings.get(structure.name, ()):
                placement = world.get_placement(placement_id)
                groups.setdefault((index, structure.assembly, placement.transform.origin), set()).add(placement_id)
    removed = set()
    for ids in sorted(groups.values(), key=max, reverse=True):
        complete = True
        for placement_id in ids:
            placement = world.get_placement(placement_id)
            for local, block_id in placement.field.iter_block_ids():
                if placement.field.palette.get_state(block_id).name in AIR_NAMES:
                    continue
                p = placement.transform.apply(local)
                source = world.source_at(p)
                foreign = p in foundations and not any(abs(placement.transform.origin.y - y) <= 1 for y in foundations[p])
                if foreign or p in protected or not placement.occupies(p) or source is None or source.placement_id not in ids:
                    complete = False
                    break
            if not complete:
                break
        if not complete:
            for placement_id in ids:
                world.remove_placement(placement_id)
            removed.update(ids)
    for fittings in placed.fitting_ids:
        for name, ids in fittings.items():
            fittings[name] = [placement_id for placement_id in ids if placement_id not in removed]
