from ..Core.Field.block_field import BlockField
from ..Core.block.rotation import rotate_block_state
from ..Format.schematics import parse_state_key


SIDES = ((-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1))


def lay_formation(world, domains, spec, traces, *, placements=None, owns=lambda p: True, holds=lambda p: True):
    sources = {p: world.source_at(p) for p in world.iter_source_positions()
               if owns(p)}
    cells = {p: world.registry.get_state(source.block_id) for p, source in sources.items()}
    replace = {domains.id_of(name) for name in spec.replace_domains}
    rail_domains = {domains.id_of(name) for name in spec.rail_domains}
    rails = {p: state for p, state in cells.items() if sources[p].domain in rail_domains}
    if not rails:
        return
    if not traces["shoulders"]:
        raise ValueError(f"formation {spec.name!r} needs native rail lane paths")
    floor = {}
    for placement in world.instances if placements is None else placements:
        if placement.connectivity_domain not in replace:
            continue
        for local, block_id in placement.field.iter_block_ids():
            state = placement.field.palette.get_state(block_id)
            p = placement.transform.apply(local)
            if state.name in spec.foundation_blocks and placement.occupies(p):
                floor[p + (0, 1, 0)] = state
    if not floor:
        raise ValueError(f"formation {spec.name!r} has rails but no authored foundation")
    supports = {p + (0, -1, 0): state for p, state in rails.items()}
    shoulders = traces["shoulders"] - supports.keys()
    flanks = traces["flanks"] - supports.keys() - shoulders
    bore = {p + (0, dy, 0) for p in rails for dy in range(5)}
    support, flank, shoulder, edge = (parse_state_key(value) for value in
                                    (spec.support, spec.flank, spec.shoulder, spec.edge))
    air = parse_state_key("minecraft:air")
    field = BlockField(world.registry.palette)

    def put(p, state):
        current = world.source_at(p)
        if holds(p) and p not in bore and (current is None or (current.domain in replace and owns(p))):
            field.set_block(p, state)

    for p, state in cells.items():
        if sources[p].domain in replace and (state.name == edge.name or (
            p in floor and state.name in {support.name, flank.name, shoulder.name})):
            put(p, air)
    authored = dict(floor)
    columns = ({}, {})
    for p in supports:
        columns[0].setdefault((p.x, p.y), []).append(p.z)
        columns[1].setdefault((p.z, p.y), []).append(p.x)
    levels = {p.y for p in authored}
    margin = spec.clearance + 1
    offsets = sorted(((dx, 0, dz) for dx in range(-margin, margin + 1) for dz in range(-margin, margin + 1)),
                     key=lambda step: (abs(step[0]) + abs(step[2]), step))
    for p, rail in supports.items():
        axis = int(rail.get("shape") in ("north_south", "ascending_north", "ascending_south"))
        for step in offsets:
            q = p + step
            along = q.z if axis else q.x
            across = q.x if axis else q.z
            held = columns[axis].get((along, q.y), ())
            held = any(abs(across - value) <= margin * 2 for value in held)
            if q in floor or not held or not holds(q + (0, -1, 0)):
                continue
            source = next((authored[candidate] for y in sorted(levels, key=lambda y: (abs(q.y - y), y))
                           for offset in offsets if (candidate := q + offset + (0, y - q.y, 0)) in authored), None)
            if source is not None and q + (0, -1, 0) not in bore:
                put(q + (0, -1, 0), source)
                if field.get_block(q + (0, -1, 0)) is not None:
                    floor[q] = source

    interior = supports.keys() | flanks | shoulders
    clearance = {p + (0, dy, 0) for p in interior for dy in range(5)}

    def outside(p):
        for step in SIDES:
            if p + step not in shoulders or p + step not in floor:
                continue
            q = p
            while q in floor and q not in interior:
                q += (-step[0], 0, -step[2])
            if q not in floor:
                return True
        return False

    edges = {p for p in (traces["edges"] - interior - clearance) & floor.keys()
             if outside(p)}
    for positions, state in ((shoulders, shoulder), (flanks, flank), (supports, support)):
        for p in positions:
            if p in floor or p in supports:
                if state is flank:
                    nearby = sorted((r for step in SIDES if (r := p + step) in supports),
                                    key=lambda r: r.as_tuple())
                    rail = supports[nearby[0]] if nearby else None
                    turns = int(rail is not None and rail.get("shape") in ("north_south", "ascending_north", "ascending_south"))
                    oriented = rotate_block_state(flank, world.registry.get_info(world.registry.register(flank)), turns)
                    put(p, oriented)
                else:
                    put(p, state)
    for p in edges:
        base = p + (0, -1, 0)
        below = field.get_block(base)
        if below is None:
            below = world.get_block(base)
        if below is None or below == air:
            put(base, floor[p])
        for dy in range(spec.edge_height):
            q = p + (0, dy, 0)
            if q not in clearance:
                put(q, edge)
    for p, foundation in floor.items():
        above = world.source_at(p + (0, 1, 0))
        state = field.get_block(p)
        if (state == air and above is not None and above.domain not in replace
                and world.registry.get_state(above.block_id).name == edge.name):
            put(p, foundation)
    if len(field):
        return domains.place(world, field, spec.domain)
