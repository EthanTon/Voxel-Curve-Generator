from ...Assembly.formations import lay_formation as construct
from .regions import contains


def lay_formation(world, domains, rule, traces):
    components = frozenset(rule.components)
    placements = tuple(p for p in world.instances if world.owners.get(p.placement_id) in components)

    def holds(p):
        return (not rule.boundary or contains(rule.boundary, p.x + .5, p.z + .5)) and (
            rule.heights is None or rule.heights[0] <= p.y <= rule.heights[1])

    placement = construct(world, domains, rule.formation, traces, placements=placements, holds=holds,
                          owns=lambda p: world.owner_at(p) in components)
    if placement is not None:
        world.owners[placement] = rule.components[0]


def lay_formations(world, domains, rules, paths):
    for rule in rules:
        traces = {role: set().union(*(paths[name].get(rule.formation.name, {}).get(role, ())
                                     for name in rule.components))
                  for role in ("flanks", "shoulders", "edges")}
        lay_formation(world, domains, rule, traces)
