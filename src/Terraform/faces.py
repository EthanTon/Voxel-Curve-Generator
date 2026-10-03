"""Bounded, synchronous morphology of construction-created terrain faces."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import binary_dilation

from ..Core.coordinates import BlockPos
from .classify import BlockClass, classify
from .components import FACES
from .smoothing import Mode, guard_steps, new_exposure
from .terrain import SOLID


def repair(terrain, grid, natural, designed, known, eligible, fill, cut,
           clearance, modes, options, overrides=None, max_drop=1,
           cut_floor=None, fill_ceiling=None):
    """Repair new thin remnants and cut-created pockets; preserve native voids.

    Read saved occupancy, not a heightfield pretending that every column is
    solid. Decisions in each iteration use one snapshot. Body columns,
    fitting buffers, water and keep-outs are excluded by the caller.
    """
    if options.face_iterations <= 0 or not eligible.any():
        return frozenset(), {}
    touched = known & (np.abs(designed - natural) >= .5)
    exposed = new_exposure(designed, natural, known)
    for p in cut:
        if grid.holds(p.x, p.z):
            touched[grid.at(p.x, p.z)] = True
    seeds = touched | exposed
    if not seeds.any():
        return frozenset(), {}
    area = eligible & binary_dilation(seeds, iterations=options.face_iterations)
    if not area.any():
        return frozenset(), {}

    @lru_cache(None)
    def column(x, z):
        return {y: state for y, state in terrain.column(x, z)}

    def original(p):
        state = column(p.x, p.z).get(p.y)
        return state, BlockClass.AIR if state is None else classify(state)

    additions = {p: fill.palette.get_state(block_id) for p, block_id in fill.iter_block_ids()}
    removed = set(cut)
    added = {}
    eroded = set()
    budget = {}
    levels = np.floor(designed + .5)
    candidates = set()
    bottom = {}
    cut_bottom = {}
    for p in cut:
        key = p.x, p.z
        cut_bottom[key] = min(cut_bottom.get(key, p.y), p.y)
    for row, cell in zip(*np.nonzero(area)):
        x, z = tuple(grid.column(row, cell))
        # Avoid treating unsurveyed columns as open space at the edit rim.
        if any(not grid.holds(x + dx, z + dz) or not known[grid.at(x + dx, z + dz)]
               for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            continue
        neighbours = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
        # A bore may remove sides beneath unchanged tops. Its actual cut
        # depth, rather than the heightfield alone, sets this repair window.
        low = min(min(levels[grid.at(x + dx, z + dz)],
                      cut_bottom.get((x + dx, z + dz), np.inf))
                  for dx, dz in neighbours) - options.face_depth
        bottom[x, z] = low
        for y, state in column(x, z).items():
            if low <= y <= levels[row, cell] and classify(state) is BlockClass.GROUND:
                candidates.add(BlockPos(x, y, z))

    def occupied(p):
        if p in added or p in additions:
            return True
        return p not in removed and original(p)[1] in (*SOLID, BlockClass.FLUID)

    def movement(p):
        at = grid.at(p.x, p.z)
        mode = modes[at]
        explicit = (options.behavior != "auto" if overrides is None else overrides[at] >= 0)
        cut_face = any(p + step in removed and p + step not in additions
                       and original(p + step)[1] is BlockClass.GROUND for step in FACES)
        if mode == Mode.STABLE and not explicit and (exposed[at] or cut_face):
            # A previously unchanged column inherits an adjacent excavation's
            # melt intent only when its exposed face is new. A sideways cut
            # can expose it even when both column tops remain unchanged.
            mode = Mode.MELT
        return mode

    def within_budget(p, counts):
        return movement(p) != Mode.STABLE or counts.get((p.x, p.z), 0) < int(options.stable_change)

    for _ in range(options.face_iterations):
        take, place = set(), {}
        next_budget = budget.copy()
        # Sorting makes per-column stable budgets independent of set order.
        for p in sorted(candidates - removed - additions.keys(), key=lambda p: (p.x, p.z, -p.y)):
            if (movement(p) == Mode.GROW or not within_budget(p, next_budget)
                    or (cut_floor is not None and p.y <= cut_floor[grid.at(p.x, p.z)])):
                continue
            neighbours = [p + step for step in FACES]
            air = [q for q in neighbours if not occupied(q)]
            old_air = [q for q in neighbours if original(q)[1] not in (*SOLID, BlockClass.FLUID)]
            if (len(air) >= options.face_min_exposed and len(old_air) < options.face_min_exposed
                    and any(q not in old_air for q in air)):
                take.add(p)
                next_budget[p.x, p.z] = next_budget.get((p.x, p.z), 0) + 1
        # Fill only voids created by our cut, never original caves or water.
        for p in sorted(removed):
            if (not grid.holds(p.x, p.z) or not area[grid.at(p.x, p.z)]
                    or (p.x, p.z) not in bottom or p.y < bottom[p.x, p.z]
                    or p.y > levels[grid.at(p.x, p.z)]
                    or (fill_ceiling is not None and p.y > fill_ceiling[grid.at(p.x, p.z)])
                    or clearance.holds(*grid.at(p.x, p.z), p.y)
                    or movement(p) == Mode.MELT or not within_budget(p, next_budget)):
                continue
            state, kind = original(p)
            if kind is not BlockClass.GROUND:
                continue
            if sum(occupied(p + step) for step in FACES) >= 5:
                place[p] = state
                next_budget[p.x, p.z] = next_budget.get((p.x, p.z), 0) + 1
        if not take and not place:
            break
        # Taking a surface corner must not introduce a new terrace step.
        # Subsurface side erosion remains fully three-dimensional.
        affected = {(p.x, p.z) for p in take | place.keys()}
        proposed = levels.copy()
        # Snapshot once per synchronous round. Rebuilding the entire cut
        # union for every surviving voxel makes wide gentle cuts quadratic.
        pending_removals = removed | take
        addition_tops = {}
        for positions in (additions, added, place):
            for p in positions:
                key = p.x, p.z
                if key in affected:
                    addition_tops[key] = max(addition_tops.get(key, p.y), p.y)
        for x, z in affected:
            survivors = [y for y, state in column(x, z).items()
                         if y <= levels[grid.at(x, z)] and classify(state) in SOLID
                         and BlockPos(x, y, z) not in pending_removals]
            if (x, z) in addition_tops:
                survivors.append(addition_tops[x, z])
            if survivors:
                proposed[grid.at(x, z)] = max(survivors)
        guarded = guard_steps(levels, proposed, area, max_drop)
        denied = {(x, z) for x, z in affected if guarded[grid.at(x, z)] != proposed[grid.at(x, z)]}
        take = {p for p in take if (p.x, p.z) not in denied}
        place = {p: state for p, state in place.items() if (p.x, p.z) not in denied}
        if not take and not place:
            break
        # Commit together; a voxel removed this round only affects the next.
        removed.update(take)
        eroded.update(take)
        removed.difference_update(place)
        added.update(place)
        candidates.difference_update(place)
        budget = budget.copy()
        for p in take | place.keys():
            budget[p.x, p.z] = budget.get((p.x, p.z), 0) + 1
        levels = np.where(guarded == proposed, proposed, levels)
    # Ground left touching nothing over an eroded block would float; it goes too.
    above = BlockPos(0, 1, 0)
    loose = {p + above for p in eroded
             if p + above not in removed and p + above not in additions
             and grid.holds(p.x, p.z) and area[grid.at(p.x, p.z)]
             and original(p + above)[1] is BlockClass.GROUND
             and not any(occupied(p + above + step) for step in FACES)}
    eroded |= loose
    return frozenset(eroded - added.keys()), added


def floating_remnants(terrain, grid, area, fill, cut, budget: int = 48) -> frozenset[BlockPos]:
    """Small groups of ground the cut left joined to nothing, over `area`'s columns.

    A cut beside a native shelf over a hollow, or into one, can leave a few
    blocks (saved or laid) hanging. Each group of ground beside the cut with
    nothing directly under it is followed through touching ground; one that
    ends within `budget` blocks, all of it ground in `area`, floats.
    """
    additions = {p: state for p, state in ((p, fill.palette.get_state(i)) for p, i in fill.iter_block_ids())}
    below = BlockPos(0, -1, 0)

    @lru_cache(None)
    def column(x, z):
        return {y: classify(state) for y, state in terrain.column(x, z)}

    def kind(p):
        if p in additions:
            return classify(additions[p])
        if p in cut:
            return BlockClass.AIR
        return column(p.x, p.z).get(p.y, BlockClass.AIR)

    def solid(p):
        return kind(p) in SOLID

    # A chunk at a time, so the saved world is read once.
    within = sorted((p for p in cut if grid.holds(p.x, p.z) and area[grid.at(p.x, p.z)]),
                    key=lambda p: (p.x >> 4, p.z >> 4, p.x, p.z, p.y))
    seeds = []
    for p in within:
        for step in FACES:
            q = p + step
            if kind(q) is BlockClass.GROUND and not solid(q + below):
                seeds.append(q)
    floating, anchored = set(), set()
    for seed in seeds:
        if seed in floating or seed in anchored:
            continue
        group, frontier, held = {seed}, [seed], False
        while frontier and not held:
            q = frontier.pop()
            for step in FACES:
                r = q + step
                if r in group or not solid(r):
                    continue
                if (r in anchored or len(group) >= budget or kind(r) is not BlockClass.GROUND
                        or not grid.holds(r.x, r.z) or not area[grid.at(r.x, r.z)]):
                    held = True
                    break
                group.add(r)
                frontier.append(r)
        (anchored if held else floating).update(group)
    return frozenset(floating)
