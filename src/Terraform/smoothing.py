"""Shared, deterministic material-direction rules for terrain smoothing."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from . import kernels
from .distance import nearest_euclidean
from .grid import Field
from .layers import smootherstep
from .terrain import ABSENT

if TYPE_CHECKING:
    from .layers import Ground
    from .plan import Plan, Smoothing, Stretch
    from .terraform import Alignment


class Mode(IntEnum):
    STABLE = 0
    GROW = 1
    MELT = 2


def choose(designed: Field, natural: Field, threshold: float = 1.0) -> Field:
    """Use local design intent; neither a station median nor the opposite side votes."""
    change = designed - natural
    return np.where(change > threshold, Mode.GROW,
                    np.where(change < -threshold, Mode.MELT, Mode.STABLE)).astype(np.int8)


def constrain(reference: Field, candidate: Field, modes: Field, stable_change: float) -> Field:
    """Grow only adds, melt only removes; stable has one budget over all passes."""
    return np.where(modes == Mode.GROW, np.maximum(reference, candidate),
                    np.where(modes == Mode.MELT, np.minimum(reference, candidate),
                             np.clip(candidate, reference - stable_change, reference + stable_change)))


def blur(values: Field, known: Field, radius: float) -> Field:
    # Unknown columns must not pull a bank down toward the ABSENT sentinel.
    return kernels.normalized(values, known, radius)


def new_exposure(designed: Field, natural: Field, known: Field) -> Field:
    """Columns whose directed vertical faces grew after neighbouring edits."""
    exposed = np.zeros(natural.shape, dtype=bool)
    for axis in range(natural.ndim):
        first, second = [slice(None)] * natural.ndim, [slice(None)] * natural.ndim
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        first, second = tuple(first), tuple(second)
        valid = known[first] & known[second]
        # Compare the lower end of each face, not only the height of its
        # owner. A saved column can acquire a new face without moving.
        for high, low in ((first, second), (second, first)):
            old_bottom = np.minimum(natural[high], natural[low])
            bottom = np.minimum(designed[high], designed[low])
            new = (designed[high] > natural[high] + .5) | (
                bottom < old_bottom - .5)
            exposed[high] |= valid & (designed[high] > designed[low] + .5) & new
    return exposed


def preserved_faces(designed: Field, natural: Field, known: Field) -> Field:
    cliff = np.zeros(natural.shape, dtype=bool)
    original_drop = np.zeros(natural.shape)
    final_drop = np.zeros(natural.shape)
    for axis in range(natural.ndim):
        first, second = [slice(None)] * natural.ndim, [slice(None)] * natural.ndim
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        first, second = tuple(first), tuple(second)
        edge = known[first] & known[second] & (np.abs(natural[first] - natural[second]) >= 2)
        cliff[first] |= edge
        cliff[second] |= edge
        for high, low in ((first, second), (second, first)):
            valid = known[high] & known[low]
            original_drop[high] = np.maximum(original_drop[high], np.where(valid, natural[high] - natural[low], 0))
            final_drop[high] = np.maximum(final_drop[high], np.where(valid, designed[high] - designed[low], 0))
    # An existing dominant cliff face may deliberately meet a portal. A
    # new side recess does not authorize flattening that entire native face.
    dominant_kept = (original_drop >= 2) & (final_drop <= original_drop + .5)
    return cliff & (np.abs(designed - natural) < .5) & (
        ~new_exposure(designed, natural, known) | dominant_kept)


def contact_floor(reference: Field, natural: Field, known: Field,
                  rock_floor: Field, cover_floor: Field) -> Field:
    """Keep required roof cover without holding newly exposed rock fins.

    The rock body's trimming depth is a preference, not structural cover.
    Once a neighbouring cut exposes a new face, allow local ledge grading
    below that preference while keeping the actual roof/fitting crest.
    Existing native faces retain their original rock floor.
    """
    return np.where(new_exposure(reference, natural, known), cover_floor,
                    np.maximum(rock_floor, cover_floor))


def contour_cut(natural: Field, sources: Field, heights: Field, editable: Field,
                reach: float, max_drop: float | Field, contour_scale: float = 3.) -> Field:
    """Use the shared connected contact kernel for excavation controls only."""
    return contour_contact(natural, sources & (heights < natural - .5), heights,
                           editable, reach, max_drop, contour_scale)


def contour_contact(natural: Field, sources: Field, heights: Field, editable: Field,
                    reach: float, max_drop: float | Field, contour_scale: float = 3.) -> Field:
    """Plan finite earthworks from real contacts, carrying the saved land's relief.

    Travel follows connected terrain, with a small penalty for crossing
    native contours. Sources carry displacement, not absolute elevation.
    A compact C2 falloff meets the original hill, with masked displacement
    diffusion to soften source seams, rather than a growing slope cone.
    """
    import heapq

    sources = sources & editable
    if not sources.any() or reach <= 0:
        return natural.copy()
    distance = np.full(natural.shape, np.inf)
    carried = np.zeros(natural.shape)
    widths = np.zeros(natural.shape)
    drop = np.broadcast_to(max_drop, natural.shape)
    queue = []
    for r, c in zip(*np.nonzero(sources)):
        distance[r, c] = 0
        carried[r, c] = heights[r, c] - natural[r, c]
        widths[r, c] = min(reach, max(2., 1.875 * abs(carried[r, c]) / max(drop[r, c], .1)))
        heapq.heappush(queue, (0., int(r), int(c)))
    rows, cells = natural.shape
    steps = ((-1, 0, 1.), (1, 0, 1.), (0, -1, 1.), (0, 1, 1.),
             (-1, -1, 2**.5), (-1, 1, 2**.5), (1, -1, 2**.5), (1, 1, 2**.5))
    while queue:
        away, r, c = heapq.heappop(queue)
        if away != distance[r, c]:
            continue
        for dr, dc, length in steps:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < rows and 0 <= cc < cells and editable[rr, cc]):
                continue
            # Both gates are required: even a diagonal cannot cross a
            # water/structure mask or connect two corners of separate cuts.
            if dr and dc and not (editable[r, cc] and editable[rr, c]):
                continue
            step = length * (1 + .15 * min(abs(natural[rr, cc] - natural[r, c])
                                          / max(contour_scale, .1), 2.))
            candidate = away + step
            if candidate < distance[rr, cc] and candidate < reach:
                distance[rr, cc] = candidate
                carried[rr, cc] = carried[r, c]
                widths[rr, cc] = widths[r, c]
                heapq.heappush(queue, (candidate, rr, cc))
    active = editable & np.isfinite(distance) & (distance < widths)
    amount = 1 - smootherstep(np.divide(distance, widths, out=np.ones_like(distance), where=widths > 0))
    displacement = np.where(active, carried * amount, 0.)
    # Remove nearest-source seams without blurring native cliffs or growing
    # the support of the operation. Contact heights remain exact.
    softened = kernels.contour(displacement, natural, active, 4, contour_scale)
    displacement = np.where(active & ~sources, .5 * (displacement + softened), displacement)
    return np.where(editable, natural + displacement, natural)


def retreat(reference: Field, natural: Field, known: Field, editable: Field,
            floor: Field, max_drop: float | Field, passes: int) -> Field:
    """Run new excavation faces back into the hill with synchronous bounds.

    Original directed cliff drops are permitted. Only the extra exposure
    created by the cut propagates; neither fills nor protected columns move.
    """
    result = reference.copy()
    drop = np.broadcast_to(max_drop, reference.shape)
    for _ in range(passes):
        ceiling = np.full(reference.shape, np.inf)
        for axis in (0, 1):
            a, b = [slice(None)] * 2, [slice(None)] * 2
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            for high, low in ((a, b), (b, a)):
                allowed = np.maximum(drop[high], natural[high] - natural[low])
                bad = known[high] & known[low] & editable[high] & (
                    result[high] <= natural[high]) & (result[high] > result[low] + allowed + 1e-9)
                ceiling[high] = np.minimum(ceiling[high], np.where(bad, result[low] + allowed, np.inf))
        candidate = np.where(editable, np.minimum(result, np.maximum(ceiling, floor)), result)
        if np.array_equal(candidate, result):
            break
        result = candidate
    return result


def meet(reference: Field, natural: Field, footprint: Field, eligible: Field,
         heights: Field, max_drop: int, passes: int,
         options: Smoothing | None = None, reach: float = 32.0,
         overrides: Field | None = None, cut_floor: Field | None = None,
         cut_drop: Field | None = None) -> Field:
    """Carry actual exterior contact heights into nearby land as slope limits.

    A fill needs enough land to support it, a cut needs the land to retreat.
    Taking only the outline keeps a trench floor from cutting through its
    own retaining walls. Suspended land and protected columns are excluded
    by the caller rather than raised toward the deck.
    """
    from scipy.ndimage import binary_dilation
    from . import slopes

    outline = footprint & binary_dilation(~footprint)
    sources = outline & eligible
    grow = sources & (heights > natural + .5)
    melt = sources & (heights < natural - .5)
    opened = eligible & ~footprint
    result = reference.copy()
    def contact(selected, bound, rising):
        allowed = max_drop if not rising or cut_drop is None else cut_drop
        # Both banks and cuttings follow connected native contours. A
        # nearest-source lookup across a keep-out can carry the other
        # bank's height through water or a structure into this slope.
        candidate = contour_contact(natural, selected, heights, opened | selected,
                                    reach, allowed,
                                    3. if options is None else options.contour_scale)
        amount = 1.0 if options is None else options.contact_blend
        shaped = reference + amount * (candidate - reference)
        if options is not None:
            modes = choose(candidate, natural, options.decision_threshold)
            if options.behavior != "auto":
                modes = np.full(reference.shape, Mode[options.behavior.upper()], dtype=np.int8)
            if overrides is not None:
                modes = np.where(overrides >= 0, overrides, modes)
            shaped = constrain(reference, shaped, modes, options.stable_change)
        if rising:
            candidate = np.minimum(shaped, bound)
            if cut_floor is not None:
                candidate = np.maximum(candidate, cut_floor)
            return np.minimum(reference, candidate)
        return np.maximum(reference, np.maximum(shaped, bound))
    if grow.any():
        lower = slopes.cone(grow, heights, opened | grow, 1 / max_drop,
                            rising=False, passes=passes, natural=natural)
        result = np.where(opened & np.isfinite(lower), contact(grow, lower, False), result)
    if melt.any():
        upper = slopes.cone(melt, heights, opened | melt,
                            1 / (max_drop if cut_drop is None else cut_drop),
                            rising=True, passes=passes, natural=natural, floor=cut_floor)
        result = np.where(opened & np.isfinite(upper),
                          np.minimum(result, contact(melt, upper, True)), result)
    return result


def brush(reference: Field, area: Field, modes: Field, options: Smoothing,
          radius: float, passes: int, known: Field | None = None,
          guide: Field | None = None) -> Field:
    if passes <= 0 or not (area > 0).any():
        return reference
    known = (reference > ABSENT) if known is None else known
    weight = np.where(known, np.clip(area, 0.0, 1.0), 0.0)
    result = reference.copy()
    for _ in range(passes):
        average = kernels.normalized(result, known, radius, options.kernel)
        if guide is not None and options.contour_strength > 0:
            # Smooth edits along the saved land. Broad absolute-height
            # averaging alone rounds off native rock and repeats terraces.
            relief = np.where(known, result - guide, 0.0)
            carried = kernels.contour(relief, guide, known & (weight > 0),
                                      max(1, int(np.ceil(radius * radius))), options.contour_scale)
            average += options.contour_strength * (guide + carried - average)
        candidate = result + weight * (average - result)
        result = np.where(weight > 0,
                          constrain(reference, candidate, modes, options.stable_change), reference)
    return result


@dataclass(frozen=True, slots=True)
class Context:
    overrides: Field
    transition_weight: Field

    def modes(self, designed: Field, natural: Field, options: Smoothing) -> Field:
        automatic = choose(designed, natural, options.decision_threshold)
        return np.where(self.overrides >= 0, self.overrides, automatic).astype(np.int8)


def context(plan: Plan, ground: Ground, alignment: Alignment,
            stretches: tuple[Stretch, ...]) -> Context:
    """Resolve profile and transition overrides once, using route geometry.

    Global behavior, then profile overrides, then matching transition rules;
    later rules win. Automatic decisions remain independent per terrain column.
    """
    from .cross_section import axes

    options, grid = plan.smoothing, ground.grid
    stations = np.clip(np.rint(ground.station).astype(int), 0, alignment.length - 1)
    profiles = np.full(alignment.length, "", dtype=object)
    labels = np.full(alignment.length, "", dtype=object)
    active = np.zeros(alignment.length, dtype=bool)
    for stretch in sorted(stretches, key=lambda each: each.order):
        start, stop = max(0, stretch.start), min(alignment.length, stretch.stop + 1)
        profiles[start:stop] = stretch.profile
        labels[start:stop] = repr((stretch.profile, stretch.cross_section))
        active[start:stop] = stretch.intensity.value == "shape"

    def code(behavior):
        return -1 if behavior == "auto" else Mode[behavior.upper()]

    overrides = grid.filled(code(options.behavior), dtype=np.int8)
    for profile, behavior in options.profiles.items():
        overrides = np.where(profiles[stations] == profile, code(behavior), overrides)
    weight = grid.filled(0.0)
    oriented = None
    columns = {}
    for column, station in alignment.station.items():
        columns.setdefault(station, []).append(column)
    for index in range(1, alignment.length):
        if labels[index] == labels[index - 1] or not (active[index] or active[index - 1]):
            continue
        source, target = profiles[index - 1], profiles[index]
        # An interval with no earthworks assignment is deliberately held;
        # it is not an implicit section requesting a new transition.
        if not source or not target:
            continue
        seeds = grid.filled(False, dtype=bool)
        for x, z in (*columns.get(index - 1, ()), *columns.get(index, ())):
            seeds[grid.at(x, z)] = True
        if not seeds.any():
            continue
        matching = [rule for rule in options.transitions if
                    (source == rule.source and target == rule.target) or (
                        rule.bidirectional and source == rule.target and target == rule.source)]
        default_reach = options.transition_reach
        if default_reach > 0 and options.adaptive_transitions:
            known = seeds & (ground.natural > ABSENT)
            if known.any():
                gap = np.maximum(np.abs(ground.natural - ground.seed_height),
                                 np.abs(ground.natural - ground.crest))
                # A tall meeting needs more room than a small bump. The
                # corridor remains bounded by the ordinary terraform reach.
                needed = 1.875 * float(np.max(gap[known])) / plan.slopes.max_drop
                default_reach = max(default_reach, min(needed, plan.reach, plan.margin))
        reach_max = max([default_reach, *(
            default_reach if rule.reach is None else rule.reach for rule in matching)])
        if reach_max <= 0:
            continue
        distance = nearest_euclidean(seeds, (), reach_max + 1).distance
        # Avoid reaching a different arm of a winding route just because it
        # passes close to this boundary in world space.
        own_arm = (stations >= index - default_reach - 1) & (
            stations <= index + default_reach + 1)
        boundary_weight = (np.where(own_arm, 1 - smootherstep(distance / default_reach), 0.0)
                           if default_reach > 0 else grid.filled(0.0))
        for rule in matching:
            reach = default_reach if rule.reach is None else rule.reach
            own_arm = (stations >= index - reach - 1) & (stations <= index + reach + 1)
            influence = (np.where(own_arm, 1 - smootherstep(distance / reach), 0.0)
                         if reach > 0 else grid.filled(0.0))
            side = np.ones(grid.shape, dtype=bool)
            if rule.side != "both":
                if oriented is None:
                    oriented = axes(alignment)
                xs = np.arange(grid.width)[None, :] + grid.origin.x
                zs = np.arange(grid.depth)[:, None] + grid.origin.z
                lateral = ((xs - oriented.x[stations]) * oriented.right_x[stations]
                           + (zs - oriented.z[stations]) * oriented.right_z[stations])
                side = lateral < 0 if rule.side == "left" else lateral > 0
            selected = (influence > 0) & side
            overrides = np.where(selected, code(rule.behavior), overrides)
            boundary_weight = np.where(side, influence, boundary_weight)
        weight = np.maximum(weight, boundary_weight)
    return Context(overrides, weight)


def guard_steps(reference: Field, candidate: Field, editable: Field, max_drop: int) -> Field:
    """A brush must not introduce new cliff steps after the slope repair.

    Existing natural/structural faces may remain. Only offending brush edits
    are withdrawn; the pre-brush surface is always a feasible fallback.
    """
    result = candidate.copy()
    for _ in range(32):
        changed = np.zeros(reference.shape, dtype=bool)
        levels, original = np.floor(result + .5), np.floor(reference + .5)
        for axis in range(reference.ndim):
            low, high = [slice(None)] * reference.ndim, [slice(None)] * reference.ndim
            low[axis], high[axis] = slice(None, -1), slice(1, None)
            low, high = tuple(low), tuple(high)
            limit = np.maximum(max_drop, np.abs(original[low] - original[high]))
            bad = np.abs(levels[low] - levels[high]) > limit
            changed[low] |= bad
            changed[high] |= bad
        changed &= editable & (result != reference)
        if not changed.any():
            return result
        result = np.where(changed, reference, result)
    return reference.copy()
