from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from ..Core.block.palette import BlockPalette
from ..Core.coordinates import BlockPos, ColumnPos
from ..Core.plan_index import NearestInPlan
from . import clearance as clearing
from . import layers as vocabulary
from . import kernels
from .cross_section import side_modes
from .clearance import Clearance
from .distance import nearest, nearest_euclidean
from .grid import Field, Grid
from .materialize import Earthworks, build
from .plan import Action, Intensity, Look, Plan, Step, Stretch
from .surroundings import Surroundings, measure
from .terrain import ABSENT, Terrain, survey
from .surface_mask import SurfaceReference
from .vegetation import clearing as growth_clearing


SAMPLE_OFFSET = 0.5
SETTLING_LIMIT = 2.0
DEAD_BAND = 0.5
# The look (smoothing, grain) is held off the track and SHOULDER beside it,
# fading in to full strength FORMATION out.
SHOULDER = 2.0
FORMATION = 5.0
DESPECKLE_PASSES = 2
# The radius over which heights carried out from the track are averaged, and
# the share of a stretch's scope over which its change fades out.
EASING = 6.0
FADE = 0.25


@dataclass(frozen=True, slots=True)
class Alignment:
    footing: dict[tuple[int, int], int]
    crest: dict[tuple[int, int], int]
    station: dict[tuple[int, int], int]

    @classmethod
    def of(
        cls,
        blocks: Iterable[BlockPos],
        stations: Sequence[int] | None = None,
    ) -> Alignment:
        footing: dict[tuple[int, int], int] = {}
        crest: dict[tuple[int, int], int] = {}
        station: dict[tuple[int, int], int] = {}
        for index, position in enumerate(blocks):
            key = (position.x, position.z)
            footing[key] = min(footing.get(key, position.y), position.y)
            crest[key] = max(crest.get(key, position.y), position.y)
            station.setdefault(key, index if stations is None else stations[index])
        return cls(footing, crest, station)

    @property
    def columns(self) -> tuple[ColumnPos, ...]:
        return tuple(ColumnPos(x, z) for x, z in self.footing)

    @property
    def length(self) -> int:
        return max(self.station.values(), default=0) + 1

    def locate(self, where) -> int:
        if isinstance(where, str):
            raise KeyError(f"marker {where!r} must be resolved before terraforming")
        where = ColumnPos.coerce(where)
        found = self.station.get((where.x, where.z))
        if found is None:
            raise KeyError(f"{tuple(where)} is not a column of the alignment")
        return found

    def crest_blocks(self) -> list[BlockPos]:
        return [BlockPos(x, y, z) for (x, z), y in self.crest.items()]


@dataclass(frozen=True, slots=True)
class Report:
    surroundings: Surroundings
    stretches: tuple[tuple[str, int, int, int], ...]
    followed: tuple[tuple[str, int], ...]
    ungenerated: int
    suppressed: int
    filled: int
    carved: int
    cleared: int

    def lines(self) -> list[str]:
        shown = [
            f"roughness {self.surroundings.roughness:.2f}"
            f"  scale {self.surroundings.feature_scale:.0f}"
            f"  repose {self.surroundings.repose:.2f}"
            f"  -> fill 1:{self.surroundings.fill_ratio:.1f}"
        ]
        for name, start, stop, columns in self.stretches:
            shown.append(f"{name:<12} {start}..{stop}  {columns} columns")
        for name, placements in self.followed:
            shown.append(f"{name:<12} follows {placements} placements")
        shown.append(f"filled {self.filled}, carved {self.carved}, cleared {self.cleared}")
        shown.append(
            f"ungenerated {self.ungenerated}, suppressed {self.suppressed}"
        )
        return shown


@dataclass(frozen=True, slots=True)
class Terraforming:
    grid: Grid
    natural: Field
    designed: Field
    earthworks: Earthworks
    report: Report
    # Space the build keeps clear at any depth; exports copying the ground
    # must apply it themselves.
    clearance: Clearance | None = None

    @property
    def fill(self):
        return self.earthworks.fill

    @property
    def cut(self) -> frozenset[BlockPos]:
        return self.earthworks.cut

    @property
    def relief(self) -> Field:
        return self.designed - self.natural


def terraform(
    plan: Plan,
    terrain: Terrain,
    alignment: Alignment,
    palette: BlockPalette | None = None,
    clearance: Clearance | None = None,
    locate: Callable[[object], int] | None = None,
    fittings: Mapping[str, Sequence[BlockPos]] | None = None,
    envelope: Mapping[tuple[int, int], Sequence[int]] | None = None,
    section_spans: Mapping[str, Sequence[tuple[int, int]]] | None = None,
) -> Terraforming:
    # With `envelope` (the build's block heights by column) the clearance is
    # the space the build encloses plus headroom; otherwise the plan's
    # headroom over the alignment's crest.
    anchors = {
        name: Alignment.of(blocks)
        for name, blocks in (fittings or {}).items()
        if blocks
    }
    grid = Grid.around(
        [*alignment.columns, *(each for held in anchors.values() for each in held.columns)],
        plan.margin,
    )
    stretches = plan.stretches(
        locate or alignment.locate, alignment.length, section_spans
    )
    if clearance is None and envelope is not None:
        clearance = clearing.enclosing(
            grid, envelope, _headroom(plan, alignment, stretches), alignment.footing
        )
    clearance = clearance or clearing.above(
        grid, alignment.crest_blocks(), plan.headroom
    )
    surface = survey(terrain, grid, clearance.holds, plan.foliage.clear)
    known = surface.available & (surface.height > ABSENT)
    natural = surface.height.astype(np.float64)
    base = _levelled(natural, known)
    seeds = grid.mark(alignment.columns)

    circular_main = any(_circular(plan.profile(stretch.profile)) for stretch in stretches)
    ground = _ground(grid, base, alignment, seeds, plan.reach, circular=circular_main)
    protected = _protected(plan, surface, grid, ground)
    beyond = known & (ground.distance >= plan.reach * SAMPLE_OFFSET)
    surroundings = measure(surface.height, beyond)

    permitted = np.where(protected, 0.0, _permitted(grid, plan, known))
    # No earthworks across water under a bridge's interior.
    water = surface.water if surface.water is not None else grid.filled(ABSENT)
    wet_bridge = grid.filled(False, dtype=bool)
    for stretch in stretches:
        if stretch.profile == "bridge":
            wet_bridge |= (
                (ground.station > stretch.start)
                & (ground.station < stretch.stop)
                & (water > surface.height)
            )
    scopes = [
        np.where(wet_bridge, 0.0, _scope(plan, ground, stretch, permitted))
        if stretch.profile != "bridge" else _scope(plan, ground, stretch, permitted)
        for stretch in stretches
    ]
    stack = _layers(
        plan, surroundings, stretches, scopes, grid, ground, alignment,
        base, water, known,
    )
    bridge_stations = grid.filled(False, dtype=bool)
    for stretch in stretches:
        if stretch.profile == "bridge":
            bridge_stations |= (
                (ground.station > stretch.start) & (ground.station < stretch.stop)
            )
    followers, followed = _followers(
        plan, surroundings, grid, base, anchors, permitted,
        bridge_stations, ground.distance <= FORMATION,
    )
    # Only shaping stretches and footings change the ground.
    writable = np.maximum(_shaping(grid, stack), followed)

    # Stretches, then the look over all of them, then the footings on top.
    designed = vocabulary.shape(ground, stack)
    designed = vocabulary.settle(
        ground, designed, _filters(plan, surroundings), _formation(ground)
    )
    designed = vocabulary.shape(ground, followers, designed)
    fade = np.clip(writable / FADE, 0.0, 1.0)
    designed = np.where(
        (writable > 0) & known,
        clearance.limit_surface(natural, natural + fade * (designed - natural)),
        natural,
    )
    designed = np.where(protected, natural, designed)
    designed = np.where(np.abs(designed - natural) < DEAD_BAND, natural, designed)
    footings = grid.mark(each for held in anchors.values() for each in held.columns)
    editable = known & (writable > 0) & ~protected & ~footings & (ground.distance > 0)
    designed = np.where(editable, clearance.limit_surface(natural, _despeckle(
        designed, natural, known, editable, ground.distance > SHOULDER,
    )), designed)
    designed = _surface_limits(
        plan, terrain, grid, ground, stretches, natural, designed, followers
    )
    designed = _join_fixed_edges(
        designed, natural, known, protected, writable, ground, plan.reach, footings,
    )
    designed = np.where(np.abs(designed - natural) < DEAD_BAND, natural, designed)

    # Growth is cleared where the ground changes or the clearance passes.
    changed = (np.rint(designed) != np.rint(natural)) & known
    cleared = _clear_growth(plan, surface, grid, changed, clearance, protected)
    earthworks = build(
        grid, natural, designed, plan.strata, palette or BlockPalette(),
        surface.state, water, clearance,
    )
    carved = earthworks.cut | frozenset(
        position for position in surface.intruding
        if not (grid.holds(position.x, position.z)
                and protected[grid.at(position.x, position.z)])
    ) | cleared

    return Terraforming(
        grid,
        natural,
        designed,
        Earthworks(earthworks.fill, frozenset(carved)),
        Report(
            surroundings=surroundings,
            stretches=tuple(
                (
                    f"{stretch.intensity.value}:{stretch.profile}",
                    stretch.start,
                    stretch.stop,
                    int((scope > 0).sum()),
                )
                for stretch, scope in zip(stretches, scopes)
            ),
            followed=tuple(
                (follows.fitting, len(anchors[follows.fitting].footing))
                for follows in plan.follows
                if follows.fitting in anchors
            ),
            ungenerated=int((~known).sum()),
            suppressed=int((writable == 0).sum()),
            filled=len(earthworks.fill),
            carved=len(carved),
            cleared=len(cleared),
        ),
        clearance,
    )


def _surface_limits(
    plan: Plan,
    terrain: Terrain,
    grid: Grid,
    ground: vocabulary.Ground,
    stretches: Sequence[Stretch],
    natural: Field,
    designed: Field,
    followers: Sequence[vocabulary.Layer] = (),
) -> Field:
    if plan.surface_mask is None and not any(
        stretch.surface_mask is not None for stretch in stretches
    ) and not any(layer.surface_mask is not None for layer in followers):
        return designed
    # Clamps each changed column to its surface mask: the one of the
    # highest-order stretch at its station, or of a follower covering it. A
    # designed reference reads the unlimited proposal.
    reference = SurfaceReference(terrain, SimpleNamespace(
        grid=grid, natural=natural, designed=designed,
    ))
    limited = designed.copy()
    ranked = sorted(enumerate(stretches), key=lambda pair: (pair[1].order, pair[0]))
    masked_followers = sorted(
        (layer for layer in followers if layer.surface_mask is not None),
        key=lambda each: each.order,
    )
    for row, cell, column in grid.columns():
        was = float(natural[row, cell])
        now = float(designed[row, cell])
        if was <= ABSENT or abs(now - was) < DEAD_BAND:
            continue
        station = int(ground.station[row, cell])
        owning = next((stretch for _, stretch in reversed(ranked)
                       if stretch.start <= station <= stretch.stop), None)
        mask = (owning.surface_mask if owning and owning.surface_mask is not None
                else plan.surface_mask)
        for layer in masked_followers:
            if (layer.ground is not None and layer.mask is not None
                    and layer.mask[row, cell] > 0
                    and layer.ground.distance[row, cell] < (
                        layer.reach if layer.ground.reach is None
                        else layer.ground.reach[row, cell]
                    )):
                mask = layer.surface_mask
        if mask is None:
            continue
        height = reference.height(column.x, column.z, mask)
        if height is None:
            limited[row, cell] = was
            continue
        boundary = height + mask.offset
        if now > was:
            # Fill grows up from the ground; a cut comes down from the top.
            limited[row, cell] = (
                max(was, min(now, boundary)) if mask.side == "below"
                else now if boundary <= was + 1 else was
            )
        else:
            limited[row, cell] = (
                min(was, max(now, boundary - 1)) if mask.side == "above"
                else now if boundary >= was else was
            )
    return limited


def _despeckle(
    designed: Field, natural: Field, known: Field, editable: Field, lowerable: Field
) -> Field:
    # A plus-shaped median of the block heights over the edited ground and its
    # rim: drops lone raised or lowered columns and fills one-column notches,
    # keeping straight edges, corners and diagonals. Columns outside
    # `lowerable` (the track's shoulder) are only ever raised.
    for _ in range(DESPECKLE_PASSES):
        height = np.rint(designed)
        edited = kernels.local_maximum((height != natural).astype(np.float64), 1) > 0
        levels = np.pad(height, 1, mode="edge")
        present = np.pad(known, 1, mode="constant")
        around = [
            np.where(present[rows, cells], levels[rows, cells], height)
            for rows, cells in (
                (slice(None, -2), slice(1, -1)), (slice(2, None), slice(1, -1)),
                (slice(1, -1), slice(None, -2)), (slice(1, -1), slice(2, None)),
            )
        ]
        median = np.sort(np.stack((height, *around)), axis=0)[2]
        moved = editable & edited & ((median > height) | (lowerable & (median < height)))
        if not moved.any():
            break
        designed = np.where(moved, median, designed)
    return designed


def _join_fixed_edges(
    designed: Field,
    natural: Field,
    known: Field,
    protected: Field,
    writable: Field,
    ground: vocabulary.Ground,
    reach: float,
    anchors: Field | None = None,
) -> Field:
    """Grade a fill into nearby untouched ground when a hard mask made a cliff.

    Protected water, structures, and keep-outs stay exactly as surveyed. The
    track's raised formation is the other end of the grade. Only a boundary
    that introduced a new step steeper than two blocks triggers the repair;
    natural cliffs and already gentle toes are left alone.
    """
    fixed = known & (
        protected | (writable <= 0) | (np.abs(designed - natural) < DEAD_BAND)
    )
    raised = known & ~fixed & (designed > natural + DEAD_BAND)
    abrupt = ground.grid.filled(False, dtype=bool)
    for row_from, row_to, cell_from, cell_to in (
        (slice(None), slice(None), slice(1, None), slice(None, -1)),
        (slice(None), slice(None), slice(None, -1), slice(1, None)),
        (slice(1, None), slice(None, -1), slice(None), slice(None)),
        (slice(None, -1), slice(1, None), slice(None), slice(None)),
    ):
        neighbor = (row_from, cell_from)
        edge = (row_to, cell_to)
        abrupt[edge] |= (
            fixed[edge] & raised[neighbor]
            & (designed[neighbor] - natural[edge] > 2.0)
            & (np.abs(natural[neighbor] - natural[edge]) <= 2.0)
        )
    if not abrupt.any():
        return designed
    high = raised & ((ground.distance <= SHOULDER)
                     | (anchors if anchors is not None else False))
    if not high.any():
        return designed
    span = max(float(reach), FORMATION + 1.0)
    lower = nearest(abrupt, (natural,), span)
    upper = nearest(high, (designed,), span)
    near = raised & (lower.distance < span) & (upper.distance < span)
    if not near.any():
        return designed
    portion = np.divide(
        lower.distance, lower.distance + upper.distance,
        out=np.zeros_like(lower.distance),
        where=(lower.distance + upper.distance) > 0,
    )
    grade = lower[0] + portion * (upper[0] - lower[0])
    repaired = np.where(near, np.maximum(natural, np.minimum(designed, grade)), designed)
    # Rounding a continuous slope to whole blocks can turn a 2.5-block
    # shoreline rise back into a three-block step. Cap the first editable
    # column to two blocks above any adjacent fixed shore column.
    cap = ground.grid.filled(np.inf)
    for row_from, row_to, cell_from, cell_to in (
        (slice(None), slice(None), slice(1, None), slice(None, -1)),
        (slice(None), slice(None), slice(None, -1), slice(1, None)),
        (slice(1, None), slice(None, -1), slice(None), slice(None)),
        (slice(None, -1), slice(1, None), slice(None), slice(None)),
    ):
        neighbor = (row_from, cell_from)
        edge = (row_to, cell_to)
        eligible = raised[edge] & fixed[neighbor]
        eligible &= np.abs(natural[edge] - natural[neighbor]) <= 2.0
        cap[edge] = np.minimum(
            cap[edge], np.where(eligible, natural[neighbor] + 2.0, np.inf)
        )
    not_anchor = ground.grid.filled(True, dtype=bool) if anchors is None else ~anchors
    return np.where(
        near & not_anchor & ((ground.distance > SHOULDER)
                | (repaired < ground.seed_height - 2.0)),
        np.maximum(natural, np.minimum(repaired, cap)), repaired,
    )


def _headroom(
    plan: Plan, alignment: Alignment, stretches: Sequence[Stretch]
) -> Callable[[int, int], int]:
    # A column's headroom: the least of the stretches at its nearest station
    # (a shared end belongs to the incoming stretch), else the plan's.
    nearest = NearestInPlan(
        (x, z, station) for (x, z), station in alignment.station.items()
    )

    def room(x: int, z: int) -> int:
        station = nearest.nearest(x, z)
        holding = [
            stretch for stretch in stretches if stretch.start <= station <= stretch.stop
        ]
        holding = [stretch for stretch in holding if stretch.start == station] or holding
        said = [stretch.headroom for stretch in holding if stretch.headroom is not None]
        return min(said) if said else plan.headroom

    return room


def _levelled(natural: Field, known: Field) -> Field:
    if not known.any():
        return np.zeros_like(natural)
    return np.where(known, natural, float(np.mean(natural[known])))


def _ground(
    grid: Grid,
    natural: Field,
    alignment: Alignment,
    seeds: Field,
    reach: float,
    circular: bool = False,
    seed_reach: Field | None = None,
) -> vocabulary.Ground:
    footing = grid.filled(0.0)
    crest = grid.filled(0.0)
    station = grid.filled(-1.0)
    for (x, z), level in alignment.footing.items():
        if grid.holds(x, z):
            footing[grid.at(x, z)] = level
    for (x, z), level in alignment.crest.items():
        if grid.holds(x, z):
            crest[grid.at(x, z)] = level
    for (x, z), index in alignment.station.items():
        if grid.holds(x, z):
            station[grid.at(x, z)] = index
    carried = (footing, crest, station) if seed_reach is None else (footing, crest, station, seed_reach)
    found = (
        nearest_euclidean(seeds, carried, reach, radii=seed_reach)
        if circular else nearest(seeds, carried, reach)
    )
    held = found.distance < (reach if seed_reach is None else found[3])
    return vocabulary.Ground(
        grid=grid,
        natural=natural,
        distance=found.distance,
        seed_height=_eased(found[0], found.distance, held),
        crest=_eased(found[1], found.distance, held),
        station=found[2],
        reach=None if seed_reach is None else found[3],
    )


def _eased(level: Field, distance: Field, held: Field) -> Field:
    # A height carried out from the nearest track column steps wherever the
    # deck does; averaged over a circle it ramps instead. Exact on the track
    # and its shoulder, fully averaged from FORMATION out.
    beside = held & (distance > 0)
    share = kernels.disk(beside.astype(np.float64), EASING, "constant")
    mean = np.divide(
        kernels.disk(np.where(beside, level, 0.0), EASING, "constant"), share,
        out=level.copy(), where=share > 1e-6,
    )
    ramp = np.clip((distance - SHOULDER) / (FORMATION - SHOULDER), 0.0, 1.0)
    return np.where(beside, level + ramp * (mean - level), level)


def _permitted(grid: Grid, plan: Plan, available: Field) -> Field:
    rows = np.arange(grid.depth)[:, None] + grid.origin.z
    cells = np.arange(grid.width)[None, :] + grid.origin.x
    allowed = vocabulary.feathered(available.astype(np.float64), 2)
    for keepout in plan.keepouts:
        inside = (
            (cells >= keepout.least.x)
            & (cells <= keepout.most.x)
            & (rows >= keepout.least.z)
            & (rows <= keepout.most.z)
        )
        allowed = np.where(inside, 0.0, allowed)
    return allowed


def _protected(plan: Plan, surface, grid: Grid, ground: vocabulary.Ground) -> Field:
    mask = grid.filled(False, dtype=bool)
    if plan.terrain_mask.preserve_water and surface.water is not None:
        wet = surface.water > surface.height
        if plan.terrain_mask.water_buffer:
            wet = kernels.local_maximum(
                wet.astype(np.float64), plan.terrain_mask.water_buffer
            ) > 0
        # Preserve the whole wet column. The old shoulder exemption let a
        # high embankment fill the water beside the deck, then stop in one
        # block at the protected river: a vertical artificial shoreline.
        mask |= wet
    if plan.terrain_mask.preserve_structures and surface.structures is not None:
        mask |= surface.structures
    return mask


def _shaping(grid: Grid, stack: Sequence[vocabulary.Layer]) -> Field:
    # The union of the masks of layers that change the ground.
    reach = grid.filled(0.0)
    for layer in stack:
        if isinstance(layer.generator, vocabulary.Keep):
            continue
        scope = grid.filled(1.0) if layer.mask is None else layer.mask
        reach = np.maximum(reach, scope)
    return reach


def _scope(
    plan: Plan, ground: vocabulary.Ground, stretch: Stretch, allowed: Field
) -> Field:
    # Each column's share of a stretch's shaping: full over its stations,
    # running out past its ends. A stretch that keeps the ground (bridge,
    # tunnel) instead yields its ends to its neighbours' run-outs.
    if stretch.intensity is not Intensity.SHAPE:
        return ground.grid.filled(0.0)
    within = (ground.station >= stretch.start) & (ground.station <= stretch.stop)
    if stretch.cross_section is None and _keeps(plan.profile(stretch.profile)):
        reach = 1.0 - _run_out(ground.grid, ~within, stretch.transition)
    else:
        reach = _run_out(ground.grid, within, stretch.transition)
    return reach * allowed


def _keeps(profile) -> bool:
    return all(step.action is Action.KEEP for step in profile.steps)


def _circular(profile) -> bool:
    return any(step.action is Action.CIRCULAR for step in profile.steps)


def _run_out(grid: Grid, mask: Field, length: int) -> Field:
    # 1 on `mask`, falling to 0 `length` columns out from it.
    if length <= 0:
        return mask.astype(np.float64)
    away = nearest(mask, (), float(length)).distance
    return np.clip(1.0 - away / length, 0.0, 1.0)


def _layers(
    plan: Plan,
    surroundings: Surroundings,
    stretches: Sequence[Stretch],
    scopes: Sequence[Field],
    grid: Grid,
    ground: vocabulary.Ground,
    alignment: Alignment,
    natural: Field,
    water: Field,
    known: Field,
) -> list[vocabulary.Layer]:
    stack: list[vocabulary.Layer] = []
    softness = _softness(plan, surroundings)
    side_cache: dict[str, dict[str, Field]] = {}
    for stretch, scope in zip(stretches, scopes):
        if not scope.any():
            continue
        if stretch.cross_section is not None:
            section = plan.cross_section(stretch.cross_section)
            if section.name not in side_cache:
                side_cache[section.name] = side_modes(
                    grid, ground, alignment, section, natural, water, known
                )
            modes = side_cache[section.name]
            operations = (
                ("embankment", Step(Action.SLOPE_DOWN, vocabulary.Blend.RAISE)),
                ("cutting", Step(Action.SLOPE_UP, vocabulary.Blend.LOWER, base=section.cut_base)),
                ("at_grade", Step(Action.SLOPE_DOWN, vocabulary.Blend.RAISE)),
                ("at_grade", Step(Action.SLOPE_UP, vocabulary.Blend.LOWER)),
            )
            for index, (mode, step) in enumerate(operations):
                mask = scope * modes[mode]
                if not mask.any():
                    continue
                stack.append(_step_layer(
                    step, surroundings, plan, plan.reach, softness, mask,
                    stretch.order * 100 + index,
                    f"{section.name}:{mode}:{step.action.value}",
                    feather=plan.feather,
                ))
            continue
        profile = plan.profile(stretch.profile)
        for index, step in enumerate(profile.steps):
            stack.append(_step_layer(
                step, surroundings, plan, step.reach or plan.reach, softness, scope,
                stretch.order * 100 + index, f"{profile.name}:{step.action.value}",
                feather=plan.feather if step.feather is None else step.feather,
            ))
    return stack


def _step_layer(
    step: Step,
    surroundings: Surroundings,
    plan: Plan,
    reach: float,
    softness: float,
    mask: Field,
    order: int,
    name: str,
    **extra,
) -> vocabulary.Layer:
    return vocabulary.Layer(
        generator=_generator(step, surroundings, reach, plan),
        blend=step.blend,
        mask=mask,
        softness=step.softness if step.softness is not None else softness,
        reach=reach,
        order=order,
        name=name,
        **extra,
    )


def _followers(
    plan: Plan,
    surroundings: Surroundings,
    grid: Grid,
    base: Field,
    anchors: Mapping[str, Alignment],
    permitted: Field,
    bridge_stations: Field | None = None,
    bridge_core: Field | None = None,
) -> tuple[list[vocabulary.Layer], Field]:
    stack: list[vocabulary.Layer] = []
    covered = grid.filled(0.0)
    softness = _softness(plan, surroundings)
    for follows in plan.follows:
        anchor = anchors.get(follows.fitting)
        if anchor is None or follows.intensity is not Intensity.SHAPE:
            continue
        profile = plan.profile(follows.profile)
        span = follows.reach or _spread(profile, surroundings)
        adaptive = follows.reach_mode == "adaptive" and not follows.reach and all(
            step.reach == 0 for step in profile.steps
        )
        seed_reach = None
        if adaptive:
            texture = kernels.box(np.abs(base - kernels.box(base, 2)), 2)
            seed_reach = grid.filled(0.0)
            for (x, z), height in anchor.footing.items():
                if grid.holds(x, z):
                    row, cell = grid.at(x, z)
                    gap = abs(height - base[row, cell])
                    seed_reach[row, cell] = np.clip(
                        span * (1 + follows.height_response * gap
                                + follows.terrain_response * texture[row, cell]),
                        follows.reach_min, follows.reach_max,
                    )
            span = follows.reach_max
        ground = _ground(
            grid, base, anchor, grid.mark(anchor.columns), span,
            circular=_circular(profile), seed_reach=seed_reach,
        )
        allowed = permitted
        if bridge_stations is not None and bridge_core is not None:
            # A land mast's apron must not spill beneath the bridge deck.
            # A mast actually stationed on the bridge keeps its footing.
            on_bridge = np.zeros(anchor.length, dtype=bool)
            for (x, z), station in anchor.station.items():
                if grid.holds(x, z):
                    on_bridge[station] = bridge_stations[grid.at(x, z)]
            source = np.clip(ground.station.astype(np.int64), 0, anchor.length - 1)
            allowed = np.where(
                bridge_stations & bridge_core & ~on_bridge[source],
                0.0, permitted,
            )
        covered = np.maximum(
            covered, np.where(ground.distance < (
                span if ground.reach is None else ground.reach
            ), allowed, 0.0)
        )
        for index, step in enumerate(profile.steps):
            stack.append(_step_layer(
                step, surroundings, plan, step.reach or span, softness, allowed,
                follows.order * 100 + index, f"{follows.fitting}:{step.action.value}",
                feather=step.feather, ground=ground, surface_mask=follows.surface_mask,
            ))
    return stack, covered


def _formation(ground: vocabulary.Ground) -> Field:
    # 1 on the track and its shoulder, fading to 0 FORMATION out from it.
    return np.clip(
        (FORMATION - ground.distance) / (FORMATION - SHOULDER), 0.0, 1.0
    )


def _spread(profile, surroundings: Surroundings) -> float:
    widest = max((step.reach for step in profile.steps), default=0.0)
    gaussian = max(
        (3.0 * (step.width or 3.0) for step in profile.steps
         if step.action in (Action.GAUSSIAN, Action.CIRCULAR)),
        default=0.0,
    )
    return max(widest * 2.0, gaussian, surroundings.fill_ratio * 4.0)


def _generator(step: Step, surroundings: Surroundings, reach: float = 0.0, plan: Plan | None = None):
    crest = step.base == "crest"
    if step.action is Action.SLOPE_DOWN:
        response = plan.embankment if plan is not None and plan.embankment.adaptive and not step.ratio else None
        return vocabulary.Slope(
            step.ratio or surroundings.fill_ratio, rising=False, from_crest=crest,
            height_response=0.0 if response is None else response.height_response,
            terrain_response=0.0 if response is None else response.terrain_response,
            max_ratio=4.5 if response is None else response.max_ratio,
        )
    if step.action is Action.SLOPE_UP:
        chosen = None if plan is None else plan.cutting.ratio
        return vocabulary.Slope(
            step.ratio or chosen or surroundings.cut_ratio, rising=True, from_crest=crest,
        )
    if step.action is Action.LEVEL:
        return vocabulary.Level(from_crest=crest)
    if step.action is Action.GAUSSIAN:
        return vocabulary.GaussianPad(step.width or 3.0, from_crest=crest)
    if step.action is Action.CIRCULAR:
        return vocabulary.CircularPad(reach, step.width or 3.0, from_crest=crest)
    if step.action is Action.MOUND:
        return vocabulary.Mound(step.height, step.width)
    if step.action is Action.OFFSET:
        return vocabulary.Offset(step.height, from_crest=crest)
    return vocabulary.Keep()


def _softness(plan: Plan, surroundings: Surroundings) -> float:
    if plan.look is Look.ENGINEERED:
        return 0.1
    if plan.look is Look.MINIMAL:
        return 0.25
    return 0.5


def _filters(plan: Plan, surroundings: Surroundings) -> tuple:
    if plan.look is Look.ENGINEERED:
        return ()
    settling = float(np.clip(surroundings.feature_scale * 0.25, 0.5, SETTLING_LIMIT))
    options = plan.smoothing
    size = settling if options.size is None else options.size
    if options.mode == "adaptive":
        smooth = vocabulary.AdaptiveSmooth(
            size, options.strength, options.passes,
            options.min_size, options.max_size, options.terrain_response,
            kernel=options.kernel,
        )
    else:
        smooth = vocabulary.Smooth(
            size, options.strength, options.passes, kernel=options.kernel
        )
    if plan.look is Look.MINIMAL:
        return (smooth,)
    return (
        smooth,
        vocabulary.Roughen(surroundings.roughness, surroundings.feature_scale),
    )


def _clear_growth(
    plan: Plan, surface, grid: Grid, changed: Field, clearance: Clearance,
    protected: Field | None = None,
):
    # Never inside a keep-out or protected column.
    if not plan.foliage.clear or not surface.growth:
        return frozenset()

    def in_the_way(position: BlockPos) -> bool:
        if not grid.holds(position.x, position.z):
            return False
        row, cell = grid.at(position.x, position.z)
        if protected is not None and protected[row, cell]:
            return False
        return bool(changed[row, cell]) or clearance.holds(row, cell, position.y)

    cleared = growth_clearing(surface.growth, in_the_way, surface.trees)
    return frozenset(
        position for position in cleared
        if not any(keepout.holds(position.x, position.z) for keepout in plan.keepouts)
        and not (protected is not None and grid.holds(position.x, position.z)
                 and protected[grid.at(position.x, position.z)])
    )
