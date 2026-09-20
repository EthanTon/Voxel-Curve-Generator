from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..Core.block.palette import BlockPalette
from ..Core.coordinates import BlockPos, ColumnPos
from . import clearance as clearing
from . import layers as vocabulary
from .clearance import Clearance
from .distance import nearest
from .grid import Field, Grid
from .materialize import Earthworks, build
from .plan import Action, Intensity, Look, Plan, Step, Stretch
from .surroundings import Surroundings, measure
from .terrain import ABSENT, Terrain, survey
from .vegetation import clearing as growth_clearing


SAMPLE_OFFSET = 0.5
SETTLING_LIMIT = 2.0
DEAD_BAND = 0.5


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
) -> Terraforming:
    anchors = {
        name: Alignment.of(blocks)
        for name, blocks in (fittings or {}).items()
        if blocks
    }
    grid = Grid.around(
        [*alignment.columns, *(each for held in anchors.values() for each in held.columns)],
        plan.margin,
    )
    clearance = clearance or clearing.above(
        grid, alignment.crest_blocks(), plan.headroom
    )
    surface = survey(terrain, grid, clearance.holds, plan.foliage.clear)
    known = surface.available & (surface.height > ABSENT)
    natural = surface.height.astype(np.float64)
    base = _levelled(natural, known)
    seeds = grid.mark(alignment.columns)

    ground = _ground(grid, base, alignment, seeds, plan.reach)
    beyond = known & (ground.distance >= plan.reach * SAMPLE_OFFSET)
    surroundings = measure(surface.height, beyond)

    permitted = _permitted(grid, plan, known)
    allowed = np.where(seeds, 0.0, permitted)
    stretches = plan.stretches(locate or alignment.locate, alignment.length)
    stack = _layers(plan, surroundings, ground, allowed, stretches)
    followers, followed = _followers(
        plan, surroundings, grid, base, anchors, permitted
    )
    stack += followers
    writable = np.maximum(allowed, followed)

    designed = vocabulary.shape(ground, stack)
    designed = clearance.limit(designed)
    designed = np.where((writable > 0) & known, designed, natural)
    designed = np.where(np.abs(designed - natural) < DEAD_BAND, natural, designed)

    cleared = _clear_growth(plan, surface.growth, grid, writable)
    earthworks = build(
        grid, natural, designed, plan.strata, palette or BlockPalette(), surface.state
    )
    carved = earthworks.cut | surface.intruding | cleared

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
                    int((_scope(ground, stretch, allowed) > 0).sum()),
                )
                for stretch in stretches
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
    )


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
    found = nearest(seeds, (footing, crest, station), reach)
    return vocabulary.Ground(
        grid=grid,
        natural=natural,
        distance=found.distance,
        seed_height=found[0],
        crest=found[1],
        station=found[2],
    )


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


def _scope(ground: vocabulary.Ground, stretch: Stretch, allowed: Field) -> Field:
    if stretch.intensity is not Intensity.SHAPE:
        return np.zeros(ground.grid.shape)
    within = (ground.station >= stretch.start) & (ground.station <= stretch.stop)
    return vocabulary.feathered(within.astype(np.float64), stretch.transition) * allowed


def _layers(
    plan: Plan,
    surroundings: Surroundings,
    ground: vocabulary.Ground,
    allowed: Field,
    stretches: Sequence[Stretch],
) -> list[vocabulary.Layer]:
    stack: list[vocabulary.Layer] = []
    softness = _softness(plan, surroundings)
    filters = _filters(plan, surroundings)
    for stretch in stretches:
        scope = _scope(ground, stretch, allowed)
        if not scope.any():
            continue
        profile = plan.profile(stretch.profile)
        for index, step in enumerate(profile.steps):
            stack.append(
                vocabulary.Layer(
                    generator=_generator(step, surroundings),
                    blend=step.blend,
                    mask=scope,
                    softness=step.softness if step.softness is not None else softness,
                    reach=step.reach or plan.reach,
                    filters=filters if index == len(profile.steps) - 1 else (),
                    order=stretch.order * 100 + index,
                    name=f"{profile.name}:{step.action.value}",
                )
            )
    return stack


def _followers(
    plan: Plan,
    surroundings: Surroundings,
    grid: Grid,
    base: Field,
    anchors: Mapping[str, Alignment],
    permitted: Field,
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
        ground = _ground(grid, base, anchor, grid.mark(anchor.columns), span)
        covered = np.maximum(
            covered, np.where(ground.distance < span, permitted, 0.0)
        )
        for index, step in enumerate(profile.steps):
            stack.append(
                vocabulary.Layer(
                    generator=_generator(step, surroundings),
                    blend=step.blend,
                    mask=permitted,
                    softness=step.softness if step.softness is not None else softness,
                    reach=step.reach or span,
                    feather=step.feather,
                    order=follows.order * 100 + index,
                    name=f"{follows.fitting}:{step.action.value}",
                    ground=ground,
                )
            )
    return stack, covered


def _spread(profile, surroundings: Surroundings) -> float:
    widest = max((step.reach for step in profile.steps), default=0.0)
    return max(widest * 2.0, surroundings.fill_ratio * 4.0)


def _generator(step: Step, surroundings: Surroundings):
    if step.action is Action.SLOPE_DOWN:
        return vocabulary.Slope(step.ratio or surroundings.fill_ratio, rising=False)
    if step.action is Action.SLOPE_UP:
        return vocabulary.Slope(step.ratio or surroundings.cut_ratio, rising=True)
    if step.action is Action.LEVEL:
        return vocabulary.Level()
    if step.action is Action.MOUND:
        return vocabulary.Mound(step.height, step.width)
    if step.action is Action.OFFSET:
        return vocabulary.Offset(step.height)
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
    if plan.look is Look.MINIMAL:
        return (vocabulary.Smooth(settling),)
    return (
        vocabulary.Smooth(settling),
        vocabulary.Roughen(surroundings.roughness, surroundings.feature_scale),
    )


def _clear_growth(plan: Plan, growth, grid: Grid, writable: Field):
    if not plan.foliage.clear or not growth:
        return frozenset()
    rows, cells = np.nonzero(writable > 0)
    return growth_clearing(growth, grid, set(zip(rows.tolist(), cells.tolist())))
