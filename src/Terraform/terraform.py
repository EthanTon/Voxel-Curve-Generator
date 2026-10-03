from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import binary_dilation

from ..Core.block.palette import BlockPalette
from ..Core.coordinates import BlockPos, ColumnPos
from ..Core.plan_index import NearestInPlan
from . import clearance as clearing
from . import layers as vocabulary
from . import kernels
from . import sculpt
from . import slopes
from . import smoothing
from .cross_section import side_modes
from .clearance import Clearance
from .distance import nearest, nearest_euclidean
from .grid import Field, Grid
from .materialize import Earthworks, build, finish_surfaces
from .plan import Action, Blend, Intensity, Look, Plan, Step, Stretch
from .subsurface import Layering, Subsurface, surface_blocks
from .surroundings import Surroundings, measure
from .classify import BlockNames, classify, ground_blocks
from .terrain import ABSENT, SOLID, Terrain, survey
from .surface_mask import SurfaceReference
from .noise import fbm
from .vegetation import clearing as growth_clearing


SAMPLE_OFFSET = 0.5
SETTLING_LIMIT = 2.0
DEAD_BAND = 0.5
# The look (smoothing, grain) is held off the track and SHOULDER beside it,
# fading in to full strength FORMATION out.
SHOULDER = 2.0
FORMATION = 5.0
# Passes of the majority rule over the one-block fringe of the earthworks,
# and over every level's edge.
FRINGE_PASSES = 2
LEVEL_PASSES = 2
# How deep a hollow under the track's footing is filled, as deep as a
# ravine; a cavern deeper than this is left open.
UNDERPIN = 48
DESPECKLE_PASSES = 2
# The radius over which heights carried out from the track are averaged, and
# the share of a stretch's scope over which its change fades out.
EASING = 6.0
# Growth is cleared this far beside a bridge's deck, from its underside up
# to this far over its top, so no crown reaches into the deck or its wires.
DECK_EDGE = 1.5
DECK_ROOM = 9
# Slope steepness wanders over this wavelength, its contours warped by up to
# WARP blocks; natural ground's own detail shows through the works by up to
# DETAIL blocks.
WAVELENGTH = 32.0
WARP = 8.0
DETAIL = 1.5
# A portal's hill may rise this many blocks a column: a portal is a track
# structure, so its face need not keep to max_drop, and the hill it cuts
# back stays a notch rather than a crater.
PORTAL_RISE = 2
FADE = 0.25
# How far below a running space a bridge column's structure may stop and
# still carry it.
DECK_REACH = 4.0


@dataclass(frozen=True, slots=True)
class Alignment:
    footing: dict[tuple[int, int], int]
    crest: dict[tuple[int, int], int]
    station: dict[tuple[int, int], int]
    # The height the ground meets, where a section sets one; else the footing.
    formation: dict[tuple[int, int], int] = field(default_factory=dict)
    # Unsupported rails meet their support level, including above a low
    # decorative base or in a profile measured from retaining wall crests.
    rail_support: dict[tuple[int, int], int] = field(default_factory=dict)
    # The path's centreline column at each station, where the path is known.
    centres: dict[int, tuple[int, int]] = field(default_factory=dict)

    @classmethod
    def of(
        cls,
        blocks: Iterable[BlockPos],
        stations: Sequence[int] | None = None,
        formation: Callable[[int, int, int], int | None] | None = None,
        covered: Sequence[bool] | None = None,
        decoration: Sequence[bool] | None = None,
    ) -> Alignment:
        # Blocks the earthworks may cover (`covered`) mark out the track but
        # set its footing and crest only where nothing else stands. The
        # crest is the top of a column's body: `decoration` on it, such as
        # a railing on a wall, is left out wherever the column has a body.
        footing: dict[tuple[int, int], int] = {}
        crest: dict[tuple[int, int], int] = {}
        body: dict[tuple[int, int], int] = {}
        station: dict[tuple[int, int], int] = {}
        firm: set[tuple[int, int]] = set()
        for index, position in enumerate(blocks):
            key = (position.x, position.z)
            station.setdefault(key, index if stations is None else stations[index])
            if covered is not None and covered[index]:
                if key in firm:
                    continue
            elif key not in firm:
                firm.add(key)
                footing.pop(key, None)
                crest.pop(key, None)
                body.pop(key, None)
            footing[key] = min(footing.get(key, position.y), position.y)
            crest[key] = max(crest.get(key, position.y), position.y)
            if decoration is not None and not decoration[index]:
                body[key] = max(body.get(key, position.y), position.y)
        crest.update((key, top) for key, top in body.items() if key in crest)
        formed: dict[tuple[int, int], int] = {}
        if formation is not None:
            for (x, z), at in station.items():
                level = formation(x, z, at)
                if level is not None:
                    formed[(x, z)] = level
        return cls(footing, crest, station, formed)

    @property
    def meets(self) -> dict[tuple[int, int], int]:
        # The footing, raised to the formation where a section sets one.
        return {**self.footing, **self.formation}

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
    # Drops beyond the configured maximum outside water, structures, rock
    # and untouched natural faces.
    steep: int = 0
    smoothing: tuple[tuple[str, int], ...] = ()
    face_carved: int = 0
    face_filled: int = 0
    tunnel_side_columns: int = 0

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
        shown.append(f"new surface drops beyond slope limit: {self.steep}")
        shown.append(f"face repair carved {self.face_carved}, filled {self.face_filled}")
        if self.tunnel_side_columns:
            shown.append(f"tunnel side contacts {self.tunnel_side_columns} columns")
        if self.smoothing:
            shown.append("smoothing " + ", ".join(f"{name} {count}" for name, count in self.smoothing))
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
    # A continuous terrain corridor for exports, including unchanged ground
    # under bridges and between the columns the earthworks change.
    coverage: Field | None = None
    terrain_top: Field | None = None
    terrain_chunks: Mapping | None = None
    water: Field | None = None
    # Material-direction decisions per terrain column, for inspection.
    smoothing_modes: Field | None = None
    # Native water and protected blocks give way inside this track space.
    track_clearance: Clearance | None = None
    # The fixed construction enclosing that space must also stay dry. Roof
    # stairs and wires can lie just outside the empty clearance interval.
    dry_blocks: frozenset[BlockPos] = frozenset()

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
    running: Mapping[tuple[int, int], int] | None = None,
) -> Terraforming:
    # A custom terrain's own ground blocks count as ground throughout.
    with ground_blocks(plan.ground.patterns):
        return _terraform(plan, terrain, alignment, palette, clearance, locate, fittings,
                          envelope, section_spans, running)


def _terraform(
    plan: Plan,
    terrain: Terrain,
    alignment: Alignment,
    palette: BlockPalette | None,
    clearance: Clearance | None,
    locate: Callable[[object], int] | None,
    fittings: Mapping[str, Sequence[BlockPos]] | None,
    envelope: Mapping[tuple[int, int], Sequence[int]] | None,
    section_spans: Mapping[str, Sequence[tuple[int, int]]] | None,
    running: Mapping[tuple[int, int], int] | None,
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
        max(plan.margin, plan.embankment.max_reach),
    )
    stretches = _continuous_keeps(plan, plan.stretches(
        locate or alignment.locate, alignment.length, section_spans
    ))
    if envelope is not None:
        # A tunnel envelope supplies its real roof if the alignment only
        # contains its bed. Open sections still meet their own body, not
        # the wires or other fittings included in a whole-build envelope.
        covered = np.zeros(alignment.length, dtype=bool)
        for stretch in stretches:
            if stretch.profile == "tunnel":
                covered[max(0, stretch.start):min(alignment.length, stretch.stop + 1)] = True
        alignment = replace(alignment, crest={
            column: max(top, max(envelope.get(column, (top,)), default=top))
            if covered[alignment.station[column]] else top
            for column, top in alignment.crest.items()
        })
    footing = {**alignment.footing,
               **{column: level + 1 for column, level in alignment.rail_support.items()}}
    track_clearance = None
    dry_blocks = frozenset()
    if plan.clear_track and envelope is not None:
        dry_blocks = frozenset(
            BlockPos(x, y, z)
            for (x, z), heights in envelope.items() for y in heights
        )
    if plan.clear_track and envelope is not None and running:
        track_clearance = clearing.enclosing(
            grid, {column: heights for column, heights in envelope.items() if column in running},
            _headroom(plan, alignment, stretches), footing, running, plan.headroom,
        )
    if clearance is None and envelope is not None:
        # A low decorative base must not turn the missing sleeper below a
        # protruding rail into clearance that forbids its support fill.
        clearance = clearing.enclosing(
            grid, envelope, _headroom(plan, alignment, stretches), footing,
            running, plan.headroom,
        )
    clearance = clearance or clearing.above(
        grid, alignment.crest_blocks(), plan.headroom
    )
    if track_clearance is not None:
        clearance = clearance.merged(track_clearance)
    include_depths = getattr(terrain, "include_depths", None)
    if include_depths is not None:
        read_to = {column: level - UNDERPIN for column, level in alignment.footing.items()}
        for row, cell in zip(*np.nonzero(clearance.floor <= clearance.ceiling)):
            column = tuple(grid.column(row, cell))
            read_to[column] = min(read_to.get(column, clearing.OPEN),
                                  int(clearance.floor[row, cell]))
        include_depths(read_to)
    capture = getattr(terrain, "capture", None)
    if capture is not None:
        capture()
    footprint = grid.mark([
        *alignment.columns, *(each for held in anchors.values() for each in held.columns),
    ])
    # A winding route can enclose most of a world in its bounding box. Read
    # only its corridor and a halo for smoothing, not every forest inside it.
    sample_area = None
    if grid.width * grid.depth >= 262144:
        radius = max(plan.margin, plan.embankment.max_reach) + 12
        sample_area = nearest(footprint, (), radius + 1).distance <= radius
    surface = survey(terrain, grid, clearance.holds, plan.foliage.clear or plan.clear_track, sample_area)
    known = surface.available & (surface.height > ABSENT)
    natural = surface.height.astype(np.float64)
    # What the land is made of under its surface, and how it lies round
    # each column: fills are built of it, and its rock stands as faces.
    below = Subsurface(surface.layers, surface.layer_states, natural, plan.cutting.rock_rule)
    layering = below.layering(known, plan.embankment.layering_tile, plan.embankment.layering_reach)
    # Besides the blocks named as tops, whatever this land only ever has on
    # top (a custom grass, a snow cover) is its top too.
    top_blocks = BlockNames(plan.embankment.tops.patterns + surface_blocks(below, known))
    base = _levelled(natural, known)
    seeds = grid.mark(alignment.columns)
    steepness = _steepness(plan, grid)

    circular_main = any(_circular(plan.profile(stretch.profile)) for stretch in stretches)
    # Distances run past the reach to the shoulder and the brush beyond it,
    # so a shoulder wider than the slopes' reach is still level, and the
    # brush can tell the shoulder from the ground around it.
    ground = _ground(grid, base, alignment, seeds, _span(plan), circular=circular_main)
    natural_bridges = grid.filled(False, dtype=bool)
    if plan.bridge.natural_only:
        for stretch in stretches:
            if stretch.intensity is not Intensity.SHAPE:
                continue
            within = (ground.station >= stretch.start) & (ground.station <= stretch.stop)
            if stretch.profile == "bridge" and stretch.cross_section is None:
                natural_bridges |= within
            elif stretch.cross_section is not None:
                modes = side_modes(grid, ground, alignment, plan.cross_section(stretch.cross_section),
                                   natural, surface.water, known)
                natural_bridges |= within & modes["bridge"]
    shaping_ground = ground
    shaping_stations = np.zeros(alignment.length, dtype=bool)
    for stretch in stretches:
        if (stretch.intensity is Intensity.SHAPE
                and (stretch.cross_section is not None
                     or not _keeps(plan.profile(stretch.profile)))):
            shaping_stations[max(stretch.start, 0):min(stretch.stop + 1, alignment.length)] = True
    if plan.bleed:
        # A bank continues from its own formation, not from the underside
        # of whichever bridge or tunnel section happens to be nearest.
        sources = grid.mark(
            ColumnPos(x, z) for (x, z), station in alignment.station.items()
            if shaping_stations[station]
        )
        if sources.any():
            shaping_ground = _ground(
                grid, base, alignment, sources, _span(plan), circular=circular_main,
            )
    protected, shore = _protected(plan, surface, grid)
    # A shaped section standing on a dry shore still gets ground under it.
    protected &= ~(shore & grid.mark(
        ColumnPos(x, z) for (x, z), station in alignment.station.items()
        if shaping_stations[station]
    ))
    clearance_protected = protected.copy()
    clearance_shore = shore.copy()
    fitting_kept = grid.mark(
        column for name in plan.preserve_fittings if name in anchors
        for column in anchors[name].columns
    )
    if fitting_kept.any():
        fitting_kept = nearest(fitting_kept, (), plan.fitting_buffer + 1.0).distance <= plan.fitting_buffer
        # A nearby pier must not leave a hole beneath the departing bank's
        # deck or its rails. Its buffer protects the surrounding ground.
        fitting_kept &= ~grid.mark(
            ColumnPos(x, z) for (x, z), station in alignment.station.items()
            if shaping_stations[station]
        )
        protected |= fitting_kept
        shore &= ~fitting_kept
    beyond = known & (ground.distance >= plan.reach * SAMPLE_OFFSET)
    surroundings = measure(surface.height, beyond)

    permitted_area = _permitted(grid, plan, known)
    # Margin is a hard limit round the alignment and fitting feet, not just
    # padding round their bounding box. The brush and repairs obey it too.
    footprint = grid.mark([
        *alignment.columns, *(each for held in anchors.values() for each in held.columns),
    ])
    within_margin = nearest(footprint, (), plan.margin + 1.0).distance <= plan.margin
    allowed = np.where(within_margin, permitted_area, 0.0)
    permitted = np.where(protected, 0.0, allowed)
    # No earthworks across water under a bridge's interior.
    water = surface.water if surface.water is not None else grid.filled(ABSENT)
    wet_bridge = grid.filled(False, dtype=bool)
    for stretch in stretches:
        if stretch.profile == "bridge":
            wet_bridge |= (
                (ground.station > stretch.start)
                & (ground.station < stretch.stop)
                & (water > surface.height) & (not plan.bleed)
            )
    scopes = [_scope(plan, shaping_ground, stretch, permitted, ground.station)
              for stretch in stretches]
    if wet_bridge.any():
        scopes = [np.where(wet_bridge, 0.0, scope) if stretch.profile != "bridge" else scope
                  for stretch, scope in zip(stretches, scopes)]
    # A bank may blend around a tunnel, but its buried lining keeps the
    # mountain above its roof. Clearing the bore is handled separately.
    tunnel_stations = np.zeros(alignment.length, dtype=bool)
    for stretch in stretches:
        if stretch.profile == "tunnel":
            tunnel_stations[max(0, stretch.start):min(alignment.length, stretch.stop + 1)] = True
    tunnel_domain = tunnel_stations[np.clip(ground.station.astype(np.int64), 0,
                                           alignment.length - 1)]
    # Hold the hillside beside a buried tunnel as well as its roof. An
    # approaching bank must not strip a flat shelf into that mountain.
    buried = tunnel_domain & (natural > ground.crest) & known
    # Every shaped approach can meet a portal, including a bridge. The
    # geometry supplies the meeting heights; smoothing chooses locally
    # whether the land needs growing, stabilizing, or melting.
    approach_stations = np.zeros(alignment.length, dtype=bool)
    for stretch in stretches:
        if stretch.intensity is Intensity.SHAPE:
            approach_stations[max(stretch.start, 0):min(stretch.stop + 1, alignment.length)] = True
    smoothing_protected = protected.copy()
    protected |= buried
    permitted = np.where(buried, 0.0, permitted)
    scopes = [np.where(buried, 0.0, scope) if scope.any() else scope for scope in scopes]
    stack = _layers(
        plan, surroundings, stretches, scopes, grid, shaping_ground, alignment,
        base, water, known,
    )
    bridge_stations = grid.filled(False, dtype=bool)
    for stretch in stretches:
        if stretch.profile == "bridge":
            bridge_stations |= (
                (ground.station > stretch.start) & (ground.station < stretch.stop)
            )
    # Adjacent section labels should not interrupt a bridge bank's taper.
    # Retaining walls and buried tunnel ground remain fixed at the handover.
    retained_stations = tunnel_stations.copy()
    for stretch in stretches:
        if _retains(plan, stretch):
            retained_stations[max(0, stretch.start):min(alignment.length, stretch.stop + 1)] = True
    retained_domain = retained_stations[np.clip(ground.station.astype(np.int64), 0,
                                                alignment.length - 1)]
    bridge_vicinity = binary_dilation(
        bridge_stations, iterations=max(1, min(12, int(plan.reach)))
    ) & ~retained_domain
    followers, followed = _followers(
        plan, surroundings, grid, base, anchors, permitted,
        None if plan.bleed else bridge_stations, ground.distance <= FORMATION,
    )
    # Only shaping stretches and footings change the ground.
    writable = np.maximum(_shaping(grid, stack), followed)

    # Stretches, then the look over all of them, then the footings on top.
    # The look and the brush both work within the brush's area: the ground
    # the works touch, a margin, and a falloff.
    footings = grid.mark(each for held in anchors.values() for each in held.columns)
    shaped = _crossfaded(
        plan, surroundings, stretches, scopes, grid, shaping_ground, alignment,
        base, water, known,
    )
    smoothing_context = smoothing.context(plan, ground, alignment, stretches)
    smoothing_modes = smoothing_context.modes(shaped, natural, plan.smoothing)
    # Natural drops the works leave alone stay as they are: the look, the
    # brush and the tidying pass them by.
    area = _brush_area(
        plan, ground, stretches, scopes, shaped,
        known & ~protected & ~footings & (permitted > 0)
        & ~(smoothing.preserved_faces(shaped, natural, known) if plan.smoothing.preserve_cliffs else False),
    )
    designed = vocabulary.settle(
        ground, shaped, _filters(plan, surroundings), _formation(ground, plan.shoulder)
    )
    designed = smoothing.constrain(shaped, designed, smoothing_modes, plan.smoothing.stable_change)
    designed = shaped + area * (designed - shaped)
    designed = vocabulary.shape(ground, followers, designed)
    fade = np.clip(writable / FADE, 0.0, 1.0)
    designed = np.where(
        (writable > 0) & known,
        clearance.limit_surface(natural, natural + fade * (designed - natural)),
        natural,
    )
    designed = np.where(protected, natural, designed)
    designed = np.where(np.abs(designed - natural) < DEAD_BAND, natural, designed)
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
    # The banks and cuttings laid on their slopes, as far out as their
    # height needs. The brush rounds their toes into the ground, and the
    # slopes hold again after it, so their tops stay crisp.
    stone = below.rock_tops() if plan.cutting.rock else grid.filled(-np.inf)
    cliff_rock = (_natural_cliffs(natural, known)
                  if plan.embankment.shape == "natural" else known)
    # Native rock belongs to the land, including the middle of an assigned
    # section. Restricting its grade to transition brushes makes the same
    # cliff change policy as it crosses a section boundary.
    rock_contact = cliff_rock & (stone > ground.crest + PORTAL_RISE)
    # Buried stone does not make a rolling hillside a cliff. Gentle land
    # sweeps to the bed/retaining crest with the normal grade throughout.
    contact_drop = np.where(rock_contact,
                            np.maximum(plan.slopes.max_drop, plan.cutting.rock_drop / steepness),
                            plan.slopes.max_drop)
    limits = _slope_limits(
        plan, grid, alignment, stretches, below, natural, known, water,
        # Ground past the margin is held, so a slope must taper toward it.
        # It is not an exempt obstacle, which would leave a cliff at the rim.
        ~known | (protected & ~shore if plan.terrain_mask.slope_to_water else protected)
        | (permitted_area <= 0) | footings,
        clearance, shore, (writable <= 0) if plan.bleed else None,
        contact_drop,
    )
    # Sculpt new rock faces into bounded ledges instead of forcing the
    # soil's one-block grade across the mountain or pinning a vertical fin.
    portal_drop = (contact_drop if plan.embankment.shape == "natural"
                   else np.maximum(PORTAL_RISE, contact_drop))
    # The hill over a buried roof only needs a local trim. Sides use the
    # rock grade; the bore is handled separately by three-dimensional air.
    rock_floor = np.where(rock_contact & tunnel_domain & (ground.distance <= 0),
                          stone - plan.cutting.rock_drop, -np.inf)
    designed = limits.shape(designed, natural, followed > 0)
    if plan.slopes.variation > 0:
        designed = limits.clamp(_natural_detail(designed, natural, known, ground.distance))
    if plan.terrain_mask.slope_to_water:
        limits = _water_banks(plan, limits, grid, alignment, natural, designed, water, allowed)
        designed = limits.clamp(designed)
    kept = ((smoothing.preserved_faces(designed, natural, known) if plan.smoothing.preserve_cliffs else False)
            | limits.held | limits.exempt)
    area = _brush_area(
        plan, ground, stretches, scopes, designed,
        known & ~protected & ~footings & (permitted > 0) & ~kept,
    )
    # A cutting's crest on the shoulder stays up to its walls. Nothing else
    # is clamped: a bound taken from the unbrushed ground puts its grain back.
    unbrushed = designed
    designed = _brushed(plan, designed, area, smoothing_modes, known)
    crest = (ground.distance <= max(SHOULDER, plan.shoulder)) & (unbrushed < natural)
    designed = np.where(crest, np.maximum(designed, unbrushed), designed)
    designed = limits.clamp(designed)
    brushed = area > 0.0
    designed = np.where(known, clearance.limit_surface(natural, designed), designed)
    designed = np.where(np.abs(designed - natural) < DEAD_BAND, natural, designed)
    # The fringe is tidied where the brush worked (or, with no brush, where
    # the works can edit), and lone pits and posts go wherever the works
    # moved the ground; the brushed shoulder is tidied up to the footprint.
    beyond = ground.distance > (0.0 if plan.brush.passes > 0 else max(SHOULDER, plan.shoulder))
    fringe = (brushed if plan.brush.passes > 0 else editable) & beyond & ~kept
    designed = _tidy_fringe(designed, natural, fringe)
    moved = np.abs(designed - natural) >= DEAD_BAND
    lone = ((editable & moved) | brushed) & beyond & ~kept
    designed = _levelled_lone_columns(designed, lone)
    designed = _tidied_levels(designed, lone)
    levels = np.where(known, slopes.level(designed), designed)
    enforced, steep = limits.enforce(levels, natural)
    designed = np.where(enforced != levels, enforced, designed)
    designed = _ease_preserved_bridge_ground(
        designed, natural, known, water, protected, footings,
        bridge_vicinity, ground.distance, plan.slopes.max_drop,
        min(12, int(plan.reach)),
    )
    # The slope solver exempts protected ground and some section boundaries.
    # A changed bank can still end against that fixed ground, especially at
    # the outer edge of a long bridge spill. Ease only the existing edits
    # back toward the saved terrain; never extend the footprint to repair it.
    edge_movable = (
        known & ~protected & ~footings & (permitted_area > 0)
        & ((ground.distance > max(3.0, plan.shoulder + 2.0))
           | (bridge_vicinity & (ground.distance > 0)))
        & ~(clearance.ceiling >= natural)
    )
    designed = _taper_edit_edges(
        designed, natural, edge_movable, plan.slopes.max_drop,
        plan.slopes.repair_passes,
    )
    # Apply this after slope repair and neighboring run-outs: neither can
    # lift a bridge's ground toward the track or erase a valley beneath it.
    bridge_ground = natural
    if natural_bridges.any():
        smoothed = _bridge_ground(plan, natural, known)
        local = (natural_bridges & known & ~protected
                 & (permitted_area > 0) & within_margin
                 & (ground.distance <= plan.bridge.reach))
        smoothed = clearance.limit_surface(natural, smoothed)
        kept = np.where(local, smoothed, natural)
        # Under a bridge's own footprint the ground is never raised, and
        # what the saved land holds over its deck is taken down under it.
        footprint = known & ~protected & (permitted_area > 0) & within_margin & (ground.distance <= 0)
        kept = np.where(footprint, np.minimum(np.minimum(kept, natural), ground.seed_height - 1), kept)
        designed = np.where(natural_bridges, kept, designed)
        bridge_ground = np.where(natural_bridges, kept, natural)

    # A saved cliff is a legitimate terrain contact. Only faces that the
    # works already altered need grading; do not turn intact natural faces
    # beside a portal into the same gentle mound used on rolling terrain.
    native_faces = (smoothing.preserved_faces(designed, natural, known) & (ground.distance > 0)
                    if plan.smoothing.preserve_cliffs else grid.filled(False, dtype=bool))
    portal_corridor = (known & within_margin & ~smoothing_protected & ~native_faces
                       & (permitted_area > 0) & ~footings
                       & (smoothing_context.transition_weight > 0))
    if (approach_stations & ~tunnel_stations).any() and tunnel_stations.any():
        # Grading can expose a neighbour without changing its own column.
        # Re-evaluate that neighbour instead of pinning a new rock fin.
        for _ in range(max(2, plan.smoothing.face_iterations + 1)):
            before = designed
            roof_contacts = (known & within_margin & (permitted_area > 0)
                             & (smoothing_context.transition_weight > 0)
                             & (natural > ground.crest + 1)
                             & ~(smoothing_protected & ~fitting_kept) & ~native_faces)
            designed = _portal_grading(plan, ground, approach_stations & ~tunnel_stations,
                                       tunnel_domain & (portal_corridor | roof_contacts), natural, designed,
                                       steepness, portal_corridor, rock_floor, portal_drop)
            native_faces = (smoothing.preserved_faces(designed, natural, known) & (ground.distance > 0)
                            if plan.smoothing.preserve_cliffs else grid.filled(False, dtype=bool))
            portal_corridor = (known & within_margin & ~smoothing_protected & ~native_faces
                               & (permitted_area > 0) & ~footings
                               & (smoothing_context.transition_weight > 0))
            if np.array_equal(before, designed):
                break
        roof_floor = np.where(tunnel_domain & (ground.distance <= 0) & (natural > ground.crest),
                              ground.crest + 1, -np.inf)
        contact_floor = np.maximum(rock_floor, np.where(retained_domain & ~tunnel_domain, ground.crest, roof_floor))
        designed = smoothing.retreat(
            designed, natural, known, portal_corridor & (ground.distance > 0),
            contact_floor, portal_drop, plan.slopes.repair_passes,
        )
        # A fitting's column protects its foot, not a tall soil remnant over
        # its head. Pier buffers likewise cannot pin a new excavation face
        # above the deck. Hard keep-outs, water and native cliffs still hold.
        overburden = known & (natural > ground.crest + 1)
        repairable = (known & within_margin & (permitted_area > 0)
                      & (smoothing_context.transition_weight > 0) & overburden
                      & ~(smoothing_protected & ~fitting_kept) & ~native_faces)
        contact_floor = np.maximum(rock_floor, np.where(ground.distance <= 0, ground.crest + 1, -np.inf))
        designed = smoothing.retreat(
            designed, natural, known, repairable, contact_floor,
            portal_drop, plan.slopes.repair_passes,
        )
    smoothing_modes = smoothing_context.modes(designed, natural, plan.smoothing)
    suspended = natural_bridges & (natural < ground.seed_height)
    for stretch in stretches:
        if stretch.cross_section is None and _keeps(plan.profile(stretch.profile)):
            suspended |= ((ground.station >= stretch.start) & (ground.station <= stretch.stop)
                          & (natural < ground.seed_height - plan.smoothing.stable_change)
                          & (np.abs(designed - natural) < DEAD_BAND))
    transition_protected = (smoothing_protected | footings | ~within_margin | native_faces
                            | (permitted_area <= 0) | suspended)
    meeting_heights = designed.copy()
    for stretch in stretches:
        within = (ground.station >= stretch.start) & (ground.station <= stretch.stop)
        walls = grid.filled(False, dtype=bool)
        if _retains(plan, stretch):
            walls = within
        elif stretch.cross_section is not None:
            section = plan.cross_section(stretch.cross_section)
            if section.cut_base == "crest":
                walls = within & side_modes(grid, ground, alignment, section, natural,
                                             water, known)["cutting"]
        meeting_heights = np.where(walls, np.maximum(meeting_heights, ground.crest), meeting_heights)
    designed = smoothing.meet(
        designed, natural, ground.distance <= 0,
        known & ~transition_protected & ~tunnel_domain
        & ((smoothing_context.transition_weight > 0) | ((writable > 0) & (followed <= 0))),
        meeting_heights, plan.slopes.max_drop, plan.slopes.cone_passes,
        (replace(plan.smoothing, contact_blend=0) if plan.look is Look.ENGINEERED
         or plan.embankment.shape == "linear" else plan.smoothing), min(plan.reach, plan.margin),
        smoothing_context.overrides,
        rock_floor,
        contact_drop,
    )
    smoothing_modes = smoothing_context.modes(designed, natural, plan.smoothing)
    designed = _smooth_transitions(
        plan, ground, designed, natural, known, smoothing_context,
        smoothing_modes, transition_protected,
        natural_bridges, tunnel_domain, clearance,
    )
    # The broad contact stage can extend a bank beyond the old brush. Check
    # the same physical slope bounds again after that stage, not before it.
    final_levels = np.where(known, slopes.level(designed), designed)
    final_enforced, _ = limits.enforce(final_levels, natural)
    # Bounds built for the approaching bank must not resurrect fill after
    # the bridge policy has deliberately restored the valley below it.
    final_enforced = np.where(natural_bridges, np.minimum(final_enforced, designed), final_enforced)
    designed = np.where(final_enforced != final_levels, final_enforced, designed)
    # Re-check exposure after the final surface constraints. A neighbour
    # graded this round can reveal the next column of a ridge; stale cliff
    # masks would freeze that column in place again.
    overburden = known & (natural > ground.crest + 1)
    roof_cover = np.where(ground.distance <= 0, ground.crest + 1, -np.inf)
    contact_floor = np.maximum(rock_floor, roof_cover)
    if plan.look is not Look.ENGINEERED and plan.embankment.shape == "natural":
        contact_floor = smoothing.contact_floor(designed, natural, known, rock_floor, roof_cover)
        if limits.least is not None:
            contact_floor = np.maximum(contact_floor, limits.least)
    for _ in range(plan.smoothing.face_iterations + 1):
        native_faces = (smoothing.preserved_faces(designed, natural, known)
                        if plan.smoothing.preserve_cliffs else grid.filled(False, dtype=bool))
        repairable = (known & within_margin & (permitted_area > 0)
                      & (smoothing_context.transition_weight > 0) & overburden
                      & ~(smoothing_protected & ~fitting_kept) & ~native_faces)
        repaired = smoothing.retreat(designed, natural, known, repairable, contact_floor,
                                     portal_drop, plan.slopes.repair_passes)
        if np.array_equal(repaired, designed):
            break
        designed = repaired
    # Where the bridge floor is fixed, taper a changed bank toward it rather
    # than pushing that floor back up to an outdated bank envelope.
    designed = _taper_edit_edges(
        designed, natural, known & within_margin & ~smoothing_protected
        & ~footings & ~natural_bridges & ~tunnel_domain & ~retained_domain
        & (permitted_area > 0) & (ground.distance > max(SHOULDER, plan.shoulder)),
        plan.slopes.max_drop, plan.slopes.repair_passes, contact_drop,
    )
    # This final hold applies only to buried rock roofs. Formation and
    # bridge columns already have their own surface/clearance constraints.
    roof_hold = np.where(np.isfinite(rock_floor), contact_floor, -np.inf)
    designed = np.where(designed < natural, np.maximum(designed, np.minimum(natural, roof_hold)), designed)
    sculpted_area = grid.filled(False, dtype=bool)
    if plan.sculpt.enabled:
        # Only a fitting standing in the ground pins it: rails on a deck or
        # inside a lining do not.
        low, high = _fitting_span(grid, anchors)
        standing = footings & (low <= designed + 1.5) & (high >= designed - 1.5)
        # The ground kept beside and under a pier may take a bank's spill,
        # never a cut, so a bank runs round a pier instead of leaving it in a pit.
        piers = grid.mark(column for name in plan.preserve_fittings if name in anchors
                          for column in anchors[name].columns)
        spillable = (fitting_kept | piers) & ~clearance_protected
        reachable = known & within_margin & (permitted_area > 0)
        designed, sculpted_area = _sculpted(
            plan, grid, ground, natural, designed, known,
            reachable & ((~smoothing_protected & ~fitting_kept & ~standing) | spillable),
            reachable & spillable, natural_bridges, bridge_ground, tunnel_domain, stone, clearance,
            _crest_field(grid, alignment, natural), surface.water, smoothing_protected,
            _hollow_tops(grid, envelope, running),
            smoothing_context,
        )

    # Tunnel sides have their own native-contour controls. They deliberately
    # bypass rail-height slope bounds while the buried mountain stays held
    # during the ordinary bank/cutting solver above.
    tunnel_side_area = grid.filled(False, dtype=bool)
    tunnel_body = grid.filled(False, dtype=bool)
    if plan.tunnel.sides:
        side_stations = np.zeros(alignment.length, dtype=bool)
        for stretch in stretches:
            if stretch.profile == "tunnel" and stretch.intensity is Intensity.SHAPE:
                side_stations[max(0, stretch.start):min(alignment.length, stretch.stop + 1)] = True
        side_domain = side_stations[np.clip(ground.station.astype(np.int64), 0, alignment.length - 1)]
        body = grid.mark(column for column in alignment.columns
                         if side_stations[alignment.station[tuple(column)]])
        # Actual column heights, not the roof elevation propagated sideways
        # through the mountain. Extra envelope columns can be lining edges.
        tops = _crest_field(grid, alignment, natural)
        bottoms = natural.copy()
        for column, height in alignment.footing.items():
            bottoms[grid.at(*column)] = height
        if envelope is not None:
            for column, heights in envelope.items():
                if not heights or not grid.holds(*column):
                    continue
                at = grid.at(*column)
                if side_domain[at]:
                    body[at] = True
                    tops[at], bottoms[at] = max(heights), min(heights)
        tunnel_body = body
        roof = (body & (tops > bottoms + 1) & known & (natural > tops)
                & within_margin & (permitted_area > 0) & ~smoothing_protected
                & (clearance.ceiling >= clearance.floor) & (clearance.ceiling < tops))
        cover_floor = tops + plan.tunnel.min_cover
        designed = np.where(roof, np.maximum(designed, cover_floor), designed)
        before_sides = designed
        designed, tunnel_side_area = _tunnel_sides(
            plan, natural, designed, body, tops, bottoms,
            known & within_margin & (permitted_area > 0) & ~smoothing_protected,
            side_domain, smoothing_context, contact_drop,
        )
        # Apply explicit height masks to this new pass as well.
        masked = _surface_limits(plan, terrain, grid, ground, stretches,
                                 natural, designed, followers)
        designed = np.where(tunnel_side_area, masked, designed)
        designed = clearance.limit_surface(natural, designed)
        designed = np.where(roof, np.maximum(designed, cover_floor), designed)
        tunnel_side_area &= np.abs(designed - before_sides) >= DEAD_BAND

    # Growth is cleared where the ground changes or the clearance passes.
    changed = (slopes.level(designed) != natural) & known
    # Preserving a riverbed or shore must not leave leaves in the bridge's
    # headroom. Structural protection and keep-outs still protect growth.
    bore_protected = clearance_protected & ~(clearance_shore | (water > natural))
    cleared = _clear_growth(plan, surface, grid, changed, clearance, bore_protected, designed,
                            natural_bridges, ground.seed_height, ground.distance, ground.crest)
    earthworks = build(
        grid, natural, designed, plan.strata, palette or BlockPalette(),
        layering, water, clearance, top_blocks,
        rock_surface=stone if plan.cutting.rock else None, subsurface=surface.layers,
    )
    _underpin(plan, terrain, alignment, stretches, earthworks, grid, layering, protected, clearance)
    carved = earthworks.cut | frozenset(
        position for position in surface.intruding
        if not (grid.holds(position.x, position.z)
                and bore_protected[grid.at(position.x, position.z)])
    ) | cleared
    if track_clearance is not None:
        # Surface protection still holds around the route, but cannot leave
        # a buried road column, water, or vegetation in its usable space.
        carved |= track_clearance.carved(surface.intruding | surface.growth, grid)

    from .faces import floating_remnants, repair as repair_faces
    # Whole body/fitting columns stay out of block morphology: this cannot
    # erase a lining corner, a wall, a pole or its supporting material.
    fixed_columns = grid.mark([*alignment.columns,
                               *(p for held in anchors.values() for p in held.columns)])
    fixed_columns |= tunnel_body
    face_area = (known & within_margin & ~smoothing_protected & ~footings & ~fixed_columns
                 & (permitted_area > 0)
                 & ((writable > 0) | (smoothing_context.transition_weight > 0) | tunnel_side_area))
    if plan.smoothing.preserve_cliffs:
        face_area &= ~smoothing.preserved_faces(designed, natural, known)
    # Budgets participate in each synchronous morphology decision, rather
    # than filtering committed cuts afterwards and invalidating neighbours.
    side_cut_floor = np.where(tunnel_side_area,
                              np.minimum(natural - plan.tunnel.max_change, designed), -np.inf)
    side_fill_ceiling = np.where(tunnel_side_area,
                                 np.maximum(natural + plan.tunnel.max_change, designed), np.inf)
    extra_cut, extra_fill = repair_faces(
        terrain, grid, natural, designed, known, face_area, earthworks.fill,
        carved, clearance, smoothing_modes, plan.smoothing,
        smoothing_context.overrides, plan.slopes.max_drop,
        cut_floor=side_cut_floor, fill_ceiling=side_fill_ceiling,
    )
    carved |= extra_cut
    for position, state in extra_fill.items():
        earthworks.fill.set_block(position, state)
    carved -= extra_fill.keys()
    hanging = floating_remnants(terrain, grid,
                                known & within_margin & ~smoothing_protected & ~footings & ~fixed_columns,
                                earthworks.fill, carved)
    for position in hanging:
        if earthworks.fill.get_block(position) is not None:
            earthworks.fill.remove_block(position)
    carved |= hanging
    extra_cut = frozenset(extra_cut | hanging)
    if extra_cut or extra_fill:
        changed_columns = {(p.x, p.z) for p in extra_cut | extra_fill.keys()}
        fill_heights: dict[tuple[int, int], list[int]] = {}
        for p, _ in earthworks.fill.iter_block_ids():
            if (p.x, p.z) in changed_columns:
                fill_heights.setdefault((p.x, p.z), []).append(p.y)
        for x, z in changed_columns:
            # Side erosion can leave an overhang: retain its actual highest
            # surviving ground instead of pretending this is a solid column.
            heights = [y for y, state in terrain.column(x, z)
                       if y <= designed[grid.at(x, z)] + .5 and classify(state) in SOLID
                       and BlockPos(x, y, z) not in carved]
            heights.extend(fill_heights.get((x, z), ()))
            if heights:
                designed[grid.at(x, z)] = max(heights)
        removed_growth = growth_clearing(
            surface.growth,
            lambda p: p in extra_cut or p + BlockPos(0, -1, 0) in extra_cut,
            surface.trees, surface.trunks,
        ) if plan.foliage.clear else frozenset()
        removed_growth = frozenset(p for p in removed_growth
                                  if grid.holds(p.x, p.z) and not bore_protected[grid.at(p.x, p.z)])
        cleared |= removed_growth
        carved |= removed_growth

    smoothing_modes = smoothing_context.modes(designed, natural, plan.smoothing)
    if plan.look is not Look.ENGINEERED and plan.embankment.shape == "natural":
        finish_surfaces(
            terrain, grid, natural, designed, known & within_margin
            & ~smoothing_protected & ~footings & ~fixed_columns & (permitted_area > 0),
            earthworks.fill, carved, plan.strata, layering, water, clearance,
            top_blocks, skin=sculpted_area,
            rock_surface=stone if plan.cutting.rock else None,
        )
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
            ungenerated=int((~known & (sample_area if sample_area is not None else True)).sum()),
            suppressed=int((writable == 0).sum()),
            filled=len(earthworks.fill),
            carved=len(carved),
            cleared=len(cleared),
            face_carved=len(extra_cut),
            face_filled=len(extra_fill),
            tunnel_side_columns=int(np.count_nonzero(tunnel_side_area)),
            steep=_final_steps(designed, natural, known & ~smoothing_protected & ~footings
                               & ~limits.exempt & ~limits.bed & ~fixed_columns,
                               np.where(sculpted_area & (designed <= stone + 1),
                                        np.maximum(contact_drop, plan.cutting.rock_drop),
                                        contact_drop)),
            smoothing=tuple((mode.name.lower(), int(np.count_nonzero(
                known & (ground.distance > 0) & (smoothing_modes == mode))))
                for mode in smoothing.Mode),
        ),
        clearance,
        (within_margin | (slopes.level(designed) != natural)) & known,
        surface.top,
        getattr(terrain, "saved_chunks", None),
        surface.water,
        smoothing_modes,
        track_clearance,
        dry_blocks,
    )


def _slope_limits(
    plan: Plan,
    grid: Grid,
    alignment: Alignment,
    stretches: Sequence[Stretch],
    below: Subsurface,
    natural: Field,
    known: Field,
    water: Field,
    blocked: Field,
    clearance: Clearance,
    shore: Field,
    held: Field | None = None,
    cut_drop: Field | None = None,
) -> slopes.Limits:
    # What each station's stretch makes of the ground beside it: shaped
    # (from the top of its walls where it retains), a bridge's ground its
    # neighbours' banks may spill onto, or held as it is.
    length = alignment.length
    role = np.full(length, slopes.HELD, dtype=np.int8)
    from_crest = np.zeros(length, dtype=bool)
    for stretch in stretches:
        start, stop = max(stretch.start, 0), min(stretch.stop, length - 1)
        if stretch.intensity is not Intensity.SHAPE or stop < start:
            continue
        stations = slice(start, stop + 1)
        if stretch.cross_section is not None or not _keeps(plan.profile(stretch.profile)):
            role[stations] = slopes.SHAPED
            from_crest[stations] |= _retains(plan, stretch)
        elif plan.embankment.spill and "bridge" in stretch.profile:
            role[stations] = np.maximum(role[stations], slopes.SPILL)
    track = grid.mark(alignment.columns)
    station = grid.filled(0.0)
    for (x, z), index in alignment.station.items():
        if grid.holds(x, z):
            station[grid.at(x, z)] = index
    meets = grid.filled(0.0)
    for (x, z), height in alignment.meets.items():
        if grid.holds(x, z):
            at = grid.at(x, z)
            walls = from_crest[int(station[at])] and (x, z) not in alignment.rail_support
            meets[at] = alignment.crest.get((x, z), height) if walls else height
    bank_margin = max(plan.margin, plan.embankment.max_reach)
    zone = nearest(track, (station,), bank_margin + 1.0)
    index = np.clip(zone[0].astype(np.int64), 0, length - 1)
    # Past the margin the ground stays as it is; a slope that reaches it is
    # steepened to meet it.
    roles = np.where(zone.distance > bank_margin, slopes.HELD, role[index])
    if plan.bleed:
        roles = np.where(
            ~track & (zone.distance <= bank_margin) & (roles == slopes.HELD),
            slopes.SPILL, roles,
        )
    if held is not None:
        # The edit boundary is fixed ground to taper toward. Treating it as
        # an exempt obstacle allows a bank to end in a vertical cliff.
        roles = np.where(held & ~track, slopes.HELD, roles)
    # Never across water under a bridge, nor on a side a cross section
    # leaves open.
    if not plan.embankment.water_spill:
        blocked = blocked | ((water > natural) & (roles == slopes.SPILL))
    rail_support = grid.mark(ColumnPos(x, z) for x, z in alignment.rail_support)
    walled = track & from_crest[index] & ~rail_support
    for stretch in stretches:
        if stretch.cross_section is None:
            continue
        section = plan.cross_section(stretch.cross_section)
        modes = side_modes(grid, SimpleNamespace(station=zone[0]), alignment, section, natural, water, known)
        within = (index >= stretch.start) & (index <= stretch.stop)
        if section.cut_base == "crest":
            # A cut side measured from its walls' tops holds the ground
            # there, as a trench does.
            side = track & modes["cutting"] & within & ~rail_support
            walled |= side
            meets = np.where(side, _crest_field(grid, alignment, meets), meets)
        if not plan.bleed:
            blocked = blocked | (modes["bridge"] & within)
    cap = np.where(clearance.ceiling >= natural, clearance.floor - 1.0, np.inf)
    natural_shape = plan.embankment.shape == "natural"
    # These are only steepness bounds for a natural design. Applying the
    # chosen bank ratio here as its target would recreate uniform cones.
    bound_run = 1.0 / plan.slopes.max_drop
    limits = slopes.limits(
        natural, track, roles, meets, walled, blocked, cap,
        plan.shoulder,
        bound_run if natural_shape else max(plan.embankment.ratio or slopes.DEFAULT_RUN, 1.0),
        (bound_run if cut_drop is None else 1 / cut_drop) if natural_shape
        else max(plan.cutting.ratio or slopes.DEFAULT_RUN, 1.0),
        plan.slopes.cone_passes,
        below if plan.cutting.rock else None,
        plan.slopes.portal_reach,
        plan.slopes.repair_passes,
        plan.cutting.rock_band,
        plan.slopes.pocket,
        plan.cutting.wall_band,
        plan.slopes.ridge_width,
        None if plan.terrain_mask.slope_to_water else shore,
        retained_area=from_crest[index] if plan.bleed else None,
        rock_area=_natural_cliffs(natural, known) if natural_shape else None,
        hold_walls=natural_shape,
    )
    limits = replace(limits, conform=not natural_shape, max_drop=plan.slopes.max_drop,
                     cut_drop=cut_drop if natural_shape else None)
    if bank_margin > plan.margin:
        # Outside the normal edit corridor only a bank may add ground.
        # Slopes and repair cannot excavate the mountain there.
        least = np.where(zone.distance > plan.margin, natural, -np.inf)
        if limits.least is not None:
            least = np.maximum(least, limits.least)
        limits = replace(limits, cut=np.maximum(limits.cut, least), least=least)
    return limits


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
    # Eased like a side slope, flat where it meets the ground and the top,
    # as far as the room allows: the eased middle climbs no more than two a
    # block, else the grade stays straight.
    rise = np.abs(upper[0] - lower[0])
    straight = np.divide(rise, lower.distance + upper.distance,
                         out=np.zeros_like(rise), where=(lower.distance + upper.distance) > 0)
    ease = np.clip(np.divide(2.0 - straight, (vocabulary.EASED_PEAK - 1.0) * straight,
                             out=np.ones_like(rise), where=straight > 0), 0.0, 1.0)
    curve = ease * vocabulary.smootherstep(portion) + (1.0 - ease) * portion
    grade = lower[0] + curve * (upper[0] - lower[0])
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


def _underpin(
    plan: Plan,
    terrain: Terrain,
    alignment: Alignment,
    stretches: Sequence[Stretch],
    earthworks: Earthworks,
    grid: Grid,
    layering: Layering,
    protected: Field | None = None,
    clearance: Clearance | None = None,
) -> None:
    # The heightfield only sees the top of the ground, so a cave or an
    # overhang under the track's footing is left hollow. Where a stretch
    # raises ground, fill it down to the rock under it, with the land's own
    # deepest layer there.
    raising = [
        stretch for stretch in stretches
        if stretch.intensity is Intensity.SHAPE
        and any(step.blend is Blend.RAISE for step in plan.profile(stretch.profile).steps)
    ]
    fill = earthworks.fill
    for (x, z), footing in alignment.footing.items():
        if protected is not None and grid.holds(x, z) and protected[grid.at(x, z)]:
            continue
        station = alignment.station[(x, z)]
        # A shared marker belongs to the incoming section, as for headroom.
        # Extending the departing bank's underpin here builds a solid wall
        # under the first bridge deck, even where only a spill was designed.
        holding = [stretch for stretch in stretches if stretch.start <= station <= stretch.stop]
        holding = [stretch for stretch in holding if stretch.start == station] or holding
        if not any(stretch in raising for stretch in holding):
            continue
        solid = {y for y, state in terrain.column(x, z) if classify(state) in SOLID}
        # Through the earthworks' own fill, down to the ground it stands on.
        hollow = []
        for y in range(footing - 1, footing - 1 - UNDERPIN, -1):
            position = BlockPos(x, y, z)
            if y in solid:
                break
            if fill.get_block(position) is None:
                hollow.append(position)
        else:
            continue
        found = layering.at(*grid.at(x, z), layering.layers.shape[0]) if grid.holds(x, z) else None
        for position in hollow:
            if clearance is not None and clearance.holds(*grid.at(position.x, position.z), position.y):
                continue
            fill.set_block(position, found or plan.strata.filler)


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


def _natural_cliffs(natural: Field, known: Field) -> Field:
    # Survey the native shape at block and hillside scales. A steep rock
    # hillside can consist entirely of one-block steps; testing adjacent
    # columns alone mistakes it for a gentle bank and grades the mountain
    # away. Four blocks of sustained rise over four columns also qualifies.
    # Unknown columns interrupt the survey, so an unsurveyed gap cannot
    # manufacture a cliff between disconnected patches.
    level = np.where(known, natural, np.nan)
    cliff = np.zeros(natural.shape, dtype=bool)
    rows = np.abs(np.diff(level, axis=0)) >= 2.0
    cells = np.abs(np.diff(level, axis=1)) >= 2.0
    cliff[1:] |= rows
    cliff[:-1] |= rows
    cliff[:, 1:] |= cells
    cliff[:, :-1] |= cells
    for axis in (0, 1):
        run = 4
        if natural.shape[axis] <= run:
            continue
        starts, stops = [slice(None)] * 2, [slice(None)] * 2
        starts[axis], stops[axis] = slice(None, -run), slice(run, None)
        starts, stops = tuple(starts), tuple(stops)
        difference = level[stops] - level[starts]
        sustained = np.abs(difference) >= run
        for offset in range(run):
            a, b = [slice(None)] * 2, [slice(None)] * 2
            a[axis] = slice(offset, natural.shape[axis] - run + offset)
            b[axis] = slice(offset + 1, natural.shape[axis] - run + offset + 1)
            step = level[tuple(b)] - level[tuple(a)]
            sustained &= np.isfinite(step) & (step * difference >= 0)
        for offset in range(run + 1):
            target = [slice(None)] * 2
            target[axis] = slice(offset, natural.shape[axis] - run + offset)
            cliff[tuple(target)] |= sustained
    return cliff


def _final_steps(designed: Field, natural: Field, known: Field, max_drop: float | Field) -> int:
    # Count newly enlarged faces after all surface stages, including columns
    # held at their native height. Original cliffs are not failures.
    levels = slopes.level(designed)
    count = 0
    permitted = np.broadcast_to(max_drop, natural.shape)
    for axis in (0, 1):
        a, b = [slice(None)] * 2, [slice(None)] * 2
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        valid = known[a] & known[b]
        difference = levels[a] - levels[b]
        allowed = np.where(difference > 0, permitted[a], permitted[b])
        count += int(np.count_nonzero(valid & (np.abs(difference)
                         > np.maximum(allowed, np.abs(natural[a] - natural[b])))))
    return count


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
    for (x, z), level in alignment.meets.items():
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


def _water_banks(
    plan: Plan, limits: slopes.Limits, grid: Grid, alignment: Alignment,
    natural: Field, designed: Field, water: Field, allowed: Field,
) -> slopes.Limits:
    # Only shores beside the works become slope anchors. Carry the actual
    # water surface, rather than the riverbed, inland at a block per column.
    # The compact corridor, protected builds, rail bed and tunnel ground hold.
    wet = water > natural
    if not wet.any():
        return limits
    dry = limits.free & (allowed > 0) & ~grid.mark(alignment.columns)
    near = nearest(wet, (water,), 2.0)
    changed = dry & (np.abs(designed - natural) >= DEAD_BAND)
    beside_works = nearest(changed, (), max(plan.reach, 1.0) + 1.0).distance <= max(plan.reach, 1.0)
    # A shoreline anchor must fit between the water and the ground that
    # cannot move. Otherwise cutting to it would leave a cliff at the margin
    # or undermine the track. Carry that fixed ground's minimum inward too.
    fixed = ~dry & ~wet & (natural > ABSENT)
    floor = slopes.cone(
        fixed, np.where(grid.mark(alignment.columns), designed, natural), dry,
        1.0, rising=False, passes=plan.slopes.cone_passes,
    )
    sources = dry & (near.distance <= 1.0) & beside_works & (floor <= near[0])
    if not sources.any():
        return limits
    cap = slopes.cone(
        sources, near[0], dry, 1.0, rising=True,
        passes=plan.slopes.cone_passes,
    )
    cap = np.maximum(cap, floor)
    # Where the existing bank needs cutting, hold its new slope height too:
    # the brush must not hollow out a step that rock-face repair would keep.
    tapered = dry & (natural > cap)
    least = np.where(tapered, cap, -np.inf)
    if limits.least is not None:
        least = np.maximum(least, limits.least)
    return replace(
        limits, fill=np.where(tapered, cap, np.minimum(limits.fill, cap)),
        cut=np.minimum(limits.cut, cap), cap=np.minimum(limits.cap, cap), least=least,
        ceiling=slopes.level(cap),
    )


def _protected(plan: Plan, surface, grid: Grid) -> tuple[Field, Field]:
    # Water and its dry buffer stay untouched. Only an explicitly requested
    # shoreline taper may grade the dry buffer, after ordinary shaping.
    mask = grid.filled(False, dtype=bool)
    shore = grid.filled(False, dtype=bool)
    if plan.terrain_mask.preserve_water and surface.water is not None:
        wet = surface.water > surface.height
        if plan.terrain_mask.water_buffer:
            near = kernels.local_maximum(
                wet.astype(np.float64), plan.terrain_mask.water_buffer
            ) > 0
            shore = near & ~wet
            wet = near
        mask |= wet
    if plan.terrain_mask.preserve_structures and surface.structures is not None:
        mask |= surface.structures
        shore &= ~surface.structures
    return mask, shore


def _span(plan: Plan) -> float:
    # How far out from the track the ground is shaped at all.
    return max(plan.reach, plan.shoulder + plan.brush.margin + plan.brush.falloff)


def _brush_area(
    plan: Plan,
    ground: vocabulary.Ground,
    stretches: Sequence[Stretch],
    scopes: Sequence[Field],
    shaped: Field,
    brushable: Field,
) -> Field:
    # Where the brush works, and how hard: 1 over the ground the works touch
    # and `margin` beyond, fading to 0 over `falloff`. Nothing on the track's
    # own footprint, under a bridge or over a tunnel, or protected. The
    # shoulder is brushed too, so its brow rolls down rather than standing
    # as a lip.
    brush = plan.brush
    held = ground.distance <= 0.0
    kept = ground.grid.filled(0.0)
    for stretch, scope in zip(stretches, scopes):
        if stretch.cross_section is None and _keeps(plan.profile(stretch.profile)):
            kept = np.maximum(kept, scope)
    touched = brushable & ~held & (np.abs(shaped - ground.natural) >= DEAD_BAND)
    if not touched.any():
        return np.broadcast_to(0.0, ground.grid.shape)
    away = nearest(touched, (), brush.margin + brush.falloff).distance
    weight = 1.0 - vocabulary.smootherstep((away - brush.margin) / brush.falloff)
    return np.where(brushable & ~held, weight * (1.0 - kept), 0.0)


def _brushed(plan: Plan, designed: Field, area: Field,
             modes: Field | None = None, known: Field | None = None) -> Field:
    # The finished ground smoothed the way a player would with a smoothing
    # brush, the ground's own bumps and all. Everything outside the brush's
    # area -- the track among it -- holds its height through every pass, so
    # the ground smooths up to it.
    if plan.brush.passes <= 0 or not (area > 0).any():
        return designed
    if modes is None:
        # A standalone brush has no cut/fill intent to classify.
        mode = (smoothing.Mode.STABLE if plan.smoothing.behavior == "auto"
                else smoothing.Mode[plan.smoothing.behavior.upper()])
        modes = np.full(designed.shape, mode, dtype=np.int8)
    return smoothing.brush(designed, area, modes, plan.smoothing,
                           plan.brush.radius, plan.brush.passes, known)


def _smooth_transitions(
    plan: Plan, ground: vocabulary.Ground, designed: Field, natural: Field,
    known: Field, context: smoothing.Context, modes: Field, protected: Field,
    bridges: Field, tunnels: Field, clearance: Clearance,
) -> Field:
    # One brush works over all section interfaces. Heights have already
    # been chosen from the actual formation, wall, deck and tunnel roof;
    # no interface recipe raises a whole valley toward a floating deck.
    area = np.where(known & ~protected & ((ground.distance > 0) | tunnels),
                    context.transition_weight, 0.0)
    if plan.smoothing.preserve_cliffs:
        native_faces = smoothing.preserved_faces(designed, natural, known) & (ground.distance > 0)
        area = np.where(native_faces, 0.0, area)
    # A section interface is a location to blend earthworks, not permission
    # to reshape intact terrain throughout its disk. Feather only the actual
    # cuts/fills and a small brush-width edge into the saved land. Fixed
    # formation/roof columns can seed that edge without being brushed.
    touched = known & ~protected & (np.abs(designed - natural) >= DEAD_BAND)
    if not touched.any():
        return designed
    feather = 2 * plan.brush.radius
    if feather > 0:
        away = nearest(touched, (), feather + 1).distance
        area *= 1 - vocabulary.smootherstep(away / feather)
    else:
        area *= touched
    result = smoothing.brush(designed, area, modes, plan.smoothing,
                             plan.brush.radius, plan.brush.passes, known, natural)
    # Physical constraints still hold after every material-direction choice.
    result = clearance.limit_surface(natural, result)
    result = np.where(tunnels & (ground.distance <= 0) & (natural > ground.crest),
                      np.maximum(result, np.minimum(natural, ground.crest + 1)), result)
    bridge_limit = np.where(ground.distance <= 0, np.minimum(natural, ground.seed_height - 1),
                            natural + plan.bridge.max_change)
    result = np.where(bridges, np.minimum(result, bridge_limit), result)
    result = np.where(area > 0, result, designed)
    result = np.where((np.abs(result - natural) < DEAD_BAND)
                      & (np.abs(designed - natural) < DEAD_BAND), natural, result)
    return smoothing.guard_steps(designed, result, area > 0, plan.slopes.max_drop)


def _tidy_fringe(designed: Field, natural: Field, free: Field) -> Field:
    # Where a fill or cut fades out it is a block or less, and grain near
    # half a block tips columns either way: specks, and a ragged edge. There
    # a column takes the change most of its eight neighbours have.
    level = np.rint(natural)
    step = np.rint(designed) - level
    for _ in range(FRINGE_PASSES):
        padded = np.pad(step, 1, mode="edge")
        rows, cells = step.shape
        around = [
            padded[1 + dz:1 + dz + rows, 1 + dx:1 + dx + cells]
            for dz in (-1, 0, 1) for dx in (-1, 0, 1) if dz or dx
        ]
        raised = sum((each >= 1).astype(np.int8) for each in around)
        lowered = sum((each <= -1).astype(np.int8) for each in around)
        fringe = free & (np.abs(step) <= 1)
        tidied = step.copy()
        tidied[fringe & (step == 1) & (raised < 3)] = 0
        tidied[fringe & (step == -1) & (lowered < 3)] = 0
        tidied[fringe & (step == 0) & (raised >= 6)] = 1
        tidied[fringe & (step == 0) & (lowered >= 6)] = -1
        step = tidied
    moved = np.rint(designed) - level != step
    return np.where(moved, level + step, designed)


def _levelled_lone_columns(designed: Field, free: Field) -> Field:
    # A column left above or below all four of its neighbours, a pit or a
    # post the passes before could make, goes to the nearest of them.
    level = np.rint(designed)
    padded = np.pad(level, 1, mode="edge")
    rows, cells = level.shape
    around = np.stack([
        padded[1 + dz:1 + dz + rows, 1 + dx:1 + dx + cells]
        for dz, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
    ])
    low, high = around.min(0), around.max(0)
    pit = free & (level < low)
    post = free & (level > high)
    return np.where(pit, low, np.where(post, high, designed))


def _tidied_levels(designed: Field, free: Field) -> Field:
    # Every level's edge, not just the fringe: a column with five or more of
    # its eight neighbours a level below is a tooth and comes down, one with
    # five or more above is a notch and goes up. A straight or a diagonal
    # edge never has five on the other side, so it stays.
    step = np.rint(designed)
    rows, cells = step.shape
    for _ in range(LEVEL_PASSES):
        padded = np.pad(step, 1, mode="edge")
        around = [
            padded[1 + dz:1 + dz + rows, 1 + dx:1 + dx + cells]
            for dz in (-1, 0, 1) for dx in (-1, 0, 1) if dz or dx
        ]
        below = sum((each < step).astype(np.int8) for each in around)
        above = sum((each > step).astype(np.int8) for each in around)
        tidied = step.copy()
        tidied[free & (below >= 5)] -= 1
        tidied[free & (above >= 5)] += 1
        step = tidied
    moved = np.rint(designed) != step
    return np.where(moved, step, designed)


def _crossfaded(
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
) -> Field:
    # Each stretch shaped on its own and the shapes weighed by their scopes,
    # so neighbours blend across a handover instead of the later overriding
    # the earlier along the line of the station where they meet. A stretch
    # with retaining walls holds its own stations -- its walls need the
    # ground at their top all along -- and its neighbours blend into it
    # beyond its ends.
    held = grid.filled(0.0)
    for stretch, scope in zip(stretches, scopes):
        if not plan.bleed and _retains(plan, stretch):
            held = np.maximum(held, scope)
    whole = grid.filled(1.0)
    change = grid.filled(0.0)
    total = grid.filled(0.0)
    candidates: dict[tuple[str, str | None], Field] = {}
    for stretch, scope in zip(stretches, scopes):
        if not scope.any():
            continue
        weight = scope if _retains(plan, stretch) else scope * (1.0 - held)
        key = (stretch.profile, stretch.cross_section)
        if key not in candidates:
            layers = _layers(
                plan, surroundings, [stretch], [whole], grid, ground, alignment,
                natural, water, known,
            )
            candidates[key] = vocabulary.shape(ground, layers) - ground.natural
        change += weight * candidates[key]
        total += weight
    return ground.natural + change / np.maximum(total, 1.0)


def _steepness(plan: Plan, grid: Grid) -> Field | float:
    # How many times gentler than its steepest a slope may run at each column.
    if plan.slopes.variation <= 0:
        return 1.0
    wander = np.clip((fbm(grid, plan.slopes.seed, WAVELENGTH, 3, WARP) + 1.0) / 2.0, 0.0, 1.0)
    return 1.0 + plan.slopes.variation * wander


def _natural_detail(designed: Field, natural: Field, known: Field, distance: Field) -> Field:
    # The saved ground's bumps carried onto the works' slopes: none on the
    # track's shoulder or where the ground barely moved, in full where it
    # moved three blocks or more.
    level = np.where(known, natural, 0.0)
    share = kernels.disk(known.astype(np.float64), 3.0, "constant")
    smooth = np.divide(kernels.disk(level, 3.0, "constant"), share,
                       out=level.copy(), where=share > 1e-6)
    detail = np.clip(natural - smooth, -DETAIL, DETAIL)
    weight = (np.clip((distance - SHOULDER) / 3.0, 0.0, 1.0)
              * np.clip(np.abs(designed - natural) / 3.0, 0.0, 1.0))
    return np.where(known, designed + weight * detail, designed)


def _sculpted(
    plan: Plan, grid: Grid, ground: vocabulary.Ground, natural: Field, designed: Field,
    known: Field, editable: Field, spillable: Field, natural_bridges: Field, bridge_ground: Field,
    tunnel_domain: Field, stone: Field, clearance: Clearance, tops: Field,
    water: Field | None, protected: Field, hollows: Field,
    policy: smoothing.Context | None = None,
) -> tuple[Field, Field]:
    # Open sections keep their structural contacts: bed, shoulder and the band
    # behind retaining walls, though ground kept beside a pier there may take
    # a bank's spill. Every other column the works may change is shaped afresh
    # from those contacts and the ground they leave alone; a tunnel's lining
    # keeps its cover.
    track = ground.distance <= 0
    # A wall's cut-back band must not become an extra flat fill shoulder.
    # Banks start at the requested shoulder; excavated contacts keep the
    # wider band needed behind retaining walls.
    band = (((ground.distance <= plan.shoulder)
             | ((ground.distance <= plan.cutting.wall_band + 1.0)
                & (designed < natural - DEAD_BAND)))
            & ~tunnel_domain & ~natural_bridges)
    lining = tunnel_domain & track
    roof = tops + plan.tunnel.min_cover
    free = editable & (~band | (spillable & ~track))
    fixed = known & ~free
    # A bridge keeps the ground its own policy leaves under it; held water
    # stands at its surface. Nothing is built up under the bridge's deck and
    # saved ground standing over the deck is taken down, but ground the land
    # already holds inside the deck's own body stays, and the deck holds the
    # land beside it at its top. An edge of the structure with no deck over
    # it lies in the land like any other ground.
    under = natural_bridges & track
    # A deck column carries the running space: its structure reaches within a
    # few blocks of that space's floor.
    deck = under & (clearance.floor <= clearance.ceiling) & (tops >= clearance.floor - DECK_REACH)
    inside = deck & (natural > ground.seed_height - 1.0) & np.isfinite(hollows)
    base = np.where(natural_bridges, bridge_ground, natural)
    base = np.where(deck, np.where(inside, np.minimum(natural, hollows),
                                   np.minimum(base, ground.seed_height - 1.0)), base)
    base = np.where(under & ~deck, natural, base)
    heights = designed.copy()
    land = natural
    if water is not None:
        wet = fixed & (water > natural)
        base = np.where(wet, water, base)
        heights = np.where(wet, water, heights)
        land = np.where(wet, water, natural)
    cap = np.where(deck, np.maximum(ground.seed_height - 1.0, base), np.inf)
    cap = np.minimum(cap, np.where(clearance.ceiling >= natural, clearance.floor - 1.0, np.inf))
    hold = np.where(deck, np.maximum(cap, np.minimum(base, tops)), cap)
    # Only a lining the land already buries keeps its cover; one standing
    # out of a valley side is not buried by the works.
    floor = np.where(lining & (natural > tops), roof, -np.inf)
    steep = (sculpt.native_steepness(natural, known & ~protected, (0.5, 0.97),
                                     origin=(grid.origin.x, grid.origin.z))
             if plan.sculpt.adapt else None)
    rates = sculpt.rates_for(grid, plan.slopes.seed, plan.sculpt.rock_rise,
                             plan.sculpt.soil_rise, plan.sculpt.bank_rise, steep)
    fill_only = free & spillable
    shaped = sculpt.design(base, fixed, heights, free, stone, rates, cap, floor, fill_only, hold)
    beside = editable & band & ~track
    textured = sculpt.texture(shaped.heights, base, fixed, free, heights, stone, shaped.cut_floor,
                              grid, plan.slopes.seed, plan.sculpt.ledges, plan.cutting.rock_drop,
                              beside, cap, land)
    textured = np.where(fill_only, np.maximum(textured, base), textured)
    textured = np.where(free | beside, np.minimum(np.maximum(textured, floor), cap), designed)
    textured = clearance.limit_surface(natural, textured)
    # Sculpting runs after the ordinary brush. Fair its bank contours last,
    # keeping every contact, cliff, toe and broad mound where it was placed.
    bank_reference = textured
    textured = sculpt.smooth_banks(
        textured, land, free & ~lining & ~deck, plan.sculpt.bank_radius,
        plan.sculpt.bank_passes, plan.smoothing.stable_change, plan.slopes.max_drop,
        stations=ground.station,
    )
    # Automatic fairing may trim excess fill as well as soften a hollow,
    # while remaining above the saved land. Explicit directional rules hold.
    if policy is None:
        mode = (smoothing.Mode.STABLE if plan.smoothing.behavior == "auto"
                else smoothing.Mode[plan.smoothing.behavior.upper()])
        modes = np.full(textured.shape, mode, dtype=np.int8)
    else:
        modes = np.where(policy.overrides >= 0, policy.overrides, smoothing.Mode.STABLE)
    textured = smoothing.constrain(bank_reference, textured, modes, plan.smoothing.stable_change)
    textured = np.where(free, np.minimum(np.maximum(textured, floor), cap), bank_reference)
    textured = clearance.limit_surface(natural, textured)
    textured = smoothing.guard_steps(bank_reference, textured, free, plan.slopes.max_drop)
    # The structural band beside the track takes the same skin as the faces.
    skin = (free | beside) & (textured < natural - DEAD_BAND)
    return textured, skin


def _hollow_tops(grid: Grid, envelope: Mapping[tuple[int, int], Sequence[int]] | None,
                 running: Mapping[tuple[int, int], int] | None) -> Field:
    # The top of the first gap inside each column's structure, below its
    # running surface: ground the land already holds there may stay.
    found = grid.filled(-np.inf)
    for (x, z), heights in (envelope or {}).items():
        if not heights or not grid.holds(x, z):
            continue
        levels = sorted(set(heights))
        surface = None if running is None else running.get((x, z))
        for below, above in zip(levels, levels[1:]):
            if surface is not None and above > surface:
                break
            if above - below > 1:
                found[grid.at(x, z)] = above - 1
                break
    return found


def _fitting_span(grid: Grid, anchors: Mapping[str, Alignment]) -> tuple[Field, Field]:
    # The lowest and highest block of any fitting over each column.
    low, high = grid.filled(np.inf), grid.filled(-np.inf)
    for held in anchors.values():
        for (x, z), y in held.footing.items():
            if grid.holds(x, z):
                at = grid.at(x, z)
                low[at] = min(low[at], y)
        for (x, z), y in held.crest.items():
            if grid.holds(x, z):
                at = grid.at(x, z)
                high[at] = max(high[at], y)
    return low, high


def _tunnel_sides(
    plan: Plan, natural: Field, designed: Field, body: Field,
    tops: Field, bottoms: Field, eligible: Field, domain: Field,
    context: smoothing.Context, contact_drop: Field,
) -> tuple[Field, Field]:
    """Blend exposed lining contacts into native relief, never into rail height.

    Deeply buried lining supplies no control. A shaped mouth can supply a
    bounded displacement, so its sides meet the approach without extending
    that excavation through the whole tunnel. Connected propagation respects
    water, keep-outs, unknown land and the owning route arm.
    """
    empty = np.zeros(body.shape, dtype=bool)
    if not body.any() or plan.tunnel.max_change <= 0:
        return designed, empty
    boundary = body & binary_dilation(~body)
    exposed = ((natural >= bottoms - plan.tunnel.max_change)
               & (natural <= tops + plan.tunnel.max_change))
    mouth = np.abs(designed - natural) >= DEAD_BAND
    sources = boundary & eligible & domain & (exposed | mouth)
    # Side contacts use the local lining extent, not a station-wide roof or
    # formation plane. Preserve existing portal controls where they exist.
    heights = np.where(mouth, designed, np.clip(natural, bottoms, tops + plan.tunnel.min_cover))
    heights = np.clip(heights, natural - plan.tunnel.max_change, natural + plan.tunnel.max_change)
    sources &= np.abs(heights - natural) >= DEAD_BAND
    if not sources.any():
        return designed, empty
    distance = nearest(body, (), plan.tunnel.reach + 1).distance
    outside = eligible & domain & ~body & (distance < min(plan.tunnel.reach, plan.margin, plan.reach))
    candidate = smoothing.contour_contact(
        natural, sources, heights, outside | sources,
        min(plan.tunnel.reach, plan.margin, plan.reach), contact_drop,
        plan.smoothing.contour_scale,
    )
    # Each bank independently chooses grow/stable/melt, including explicit
    # tunnel/profile/transition overrides. Do not restore existing earthworks.
    modes = context.modes(candidate, natural, plan.smoothing)
    candidate = smoothing.constrain(natural, candidate, modes, plan.smoothing.stable_change)
    delta = candidate - natural
    result = np.where(outside & (delta < -DEAD_BAND), np.minimum(designed, candidate),
                      np.where(outside & (delta > DEAD_BAND), np.maximum(designed, candidate), designed))
    moved = outside & (np.abs(result - designed) >= DEAD_BAND)
    result = _taper_edit_edges(result, natural, moved, plan.slopes.max_drop,
                              plan.slopes.repair_passes, contact_drop)
    result = np.clip(result, np.minimum(designed, natural - plan.tunnel.max_change),
                     np.maximum(designed, natural + plan.tunnel.max_change))
    return np.where(outside, result, designed), outside


def _portal_grading(
    plan: Plan,
    ground: vocabulary.Ground,
    approaching: np.ndarray,
    tunnel: Field,
    natural: Field,
    designed: Field,
    steepness: Field | float = 1.0,
    corridor: Field | None = None,
    rock_floor: Field | None = None,
    contact_drop: Field | None = None,
) -> Field:
    # The hill over and beside a tunnel carries on from its approach
    # cutting: it grows from the cutting's ground beside the track and from a
    # block over the lining at the portal. Its local material grade sets
    # the rise until it meets the hill; actual roof cover remains a floor.
    # The cutting's own sides dip to the portal too, never below its walls.
    station = np.clip(ground.station.astype(np.int64), 0, len(approaching) - 1)
    approach = approaching[station] & ~tunnel & (natural > ABSENT)
    if corridor is not None:
        approach &= corridor
    if not approach.any():
        return designed
    bed = approach & (ground.distance <= 0)
    sides = approach & ~bed
    cover = ground.crest + 1.0
    mouth = tunnel & (ground.distance <= 0) & (nearest(bed, (), 2.5).distance <= 1.5)
    if plan.look is not Look.ENGINEERED and plan.embankment.shape == "natural":
        # Only the actual mouth and the approach's exterior are controls.
        # Seeding every already-cut side column turns a small portal notch
        # into a repeated, ever-widening cone across the mountain.
        outline = bed & binary_dilation(~bed)
        domain = (approach | tunnel) & (natural > ABSENT)
        targets = np.where(mouth, np.minimum(natural, cover), designed)
        graded = smoothing.contour_cut(
            natural, mouth | outline, targets, domain, min(plan.reach, plan.margin),
            contact_drop if contact_drop is not None else max(plan.slopes.max_drop, PORTAL_RISE),
            plan.smoothing.contour_scale,
        )
        floor = np.where((ground.distance <= 0) & (natural > ground.crest), cover, -np.inf)
        # Replan the adjoining cut sides in the SAME connected domain. A
        # tunnel-only restoration would leave a new wall at its station
        # boundary against the old approach excavation. Existing fills and
        # the actual formation remain owned by their original operations.
        replanned = tunnel | (sides & (designed <= natural))
        floor = np.maximum(floor, np.where(sides, ground.crest, -np.inf))
        return np.where(replanned, np.minimum(natural, np.maximum(graded, floor)), designed)
    run = (steepness / max(plan.slopes.max_drop, PORTAL_RISE) if contact_drop is None
           else 1 / contact_drop)
    graded = slopes.cone(sides | mouth, np.where(mouth, cover, designed), sides | tunnel,
                         run, rising=True, passes=slopes.PASSES, natural=natural, floor=rock_floor)
    dipped = slopes.cone(mouth, cover, sides | tunnel, run, rising=True,
                         passes=slopes.PASSES, natural=natural, floor=rock_floor)
    designed = np.where(sides, np.minimum(designed, np.maximum(dipped, ground.crest)), designed)
    # Minimum soil cover belongs to buried roof columns only, not the
    # propagated roof height across both sides of the mountain.
    floor = np.where((ground.distance <= 0) & (natural > ground.crest), cover, -np.inf)
    if rock_floor is not None:
        floor = np.maximum(floor, rock_floor)
    result = np.where(tunnel, np.minimum(designed, np.maximum(graded, floor)), designed)
    if rock_floor is not None and mouth.any():
        # A small roof-contact trim follows native relief and fades over a
        # finite distance. A cone starting far below the rock either opens
        # the whole hill or stops before reaching the neighbouring roof.
        width = max(plan.slopes.portal_reach, 1.0)
        away = nearest(mouth, (), width + 1).distance
        amount = 1 - vocabulary.smootherstep(away / width)
        trim_depth = np.where(np.isfinite(rock_floor), np.maximum(natural - rock_floor, 0), 0)
        trimmed = np.maximum(floor, natural - trim_depth * amount)
        roof = tunnel & np.isfinite(rock_floor) & (amount > 0)
        result = np.where(roof, np.minimum(result, trimmed), result)
    return result


def _crest_field(grid: Grid, alignment: Alignment, otherwise: Field) -> Field:
    crest = otherwise.copy()
    for (x, z), height in alignment.crest.items():
        if grid.holds(x, z):
            crest[grid.at(x, z)] = height
    return crest


def _retains(plan: Plan, stretch: Stretch) -> bool:
    # Measured from the top of the track: walls the ground is held against.
    return stretch.cross_section is None and any(
        step.base == "crest" for step in plan.profile(stretch.profile).steps
    )


def _shaping(grid: Grid, stack: Sequence[vocabulary.Layer]) -> Field:
    # The union of the masks of layers that change the ground.
    reach = grid.filled(0.0)
    for layer in stack:
        if isinstance(layer.generator, vocabulary.Keep):
            continue
        scope = grid.filled(1.0) if layer.mask is None else layer.mask
        reach = np.maximum(reach, scope)
    return reach


def _continuous_keeps(plan: Plan, stretches: Sequence[Stretch]) -> tuple[Stretch, ...]:
    # A change of bridge schematic is not an abutment. Matching adjacent
    # keep profiles share one interior, so their run-outs, water protection
    # and brush mask only transition at the ends of the whole run.
    continuous: list[Stretch] = []
    for stretch in stretches:
        if continuous and stretch.intensity is Intensity.SHAPE and _keeps(plan.profile(stretch.profile)):
            previous = continuous[-1]
            if (previous.start <= stretch.start <= previous.stop + 1
                    and replace(stretch, start=previous.start, stop=previous.stop) == previous):
                continuous[-1] = replace(previous, stop=max(previous.stop, stretch.stop))
                continue
        continuous.append(stretch)
    return tuple(continuous)


def _scope(
    plan: Plan, ground: vocabulary.Ground, stretch: Stretch, allowed: Field,
    station: Field | None = None,
) -> Field:
    # Each column's share of a stretch's shaping: full over its stations,
    # running out past its ends. A stretch that keeps the ground (bridge,
    # tunnel) instead yields its ends to its neighbours' run-outs.
    if stretch.intensity is not Intensity.SHAPE:
        return np.broadcast_to(0.0, ground.grid.shape)
    within = (ground.station >= stretch.start) & (ground.station <= stretch.stop)
    if plan.bleed and _retains(plan, stretch):
        # Retaining walls end at their own section's boundary. Use the
        # complete alignment here: a nearest shaping source keeps carrying
        # its endpoint's station through the adjacent bridge.
        station = ground.station if station is None else station
        reach = ((station >= stretch.start) & (station <= stretch.stop)).astype(float)
    elif stretch.cross_section is None and _keeps(plan.profile(stretch.profile)):
        if plan.bleed:
            return np.broadcast_to(0.0, ground.grid.shape)
        reach = 1.0 - _run_out(ground.grid, ~within, stretch.transition)
    else:
        if plan.bleed:
            # Stations already came from the nearest shaping source. A
            # longitudinal fade needs no new distance transform per span.
            transition = max(stretch.transition, plan.reach, 1.0)
            away = np.maximum(np.maximum(stretch.start - ground.station,
                                         ground.station - stretch.stop), 0.0)
            reach = 1.0 - vocabulary.smootherstep(away / transition)
        else:
            reach = _run_out(ground.grid, within, stretch.transition)
    if plan.bleed:
        reach *= ground.distance < _span(plan)
    return reach * allowed


def _keeps(profile) -> bool:
    return all(step.action is Action.KEEP for step in profile.steps)


def _circular(profile) -> bool:
    return any(step.action is Action.CIRCULAR for step in profile.steps)


def _run_out(grid: Grid, mask: Field, length: int) -> Field:
    # 1 on `mask`, easing to 0 `length` columns out from it.
    if length <= 0:
        return mask.astype(np.float64)
    away = nearest(mask, (), float(length)).distance
    return 1.0 - vocabulary.smootherstep(away / length)


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
                modes = side_modes(
                    grid, ground, alignment, section, natural, water, known
                )
                side_cache[section.name] = {
                    mode: _run_out(grid, mask, int(np.ceil(plan.reach)))
                    if plan.bleed else mask
                    for mode, mask in modes.items()
                    if mode in ("embankment", "cutting", "at_grade")
                }
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
    # A slope eases out to the ground within the reach itself; fading it
    # again across the reach would put a crease back in.
    slope = step.action in (Action.SLOPE_DOWN, Action.SLOPE_UP)
    return vocabulary.Layer(
        generator=_generator(step, surroundings, reach, plan),
        blend=step.blend,
        mask=mask,
        softness=step.softness if step.softness is not None else softness,
        reach=0.0 if slope else reach,
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


def _formation(ground: vocabulary.Ground, shoulder: float = 0.0) -> Field:
    # 1 on the track and its shoulder, fading to 0 FORMATION out from it; a
    # wider designed shoulder is held level as far.
    held = max(SHOULDER, shoulder)
    return np.clip(
        (held + FORMATION - SHOULDER - ground.distance) / (FORMATION - SHOULDER), 0.0, 1.0
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
    shoulder = step.shoulder if step.shoulder is not None else (plan.shoulder if plan else 0.0)
    if step.action is Action.SLOPE_DOWN:
        response = plan.embankment if plan is not None and plan.embankment.adaptive and not step.ratio else None
        return vocabulary.Slope(
            step.ratio or (plan.embankment.ratio if plan is not None else None)
            or surroundings.fill_ratio, rising=False, from_crest=crest,
            height_response=0.0 if response is None else response.height_response,
            terrain_response=0.0 if response is None else response.terrain_response,
            max_ratio=4.5 if response is None else response.max_ratio,
            shoulder=shoulder,
            reach=reach,
            linear=plan is not None and plan.embankment.shape == "linear",
            natural=plan is not None and plan.embankment.shape == "natural",
        )
    if step.action is Action.SLOPE_UP:
        chosen = None if plan is None else plan.cutting.ratio
        return vocabulary.Slope(
            step.ratio or chosen or surroundings.cut_ratio, rising=True, from_crest=crest,
            shoulder=shoulder, reach=reach,
            natural=plan is not None and plan.embankment.shape == "natural",
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


def _ease_preserved_bridge_ground(
    designed: Field, natural: Field, known: Field, water: Field,
    protected: Field, footings: Field, bridge: Field, distance: Field,
    max_drop: int, reach: int,
) -> Field:
    """Keep a bridge bank from forming a collar around fixed dry ground.

    Each protected column carries its original height outward at the allowed
    drop per horizontal block. Taking the lowest of these bounds rounds the
    bank into nearby pier buffers and shore without touching the water or the
    bridge's own track columns.
    """
    seeds = known & protected & (water <= natural)
    if not seeds.any() or not bridge.any() or reach <= 0:
        return designed
    cap = np.where(seeds, natural, np.inf)
    for _ in range(reach):
        previous = cap.copy()
        np.minimum(cap[1:, :], previous[:-1, :] + max_drop, out=cap[1:, :])
        np.minimum(cap[:-1, :], previous[1:, :] + max_drop, out=cap[:-1, :])
        np.minimum(cap[:, 1:], previous[:, :-1] + max_drop, out=cap[:, 1:])
        np.minimum(cap[:, :-1], previous[:, 1:] + max_drop, out=cap[:, :-1])
    fill = (known & bridge & ~protected & ~footings & (water <= natural)
            & (distance > 0) & (designed > natural) & np.isfinite(cap))
    return np.where(fill, np.maximum(natural, np.minimum(designed, cap)), designed)


def _taper_edit_edges(
    designed: Field, natural: Field, movable: Field, max_drop: int, passes: int,
    cut_drop: Field | None = None,
) -> Field:
    """Ease a new face into fixed ground without enlarging the edit area.

    Restore cuts toward the source land first. If a fixed bore still leaves a
    new cliff, carry its cut back through the existing footprint. Keeping
    these phases separate prevents a cut from chasing its own edge downhill.
    """
    heights = slopes.level(designed)
    original = heights.copy()
    pairs = (
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
    )
    # A shallow cut beside unchanged high ground is best repaired by filling
    # that cut back in, not by digging a deeper face into the mountain.
    for _ in range(passes):
        up = np.full(heights.shape, -np.inf)
        for a, b in pairs:
            for high, low in ((a, b), (b, a)):
                allowed = max_drop if cut_drop is None else cut_drop[high]
                drop = heights[high] - heights[low]
                old = natural[high] - natural[low]
                newly_steep = (drop > allowed) & (drop > old)
                np.maximum(
                    up[low],
                    np.where(newly_steep & movable[low] & (heights[low] < natural[low]),
                             np.minimum(natural[low], heights[high] - allowed), -np.inf),
                    out=up[low],
                )
        changed = np.where(movable & (heights < natural), np.maximum(heights, up), heights)
        if np.array_equal(changed, heights):
            break
        heights = changed

    # Any face that cannot be eased by restoring a cut must fall back toward
    # a fixed lower column. This only lowers columns already edited by the
    # route, so it cannot enlarge the bank or cut an untouched hillside.
    for _ in range(passes):
        down = np.full(heights.shape, np.inf)
        for a, b in pairs:
            for high, low in ((a, b), (b, a)):
                allowed = max_drop if cut_drop is None else np.where(
                    heights[high] > natural[high], max_drop, cut_drop[high])
                drop = heights[high] - heights[low]
                old = natural[high] - natural[low]
                newly_steep = (drop > allowed) & (drop > old)
                high_fill = heights[high] > natural[high]
                np.minimum(
                    down[high],
                    np.where(newly_steep & movable[high] & (heights[high] != natural[high]),
                             np.where(high_fill, np.maximum(natural[high], heights[low] + allowed),
                                      heights[low] + allowed), np.inf),
                    out=down[high],
                )
        changed = np.where(movable, np.minimum(heights, down), heights)
        if np.array_equal(changed, heights):
            break
        heights = changed
    return np.where(heights != original, heights, designed)


def _bridge_ground(plan: Plan, natural: Field, known: Field) -> Field:
    """Blur existing ground with normalized weights and a bounded height change."""
    bridge = plan.bridge
    target = smoothing.blur(natural, known, bridge.radius)
    change = np.clip(bridge.strength * (target - natural), -bridge.max_change, bridge.max_change)
    behavior = plan.smoothing.profiles.get("bridge", plan.smoothing.behavior)
    mode = smoothing.Mode.STABLE if behavior == "auto" else smoothing.Mode[behavior.upper()]
    modes = np.full(natural.shape, mode, dtype=np.int8)
    candidate = smoothing.constrain(natural, natural + change, modes, plan.smoothing.stable_change)
    if plan.smoothing.preserve_cliffs:
        candidate = np.where(_natural_cliffs(natural, known), natural, candidate)
    return np.where(known, candidate, natural)


def _clear_growth(
    plan: Plan, surface, grid: Grid, changed: Field, clearance: Clearance,
    protected: Field | None = None,
    designed: Field | None = None,
    bridges: Field | None = None,
    bridge_footing: Field | None = None,
    distance: Field | None = None,
    bridge_crest: Field | None = None,
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
        if bridges is not None and bridges[row, cell]:
            # A bridge clears growth beneath its actual footprint and in its
            # deck's way, without sweeping the deck elevation out over the
            # surrounding hillside.
            beneath = (plan.bridge.clear_underneath and distance[row, cell] <= 0
                       and surface.height[row, cell] < position.y < bridge_footing[row, cell])
            deck = (bridge_crest is not None and distance[row, cell] <= DECK_EDGE
                    and bridge_footing[row, cell] <= position.y <= bridge_crest[row, cell] + DECK_ROOM)
            return beneath or deck or clearance.holds(row, cell, position.y)
        soil = surface.height[row, cell]
        lowest = soil
        if designed is not None:
            lowest = min(soil, designed[row, cell])
            soil = max(soil, designed[row, cell])
        return (bool(changed[row, cell]) and lowest < position.y <= soil + 2
                or clearance.holds(row, cell, position.y))

    cleared = growth_clearing(surface.growth, in_the_way, surface.trees, surface.trunks)
    return frozenset(
        position for position in cleared
        if not any(keepout.holds(position.x, position.z) for keepout in plan.keepouts)
        and not (protected is not None and grid.holds(position.x, position.z)
                 and protected[grid.at(position.x, position.z)])
    )
