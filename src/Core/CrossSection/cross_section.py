from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import NamedTuple

from ..coordinates import BlockPos, Vec2
from ..modes import StepMode
from .annotations import FeatureAnnotation
from .steps import STEP_RULES, StepRule, column_pushes, lead
from .turns import (
    CLOCKWISE,
    COUNTER_CLOCKWISE,
    Rotation,
    Turn,
    heading,
    is_quarter,
    neighbors,
    normalize,
    place_along,
    step_along,
)

# Section-local axes: x to the right of travel, y height above the path, z
# depth with forward at -z.
LocalPoint = BlockPos

# Mirroring across the section's long edge negates depth and everything
# measured along travel; local x is unchanged.
MIRRORED = -1
PLAIN = 1


def mirror_of(quarter: Turn) -> int:
    # The default lean when the heading gives none: odd quarters mirrored.
    return MIRRORED if (normalize(quarter) // 90) % 2 else PLAIN


def lean_of(quarter: Turn, travel: Vec2, previous: int | None = None) -> int:
    # Leans a quarter's staircase with the heading so the row stays square
    # to travel: plain when travel is clockwise of the quarter, mirrored when
    # counter-clockwise, and `previous` when dead on it.
    side = heading(quarter).cross(travel)
    if abs(side) <= 1e-9:
        return mirror_of(quarter) if previous is None else previous
    return PLAIN if side > 0 else MIRRORED


class PointField:
    # Named feature annotations in local coordinates.

    def __init__(self, annotations: Iterable[FeatureAnnotation] = ()) -> None:
        self._annotations: dict[str, FeatureAnnotation] = {}
        self._revision = 0
        for annotation in annotations:
            self.add(annotation)

    @property
    def revision(self) -> int:
        return self._revision

    def add(self, annotation: FeatureAnnotation) -> None:
        if annotation.name in self._annotations:
            raise ValueError(f"Annotation name {annotation.name!r} is already used")
        self._annotations[annotation.name] = annotation
        self._revision += 1

    def get(self, name: str) -> FeatureAnnotation:
        return self._annotations[name]

    def __contains__(self, name: object) -> bool:
        return name in self._annotations

    def __iter__(self) -> Iterator[FeatureAnnotation]:
        return iter(tuple(self._annotations.values()))

    def __len__(self) -> int:
        return len(self._annotations)


class SectionOffset(NamedTuple):
    # Where one annotation delivers relative to a path point, in path order:
    # x and z in world axes, y as height above the path. `layout` and
    # `mirrored` say which of the layouts placed them.
    annotation: FeatureAnnotation
    offsets: tuple[BlockPos, ...]
    layout: Turn = 0
    mirrored: bool = False


OrientedSection = tuple[SectionOffset, ...]


def orient_annotation(
    annotation: FeatureAnnotation,
    origin: LocalPoint,
    turn: Turn,
    lead: int,
    steps: tuple[int, ...] = (0,),
    shift: tuple[int, int] = (0, 0),
    depth: int = 1,
) -> SectionOffset:
    # One annotation in a layout. `lead` (how far its section pushes it) and
    # `steps` are blocks forward; `shift` is added in world axes. A `depth` of
    # -1 mirrors across the long edge.
    turn = normalize(turn)
    lx, ly, lz = annotation.position - origin
    fx, fz = step_along(turn)
    x, z = place_along(turn, lx, depth * lz)
    x += shift[0]
    z += shift[1]
    return SectionOffset(
        annotation,
        tuple(
            BlockPos(
                x + fx * depth * (lead + step), ly, z + fz * depth * (lead + step)
            )
            for step in steps
        ),
        turn,
        depth < 0,
    )


class _Profile(NamedTuple):
    # Per-revision summary of a section's points. `entries` is (annotation,
    # steps, lead); `pushes` is how far each column moves those outboard of it.
    entries: tuple[tuple[FeatureAnnotation, tuple[int, ...], int], ...]
    right: int
    long_length: int
    depth: int
    pushes: Mapping[int, int] = MappingProxyType({})
    left: int = 0


def _profile(
    points: PointField,
    origin: LocalPoint,
    rules: Mapping[StepMode, StepRule],
) -> _Profile:
    annotations = tuple(points)
    if not annotations:
        return _Profile((), 0, 0, 0)
    first = annotations[0].position
    min_x = max_x = first.x
    min_z = max_z = first.z
    stepped = []
    for annotation in annotations:
        position = annotation.position
        column = position.x - origin.x
        stepped.append((annotation, column, rules[annotation.step_mode](column)))
        left, right_, back, front = _covered(annotation)
        min_x, max_x = min(min_x, left), max(max_x, right_)
        min_z, max_z = min(min_z, back), max(max_z, front)
    pushes = column_pushes((column, steps) for _, column, steps in stepped)
    return _Profile(
        entries=tuple(
            (annotation, steps, lead(column, pushes))
            for annotation, column, steps in stepped
        ),
        right=max_x - origin.x,
        left=min_x - origin.x,
        long_length=max(max_x - min_x, max_z - min_z) + 1,
        depth=max_z - min_z + 1,
        pushes=MappingProxyType(pushes),
    )


def _covered(annotation: FeatureAnnotation) -> tuple[int, int, int, int]:
    # (left, right, back, front) of the annotation widened by its feature's
    # blocks.
    position = annotation.position
    blocks = getattr(annotation.feature, "blocks", None)
    bounds = None if blocks is None else blocks.bounds
    if bounds is None:
        return position.x, position.x, position.z, position.z
    return (
        position.x + bounds.minimum.x,
        position.x + bounds.maximum.x,
        position.z + bounds.minimum.z,
        position.z + bounds.maximum.z,
    )


def _in_travel_order(
    section: OrientedSection, ahead: tuple[int, int]
) -> OrientedSection:
    # Each annotation's positions sorted along `ahead`.
    return tuple(
        SectionOffset(
            offset.annotation,
            tuple(sorted(
                offset.offsets, key=lambda p: p.x * ahead[0] + p.z * ahead[1]
            )),
            offset.layout,
            offset.mirrored,
        )
        for offset in section
    )


def _centred(section: OrientedSection) -> OrientedSection:
    # Shifts a join so the path point is at its middle.
    places = [(p.x, p.z) for offset in section for p in offset.offsets]
    if not places:
        return section
    middle_x = (max(x for x, _ in places) + min(x for x, _ in places)) // 2
    middle_z = (max(z for _, z in places) + min(z for _, z in places)) // 2
    if not (middle_x or middle_z):
        return section
    return _repeated(section, ((-middle_x, -middle_z),))


def _repeated(section: OrientedSection, shifts: Sequence[tuple[int, int]]) -> OrientedSection:
    # The layout laid at every shift, one offset per annotation, shift-major.
    return tuple(
        SectionOffset(
            offset.annotation,
            tuple(dict.fromkeys(
                BlockPos(position.x + x, position.y, position.z + z)
                for x, z in shifts
                for position in offset.offsets
            )),
            offset.layout,
            offset.mirrored,
        )
        for offset in section
    )


def _onto(
    section: OrientedSection,
    other: OrientedSection,
    origin: LocalPoint,
    edge: int,
) -> tuple[int, int]:
    # The shift that puts `other`'s column `edge` on `section`'s.
    here = _column(section, origin, edge)
    there = _column(other, origin, edge)
    return here.x - there.x, here.z - there.z


def _column(section: OrientedSection, origin: LocalPoint, edge: int) -> BlockPos:
    for offset in section:
        if offset.annotation.position.x - origin.x == edge:
            return offset.offsets[0]
    raise ValueError(f"The section has no point in column {edge}")


class _OrientationCache:
    # Layouts keyed by (turn, direction, lean).
    __slots__ = ("layouts", "profile", "revision")

    def __init__(self) -> None:
        self.layouts: dict[tuple[Turn, Rotation, int | None], OrientedSection] = {}
        self.profile: _Profile | None = None
        self.revision: int | None = None


@dataclass(frozen=True, slots=True)
class CrossSection:
    # Where each feature sits across the path; blocks belong to the features.

    points: PointField = field(default_factory=PointField)
    # The local point on the path centreline.
    origin: LocalPoint = BlockPos(0, 0, 0)
    step_rules: Mapping[StepMode, StepRule] = field(default_factory=lambda: STEP_RULES)
    _orientations: _OrientationCache = field(
        default_factory=_OrientationCache,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", BlockPos.coerce(self.origin))

    @property
    def depth(self) -> int:
        return self._current().profile.depth

    @property
    def long_length(self) -> int:
        # The widest horizontal extent.
        return self._current().profile.long_length

    def step_mode_at(self, column: int) -> StepMode:
        # MANHATTAN if any point in the column steps.
        stepping = any(
            annotation.step_mode is StepMode.MANHATTAN
            for annotation, _, _ in self._current().profile.entries
            if annotation.position.x - self.origin.x == column
        )
        return StepMode.MANHATTAN if stepping else StepMode.CHEBYSHEV

    def along(
        self, column: int, step_mode: StepMode = StepMode.CHEBYSHEV
    ) -> tuple[int, ...]:
        # Blocks forward, sorted back to front, where the section lays its own
        # column at `column`: the push of stepping columns inboard of it plus
        # its own steps. Fittings use this to ride a column.
        profile = self._current().profile
        carried = lead(column, profile.pushes)
        return tuple(
            sorted(carried + step for step in self.step_rules[step_mode](column))
        )

    def reach(self, turn: Turn | None = None) -> float:
        # The furthest any oriented position lies along travel from the path
        # point, at `turn` or over every turn.
        turns = range(0, 360, 45) if turn is None else (turn,)
        furthest = 0.0
        for each in turns:
            direction = heading(each)
            for offset in (o for laid in self.layouts(each) for o in laid):
                for position in offset.offsets:
                    along = position.x * direction.x + position.z * direction.z
                    furthest = max(furthest, float(abs(along)))
        return furthest

    def orient(
        self,
        turn: Turn,
        direction: Rotation = CLOCKWISE,
        lean: int | None = None,
    ) -> OrientedSection:
        # `turn` counted `direction` from the authored heading. `lean` (see
        # `lean_of`) applies to quarters; a diagonal sets its arms' own.
        turn = normalize(turn)
        key = (turn, direction, lean)
        cache = self._current()
        oriented = cache.layouts.get(key)
        if oriented is None:
            oriented = self._orient(normalize(turn * direction), direction, lean, cache.profile)
            cache.layouts[key] = oriented
        return oriented

    def layouts(self, turn: Turn) -> Iterator[OrientedSection]:
        # Every direction and lean a turn can be laid out with.
        for direction in (CLOCKWISE, COUNTER_CLOCKWISE):
            for lean in (PLAIN, MIRRORED):
                yield self.orient(normalize(turn * direction), direction, lean)

    def _current(self) -> _OrientationCache:
        cache = self._orientations
        if cache.profile is None or cache.revision != self.points.revision:
            cache.layouts.clear()
            cache.profile = _profile(self.points, self.origin, self.step_rules)
            cache.revision = self.points.revision
        return cache

    def _orient(
        self,
        travelling: Turn,
        direction: Rotation,
        lean: int | None,
        profile: _Profile,
    ) -> OrientedSection:
        if not profile.entries:
            return ()
        if is_quarter(travelling):
            return self._layout(
                travelling,
                profile,
                (0, 0),
                mirror_of(travelling) if lean is None else lean,
            )
        return self._joined(travelling, profile, direction)

    def _joined(
        self, diagonal: Turn, profile: _Profile, direction: Rotation = CLOCKWISE
    ) -> OrientedSection:
        # A diagonal is one join piece: the quarter being left, repeated
        # forward, and the quarter being taken up, repeated backward, crossing
        # each other so neither leaves a wedge on the inside of the turn. The
        # lower quarter is laid plain and the upper mirrored, lined up on the
        # inside edge (right when turning clockwise, else left).
        lower, upper = neighbors(diagonal)
        forward_x, forward_z = step_along(diagonal)
        clockwise = direction == CLOCKWISE
        leaving, taking = (lower, upper) if clockwise else (upper, lower)
        leaving_lean, taking_lean = (PLAIN, MIRRORED) if clockwise else (MIRRORED, PLAIN)
        inside = profile.right if clockwise else profile.left
        half = max(1, profile.long_length // 2)
        taking_up = self._layout(taking, profile, (0, 0), taking_lean)
        being_left = self._layout(leaving, profile, (0, 0), leaving_lean)
        shift = _onto(being_left, taking_up, self.origin, inside)
        ahead = tuple((forward_x * k, forward_z * k) for k in range(half))
        behind = tuple(
            (shift[0] - forward_x * k, shift[1] - forward_z * k)
            for k in reversed(range(half))
        )
        return _centred(_in_travel_order(
            _repeated(being_left, ahead) + _repeated(taking_up, behind),
            (forward_x, forward_z),
        ))

    def _layout(
        self,
        turn: Turn,
        profile: _Profile,
        shift: tuple[int, int],
        depth: int = 1,
    ) -> OrientedSection:
        return tuple(
            orient_annotation(
                annotation, self.origin, turn, lead, steps, shift, depth
            )
            for annotation, steps, lead in profile.entries
        )
